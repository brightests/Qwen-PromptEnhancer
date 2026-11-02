from __future__ import annotations

import atexit
import gc
import logging
import re
import threading
import time
from pathlib import Path

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .media import (
    image_content,
    image_grid_content,
    sample_indices_per_second,
    text_content,
)
from .agent import run_agent_loop
from .runtime import LlamaServerManager
from .skills import (
    _skill_path,
    discover_skill_registry,
    parse_skill_selection,
    resolve_skill,
    router_prompt,
    skill_ids,
    system_prompt,
)

MODEL_DIR = Path(folder_paths.models_dir) / "LLM"
LLM_FOLDER = "LLM"
DEFAULT_MODEL = "Qwen3.8/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
DEFAULT_MMPROJ = "Qwen3.8/Qwen3.8-27B-mmproj-BF16.gguf"
# Sentinel dropdown value: run the GGUF as a text-only model without a projector.
NO_MMPROJ = "none"
# Sentinel dropdown value: answer the prompt directly without loading any Skill.
NO_SKILL = "none"
NO_SKILL_SYSTEM_PROMPT = (
    "You are a local assistant running inside ComfyUI. Answer the user's request directly, "
    "following any instructions embedded in the request itself. Produce exactly the deliverable "
    "asked for, with no wrappers, prefaces, or output schemas that neither the user nor the "
    "request asks for. There is no network access: output text only."
)
REFERENCE_VIDEO_FPS = 24.0
INFERENCE_LOCK = threading.Lock()
SERVER_MANAGER = LlamaServerManager()
LOGGER = logging.getLogger("ComfyUI.QwenPromptEnhancer")
# Routing budget for the short Skill-selection call; aligned with upstream 0.1.7.
ROUTING_MAX_TOKENS = 64

# Added rather than overwritten so the roots of other llama.cpp packs stay usable,
# and so folders declared in extra_model_paths.yaml show up in the dropdowns.
folder_paths.add_model_folder_path(LLM_FOLDER, str(MODEL_DIR))
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

atexit.register(SERVER_MANAGER.release)


def _success(message: str) -> None:
    LOGGER.info("%s%s%s", GREEN, message, RESET)


def _failure(message: str) -> None:
    LOGGER.error("%s%s%s", RED, message, RESET)


def _usage_summary(usage: dict[str, int]) -> str:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        return "token usage unavailable"
    return (
        f"prompt {prompt_tokens or 0}, completion {completion_tokens or 0}, "
        f"total {total_tokens} tokens"
    )


def _release_node_resources() -> None:
    SERVER_MANAGER.release()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)


def _llm_gguf_names() -> list[str]:
    """GGUF names from every registered ``LLM`` folder, rescanned on each call."""
    names = folder_paths.get_filename_list(LLM_FOLDER)
    return [name for name in names if name.lower().endswith(".gguf")]


def _is_mmproj(name: str) -> bool:
    return "mmproj" in Path(name).name.lower()


def _resolve_model(name: str) -> Path:
    """Resolve a dropdown value through folder_paths so every LLM root is searched.

    Ownership and existence are left to folder_paths: it rejects ``..`` escapes and
    reports a file that is present, which keeps symlinked or oddly named payloads
    usable while dangling links still fail.
    """
    if not name or name == NO_MMPROJ:
        raise FileNotFoundError(f"No GGUF model was selected: {name!r}")
    full = folder_paths.get_full_path(LLM_FOLDER, name)
    if not full or Path(full).suffix.lower() != ".gguf":
        raise FileNotFoundError(f"Invalid or missing GGUF model in the {LLM_FOLDER} folders: {name}")
    return Path(full).resolve()


def _autogrow_values(inputs) -> list:
    if not inputs:
        return []

    def index(item):
        match = re.search(r"(\d+)$", item[0])
        return int(match.group(1)) if match else 0

    return [value for _, value in sorted(inputs.items(), key=index) if value is not None]


def _validate_reference_images(images: list) -> None:
    if len(images) > 9:
        raise ValueError(f"At most 9 reference images are supported; got {len(images)}.")
    for image_index, image in enumerate(images, 1):
        image_count = int(image.shape[0])
        if image_count != 1:
            raise ValueError(
                f"Reference image {image_index} must contain exactly one image; got a batch of {image_count}."
            )


def _reference_video_details(
    videos: list,
    sample_frames_per_second: int,
) -> list[tuple]:
    if len(videos) > 3:
        raise ValueError(f"At most 3 reference videos are supported; got {len(videos)}.")
    details = []
    total_duration = 0.0
    for video_index, frames in enumerate(videos, 1):
        frame_count = int(frames.shape[0])
        if frame_count < 1:
            raise ValueError(
                f"Reference video {video_index} must contain at least 1 frame; got {frame_count}."
            )
        source_duration = frame_count / REFERENCE_VIDEO_FPS
        if source_duration > 15.0:
            raise ValueError(
                f"Reference video {video_index} must be at most 15 seconds at 24 fps; "
                f"got {source_duration:.2f}s."
            )
        total_duration += source_duration
        details.append(
            (
                frames,
                frame_count,
                source_duration,
                sample_indices_per_second(
                    frame_count,
                    REFERENCE_VIDEO_FPS,
                    sample_frames_per_second,
                ),
            )
        )
    if total_duration > 15.0:
        raise ValueError(
            f"Reference videos may total at most 15 seconds; got {total_duration:.2f}s."
        )
    return details


def _sampling_settings(think_mode: bool) -> dict[str, float | int]:
    if think_mode:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }


def _user_content(
    prompt: str,
    images: list,
    video_details: list[tuple],
    duration: float = 0.0,
) -> tuple[list[dict[str, object]], str]:
    request = f"User request:\n{prompt}"
    if duration > 0.0:
        # Same injection format as the upstream H3 node, so video/sound skills pace
        # the generated prompt toward the target clip length. 0 keeps images clean.
        request += f"\n\nTarget duration: {duration:.2f} seconds."
    content: list[dict[str, object]] = [text_content(request)]
    assets = []
    for picture_index, image in enumerate(images, 1):
        content.append(text_content(f"<Picture {picture_index}>: connected reference image."))
        content.append(image_content(image))
        assets.append(f"Picture {picture_index}")

    for video_index, details in enumerate(video_details, 1):
        frames, frame_count, source_duration, sample_groups = details
        sampled_frame_count = sum(len(indices) for indices in sample_groups)
        content.append(
            text_content(
                f"<Video {video_index}>: {frame_count} ordered frames at {REFERENCE_VIDEO_FPS:.3f} fps "
                f"({source_duration:.2f} seconds), represented by {sampled_frame_count} sampled frames "
                f"grouped into {len(sample_groups)} chronological one-second contact sheets."
            )
        )
        for second_index, indices in enumerate(sample_groups):
            timestamps = ", ".join(
                f"{frame_index / REFERENCE_VIDEO_FPS:.3f}s" for frame_index in indices
            )
            content.append(
                text_content(
                    f"<Video {video_index}> second {second_index + 1}/{len(sample_groups)} contact sheet. "
                    f"Read cells in row-major chronological order at: {timestamps}."
                )
            )
            content.append(image_grid_content(frames, indices))
        assets.append(
            f"Video {video_index}={source_duration:.2f}s/{sampled_frame_count} sampled frames"
        )
    return content, ", ".join(assets) if assets else "none"


class QwenPromptEnhancer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = _llm_gguf_names()
        projectors = [name for name in files if _is_mmproj(name)]
        models = [name for name in files if not _is_mmproj(name)] or [DEFAULT_MODEL]
        # Rescanned on every schema build: ComfyUI re-runs define_schema for each
        # /object_info request, so adding or renaming a Skill only needs the
        # frontend's "Refresh Node Definitions" (or a page reload) -- no restart.
        available_skills = skill_ids()
        return io.Schema(
            node_id="QwenPromptEnhancerLocal",
            display_name="Qwen Prompt Enhancer · 本地提示词",
            category="😺Bright/Qwen_PromptEnhancer",
            description="在本地 llama.cpp 服务上运行所选 GGUF 模型（默认 Qwen3.8），并借助本地 function calling 工具，"
            "让模型以多步 agent 方式执行任意已发现的 Skill。（by Bright）",
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="Describe the task.",
                ),
                io.Combo.Input(
                    "skill",
                    options=[NO_SKILL, "auto", *available_skills],
                    default="auto",
                    tooltip="内置 skills/ 与 custom_skills/ 目录中的 Skill。选 'auto' 时由本地模型自行挑选，"
                    "范围包括 skill_root 下发现的 Skill；选 'none' 时不加载 Skill，直接按提示词回答。",
                ),
                io.String.Input(
                    "skill_root",
                    default="",
                    advanced=True,
                    tooltip="可选：额外扫描的 Skill 根目录，例如其他节点的 skills/ 目录。其中的 Skill 会加入 "
                    "'auto' 自动选择的候选池；若想让某个 Skill 出现在 skill 下拉框中，请把它放进 custom_skills/。",
                ),
                io.Combo.Input(
                    "llm_model",
                    options=models,
                    default=DEFAULT_MODEL if DEFAULT_MODEL in models else models[0],
                    tooltip="models/LLM 及所有已注册 LLM 目录（含 extra_model_paths.yaml）中的 GGUF 语言模型。"
                    "任何支持工具调用的模型都能替换 Qwen3.8。",
                ),
                io.Combo.Input(
                    "vision_model",
                    options=[*projectors, NO_MMPROJ],
                    default=(
                        DEFAULT_MMPROJ
                        if DEFAULT_MMPROJ in projectors
                        else projectors[0] if projectors else NO_MMPROJ
                    ),
                    tooltip="GGUF 多模态投影器（mmproj），与语言模型放在同一目录。选 "
                    + NO_MMPROJ
                    + " 时按纯文本模型运行：启动参数不再传 --mmproj，也不能使用参考图片/参考视频。",
                ),
                io.Boolean.Input(
                    "think_mode",
                    display_name="think_mode",
                    default=False,
                    tooltip="关闭时使用 instruct 采样参数；开启后启用思考模式并使用思考采样参数，"
                    "两套都取自 Qwen 官方推荐值。只在所选模型的 chat template 含 enable_thinking 时生效，"
                    "否则自动按 instruct 参数运行。",
                ),
                io.Combo.Input(
                    "reasoning_effort",
                    options=["low", "medium", "xhigh"],
                    default="medium",
                    tooltip="仅在思考模式下生效，且只在模型模板支持 reasoning_effort 时才随请求下发。"
                    "medium 为 RTX 3090 上的均衡预设。",
                ),
                io.Boolean.Input(
                    "output_json",
                    display_name="json output",
                    default=False,
                    tooltip="开启后模型会额外给出与结果等价的结构化 JSON，从『结果 JSON』输出端口返回；"
                    "关闭时该端口输出空字符串。适合下游需要按字段解析提示词的节点。",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="ComfyUI 随机种子，会确定性地映射到 llama.cpp 的 32 位种子范围。",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=8192,
                    min=256,
                    max=16384,
                    step=128,
                    tooltip="每个 agent 步的最大生成 token 数（开启思考模式时包含思考内容）。"
                    "每步输出会随对话历史重新占用上下文窗口，调大时注意 context_size 是否足够。",
                ),
                io.Int.Input(
                    "context_size",
                    default=32768,
                    min=1024,
                    max=327680,
                    step=256,
                    advanced=True,
                    tooltip="llama.cpp 上下文窗口（--ctx-size），提示词、多步对话历史与输出共享，默认 32768。"
                    "调大会增加 KV cache 显存占用；上下文不足时请求会被 llama.cpp 拒绝"
                    "（HTTP 400，报错显示在 ComfyUI 控制台），需调大本参数或减少参考图片与视频时长。",
                ),
                io.Int.Input(
                    "video_sample_frames_per_sec",
                    default=2,
                    min=1,
                    max=8,
                    step=1,
                    advanced=True,
                    tooltip="每个参考视频每秒采样的帧数。同一秒内的帧会合并为一张按时间排序的拼图（contact sheet）一并发送。",
                ),
                io.Float.Input(
                    "duration",
                    display_name="prompt duration",
                    default=0.0,
                    min=0.0,
                    max=30.0,
                    step=0.5,
                    advanced=True,
                    tooltip="目标成片时长（秒）。0 表示不注入，图片类输出无需时长；"
                    "大于 0 时会在用户请求中追加目标时长，供视频/声音类 Skill 安排动作与声音节奏。"
                    "参考视频 15 秒的时长上限与此参数无关。",
                ),
                io.Boolean.Input(
                    "force_unload_model",
                    display_name="force unload model",
                    default=True,
                    tooltip="开启时每次运行结束后都会停止节点的 llama.cpp 服务并释放其显存与内存。运行出错时始终强制卸载。",
                ),
                io.Int.Input(
                    "max_agent_steps",
                    display_name="max agent steps",
                    default=6,
                    min=1,
                    max=12,
                    step=1,
                    advanced=True,
                    tooltip="agent 循环中工具调用轮数的上限。",
                ),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_image",
                            tooltip="一张参考图。需要时请在 prompt 中说明它的角色与用途。",
                        ),
                        prefix="reference_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "reference_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_video",
                            tooltip="以 IMAGE 批次提供的、按顺序排列的视频帧，兼容 VHS Load Video。",
                        ),
                        prefix="reference_video_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[
                io.String.Output("result"),
                io.String.Output("result_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        skill,
        skill_root,
        llm_model,
        vision_model,
        think_mode,
        reasoning_effort,
        seed,
        max_tokens,
        context_size,
        video_sample_frames_per_sec,
        duration,
        output_json,
        force_unload_model,
        max_agent_steps,
        reference_images=None,
        reference_videos=None,
    ) -> io.NodeOutput:
        total_started = time.perf_counter()
        content = None
        messages = None
        LOGGER.info(
            "[Qwen PromptEnhancer] Node execution started | skill=%s | skill_root=%s | think_mode=%s | seed=%d | force_unload_model=%s",
            skill,
            skill_root,
            think_mode,
            seed,
            force_unload_model,
        )
        try:
            if not prompt.strip():
                raise ValueError("prompt must not be empty")

            stage_started = time.perf_counter()
            LOGGER.info("[Qwen PromptEnhancer] Preparing multimodal input")
            images = _autogrow_values(reference_images)
            videos = _autogrow_values(reference_videos)
            _validate_reference_images(images)
            video_details = _reference_video_details(
                videos,
                video_sample_frames_per_sec,
            )
            model = _resolve_model(llm_model)
            projector = None if vision_model == NO_MMPROJ else _resolve_model(vision_model)
            if projector is None and (images or video_details):
                raise ValueError(
                    "参考图片/参考视频需要多模态投影器：请把 vision_model 指向 mmproj 文件，或移除参考素材。"
                )

            extra_root = str(skill_root or "").strip()
            registry = discover_skill_registry(
                extra_roots=(Path(extra_root),) if extra_root else (),
            )
            LOGGER.info(
                "[Qwen PromptEnhancer] Skill registry ready | extra_root=%s | skills=%s | elapsed %.2f s",
                extra_root or "none",
                ", ".join(spec.id for spec in registry) or "none",
                time.perf_counter() - stage_started,
            )
            selected = str(skill or "").strip() or "auto"
            if not registry and selected != NO_SKILL:
                raise ValueError(
                    "No Skills were found. Add a SKILL.md under skills/ or custom_skills/, "
                    "or point skill_root at a folder that contains Skill subfolders."
                )

            with INFERENCE_LOCK:
                stage_started = time.perf_counter()
                LOGGER.info("[Qwen PromptEnhancer] Unloading ComfyUI models for Qwen")
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                LOGGER.info(
                    "[Qwen PromptEnhancer] ComfyUI model cleanup complete | elapsed %.2f s",
                    time.perf_counter() - stage_started,
                )

                stage_started = time.perf_counter()
                LOGGER.info(
                    "[Qwen PromptEnhancer] Model loading started | model=%s | mmproj=%s | ctx=%d",
                    model.name,
                    projector.name if projector is not None else "none (text-only)",
                    context_size,
                )
                server, loaded_new = SERVER_MANAGER.acquire(model, projector, context_size=context_size)
                LOGGER.info(
                    "[Qwen PromptEnhancer] Model loading complete | %s | platform=%s | backend=%s | port=%d | elapsed %.2f s",
                    "new model loaded" if loaded_new else "resident model reused",
                    server.runtime_spec.platform_name,
                    server.backend,
                    server.port,
                    time.perf_counter() - stage_started,
                )

                # Template options follow the loaded model: only its own chat template
                # decides whether think mode, reasoning effort and tool calling exist.
                abilities = server.capabilities
                if think_mode and not abilities["thinking"]:
                    LOGGER.warning(
                        "[Qwen PromptEnhancer] The loaded model's chat template has no enable_thinking "
                        "switch; think_mode is ignored and instruct sampling is used"
                    )
                if not abilities["tools"]:
                    LOGGER.warning(
                        "[Qwen PromptEnhancer] The loaded model's chat template has no tool support; "
                        "the agent's function calling will most likely fail"
                    )
                effective_think = bool(think_mode) and bool(abilities["thinking"])
                settings = _sampling_settings(effective_think)
                LOGGER.info(
                    "[Qwen PromptEnhancer] Sampling configuration selected | %s | temperature=%.2f | top_p=%.2f | top_k=%d | min_p=%.2f | presence_penalty=%.2f | repetition_penalty=%.2f",
                    "thinking" if effective_think else "instruct",
                    settings["temperature"],
                    settings["top_p"],
                    settings["top_k"],
                    settings["min_p"],
                    settings["presence_penalty"],
                    settings["repetition_penalty"],
                )

                stage_started = time.perf_counter()
                content, asset_summary = _user_content(
                    prompt,
                    images,
                    video_details,
                    duration=max(float(duration or 0.0), 0.0),
                )
                LOGGER.info(
                    "[Qwen PromptEnhancer] Multimodal input ready | assets=%s | elapsed %.2f s",
                    asset_summary,
                    time.perf_counter() - stage_started,
                )

                if selected == "auto":
                    stage_started = time.perf_counter()
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Starting automatic Skill routing | candidates=%d",
                        len(registry),
                    )
                    # The enum pins the decoder to the exact candidate ids, so the
                    # routing call can no longer produce an unparseable selection.
                    routing_schema = {
                        "type": "object",
                        "properties": {
                            "skill": {
                                "type": "string",
                                "enum": [spec.id for spec in registry],
                            }
                        },
                        "required": ["skill"],
                        "additionalProperties": False,
                    }
                    selection, routing_usage = server.chat(
                        router_prompt(prompt, asset_summary, registry),
                        seed=seed,
                        max_tokens=ROUTING_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        min_p=0.0,
                        presence_penalty=0.0,
                        repetition_penalty=1.0,
                        think_mode=False,
                        reasoning_effort="low",
                        schema=routing_schema,
                    )
                    selected = parse_skill_selection(selection, registry)
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Skill routing complete | selected=%s | %s | elapsed %.2f s",
                        selected,
                        _usage_summary(routing_usage),
                        time.perf_counter() - stage_started,
                    )
                elif selected == NO_SKILL:
                    # "none" is a valid manual choice: skip Skill resolution and fall
                    # through to the direct-chat branch below.
                    pass
                else:
                    selected = resolve_skill(selected, registry).id
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Skill selection resolved | selected=%s (manual)",
                        selected,
                    )

                if selected == NO_SKILL:
                    messages = [
                        {"role": "system", "content": NO_SKILL_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ]
                    stage_started = time.perf_counter()
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Direct chat started | skill=none | max_tokens=%d",
                        max_tokens,
                    )
                    # No Skill tools exist in this mode, so a single direct chat replaces
                    # the tool-calling loop -- models without function calling still work.
                    result, agent_usage = server.chat(
                        messages,
                        seed=seed,
                        max_tokens=max_tokens,
                        **settings,
                        think_mode=effective_think,
                        reasoning_effort=reasoning_effort,
                    )
                    agent_steps = 1
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Direct chat complete | %s | elapsed %.2f s",
                        _usage_summary(agent_usage),
                        time.perf_counter() - stage_started,
                    )
                    # No tools exist in this mode either, so JSON output is produced by
                    # one extra conversion turn inside the same conversation.
                    result_json = ""
                    if output_json:
                        messages.append({"role": "assistant", "content": result})
                        messages.append(
                            {
                                "role": "user",
                                "content": "Convert your previous answer into ONE JSON object that preserves all of "
                                "its information (meaningful field names; arrays or nested objects where "
                                "appropriate). Reply with only the JSON object, no markdown fences and no commentary.",
                            }
                        )
                        # The schema grammar constrains the whole completion, which would
                        # collide with thinking tokens, and conversion is mechanical work
                        # that never needs reasoning -- so this turn always runs instruct.
                        convert_settings = _sampling_settings(False)
                        convert_options = {
                            "seed": seed,
                            "max_tokens": max_tokens,
                            **convert_settings,
                            "think_mode": False,
                            "reasoning_effort": "low",
                            "schema": {"type": "object"},
                        }
                        result_json, json_usage = server.chat(messages, **convert_options)
                        try:
                            json.loads(result_json)
                        except json.JSONDecodeError as error:
                            # Only reachable when llama.cpp rejected the schema and the
                            # request degraded to free-form generation: feed the parse
                            # error back and repair once, as the last safety net.
                            LOGGER.warning(
                                "[Qwen PromptEnhancer] JSON conversion failed validation (%s); repairing once",
                                error,
                            )
                            messages.extend(
                                [
                                    {"role": "assistant", "content": result_json},
                                    {
                                        "role": "user",
                                        "content": "That was not valid JSON "
                                        f"({error}). Return ONE corrected JSON object that "
                                        "preserves all information from your previous answer. "
                                        "Reply with only the JSON object, no markdown fences "
                                        "and no commentary.",
                                    },
                                ]
                            )
                            result_json, repair_usage = server.chat(messages, **convert_options)
                            json_usage = {
                                key: (json_usage.get(key, 0) + repair_usage.get(key, 0))
                                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                            }
                            try:
                                json.loads(result_json)
                            except json.JSONDecodeError as final_error:
                                raise RuntimeError(
                                    "JSON 输出在自动修复后仍不是合法 JSON："
                                    f"{final_error}\n模型输出：{result_json[:2000]}"
                                ) from final_error
                        LOGGER.info(
                            "[Qwen PromptEnhancer] JSON conversion complete | %s",
                            _usage_summary(json_usage),
                        )
                else:
                    skill_path = _skill_path(selected, registry)
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt(selected, registry),
                        },
                        {"role": "user", "content": content},
                    ]
                    stage_started = time.perf_counter()
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Agent loop started | skill=%s | max_steps=%d | max_tokens=%d",
                        selected,
                        max_agent_steps,
                        max_tokens,
                    )
                    result, result_json, agent_steps, agent_usage = run_agent_loop(
                        server,
                        messages,
                        skill_root=skill_path,
                        settings=settings,
                        seed=seed,
                        max_tokens=max_tokens,
                        # effective_think was resolved from the loaded model's template above,
                        # so a model without enable_thinking never gets thinking sampling.
                        think_mode=effective_think,
                        reasoning_effort=reasoning_effort,
                        max_steps=max_agent_steps,
                        json_output=bool(output_json),
                    )
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Agent loop complete | skill=%s | steps=%d | %s | elapsed %.2f s",
                        selected,
                        agent_steps,
                        _usage_summary(agent_usage),
                        time.perf_counter() - stage_started,
                    )

                if output_json and not result_json.strip():
                    LOGGER.warning(
                        "[Qwen PromptEnhancer] JSON output was requested but the model did not "
                        "provide a JSON payload; the 'result_json' output stays empty"
                    )

                content = None
                messages = None
                if force_unload_model:
                    stage_started = time.perf_counter()
                    LOGGER.info("[Qwen PromptEnhancer] Force-releasing node model, VRAM, and system memory")
                    _release_node_resources()
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Forced resource release complete | elapsed %.2f s",
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[Qwen PromptEnhancer] Model unload disabled; the bundled llama.cpp model will remain resident for reuse"
                    )

            _success(
                f"[Qwen PromptEnhancer] Execution completed successfully | skill={selected} | steps={agent_steps} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            return io.NodeOutput(result, result_json)
        except Exception as error:
            content = None
            messages = None
            try:
                with INFERENCE_LOCK:
                    _release_node_resources()
            except Exception as cleanup_error:  # noqa: BLE001
                _failure(
                    f"[Qwen PromptEnhancer] Cleanup after failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            _failure(
                f"[Qwen PromptEnhancer] Execution failed | {type(error).__name__}: {error} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            raise


class QwenPromptEnhancerExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenPromptEnhancer]


async def comfy_entrypoint() -> QwenPromptEnhancerExtension:
    return QwenPromptEnhancerExtension()
