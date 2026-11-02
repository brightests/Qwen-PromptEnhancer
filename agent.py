"""Bounded tool-calling agent loop for multi-step local Skill execution."""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import comfy.model_management

LOGGER = logging.getLogger("ComfyUI.QwenPromptEnhancer.tools")

AGENT_DEADLINE_SECONDS = 3600.0
AGENT_TOKEN_BUDGET = 65536
MAX_TOOL_FILE_BYTES = 200_000
SCRIPT_TIMEOUT_SECONDS = 60.0
MAX_SCRIPT_OUTPUT_BYTES = 100_000

AGENT_ADDENDUM = """

# Agent mode: local tools

You are running as a multi-step agent and may call the provided local tools:

- `list_skill_files`: enumerate the documents bundled with the selected Skill.
- `read_skill_reference`: read one bundled Skill document by its relative path.
- `run_skill_script`: run one Python script bundled with the selected Skill
  (only files under the Skill's `scripts/` directory; offline, sandboxed).
- `finish`: return the complete final deliverable and end the run.

Read the Skill documents you need, follow the selected Skill's process step by step, then call `finish`
exactly once with the complete final answer. Tools are local and offline; there is no network access.
"""

AGENT_ADDENDUM_JSON = """

# JSON output mode

JSON output is enabled: call `finish` with BOTH arguments.

- `answer`: the complete final deliverable, verbatim.
- `json`: the same deliverable re-expressed as ONE JSON object (meaningful field names,
  arrays or nested objects where appropriate). The value must be a JSON object, not prose.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_skill_files",
            "description": "List the documents bundled with the selected Skill (SKILL.md and references).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill_reference",
            "description": "Read one text document bundled with the selected Skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Relative path inside the Skill directory, for example references/shot-schema.md",
                    }
                },
                "required": ["file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_skill_script",
            "description": "Run one Python script bundled with the selected Skill (must live under the "
            "Skill's scripts/ directory). The script runs offline in a temporary working directory. "
            "For example, moss-image-prompt-enhancer exposes scripts/compile_prompt.py, which compiles "
            "a Layout draft (or a plain caption) into the final MOSS prompt. Pass your complete draft "
            "via input_json, then call finish with the compiled result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Script path relative to the Skill's scripts/ directory, e.g. compile_prompt.py",
                    },
                    "input_json": {
                        "type": "object",
                        "description": "A JSON payload written to input.json and passed to the script as its "
                        "first positional argument. Provide exactly one of input_json / input_text.",
                    },
                    "input_text": {
                        "type": "string",
                        "description": "A plain-text payload written to input.txt and passed to the script as its "
                        "first positional argument. Provide exactly one of input_json / input_text.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra command-line arguments, e.g. [\"--output\", \"prompt.txt\"]",
                    },
                    "output_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files written by the script into its working directory to read back, "
                        "e.g. [\"prompt.txt\"]",
                    },
                },
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Return the complete final deliverable and end the run. Call this exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete final deliverable, verbatim.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

# Same tool set with an extra structured-output argument on `finish`; used only when
# the node's json_output switch is on, so the default request shape never changes.
TOOL_SCHEMAS_JSON = copy.deepcopy(TOOL_SCHEMAS)
TOOL_SCHEMAS_JSON[-1]["function"]["parameters"]["properties"]["json"] = {
    "type": "object",
    "description": "The complete deliverable re-expressed as a single JSON object "
    "(meaningful field names; arrays or nested objects where appropriate).",
}


def _tool_schemas(json_output: bool) -> list[dict]:
    return TOOL_SCHEMAS_JSON if json_output else TOOL_SCHEMAS


def _safe_skill_file(skill_root: Path, relative: object) -> Path:
    root = skill_root.resolve()
    raw = str(relative or "").strip().lstrip("/\\")
    if not raw:
        raise ValueError("file must not be empty")
    candidate = (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("file must stay inside the Skill directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"Skill document not found: {relative}")
    return candidate


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _run_skill_script(skill_root: Path, args: dict[str, object]) -> str:
    """Execute a whitelisted Skill script offline in a throwaway directory."""
    script_raw = str(args.get("script") or "").strip().lstrip("/\\")
    if not script_raw:
        raise ValueError("script must not be empty")

    scripts_dir = skill_root.resolve() / "scripts"
    if not scripts_dir.is_dir():
        raise FileNotFoundError("this Skill has no scripts/ directory")
    script_path = (scripts_dir / script_raw).resolve()
    if script_path != scripts_dir and scripts_dir not in script_path.parents:
        raise ValueError("script must stay inside the Skill's scripts/ directory")
    if script_path.suffix.lower() != ".py":
        raise ValueError("only .py scripts can be run")
    if not script_path.is_file():
        raise FileNotFoundError(f"script not found: {script_raw}")

    has_json = "input_json" in args and args.get("input_json") is not None
    has_text = isinstance(args.get("input_text"), str) and bool(args["input_text"])
    if has_json == has_text:
        raise ValueError("provide exactly one of input_json / input_text")

    extra_args = args.get("args") or []
    if not isinstance(extra_args, list) or not all(isinstance(a, str) for a in extra_args):
        raise ValueError("args must be a list of strings")

    output_files = args.get("output_files") or []
    if not isinstance(output_files, list) or not all(isinstance(f, str) and f.strip() for f in output_files):
        raise ValueError("output_files must be a list of file names")

    with tempfile.TemporaryDirectory(prefix="qwen_skill_") as tmp:
        work_dir = Path(tmp)
        argv = [sys.executable, str(script_path)]
        if has_json:
            payload_path = work_dir / "input.json"
            payload_path.write_text(
                json.dumps(args["input_json"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            argv.append(str(payload_path))
        else:
            payload_path = work_dir / "input.txt"
            payload_path.write_text(str(args["input_text"]), encoding="utf-8")
            argv.append(str(payload_path))
        argv.extend(extra_args)

        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, whitelisted path
                argv,
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=SCRIPT_TIMEOUT_SECONDS,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            raise ValueError(f"script timed out after {SCRIPT_TIMEOUT_SECONDS:.0f}s") from None

        sections = [f"exit_code: {proc.returncode}", f"stdout:\n{_truncate(proc.stdout or '', MAX_SCRIPT_OUTPUT_BYTES)}"]
        if (proc.stderr or "").strip():
            sections.append(f"stderr:\n{_truncate(proc.stderr, MAX_SCRIPT_OUTPUT_BYTES)}")
        for name in output_files:
            out_path = (work_dir / name.strip().lstrip("/\\")).resolve()
            if out_path != work_dir and work_dir not in out_path.parents:
                sections.append(f"--- {name}: ERROR (path escapes the script working directory)")
                continue
            if out_path.is_file():
                sections.append(
                    f"--- {name} ---\n{_truncate(out_path.read_text(encoding='utf-8', errors='replace'), MAX_TOOL_FILE_BYTES)}"
                )
            else:
                sections.append(f"--- {name}: not found (script may not have written it)")
        return "\n".join(sections)


def dispatch_tool(name: object, arguments: object, *, skill_root: Path) -> tuple[str, bool, str]:
    """Run one local tool. Returns (result text, finished, optional JSON payload)."""
    try:
        args = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
    except (TypeError, ValueError) as error:
        return f"ERROR: tool arguments are not valid JSON: {error}", False, ""
    if not isinstance(args, dict):
        return "ERROR: tool arguments must be a JSON object", False, ""

    try:
        if name == "list_skill_files":
            root = skill_root.resolve()
            entries = ["SKILL.md"]
            references = root / "references"
            if references.is_dir():
                entries += sorted(
                    path.relative_to(root).as_posix()
                    for path in references.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".md", ".txt"}
                )
            return "Skill documents:\n" + "\n".join(entries), False, ""

        if name == "read_skill_reference":
            path = _safe_skill_file(skill_root, args.get("file"))
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_TOOL_FILE_BYTES:
                text = text[:MAX_TOOL_FILE_BYTES] + "\n...[truncated]"
            return (
                f"# {path.relative_to(skill_root.resolve()).as_posix()}\n\n{text}",
                False,
                "",
            )

        if name == "run_skill_script":
            return _run_skill_script(skill_root, args), False, ""

        if name == "finish":
            payload = args.get("json")
            if isinstance(payload, (dict, list)):
                json_text = json.dumps(payload, ensure_ascii=False)
            elif isinstance(payload, str) and payload.strip():
                # A string payload has no grammar guarantee (tool calls bypass the
                # JSON-schema response format), so validate it here: bounce invalid
                # JSON back as a tool error so the model repairs it and finishes
                # again, instead of shipping a broken payload to the output port.
                json_text = payload.strip()
                try:
                    json.loads(json_text)
                except json.JSONDecodeError as error:
                    return (
                        "ERROR: the 'json' argument is not valid JSON: "
                        f"{error}. Call finish again with the same answer and a "
                        "corrected 'json' argument holding exactly one valid JSON object.",
                        False,
                        "",
                    )
            else:
                json_text = str(payload or "").strip()
            return str(args.get("answer", "")), True, json_text

        return f"ERROR: unknown tool: {name}", False, ""
    except Exception as error:  # noqa: BLE001 - surfaced to the model, never fatal
        LOGGER.warning("[Qwen PromptEnhancer] Tool %s failed: %s", name, error)
        return f"ERROR: {type(error).__name__}: {error}", False, ""


def run_agent_loop(
    server,
    messages: list[dict[str, object]],
    *,
    skill_root: Path,
    settings: dict[str, float | int],
    seed: int,
    max_tokens: int,
    think_mode: bool,
    reasoning_effort: str,
    max_steps: int,
    json_output: bool = False,
) -> tuple[str, str, int, dict[str, int]]:
    """Drive the resident local model through bounded tool-calling steps."""
    deadline = time.perf_counter() + AGENT_DEADLINE_SECONDS
    conversation: list[dict[str, object]] = [dict(message) for message in messages]
    addendum = AGENT_ADDENDUM + (AGENT_ADDENDUM_JSON if json_output else "")
    conversation[0] = {
        **conversation[0],
        "content": str(conversation[0]["content"]) + addendum,
    }

    steps = 0
    prompt_tokens = 0
    completion_tokens = 0
    tokens_used = 0
    final = ""
    final_json = ""
    while steps < max_steps:
        comfy.model_management.throw_exception_if_processing_interrupted()
        if time.perf_counter() > deadline:
            raise TimeoutError(f"Agent loop exceeded its time budget after {steps} step(s)")
        if tokens_used > AGENT_TOKEN_BUDGET:
            raise RuntimeError(f"Agent loop exceeded its token budget after {steps} step(s)")

        turn = server.chat_turn(
            conversation,
            seed=seed,
            max_tokens=max_tokens,
            think_mode=think_mode,
            reasoning_effort=reasoning_effort,
            tools=_tool_schemas(json_output),
            tool_choice="auto",
            **settings,
        )
        steps += 1
        usage = turn.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        tokens_used += int(usage.get("total_tokens") or 0)

        calls = turn.get("tool_calls") or []
        if not calls:
            content = str(turn.get("content") or "").strip()
            if not content:
                raise RuntimeError(f"The model returned an empty turn without calling a tool (step {steps})")
            final = content
            break

        conversation.append(
            {
                "role": "assistant",
                "content": turn.get("content") or "",
                "tool_calls": calls,
            }
        )
        finished = False
        for call in calls:
            function = (call or {}).get("function") or {}
            name = function.get("name")
            result, finished, tool_json = dispatch_tool(
                name, function.get("arguments"), skill_root=skill_root
            )
            if name == "finish":
                final = result
                final_json = tool_json
            LOGGER.info(
                "[Qwen PromptEnhancer] step %d/%d | tool=%s | %s",
                steps,
                max_steps,
                name,
                (result.splitlines() or [""])[0][:160],
            )
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": result,
                }
            )
        if finished:
            break

    if not final.strip():
        raise RuntimeError(f"The agent produced no final answer within {max_steps} step(s)")
    return final.strip(), final_json.strip(), steps, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": tokens_used,
    }
