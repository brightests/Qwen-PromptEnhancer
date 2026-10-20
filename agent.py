"""Bounded tool-calling agent loop for multi-step local Skill execution."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import comfy.model_management

LOGGER = logging.getLogger("ComfyUI.QwenPromptEnhancer.tools")

AGENT_DEADLINE_SECONDS = 3600.0
AGENT_TOKEN_BUDGET = 65536
MAX_TOOL_FILE_BYTES = 200_000

AGENT_ADDENDUM = """

# Agent mode: local tools

You are running as a multi-step agent and may call the provided local tools:

- `list_skill_files`: enumerate the documents bundled with the selected Skill.
- `read_skill_reference`: read one bundled Skill document by its relative path.
- `finish`: return the complete final deliverable and end the run.

Read the Skill documents you need, follow the selected Skill's process step by step, then call `finish`
exactly once with the complete final answer. Tools are local and offline; there is no network access.
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


def dispatch_tool(name: object, arguments: object, *, skill_root: Path) -> tuple[str, bool]:
    """Run one local tool. Returns (result text, finished)."""
    try:
        args = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
    except (TypeError, ValueError) as error:
        return f"ERROR: tool arguments are not valid JSON: {error}", False
    if not isinstance(args, dict):
        return "ERROR: tool arguments must be a JSON object", False

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
            return "Skill documents:\n" + "\n".join(entries), False

        if name == "read_skill_reference":
            path = _safe_skill_file(skill_root, args.get("file"))
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_TOOL_FILE_BYTES:
                text = text[:MAX_TOOL_FILE_BYTES] + "\n...[truncated]"
            return (
                f"# {path.relative_to(skill_root.resolve()).as_posix()}\n\n{text}",
                False,
            )

        if name == "finish":
            return str(args.get("answer", "")), True

        return f"ERROR: unknown tool: {name}", False
    except Exception as error:  # noqa: BLE001 - surfaced to the model, never fatal
        LOGGER.warning("[Qwen PromptEnhancer] Tool %s failed: %s", name, error)
        return f"ERROR: {type(error).__name__}: {error}", False


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
) -> tuple[str, int, dict[str, int]]:
    """Drive the resident local model through bounded tool-calling steps."""
    deadline = time.perf_counter() + AGENT_DEADLINE_SECONDS
    conversation: list[dict[str, object]] = [dict(message) for message in messages]
    conversation[0] = {
        **conversation[0],
        "content": str(conversation[0]["content"]) + AGENT_ADDENDUM,
    }

    steps = 0
    prompt_tokens = 0
    completion_tokens = 0
    tokens_used = 0
    final = ""
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
            tools=TOOL_SCHEMAS,
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
            result, finished = dispatch_tool(name, function.get("arguments"), skill_root=skill_root)
            if name == "finish":
                final = result
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
    return final.strip(), steps, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": tokens_used,
    }
