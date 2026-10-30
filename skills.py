from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SKILL_ROOT = Path(__file__).with_name("skills")
CUSTOM_SKILL_ROOT = Path(__file__).with_name("custom_skills")
LOGGER = logging.getLogger("ComfyUI.QwenPromptEnhancer.skills")

# A Skill id may use any printable character except path separators and
# Windows-reserved filename characters. This keeps mixed-case ids such as
# "H3-Singularity" and non-ASCII folder names such as "my-skill-图片真实感".
SKILL_ID_RE = re.compile(r"^[^\x00-\x1f\\/:*?\"<>|]+$")
RESERVED_SKILL_IDS = {"auto"}


@dataclass(frozen=True)
class SkillSpec:
    """A discovered text-only Skill available to the local node."""

    id: str
    path: Path
    description: str
    source: str
    display_name: str | None = None
    version: str | None = None
    tags: tuple[str, ...] = ()


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _list_values(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(item for item in value if item.strip())
    if not isinstance(value, str) or not value.strip():
        return ()
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
        return tuple(
            item
            for item in (_scalar(part) for part in value.split(","))
            if item.strip()
        )
    return (value,)


def _front_matter(path: Path) -> dict[str, str]:
    """Read the small subset of YAML front matter needed for Skill discovery."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}

    end = next(
        (
            index
            for index in range(1, len(lines))
            if lines[index].strip() == "---"
            and not lines[index].startswith((" ", "\t"))
        ),
        None,
    )
    if end is None:
        return {}

    values: dict[str, str] = {}
    index = 1
    while index < end:
        line = lines[index]
        match = re.match(
            r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$",
            line,
        )
        if not match or match.group("indent"):
            index += 1
            continue

        key = match.group("key").replace("-", "_").casefold()
        value = match.group("value").strip()
        if value.startswith(("|", ">")):
            block: list[str] = []
            cursor = index + 1
            while cursor < end:
                candidate = lines[cursor]
                if candidate.strip() and not candidate.startswith((" ", "\t")):
                    break
                block.append(candidate)
                cursor += 1
            non_empty = [line for line in block if line.strip()]
            indent = min((len(line) - len(line.lstrip()) for line in non_empty), default=0)
            block = [line[indent:] if line.strip() else "" for line in block]
            values[key] = ("\n" if value.startswith("|") else " ").join(block).strip()
            index = cursor
            continue
        if value:
            values[key] = _scalar(value)
        index += 1
    return values


def _meta_yaml(path: Path) -> dict[str, str | tuple[str, ...]]:
    """Read simple scalar and list metadata from the bundled meta.yaml convention."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    values: dict[str, str | tuple[str, ...]] = {}
    index = 0
    while index < len(lines):
        match = re.match(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$", lines[index])
        if not match:
            index += 1
            continue
        key = match.group("key").replace("-", "_").casefold()
        value = match.group("value").strip()
        if value:
            values[key] = _scalar(value)
            index += 1
            continue

        items: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            item = re.match(r"^\s*-\s*(.+?)\s*$", lines[cursor])
            if not item:
                break
            items.append(_scalar(item.group(1)))
            cursor += 1
        if items:
            values[key] = tuple(items)
        index = cursor
    return values


def _metadata(skill_root: Path) -> tuple[str, str, str | None, str | None, tuple[str, ...]]:
    front = _front_matter(skill_root / "SKILL.md")
    meta = _meta_yaml(skill_root / "meta.yaml")
    # The Skill id is the folder name: renaming the folder in skills/ or
    # custom_skills/ is all it takes to change the name shown in the dropdown.
    # SKILL.md's `name:` field is deliberately not used for the id so the two
    # can never drift apart.
    skill_id = skill_root.name.strip()
    description = (
        front.get("description", "").strip()
        or str(meta.get("summary_en", "")).strip()
        or str(meta.get("desc_en", "")).strip()
        or skill_id
    )
    description = " ".join(description.split())
    display_name = (
        front.get("display_name", "").strip()
        or front.get("display_name_en", "").strip()
        or front.get("display_name_zh", "").strip()
        or str(meta.get("display_name_en", "")).strip()
        or str(meta.get("display_name_zh", "")).strip()
        or None
    )
    version = (
        front.get("version", "").strip()
        or str(meta.get("version", "")).strip()
        or None
    )
    raw_tags: list[str] = []
    for values in (
        _list_values(front.get("tags")),
        _list_values(front.get("tag_en")),
        _list_values(front.get("tag_cn")),
        _list_values(meta.get("tags")),
        _list_values(meta.get("tag_en")),
        _list_values(meta.get("tag_cn")),
        _list_values(meta.get("complete_tags_en")),
        _list_values(meta.get("complete_tags_cn")),
    ):
        raw_tags.extend(values)
    return skill_id, description, display_name, version, tuple(dict.fromkeys(raw_tags))


def _valid_skill_id(value: str) -> bool:
    value = value.strip()
    if not value or value in {".", ".."} or value.casefold() in RESERVED_SKILL_IDS:
        return False
    return bool(SKILL_ID_RE.fullmatch(value))


def _discover_skills(root: Path, source: str) -> list[SkillSpec]:
    if not root.is_dir():
        return []
    resolved_root = root.resolve()
    discovered: list[SkillSpec] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not candidate.is_dir():
            continue
        skill_path = candidate.resolve()
        if skill_path.parent != resolved_root:
            LOGGER.warning("Ignoring Skill outside %s: %s", resolved_root, candidate)
            continue
        skill_file = skill_path / "SKILL.md"
        if not skill_file.is_file() or skill_file.resolve().parent != skill_path:
            LOGGER.warning("Ignoring Skill without SKILL.md: %s", skill_path)
            continue
        skill_id, description, display_name, version, tags = _metadata(skill_path)
        if not _valid_skill_id(skill_id):
            LOGGER.warning("Ignoring Skill with invalid id %r: %s", skill_id, skill_path)
            continue
        discovered.append(
            SkillSpec(
                id=skill_id,
                path=skill_path,
                description=description,
                source=source,
                display_name=display_name,
                version=version,
                tags=tags,
            )
        )
    return discovered


def discover_skill_registry(
    builtin_root: Path | None = None,
    custom_root: Path | None = None,
    extra_roots: Sequence[Path] = (),
) -> tuple[SkillSpec, ...]:
    """Discover bundled Skills first, then custom and user-supplied Skill roots."""
    builtin_root = SKILL_ROOT if builtin_root is None else builtin_root
    custom_root = CUSTOM_SKILL_ROOT if custom_root is None else custom_root
    registry: list[SkillSpec] = []
    seen: set[str] = set()
    candidates = [
        *_discover_skills(builtin_root, "builtin"),
        *_discover_skills(custom_root, "custom"),
    ]
    for extra_root in extra_roots:
        if extra_root is None:
            continue
        try:
            candidates.extend(_discover_skills(Path(extra_root).expanduser(), "extra"))
        except OSError as error:  # noqa: BLE001 - a broken extra root must not abort the node
            LOGGER.warning("Ignoring unusable Skill root %s: %s", extra_root, error)
    for spec in candidates:
        key = spec.id.casefold()
        if key in seen:
            LOGGER.warning("Ignoring duplicate Skill id %r from %s", spec.id, spec.path)
            continue
        seen.add(key)
        registry.append(spec)
    return tuple(registry)


# Skill discovery is deliberately lazy: the folders are re-scanned on every call
# so a running ComfyUI never serves a list that was frozen at import time.
PREFERRED_SKILL_ID = "h3-prompt-writing"


def skill_ids() -> tuple[str, ...]:
    """Skill ids from the default folders. Re-scanned on every call."""
    return tuple(spec.id for spec in discover_skill_registry())


def _preferred_skill_id(available: Sequence[SkillSpec]) -> str:
    if not available:
        return ""
    return next(
        (spec.id for spec in available if spec.id == PREFERRED_SKILL_ID),
        available[0].id,
    )


def _registry_lookup(registry: Sequence[SkillSpec], skill: object) -> SkillSpec | None:
    key = " ".join(str(skill or "").split()).casefold()
    if not key:
        return None
    for spec in registry:
        if spec.id.casefold() == key:
            return spec
        if spec.display_name and spec.display_name.casefold() == key:
            return spec
    return None


def resolve_skill(skill: object, registry: Sequence[SkillSpec] | None = None) -> SkillSpec:
    available = discover_skill_registry() if registry is None else registry
    spec = _registry_lookup(available, skill)
    if spec is None:
        names = ", ".join(spec.id for spec in available) or "none"
        raise ValueError(f"Unknown Skill {skill!r}. Available Skills: {names}")
    return spec


def _skill_spec(skill: object, registry: Sequence[SkillSpec] | None = None) -> SkillSpec:
    return resolve_skill(skill, registry)


def _skill_path(skill: object, registry: Sequence[SkillSpec] | None = None) -> Path:
    spec = resolve_skill(skill, registry)
    root = spec.path.resolve()
    skill_file = root / "SKILL.md"
    if (
        not root.is_dir()
        or not skill_file.is_file()
        or skill_file.resolve().parent != root
    ):
        raise FileNotFoundError(f"Skill is missing SKILL.md: {spec.id}")
    return root


def router_prompt(
    user_prompt: str,
    asset_summary: str,
    skills: Sequence[SkillSpec] | None = None,
) -> list[dict[str, str]]:
    available = discover_skill_registry() if skills is None else skills
    catalog = "\n".join(f"- {spec.id}: {spec.description}" for spec in available)
    system = (
        "Select exactly one available Skill for the request. Reply with one JSON object "
        '{"skill": "<exact skill id>"} and nothing else.\n\n' + catalog
    )
    detail = f"Connected assets: {asset_summary}\n" if asset_summary and asset_summary != "none" else ""
    user = f"{detail}Request:\n{user_prompt}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_skill_selection(
    text: str,
    skills: Sequence[SkillSpec] | None = None,
) -> str:
    available = discover_skill_registry() if skills is None else skills
    by_id = {spec.id.casefold(): spec.id for spec in available}
    normalized = text.strip().casefold()
    if normalized in by_id:
        return by_id[normalized]
    matches = [
        spec.id
        for spec in available
        if re.search(
            rf"(?<![a-z0-9_.-]){re.escape(spec.id.casefold())}(?![a-z0-9_.-])",
            normalized,
        )
    ]
    if not matches:
        matches = [
            spec.id
            for spec in available
            if re.search(
                rf"(?<![a-z0-9_-]){re.escape(spec.id.casefold())}(?![a-z0-9_-])",
                normalized,
            )
        ]
    if len(matches) == 1:
        return matches[0]
    return _preferred_skill_id(available)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def system_prompt(
    skill: object,
    registry: Sequence[SkillSpec] | None = None,
) -> str:
    """Minimal, Skill-agnostic contract. All Skill-specific guidance lives in SKILL.md."""
    spec = resolve_skill(skill, registry)
    root = _skill_path(skill, registry)
    body = _read(root / "SKILL.md")
    label = f"{spec.display_name} ({spec.id})" if spec.display_name else spec.id
    return f"""# Selected Skill: {label}

{body}

# Local agent contract

You are a local agent running inside ComfyUI with offline tools. Follow the selected Skill's
process step by step. Use the provided tools to list and read any additional Skill documents you
need before producing the result.

Produce exactly the deliverable the user asks for. Do not add wrappers, prefaces, or output
schemas that neither the Skill nor the user requests. There is no network access: never attempt to
call remote services, generate media, or queue ComfyUI jobs. Output text only."""
