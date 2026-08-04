"""
Dataset and application profiles for flexipipe.

- dataset profiles: training/convert inputs (project-local YAML)
- application profiles: runtime TEITOK deployment policy (writeback, attrs, ...)
- model manifest files: optional portable bundle metadata only (not TEITOK settings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .corpus_config import _load_structured_file
from .segmentation_policy import HeuristicConfig, ModelSegmenterConfig, SegmentationPolicy


def _segmentation_policy_from_block(data: Optional[Dict[str, Any]]) -> Optional[SegmentationPolicy]:
    if not data or not isinstance(data, dict):
        return None
    mode = data.get("mode")
    if not mode:
        return None
    heuristic = HeuristicConfig.from_dict(data.get("heuristic"))
    model_cfg = ModelSegmenterConfig.from_dict(data.get("model"))
    return SegmentationPolicy(
        mode=str(mode),
        heuristic=heuristic,
        model=model_cfg,
    )


@dataclass
class ApplicationProfile:
    """How a model should be applied in a target environment (e.g. TEITOK writeback)."""

    output_format: Optional[str] = None
    tokenize: Optional[bool] = None
    writeback: Optional[bool] = None
    writeback_engine: Optional[str] = None
    rejoin_linebreaks: Optional[bool] = None
    segment: Optional[bool] = None
    writeback_insert_sentences: Optional[bool] = None
    segmenter: Optional[str] = None
    segmentation: Optional[SegmentationPolicy] = None
    attrs: Dict[str, str] = field(default_factory=dict)
    tasks: List[str] = field(default_factory=list)
    backend_options: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicationProfile":
        block = data.get("application") or data.get("process") or data
        if not isinstance(block, dict):
            block = {}
        attrs = dict(block.get("attrs") or {})
        tasks_raw = block.get("tasks") or []
        tasks = list(tasks_raw) if isinstance(tasks_raw, list) else []
        options = block.get("options") if isinstance(block.get("options"), dict) else {}
        segment_val = block.get("segment")
        if segment_val is not None:
            segment_val = bool(segment_val)
        insert_sentences = block.get("writeback_insert_sentences")
        if insert_sentences is not None:
            insert_sentences = bool(insert_sentences)
        segmenter = block.get("segmenter")
        if segmenter is not None:
            segmenter = str(segmenter)
        return cls(
            output_format=block.get("output_format"),
            tokenize=block.get("tokenize"),
            writeback=block.get("writeback"),
            writeback_engine=block.get("writeback_engine"),
            rejoin_linebreaks=block.get("rejoin_linebreaks"),
            segment=segment_val,
            writeback_insert_sentences=insert_sentences,
            segmenter=segmenter,
            segmentation=_segmentation_policy_from_block(block.get("segmentation")),
            attrs=attrs,
            tasks=tasks,
            backend_options=dict(options),
            raw=data,
        )

    def as_process_dict(self) -> Dict[str, Any]:
        """Return a process-shaped dict for _apply_process_profile_defaults."""
        block: Dict[str, Any] = {}
        for key in (
            "output_format",
            "tokenize",
            "writeback",
            "writeback_engine",
            "rejoin_linebreaks",
        ):
            value = getattr(self, key)
            if value is not None:
                block[key] = value
        if self.attrs:
            block["attrs"] = dict(self.attrs)
        if self.tasks:
            block["tasks"] = list(self.tasks)
        if self.backend_options:
            block["options"] = dict(self.backend_options)
        return block


@dataclass
class ModelManifest:
    """Portable model identity and capabilities (TEITOK-agnostic)."""

    id: Optional[str] = None
    backend: Optional[str] = None
    model: Optional[str] = None
    model_path: Optional[Path] = None
    language: Optional[str] = None
    tasks: List[str] = field(default_factory=list)
    version: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, base_dir: Optional[Path] = None) -> "ModelManifest":
        block = data.get("manifest") if isinstance(data.get("manifest"), dict) else data
        if not isinstance(block, dict):
            block = data
        model_path_raw = block.get("model_path") or block.get("artifact")
        model_path = None
        if model_path_raw:
            model_path = Path(str(model_path_raw)).expanduser()
            if base_dir and not model_path.is_absolute():
                model_path = (base_dir / model_path).resolve()
        tasks_raw = block.get("tasks") or []
        tasks = list(tasks_raw) if isinstance(tasks_raw, list) else []
        return cls(
            id=block.get("id") or block.get("name"),
            backend=block.get("backend"),
            model=block.get("model"),
            model_path=model_path,
            language=block.get("language"),
            tasks=tasks,
            version=str(block.get("version")) if block.get("version") is not None else None,
            raw=data,
        )


def load_application_profile(path: Path) -> ApplicationProfile:
    path = path.expanduser().resolve()
    data = _load_structured_file(path)
    return ApplicationProfile.from_dict(data)


def load_model_manifest(path: Path) -> ModelManifest:
    path = path.expanduser().resolve()
    data = _load_structured_file(path)
    return ModelManifest.from_dict(data, base_dir=path.parent)


class _ProfileAdapter:
    """Minimal adapter so application profiles can reuse corpus process defaults."""

    def __init__(self, application: ApplicationProfile):
        self.raw = {"process": application.as_process_dict()}
        self.attrs = dict(application.attrs)
        self.language = None
        corpus = application.raw.get("corpus")
        if isinstance(corpus, dict) and corpus.get("language"):
            self.language = str(corpus.get("language"))


def _parse_attrs_map_cli(attrs_map_arg) -> Dict[str, str]:
    if not attrs_map_arg:
        return {}
    if isinstance(attrs_map_arg, dict):
        return {str(k): str(v) for k, v in attrs_map_arg.items()}
    result: Dict[str, str] = {}
    items = attrs_map_arg if isinstance(attrs_map_arg, list) else [attrs_map_arg]
    for item in items:
        if not item:
            continue
        text = str(item)
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def apply_application_profile_defaults(args, profile: ApplicationProfile) -> None:
    """Apply application profile values when CLI did not set them."""
    adapter = _ProfileAdapter(profile)
    raw = adapter.raw or {}
    process_cfg = raw.get("process") or {}
    if not isinstance(process_cfg, dict):
        process_cfg = {}

    if adapter.language and not getattr(args, "language", None):
        args.language = adapter.language

    if not getattr(args, "output_format", None) and process_cfg.get("output_format"):
        args.output_format = str(process_cfg.get("output_format"))
    if process_cfg.get("tokenize") is True and not getattr(args, "tokenize", False):
        args.tokenize = True
    if process_cfg.get("writeback") is not None and getattr(args, "writeback", None) is None:
        args.writeback = bool(process_cfg.get("writeback"))
    if (
        process_cfg.get("writeback_engine")
        and getattr(args, "writeback_engine", "auto") == "auto"
    ):
        args.writeback_engine = str(process_cfg.get("writeback_engine"))
    if process_cfg.get("rejoin_linebreaks") is not None:
        args.rejoin_linebreaks = bool(process_cfg.get("rejoin_linebreaks"))

    if getattr(args, "tasks", None) is None and process_cfg.get("tasks"):
        tasks = process_cfg.get("tasks")
        if isinstance(tasks, list):
            args.tasks = ",".join(str(t) for t in tasks)
        else:
            args.tasks = str(tasks)

    attrs_map = _parse_attrs_map_cli(getattr(args, "attrs_map", None))
    attrs_cfg = process_cfg.get("attrs")
    if not isinstance(attrs_cfg, dict):
        attrs_cfg = {}
    for key in ("xpos", "reg", "expan", "lemma", "trslit"):
        value = attrs_map.get(key) or attrs_cfg.get(key) or adapter.attrs.get(key)
        if value and key not in attrs_map:
            attrs_map[key] = str(value)
    if attrs_map:
        args.attrs_map = [f"{k}:{v}" for k, v in attrs_map.items()]

    profile_backend_options = process_cfg.get("options")
    if isinstance(profile_backend_options, dict) and profile_backend_options:
        setattr(args, "_profile_backend_options", dict(profile_backend_options))

    block = profile.raw.get("application") or profile.raw.get("process") or profile.raw
    if not isinstance(block, dict):
        block = {}

    if profile.segment is False:
        setattr(args, "_skip_segment_task", True)
    elif profile.segment is True:
        setattr(args, "_skip_segment_task", False)

    if profile.segmentation is not None:
        setattr(args, "_runtime_segmentation_policy", profile.segmentation)

    if profile.writeback_insert_sentences is not None:
        args.writeback_insert_sentences = profile.writeback_insert_sentences

    if profile.segmenter is not None and getattr(args, "segmenter", None) is None:
        args.segmenter = profile.segmenter


def apply_model_manifest_defaults(args, manifest: ModelManifest) -> None:
    if manifest.language and not getattr(args, "language", None):
        args.language = manifest.language
    if manifest.backend and not getattr(args, "backend", None):
        args.backend = str(manifest.backend)
    if not getattr(args, "model", None):
        if manifest.model_path and manifest.model_path.exists():
            args.model = str(manifest.model_path)
        elif manifest.model:
            args.model = str(manifest.model)
    if getattr(args, "tasks", None) is None and manifest.tasks:
        args.tasks = ",".join(str(t) for t in manifest.tasks)
