"""
Corpus profile configuration for flexipipe train/convert workflows.

Loads YAML or JSON project profiles and merges them with global config and CLI overrides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_storage import read_config
from .segmentation_policy import HeuristicConfig, ModelSegmenterConfig, SegmentationPolicy


def _load_structured_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".json",):
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"PyYAML is required to load {path}. Install with: pip install pyyaml "
                "or use a .json config file."
            ) from exc
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    f"Cannot parse {path}: install pyyaml for YAML or use .json"
                ) from exc
            data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def discover_corpus_config(corpus_dir: Optional[Path]) -> Optional[Path]:
    if not corpus_dir or not corpus_dir.is_dir():
        return None
    for name in ("flexipipe.yaml", "flexipipe.yml", "flexipipe.json"):
        candidate = corpus_dir / name
        if candidate.is_file():
            return candidate
    return None


@dataclass
class CorpusProfile:
    """Merged corpus configuration used by train/convert commands."""

    name: Optional[str] = None
    language: Optional[str] = None
    teitok_input: Optional[Path] = None
    attrs: Dict[str, str] = field(default_factory=dict)
    segmentation_flexitag: SegmentationPolicy = field(
        default_factory=lambda: SegmentationPolicy(mode="document")
    )
    segmentation_conllu: SegmentationPolicy = field(
        default_factory=lambda: SegmentationPolicy(mode="heuristic")
    )
    train_backends: List[str] = field(default_factory=list)
    tagpos: Optional[str] = None
    nlpform: str = "form"
    train_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "dev": 0.1, "test": 0.1})
    ud_folder: Optional[Path] = None
    output_dir: Optional[Path] = None
    model_name: Optional[str] = None
    seed: int = 42
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusProfile":
        corpus = data.get("corpus") or {}
        teitok = data.get("teitok") or {}
        segmentation = data.get("segmentation") or {}
        train = data.get("train") or {}

        attrs = dict(teitok.get("attrs") or {})
        teitok_input = teitok.get("input")
        input_path = Path(teitok_input).expanduser() if teitok_input else None

        flexitag_mode = segmentation.get("flexitag", "document")
        conllu_mode = segmentation.get("conllu", "heuristic")
        heuristic_data = segmentation.get("heuristic")
        model_data = segmentation.get("model")

        heuristic = HeuristicConfig.from_dict(heuristic_data)
        model_cfg = ModelSegmenterConfig.from_dict(model_data)

        seg_flexitag = SegmentationPolicy(
            mode=flexitag_mode if isinstance(flexitag_mode, str) else "document",
            heuristic=heuristic,
            model=model_cfg,
        )
        seg_conllu = SegmentationPolicy(
            mode=conllu_mode if isinstance(conllu_mode, str) else "heuristic",
            heuristic=heuristic,
            model=model_cfg,
        )

        ratios = train.get("ratios") or {"train": 0.8, "dev": 0.1, "test": 0.1}
        ud_folder = train.get("ud_folder")
        output_dir = train.get("output_dir")
        model_name = train.get("name")

        return cls(
            name=corpus.get("name"),
            language=corpus.get("language"),
            teitok_input=input_path,
            attrs=attrs,
            segmentation_flexitag=seg_flexitag,
            segmentation_conllu=seg_conllu,
            train_backends=list(train.get("backends") or []),
            tagpos=train.get("tagpos"),
            nlpform=train.get("nlpform") or "form",
            train_ratios=dict(ratios),
            ud_folder=Path(ud_folder).expanduser() if ud_folder else None,
            output_dir=Path(output_dir).expanduser() if output_dir else None,
            model_name=model_name,
            seed=int(train.get("seed", 42)),
            raw=data,
        )

    def resolved_model_name(self) -> Optional[str]:
        """Model label for --name: train.name, else corpus.name."""
        return self.model_name or self.name

    def segmentation_for_backend(self, backend_type: str) -> SegmentationPolicy:
        if (backend_type or "").lower() == "flexitag":
            return self.segmentation_flexitag
        return self.segmentation_conllu

    def apply_cli_overrides(
        self,
        *,
        language: Optional[str] = None,
        train_data: Optional[Path] = None,
        xpos_attr: Optional[str] = None,
        reg_attr: Optional[str] = None,
        expan_attr: Optional[str] = None,
        lemma_attr: Optional[str] = None,
        trslit_attr: Optional[str] = None,
        segment_mode: Optional[str] = None,
        segment_flexitag: Optional[str] = None,
        segment_conllu: Optional[str] = None,
        max_sentence_tokens: Optional[int] = None,
        tagpos: Optional[str] = None,
        nlpform: Optional[str] = None,
        ud_folder: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        seed: Optional[int] = None,
        train_ratio: Optional[float] = None,
        dev_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> "CorpusProfile":
        merged = CorpusProfile(
            name=self.name,
            language=language or self.language,
            teitok_input=train_data or self.teitok_input,
            attrs=dict(self.attrs),
            segmentation_flexitag=SegmentationPolicy(
                mode=self.segmentation_flexitag.mode,
                heuristic=HeuristicConfig(
                    split_on=self.segmentation_flexitag.heuristic.split_on,
                    punct_xpos=self.segmentation_flexitag.heuristic.punct_xpos,
                    closing_punct_xpos=self.segmentation_flexitag.heuristic.closing_punct_xpos,
                    max_tokens=self.segmentation_flexitag.heuristic.max_tokens,
                ),
                model=self.segmentation_flexitag.model,
            ),
            segmentation_conllu=SegmentationPolicy(
                mode=self.segmentation_conllu.mode,
                heuristic=HeuristicConfig(
                    split_on=self.segmentation_conllu.heuristic.split_on,
                    punct_xpos=self.segmentation_conllu.heuristic.punct_xpos,
                    closing_punct_xpos=self.segmentation_conllu.heuristic.closing_punct_xpos,
                    max_tokens=self.segmentation_conllu.heuristic.max_tokens,
                ),
                model=self.segmentation_conllu.model,
            ),
            train_backends=list(self.train_backends),
            tagpos=tagpos or self.tagpos,
            nlpform=nlpform or self.nlpform,
            train_ratios=dict(self.train_ratios),
            ud_folder=ud_folder or self.ud_folder,
            output_dir=output_dir or self.output_dir,
            model_name=model_name or self.model_name,
            seed=seed if seed is not None else self.seed,
            raw=dict(self.raw),
        )

        if xpos_attr:
            merged.attrs["xpos"] = xpos_attr
        if reg_attr:
            merged.attrs["reg"] = reg_attr
        if expan_attr:
            merged.attrs["expan"] = expan_attr
        if lemma_attr:
            merged.attrs["lemma"] = lemma_attr
        if trslit_attr:
            merged.attrs["trslit"] = trslit_attr

        if max_sentence_tokens is not None:
            for policy in (merged.segmentation_flexitag, merged.segmentation_conllu):
                policy.heuristic.max_tokens = int(max_sentence_tokens)

        if segment_flexitag:
            merged.segmentation_flexitag.mode = segment_flexitag
        if segment_conllu:
            merged.segmentation_conllu.mode = segment_conllu
        if segment_mode:
            merged.segmentation_flexitag.mode = segment_mode
            merged.segmentation_conllu.mode = segment_mode

        if train_ratio is not None or dev_ratio is not None or test_ratio is not None:
            merged.train_ratios = {
                "train": train_ratio if train_ratio is not None else merged.train_ratios.get("train", 0.8),
                "dev": dev_ratio if dev_ratio is not None else merged.train_ratios.get("dev", 0.1),
                "test": test_ratio if test_ratio is not None else merged.train_ratios.get("test", 0.1),
            }

        return merged


def load_corpus_profile(
    config_path: Optional[Path] = None,
    *,
    corpus_dir: Optional[Path] = None,
) -> Optional[CorpusProfile]:
    path = config_path
    if path is None:
        path = discover_corpus_config(corpus_dir)
    if path is None:
        return None
    path = path.expanduser().resolve()
    data = _load_structured_file(path)
    return CorpusProfile.from_dict(data)


def resolve_corpus_profile(
    *,
    config_path: Optional[Path] = None,
    corpus_dir: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> CorpusProfile:
    """
    Load and merge configuration: global config.json defaults → corpus profile → CLI.
    """
    _ = read_config()  # reserved for future global defaults
    profile = load_corpus_profile(config_path, corpus_dir=corpus_dir)
    if profile is None:
        profile = CorpusProfile()
    if cli_overrides:
        profile = profile.apply_cli_overrides(**cli_overrides)
    return profile
