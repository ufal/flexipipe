"""Which backend does which step of a run (dev/TASK-ROUTING.md).

The main backend does the steps it provides; a step it does not provide (UDPipe: no named entities,
no normalization, no pronunciation) goes to the backend chosen for that step — on the command line
(``--task-backend TASK=BACKEND[:MODEL]``, or the older ``--ner-backend``, ``--pron-backend`` +
``--pron-model``, ``--normalizer``), else in the corpus's settings (settings.xml
``<flexipipe><tasks><item key="ner" backend="nametag" model="…"/></tasks>``, and EasyCorp's ``ner_backend`` /
``normalize_backend`` attributes). A step nobody provides is named before anything runs, and left
out, instead of coming out silently empty.

Steps another backend can take over: ``ner`` (NameTag), ``pron`` (light, phonetisaurus),
``normalize`` (hunspell; it runs before the main backend, through the existing normalizer).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# what a backend provides (main pipelines; the model can add to it: model_tasks)
_PIPELINE = {"tokenize", "segment", "lemmatize", "tag", "xpos", "parse"}
CAPABILITIES: Dict[str, Set[str]] = {
    "udpipe": set(_PIPELINE),
    "udpipe1": set(_PIPELINE),
    "udmorph": {"tokenize", "segment", "lemmatize", "tag", "xpos"},
    "udkanbun": set(_PIPELINE),
    "spacy": _PIPELINE | {"ner"},
    "stanza": _PIPELINE | {"ner"},
    "classla": _PIPELINE | {"ner"},
    "flair": {"tag", "xpos", "ner"},
    "transformers": {"tag", "ner"},
    "treetagger": {"tokenize", "lemmatize", "tag", "xpos"},
    "ctext": {"tokenize", "segment", "lemmatize", "tag", "xpos", "ner"},
    "flexitag": {"tokenize", "segment", "lemmatize", "tag", "xpos", "normalize"},
    "fasttext": {"tag", "xpos"},  # UPOS, XPOS, FEATS on given tokens
    "nametag": {"ner"},
    "hunspell": {"normalize"},
    "light": {"pron"},
    "phonetisaurus": {"pron"},
}
# steps only some of a backend's models do (a flexitag model with a reg layer or a pronunciation model,
# a UDPipe model trained without a parser …): model_tasks() says it for a given model where it can
MODEL_DEPENDENT: Dict[str, Set[str]] = {
    "flexitag": {"normalize", "pron"},
    "udpipe": {"parse", "lemmatize"},
    "udpipe1": {"parse", "lemmatize"},
    "spacy": {"ner", "parse"},
    "transformers": {"tag", "ner"},
}
# the steps flexipipe itself always provides around any backend
ALWAYS = {"tokenize", "segment"}
# steps a second backend can do after (or, normalize: before) the main one, and who can
SECONDARY: Dict[str, Tuple[str, ...]] = {
    "ner": ("nametag",),
    "pron": ("light", "phonetisaurus"),
    "normalize": ("hunspell",),
}


@dataclass
class Route:
    task: str
    backend: str
    model: Optional[str] = None
    # main | given | corpus | corpus-model (trained on the corpus) | flexipipe | default (installed for the language)
    source: str = "main"

    def label(self) -> str:
        return self.backend + (f" / {self.model}" if self.model else "")


def model_tasks(backend: str, model: Optional[str]) -> Set[str]:
    """Steps a model adds to its backend's: a flexitag model trained with --pron gives pronunciations."""
    extra: Set[str] = set()
    if (backend or "").lower() == "flexitag" and model:
        p = Path(model)
        folder = p if p.is_dir() else p.parent
        if (folder / "model_pron.json").exists():
            extra.add("pron")
    return extra


def _model_language(folder: Path) -> Optional[str]:
    """The language of an installed flexitag model: its pronunciation model's meta, else the folder
    name (UD naming: czech-cac-ud216 → cs)."""
    import json

    try:
        with open(folder / "model_pron.json", encoding="utf-8") as fh:
            head = fh.read(8192)  # the meta comes first (PronunciationModel.save): no need to read the model
        prefix = '{"meta": '
        if head.startswith(prefix):
            meta, _ = json.JSONDecoder().raw_decode(head[len(prefix):])
            if meta.get("language"):
                return str(meta["language"]).lower()
    except (OSError, ValueError):
        pass
    try:
        from .language_mapping import normalize_language_code

        iso_1, _, iso_3 = normalize_language_code(folder.name.split("-")[0])
        return (iso_1 or iso_3 or "").lower() or None
    except Exception:
        return None


def default_pron_models(language: Optional[str]) -> List[Path]:
    """Installed flexitag models for `language` with a pronunciation model (model_pron.json), by name."""
    if not language:
        return []
    try:
        from .model_storage import get_backend_models_dir

        models_dir = get_backend_models_dir("flexitag", create=False)
        folders = sorted(p.parent for p in models_dir.glob("*/model_pron.json"))
    except OSError:
        return []
    want = language.lower().split("-")[0].split("_")[0]
    return [f for f in folders if _model_language(f) == want]


def backend_tasks(backend: Optional[str], model: Optional[str] = None) -> Set[str]:
    b = (backend or "").lower()
    return set(CAPABILITIES.get(b, set())) | model_tasks(b, model) | ALWAYS


def parse_routes(values: Iterable[str]) -> Dict[str, Tuple[str, Optional[str]]]:
    """``TASK=BACKEND[:MODEL]`` → {task: (backend, model)}; raises ValueError for a bad one."""
    from .task_registry import TASK_LOOKUP

    normalize_task_name = lambda name: TASK_LOOKUP.get(name.lower())
    out: Dict[str, Tuple[str, Optional[str]]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--task-backend {value}: expected TASK=BACKEND[:MODEL], e.g. ner=nametag")
        task, spec = value.split("=", 1)
        task = normalize_task_name(task.strip()) or task.strip()
        backend, _, model = spec.strip().partition(":")
        if not backend:
            raise ValueError(f"--task-backend {value}: no backend")
        out[task] = (backend.lower(), model or None)
    return out


def routes_from_args(args) -> Dict[str, Tuple[str, Optional[str]]]:
    """The routes given on the command line: --task-backend, and the older per-step options."""
    routes = parse_routes(getattr(args, "task_backend", None) or [])
    if getattr(args, "ner_backend", None) and "ner" not in routes:
        routes["ner"] = (args.ner_backend.lower(), getattr(args, "nametag_model", None))
    if getattr(args, "pron_backend", None) and "pron" not in routes:
        routes["pron"] = (args.pron_backend.lower(), getattr(args, "pron_model", None))
    normalizer = getattr(args, "normalizer", None)
    if normalizer and "normalize" not in routes:
        backend, _, model = normalizer.partition(":")
        routes["normalize"] = (backend.lower(), model or None)
    return routes


def plan(
    tasks: Iterable[str],
    main_backend: Optional[str],
    main_model: Optional[str] = None,
    given: Optional[Dict[str, Tuple[str, Optional[str]]]] = None,
    corpus: Optional[Dict[str, Tuple[str, Optional[str]]]] = None,
    language: Optional[str] = None,
    corpus_model: Optional[str] = None,
) -> Tuple[Dict[str, Route], List[str]]:
    """(route per task, tasks nobody provides). Order: given, the main backend, the corpus's setting,
    the model trained on the corpus (`corpus_model`: its folder; corpus_models.in_use), a default
    for the language (pron: an installed flexitag model with pronunciations)."""
    provided = backend_tasks(main_backend, main_model)
    routes: Dict[str, Route] = {}
    missing: List[str] = []
    for task in sorted(set(tasks)):
        if given and task in given:
            routes[task] = Route(task, given[task][0], given[task][1], "given")
        elif task in provided:
            routes[task] = Route(task, (main_backend or "flexipipe").lower(), main_model, "main" if main_backend else "flexipipe")
        elif corpus and task in corpus:
            routes[task] = Route(task, corpus[task][0], corpus[task][1], "corpus")
        elif task == "pron" and corpus_model and (Path(corpus_model) / "model_pron.json").exists():
            routes[task] = Route(task, "light", str(corpus_model), "corpus-model")
        elif task == "pron" and default_pron_models(language):
            routes[task] = Route(task, "light", str(default_pron_models(language)[0]), "default")
        else:
            missing.append(task)
    return routes, missing


def routed_tasks(document) -> Set[str]:
    """The steps another backend did in this run (run_secondary: meta ``_task_routes_used``)."""
    return {entry.split(":", 1)[0] for entry in (document.meta.get("_task_routes_used") or [])}


def routed_note(document) -> str:
    """For the <change> text: "; pron: light / ortofon" (empty when no step was routed)."""
    entries = document.meta.get("_task_routes_used") or []
    return "; " + "; ".join(entries) if entries else ""


def run_secondary(document, routes: Dict[str, Route], main_backend: Optional[str], *, language: Optional[str] = None, verbose: bool = False) -> List[str]:
    """Run the steps routed to another backend than the main one, after it (ner, pron). Returns
    labels for provenance ("ner: nametag / czech-cnec2.0")."""
    import sys

    done: List[str] = []
    main = (main_backend or "").lower()
    for task, route in sorted(routes.items()):
        if route.backend == main or route.source in ("main", "flexipipe") or task == "normalize":
            continue
        if task == "pron":
            from .pron_backends import annotate_pronunciation

            if not route.model:
                raise ValueError("pronunciation by another backend needs a model: pron=light:/path/to/model")
            annotate_pronunciation(document, route.model, route.backend, replace_generated=True, verbose=verbose)
        elif task == "ner" and route.backend == "nametag":
            from .backend_registry import create_backend

            kwargs = {"endpoint_url": "https://lindat.mff.cuni.cz/services/nametag/api/recognize", "version": "3", "log_requests": verbose}
            if route.model:
                kwargs["model"] = route.model
            if language:
                kwargs["language"] = language
            ner = create_backend("nametag", training=False, **kwargs)
            result = ner.tag(document, use_raw_text=False, components=["ner"])
            from ._cli import _merge_ner_entities

            _merge_ner_entities(document, result.document)
        else:
            raise ValueError(f"the step {task} cannot be done by {route.backend} after the main backend "
                             f"(it can: {', '.join(SECONDARY.get(task, ())) or 'only the main backend'})")
        done.append(f"{task}: {route.label()}")
        if verbose:
            print(f"[flexipipe] {task} by {route.label()} ({route.source})", file=sys.stderr)
    return done
