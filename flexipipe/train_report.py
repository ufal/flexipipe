"""After training: what the new model is and how good it is, for any trainable backend.

- TRAINABLE: the backends `flexipipe train` can train from a corpus, with what they learn (when the
  data has it) and what they need besides.
- evaluate_trained(): the model on held-out data — the gold file with its annotation removed, run
  through `flexipipe process` with the model (as any later use would), then compared token by token
  with the gold annotation: upos, xpos, feats, lemma, UAS / LAS. Sentences whose tokens do not line up
  are left out (and counted).
- write_manifest(): ``flexipipe-model.json`` in the model folder — backend, the model file or folder,
  created, the training data, language, steps, scores, options — so a corpus's model list
  (corpus_models.py, model_card.py) describes models of every backend the same way.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

MANIFEST = "flexipipe-model.json"

# backend -> (steps it learns when the data has them, what it needs besides the data)
TRAINABLE: Dict[str, dict] = {
    "flexitag": {"learns": ["tag", "xpos", "lemmatize", "normalize"], "needs": [],
                 "about": "a fast lexicon + HMM tagger and lemmatizer; also learns pronunciations"},
    "udpipe1": {"learns": ["tokenize", "segment", "tag", "xpos", "lemmatize", "parse"], "needs": [],
                "about": "UDPipe 1: tokenizer, tagger, lemmatizer and parser in one model"},
    "spacy": {"learns": ["tag", "xpos", "lemmatize", "parse"], "needs": ["language"],
              "about": "a spaCy pipeline (tagger, morphologizer, lemmatizer, parser)"},
    "classla": {"learns": ["tag", "xpos", "lemmatize", "parse"], "needs": ["language"],
                "about": "Stanza / ClassLA neural models (slow to train; a GPU helps)"},
    "transformers": {"learns": ["tag"], "needs": ["base model"],
                     "about": "fine-tunes a transformer (a base model is needed; a GPU helps)"},
}


def _rows(path: Path) -> List[List[List[str]]]:
    """Sentences of a CoNLL-U file: lists of the 10 columns of each word line (no ranges, no empty nodes)."""
    sentences, cur = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    sentences.append(cur)
                cur = []
                continue
            if line.startswith("#"):
                continue
            c = line.split("\t")
            if len(c) >= 10 and c[0].isdigit():
                cur.append(c)
    if cur:
        sentences.append(cur)
    return sentences


def _bare(gold: Path, target: Path) -> None:
    """The gold file without lemma, tags, feats, heads and relations (MISC kept but for a pronunciation;
    a normalization stays: it is part of the input a model trained on normalized forms tags by)."""
    with open(gold, encoding="utf-8") as fh, open(target, "w", encoding="utf-8") as out:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 10 and not line.startswith("#"):
                c[2:9] = ["_"] * 7
                misc = [p for p in c[9].split("|") if p.split("=", 1)[0] not in ("Pron", "PronSource")]
                c[9] = "|".join(misc) if misc and misc != ["_"] else "_"
                line = "\t".join(c) + "\n"
            out.write(line)


def score(gold: Path, predicted: Path) -> dict:
    """Token-level accuracy of `predicted` against `gold` (same tokens: sentences that line up)."""
    g_sents, p_sents = _rows(gold), _rows(predicted)
    counts = {k: [0, 0] for k in ("upos", "xpos", "feats", "lemma", "uas", "las")}
    used = skipped = 0
    for g, p in zip(g_sents, p_sents):
        # the same tokens: the written form, or the normalized one (a model that tags by those outputs them)
        norm = lambda c: dict(x.split("=", 1) for x in c[9].split("|") if "=" in x).get("Normalization", c[1])
        if len(g) != len(p) or any(b[1] not in (a[1], norm(a)) for a, b in zip(g, p)):
            skipped += 1
            continue
        used += 1
        for a, b in zip(g, p):
            for key, col in (("upos", 3), ("xpos", 4), ("feats", 5), ("lemma", 2)):
                if a[col] not in ("_", ""):
                    counts[key][0] += 1
                    counts[key][1] += a[col] == b[col]
            if a[6] not in ("_", ""):
                counts["uas"][0] += 1
                counts["uas"][1] += a[6] == b[6]
                counts["las"][0] += 1
                counts["las"][1] += a[6] == b[6] and a[7].split(":")[0] == b[7].split(":")[0]
    out = {k: round(ok / n, 4) for k, (n, ok) in counts.items() if n}
    out["sentences"] = used
    if skipped:
        out["sentences_skipped"] = skipped
    return out


def evaluate_trained(backend: str, model: str, gold: Path, *, language: Optional[str] = None) -> dict:
    """The new model on `gold` (a held-out CoNLL-U split): run it, score it. {} when it cannot run."""
    with tempfile.TemporaryDirectory(prefix="flexipipe-eval-") as tmp:
        bare, predicted = Path(tmp, "bare.conllu"), Path(tmp, "predicted.conllu")
        _bare(gold, bare)
        cmd = [sys.executable, "-m", "flexipipe", "process", "--input", str(bare), "--input-format", "conllu",
               "--backend", backend, "--model", model, "--output-format", "conllu", "--output", str(predicted)]
        if language:
            cmd += ["--language", language]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0 or not predicted.exists():
            tail = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
            print(f"[flexipipe] the new model could not be evaluated: {tail[0]}", flush=True)
            return {}
        return score(gold, predicted)


LEARNED = (("upos", 3, "UD part of speech"), ("xpos", 4, "tags"), ("feats", 5, "morphological features"),
           ("lemma", 2, "lemmas"), ("head", 6, "dependencies"))


def learned(train_split: Optional[Path], backend: str) -> List[str]:
    """What a model learned: the annotations its training split has that the backend learns."""
    if not train_split or not Path(train_split).exists():
        return []
    steps = set(TRAINABLE.get(backend, {}).get("learns", []))
    seen = set()
    for sent in _rows(Path(train_split))[:2000]:
        for c in sent:
            for key, col, _ in LEARNED:
                if c[col] not in ("_", ""):
                    seen.add(key)
    step_of = {"upos": "tag", "xpos": "xpos", "feats": "tag", "lemma": "lemmatize", "head": "parse"}
    return [label for key, _, label in LEARNED if key in seen and step_of[key] in steps]


def write_manifest(folder: Path, backend: str, model: str, *, train_data: str, language: Optional[str],
                   scores: Optional[dict] = None, options: Optional[dict] = None, held_out: Optional[str] = None,
                   train_split: Optional[Path] = None) -> Path:
    """flexipipe-model.json: what the model in `folder` is (`model`: the file / folder to load, relative
    to `folder` where it is inside it)."""
    folder = Path(folder)
    try:
        model_rel = str(Path(model).resolve().relative_to(folder.resolve()))
    except ValueError:
        model_rel = str(model)
    data = {
        "backend": backend,
        "model": model_rel,
        "created": datetime.now().isoformat(timespec="seconds"),
        "trained_on": str(Path(train_data).resolve()),
        "language": language,
        "learns": learned(train_split, backend),
        "scores": scores or {},
        "held_out": held_out,
        "options": options or {},
    }
    try:  # the flexipipe that made it (a later one reads it; an earlier one may not)
        from .__main__ import _VERSION
        data["flexipipe_version"] = _VERSION
    except Exception:
        pass
    # what the training files' TEI headers say about their licence (a TEITOK corpus): the model holds
    # material from those texts, so publishing it shows this next to the licence the sender gives
    try:
        from .corpus_licence import licences_in
        corpus_licence = licences_in(train_data)
    except Exception:
        corpus_licence = None
    if corpus_licence:
        data["corpus_licence"] = corpus_licence
    path = folder / MANIFEST
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
