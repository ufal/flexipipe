"""The pronunciation step (task ``pron``): the vocabulary chain of a pronunciation model
(g2p_light.PronunciationModel), then a G2P engine for forms not in the vocabulary:

- ``light`` — g2p_light's own G2P (no dependency);
- ``phonetisaurus`` — Phonetisaurus (WFST joint n-gram G2P; BSD; github.com/AdolfVonKleist/Phonetisaurus):
  ``phonetisaurus-g2pfst`` to apply a model, ``phonetisaurus-align`` + MITLM's ``estimate-ngram`` +
  ``phonetisaurus-arpa2wfst`` to train one. The programs are found on PATH or in
  $FLEXIPIPE_PHONETISAURUS_DIR. Unknown forms go to it in one batch.

A model folder holds ``model_pron.json`` (vocabulary chain + light G2P) and, when trained with
``--pron-g2p phonetisaurus``, ``model_pron.fst``. flexitag applies it after tagging when its model
folder has one; ``process --pron-backend light|phonetisaurus --pron-model PATH`` applies one after any
backend (a tagger for written text with a corpus's pronunciations). Tokens get ``pron``,
``pron_source`` (``vocab:<level>``, ``g2p:light``, ``g2p:phonetisaurus``) and ``pron_conf``; a
pronunciation the input already has is kept.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .g2p_light import PronunciationModel, next_context

SPACE = "▁"  # a space inside a pronunciation, as one symbol for Phonetisaurus


# ── Phonetisaurus ────────────────────────────────────────────────────────

def phonetisaurus_program(name: str) -> Optional[str]:
    folder = os.environ.get("FLEXIPIPE_PHONETISAURUS_DIR")
    if folder and Path(folder, name).is_file():
        return str(Path(folder, name))
    return shutil.which(name)


def phonetisaurus_available(training: bool = False) -> Tuple[bool, List[str]]:
    """(available, missing programs)."""
    needed = ["phonetisaurus-g2pfst"]
    if training:
        needed += ["phonetisaurus-align", "estimate-ngram", "phonetisaurus-arpa2wfst"]
    missing = [n for n in needed if not phonetisaurus_program(n)]
    return not missing, missing


def _symbols(pron: str) -> str:
    return " ".join(SPACE if ch == " " else ch for ch in pron)


def train_phonetisaurus(pairs: Iterable[Tuple[str, str]], output: Path, order: int = 7) -> Path:
    """A Phonetisaurus model (output, .fst) from (form, pronunciation) pairs — one per form (the most
    frequent pronunciation): letters and sounds as symbols."""
    ok, missing = phonetisaurus_available(training=True)
    if not ok:
        raise RuntimeError("Phonetisaurus training needs " + ", ".join(missing) + " (on PATH or in $FLEXIPIPE_PHONETISAURUS_DIR)")
    with tempfile.TemporaryDirectory(prefix="flexipipe-phonetisaurus-") as tmp:
        lexicon, corpus, arpa = Path(tmp, "lexicon.dict"), Path(tmp, "lexicon.corpus"), Path(tmp, "lexicon.arpa")
        with open(lexicon, "w", encoding="utf-8") as fh:
            for form, pron in pairs:
                if form and pron and "\t" not in form:
                    fh.write(f"{form.lower()}\t{_symbols(pron)}\n")
        run = lambda args: subprocess.run(args, check=True, capture_output=True, text=True)
        run([phonetisaurus_program("phonetisaurus-align"), f"--input={lexicon}", f"--ofile={corpus}",
             "--seq1_del=true", "--seq2_del=true", "--seq1_max=2", "--seq2_max=2", "--grow=true"])
        run([phonetisaurus_program("estimate-ngram"), "-o", str(order), "-t", str(corpus), "-wl", str(arpa)])
        run([phonetisaurus_program("phonetisaurus-arpa2wfst"), f"--lm={arpa}", f"--ofile={output}"])
    return output


class PhonetisaurusG2P:
    def __init__(self, model: Path):
        self.model = Path(model)
        self.program = phonetisaurus_program("phonetisaurus-g2pfst")
        if not self.program:
            raise RuntimeError("phonetisaurus-g2pfst not found (PATH or $FLEXIPIPE_PHONETISAURUS_DIR)")

    def predict_many(self, forms: Iterable[str]) -> Dict[str, Tuple[str, float]]:
        """form (lowercase) -> (pronunciation, confidence from the path score), one batch call."""
        words = sorted({f.lower() for f in forms if f and not f.isspace()})
        if not words:
            return {}
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("\n".join(words) + "\n")
            wordlist = fh.name
        try:
            out = subprocess.run(
                [self.program, f"--model={self.model}", f"--wordlist={wordlist}", "--nbest=1"],
                check=True, capture_output=True, text=True,
            ).stdout
        finally:
            os.unlink(wordlist)
        result: Dict[str, Tuple[str, float]] = {}
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 3 or parts[0] in result:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                score = 0.0
            pron = "".join(" " if s == SPACE else s for s in parts[2].split())
            result[parts[0]] = (pron, round(math.exp(-score / max(1, len(parts[0]))), 2))  # per-letter cost → 0..1
        return result


# ── the step ─────────────────────────────────────────────────────────────

def model_files(path) -> Tuple[Optional[Path], Optional[Path]]:
    """(model_pron.json, model_pron.fst) of a model folder or file."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    js = p if p.is_file() and p.suffix == ".json" and p.name != "model_vocab.json" else folder / "model_pron.json"
    fst = folder / "model_pron.fst"
    return (js if js.exists() else None), (fst if fst.exists() else None)


def annotate_pronunciation(document, model_path, engine: str = "auto", *, replace_generated: bool = False, verbose: bool = False) -> Dict[str, int]:
    """Give the document's tokens their pronunciation. engine: light | phonetisaurus | auto
    (Phonetisaurus when the model has an .fst and the program is there, else light). Returns counts
    per source. `replace_generated`: overwrite pronunciations an earlier backend generated (not the
    input's own)."""
    js, fst = model_files(model_path)
    if js is None:
        raise FileNotFoundError(f"no pronunciation model (model_pron.json) at {model_path}")
    model = PronunciationModel.load(js)
    use_ps = engine == "phonetisaurus" or (engine == "auto" and fst is not None and phonetisaurus_available()[0])
    if use_ps and fst is None:
        raise FileNotFoundError(f"no Phonetisaurus model (model_pron.fst) next to {js}")
    counts: Dict[str, int] = {}
    unknown: List = []
    for sentence in document.sentences:
        units = [t for token in sentence.tokens for t in ((token, *token.subtokens) if token.subtokens else (token,))]
        for i, t in enumerate(units):
            has = (t.attrs.get("pron") or "").strip() not in ("", "_")
            if has and not (replace_generated and t.attrs.get("pron_source")):
                continue
            # what follows (a model trained with context="next": the next word's first letter)
            following = next_context(units[i + 1].form if i + 1 < len(units) else None) if model.uses_context else "#"
            found = model.lookup(t.form, t.xpos or "", t.lemma or "", t.upos or "", t.feats or "", following)
            if found:
                t.attrs["pron"], t.attrs["pron_source"], conf = found
                t.attrs["pron_conf"] = f"{conf:.2f}"
                counts[found[1]] = counts.get(found[1], 0) + 1
            else:
                unknown.append(t)
    if unknown:
        if use_ps:
            got = PhonetisaurusG2P(fst).predict_many(t.form for t in unknown)
        for t in unknown:
            if use_ps and t.form.lower() in got:
                pron, conf = got[t.form.lower()]
                source = "g2p:phonetisaurus"
            else:
                pron, conf = model.g2p.predict(t.form)
                source = "g2p:light"
            t.attrs["pron"], t.attrs["pron_source"], t.attrs["pron_conf"] = pron, source, f"{conf:.2f}"
            counts[source] = counts.get(source, 0) + 1
    if verbose:
        import sys

        print("[flexipipe] pronunciation: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())), file=sys.stderr)
    return counts
