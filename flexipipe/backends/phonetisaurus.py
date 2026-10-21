"""Phonetisaurus as a pronunciation backend (task pron): not a tagger — it runs after the main
backend (process --pron-backend phonetisaurus --pron-model PATH), or inside flexitag when a model trained
with --pron --pron-g2p phonetisaurus has a model_pron.fst. The work is in pron_backends.py; this spec
makes it visible in `flexipipe info backends` with its status (programs present or not)."""

from ..backend_spec import BackendSpec


def _create_phonetisaurus_backend(*args, **kwargs):
    raise ValueError(
        "phonetisaurus is a pronunciation backend, not a tagger: use "
        "--tasks …,pron --pron-backend phonetisaurus --pron-model PATH after the main backend"
    )


BACKEND_SPEC = BackendSpec(
    name="phonetisaurus",
    description="Phonetisaurus - grapheme-to-phoneme (pronunciation of unknown forms, task pron; WFST joint n-gram)",
    factory=_create_phonetisaurus_backend,
    url="https://github.com/AdolfVonKleist/Phonetisaurus",
    install_instructions="Build Phonetisaurus (needs OpenFst; training also MITLM) and put its programs on PATH "
    "or in $FLEXIPIPE_PHONETISAURUS_DIR",
)
