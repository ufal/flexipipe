"""CTexT backend implementation and registry spec."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend_spec import BackendSpec
from ..doc import Document, Entity, Sentence, Token
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
    clean_language_name,
)
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult

# Lazy import - only import ctextcore when backend is actually used
# This allows BACKEND_SPEC to be defined even if ctextcore is not installed


# CTexT language code mapping: flexiPipe ISO codes -> CTexT codes
# CTexT supports: af, nr, nso, ss, st, tn, ts, xh, zu, ve
CTEXT_LANGUAGE_MAP = {
    "af": "af",  # Afrikaans
    "afr": "af",
    "nr": "nr",  # isiNdebele
    "nde": "nr",
    "nso": "nso",  # Sesotho sa Leboa
    "ss": "ss",  # Siswati
    "ssw": "ss",
    "st": "st",  # Sesotho
    "sot": "st",
    "tn": "tn",  # Setswana
    "tsn": "tn",
    "ts": "ts",  # Xitsonga
    "tso": "ts",
    "xh": "xh",  # isiXhosa
    "xho": "xh",
    "zu": "zu",  # isiZulu
    "zul": "zu",
    "ve": "ve",  # Tshivenda
    "ven": "ve",
}

# CTexT technologies mapping to flexiPipe components
CTEXT_TECH_MAP = {
    "tokenizer": "tok",
    "tagger": "pos",
    "upos": "upos",
    "lemmatizer": "lemma",
    "morph": "morph",
    "ner": "ner",
    "chunker": "pc",
    "sent": "sent",
}

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def _map_language_to_ctext(language: str) -> Optional[str]:
    """Map flexiPipe language code to CTexT language code."""
    lang_lower = language.lower().strip()
    return CTEXT_LANGUAGE_MAP.get(lang_lower)


def _merge_ctext_results(
    results_by_tech: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Merge results from multiple CTexT technologies into a single result.
    
    Token IDs are per-sentence, so we match tokens by sentence position and token ID,
    or by character position if IDs don't match.
    
    Args:
        results_by_tech: Dictionary mapping technology names to their JSON outputs
        
    Returns:
        Merged JSON output combining all technologies
    """
    if not results_by_tech:
        return []
    
    # Use the first result as the base structure
    base_tech = list(results_by_tech.keys())[0]
    merged = results_by_tech[base_tech].copy()
    
    # Create sentence-level token lookup maps for each technology
    # Structure: tech -> sentence_index -> token_id -> token_data
    sentence_token_maps = {}
    for tech, result in results_by_tech.items():
        if tech == base_tech:
            continue
        sentence_map = {}
        sent_idx = 0
        for doc_item in result:
            doc_data = doc_item.get("doc", {})
            paragraphs = doc_data.get("p", [])
            if not isinstance(paragraphs, list):
                paragraphs = [paragraphs] if paragraphs else []
            
            for para in paragraphs:
                if not isinstance(para, dict):
                    continue
                sentences_data = para.get("sent", [])
                if not isinstance(sentences_data, list):
                    sentences_data = [sentences_data] if sentences_data else []
                
                for sent_data in sentences_data:
                    if not isinstance(sent_data, dict):
                        continue
                    tokens_data = sent_data.get("tokens", [])
                    if not isinstance(tokens_data, list):
                        continue
                    
                    token_map = {}
                    for tok in tokens_data:
                        if isinstance(tok, dict):
                            token_id = tok.get("id")
                            if token_id is not None:
                                token_map[token_id] = tok
                    if token_map:
                        sentence_map[sent_idx] = token_map
                    sent_idx += 1
        sentence_token_maps[tech] = sentence_map
    
    # Merge token data from all technologies into the base result
    sent_idx = 0
    for doc_item in merged:
        doc_data = doc_item.get("doc", {})
        paragraphs = doc_data.get("p", [])
        if not isinstance(paragraphs, list):
            paragraphs = [paragraphs] if paragraphs else []
        
        for para in paragraphs:
            if not isinstance(para, dict):
                continue
            sentences_data = para.get("sent", [])
            if not isinstance(sentences_data, list):
                sentences_data = [sentences_data] if sentences_data else []
            
            for sent_data in sentences_data:
                if not isinstance(sent_data, dict):
                    continue
                tokens_data = sent_data.get("tokens", [])
                if not isinstance(tokens_data, list):
                    continue
                
                # Get corresponding sentence from other technologies
                for tech, sentence_map in sentence_token_maps.items():
                    other_sentence_tokens = sentence_map.get(sent_idx)
                    if not other_sentence_tokens:
                        continue
                    
                    # Merge tokens by matching ID
                    for tok in tokens_data:
                        if not isinstance(tok, dict):
                            continue
                        token_id = tok.get("id")
                        if token_id is None:
                            continue
                        
                        other_tok = other_sentence_tokens.get(token_id)
                        if other_tok:
                            # Verify tokens match by text/position to avoid mismatches
                            # Match by text form or character position
                            base_text = tok.get("text", "")
                            other_text = other_tok.get("text", "")
                            base_start = tok.get("start_char")
                            other_start = other_tok.get("start_char")
                            
                            # Only merge if tokens match (same text or same position)
                            if base_text == other_text or (base_start is not None and other_start is not None and base_start == other_start):
                                # Merge lemma
                                if "lemma" in other_tok and other_tok["lemma"]:
                                    tok["lemma"] = other_tok["lemma"]
                                # Merge NER
                                if "ner" in other_tok and other_tok["ner"]:
                                    tok["ner"] = other_tok["ner"]
                                # Merge POS tags - both 'pos' (xpos) and 'upos' can coexist
                                # 'pos' technology gives language-specific tags (xpos)
                                # 'upos' technology gives universal POS tags
                                if "pos" in other_tok and other_tok["pos"]:
                                    tok["pos"] = other_tok["pos"]
                                if "upos" in other_tok and other_tok["upos"]:
                                    tok["upos"] = other_tok["upos"]
                
                sent_idx += 1
    
    return merged


def _ctext_json_to_document(
    ctext_output: List[Dict[str, Any]], original_doc: Optional[Document] = None
) -> Document:
    """Convert CTexT JSON output to flexiPipe Document format."""
    doc = Document(id=original_doc.id if original_doc else "")
    if original_doc:
        doc.meta = original_doc.meta.copy()
        doc.attrs = original_doc.attrs.copy()

    # CTexT output structure: [{'doc': {'p': {'sent': {'tokens': [...]}}}}]
    for doc_item in ctext_output:
        if not isinstance(doc_item, dict):
            continue
        doc_data = doc_item.get("doc", {})
        if not isinstance(doc_data, dict):
            continue
        
        # Handle paragraph level
        paragraphs = doc_data.get("p", [])
        if not isinstance(paragraphs, list):
            paragraphs = [paragraphs] if paragraphs else []
        
        for para in paragraphs:
            if not isinstance(para, dict):
                continue
            
            # Handle sentence level
            sentences_data = para.get("sent", [])
            if not isinstance(sentences_data, list):
                sentences_data = [sentences_data] if sentences_data else []
            
            for sent_data in sentences_data:
                if not isinstance(sent_data, dict):
                    continue
                
                tokens_data = sent_data.get("tokens", [])
                if not isinstance(tokens_data, list):
                    continue
                
                # Extract sentence text if available
                sent_text = sent_data.get("text", "")
                if not sent_text and tokens_data:
                    # Reconstruct from tokens
                    token_forms = []
                    for tok in tokens_data:
                        if isinstance(tok, dict):
                            token_forms.append(tok.get("text", ""))
                    sent_text = " ".join(token_forms)
                
                sentence = Sentence(
                    id=len(doc.sentences),
                    text=sent_text,
                    attrs={}
                )
                
                # Process tokens
                for token_idx, tok_data in enumerate(tokens_data, start=1):
                    if not isinstance(tok_data, dict):
                        continue
                    
                    form = tok_data.get("text", "")
                    if not form:
                        continue
                    
                    # Extract token ID (CTexT provides 'id' field)
                    token_id = tok_data.get("id")
                    if token_id is None:
                        # Fallback to sequential ID if not provided
                        token_id = token_idx
                    else:
                        # Ensure it's an integer
                        try:
                            token_id = int(token_id)
                        except (ValueError, TypeError):
                            token_id = token_idx
                    
                    # Extract POS tags - CTexT provides both 'upos' (universal) and 'pos' (language-specific/xpos)
                    # 'upos' technology gives universal POS tags in the 'upos' field
                    # 'pos' technology gives language-specific POS tags in the 'pos' field (these are xpos)
                    upos_tag = tok_data.get("upos", "")
                    xpos_tag = tok_data.get("pos", "")  # CTexT's 'pos' field maps to flexipipe's 'xpos'
                    
                    # Extract lemma
                    # CTexT provides lemmas via the "lemma" technology (separate from UPOS/POS)
                    # If lemma is not provided in this result, it will be merged from other technology calls
                    # Always use the lemma from CTexT if provided, even if it matches the form
                    lemma = tok_data.get("lemma", "")
                    if not lemma:
                        lemma = "_"
                    
                    # Extract morphological features (if available as a dict or string)
                    feats_str = ""
                    feats = tok_data.get("feats") or tok_data.get("morph")
                    if isinstance(feats, dict):
                        feats_str = "|".join(f"{k}={v}" for k, v in feats.items())
                    elif isinstance(feats, str):
                        feats_str = feats
                    
                    # Extract NER if available
                    ner_tag = tok_data.get("ner") or tok_data.get("entity", "")
                    
                    # Extract character positions
                    char_start = tok_data.get("start_char") or tok_data.get("char_start")
                    char_end = tok_data.get("end_char") or tok_data.get("char_end")
                    
                    token = Token(
                        id=token_id,
                        form=form,
                        lemma=lemma,
                        upos=upos_tag or "_",
                        xpos=xpos_tag or "_",  # Map CTexT's 'pos' field to flexipipe's 'xpos'
                        feats=feats_str or "_",
                        head=0,  # CTexT doesn't provide dependency parsing
                        deprel="_",
                        misc="_",
                        char_start=char_start,
                        char_end=char_end,
                    )
                    
                    # Add NER if available
                    if ner_tag and ner_tag != "_":
                        token.misc = f"NE={ner_tag}"
                    
                    sentence.tokens.append(token)
                
                # Note: space_after will be inferred later from original document text
                # CTexT may return text without spaces, so we'll use the original input text
                
                if sentence.tokens:
                    doc.sentences.append(sentence)
    
    return doc


def get_ctext_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Get available CTexT models."""
    # Check for Java first (before importing ctextcore, which will try to start Java)
    import os
    import shutil
    import subprocess
    from pathlib import Path
    
    def _test_java(java_path: str) -> bool:
        """Test if a Java executable is functional."""
        # On macOS, /usr/bin/java is a stub that shows a GUI dialog
        # Check if it's the macOS stub by checking the path
        if java_path == "/usr/bin/java":
            # Check if it's actually a real Java or the stub
            # The stub is a binary, real Java would be elsewhere
            # Skip testing the stub to avoid GUI dialog
            return False
        
        try:
            result = subprocess.run(
                [java_path, "-version"],
                capture_output=True,
                text=True,
                timeout=3,  # Shorter timeout to avoid hanging
            )
            version_output = result.stderr or result.stdout
            # Check for macOS stub error message
            if "Unable to locate a Java Runtime" in version_output or "No Java runtime present" in version_output:
                return False
            # Check if we got actual version output
            if "version" in version_output.lower() and ("openjdk" in version_output.lower() or "java" in version_output.lower()):
                return True
            return False
        except subprocess.TimeoutExpired:
            # Timeout likely means GUI dialog (macOS stub)
            return False
        except subprocess.SubprocessError:
            return False
    
    # Find Java executable - check multiple locations
    java_cmd = None
    
    # 1. Check PATH first, but verify it's functional
    path_java = shutil.which("java")
    if path_java and _test_java(path_java):
        java_cmd = path_java
    
    # 2. Check JAVA_HOME
    if not java_cmd:
        java_home = os.getenv("JAVA_HOME")
        if java_home:
            java_path = Path(java_home) / "bin" / "java"
            if java_path.exists() and _test_java(str(java_path)):
                java_cmd = str(java_path)
    
    # 3. Check common Homebrew locations (macOS)
    if not java_cmd:
        homebrew_prefix = os.getenv("HOMEBREW_PREFIX", "/opt/homebrew")
        common_java_paths = [
            Path(homebrew_prefix) / "opt" / "openjdk@17" / "bin" / "java",
            Path(homebrew_prefix) / "opt" / "openjdk@21" / "bin" / "java",
            Path(homebrew_prefix) / "opt" / "openjdk@11" / "bin" / "java",
            Path("/usr/local/opt/openjdk@17/bin/java"),
            Path("/usr/local/opt/openjdk@21/bin/java"),
            Path("/usr/local/opt/openjdk@11/bin/java"),
        ]
        for java_path in common_java_paths:
            if java_path.exists() and _test_java(str(java_path)):
                java_cmd = str(java_path)
                break
    
    if not java_cmd:
        if verbose:
            print("[flexipipe] CTexT backend requires Java (OpenJDK 17+) but 'java' command not found or not functional", file=sys.stderr)
            print("[flexipipe] On macOS with Homebrew, add Java to PATH:", file=sys.stderr)
            print("[flexipipe]   echo 'export PATH=\"/opt/homebrew/opt/openjdk@17/bin:$PATH\"' >> ~/.zshrc", file=sys.stderr)
            print("[flexipipe] Or set JAVA_HOME:", file=sys.stderr)
            print("[flexipipe]   export JAVA_HOME=/opt/homebrew/opt/openjdk@17", file=sys.stderr)
        return {}
    
    try:
        from ctextcore.core import CCore
    except ImportError as exc:
        if verbose:
            logging.warning(f"CTexT backend requires 'ctextcore' package: {exc}")
        return {}
    
    cache_key = "ctext"
    
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached and cache_entries_standardized(cached):
            return cached
    
    result: Dict[str, Dict[str, str]] = {}
    
    try:
        # Set JAVA_HOME if we found Java in a non-standard location
        # This ensures ctextcore uses the correct Java
        if java_cmd and not java_cmd.startswith("/usr/bin/java"):
            java_bin_dir = Path(java_cmd).parent
            java_home = java_bin_dir.parent
            # For Homebrew Java, the structure is: .../openjdk@17/bin/java
            # JAVA_HOME should point to .../openjdk@17
            if "openjdk" in str(java_home):
                os.environ["JAVA_HOME"] = str(java_home)
            else:
                # For standard installations, JAVA_HOME is usually the parent of bin
                os.environ["JAVA_HOME"] = str(java_home)
        
        # Suppress ctextcore download progress output and Java errors (it prints progress as floats and OCR errors)
        import io
        import contextlib
        
        # Initialize CTexT core to check available models
        # This will hang if Java is not properly installed, so we check above
        # Suppress stdout/stderr during initialization to avoid download progress spam and OCR errors
        # Java subprocess stderr needs to be redirected at the file descriptor level
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            # Save original stderr
            original_stderr = os.dup(2)  # stderr is fd 2
            try:
                # Redirect stderr to devnull
                os.dup2(devnull_fd, 2)
                # Also redirect Python's stderr
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    core = CCore()
                    available_techs = core.list_available_techs()
            finally:
                # Restore original stderr
                os.dup2(original_stderr, 2)
                os.close(original_stderr)
        finally:
            os.close(devnull_fd)
        
        # CTexT languages: af, nr, nso, ss, st, tn, ts, xh, zu, ve
        ctext_languages = {
            "af": ("af", "afr", "Afrikaans"),
            "nr": ("nr", "nde", "isiNdebele"),
            "nso": ("nso", "nso", "Sesotho sa Leboa"),
            "ss": ("ss", "ssw", "Siswati"),
            "st": ("st", "sot", "Sesotho"),
            "tn": ("tn", "tsn", "Setswana"),
            "ts": ("ts", "tso", "Xitsonga"),
            "xh": ("xh", "xho", "isiXhosa"),
            "zu": ("zu", "zul", "isiZulu"),
            "ve": ("ve", "ven", "Tshivenda"),
        }
        
        # Check which technologies are available for each language
        for ctext_code, (iso_1, iso_2, lang_name) in ctext_languages.items():
            # Check available technologies for this language
            for tech_name, tech_code in CTEXT_TECH_MAP.items():
                tech_langs = available_techs.get(tech_code, [])
                # For UPOS, also check if it's available even if not in the list
                # (the website provides UPOS, so it might be available but not listed)
                if ctext_code in tech_langs or (tech_code == "upos" and tech_langs == []):
                    # Create model entry
                    model_key = f"{ctext_code}_{tech_code}"
                    entry = build_model_entry(
                        "ctext",
                        model_key,
                        language_code=iso_1,
                        language_name=lang_name,
                        features=f"{tech_name}",
                        name=f"{lang_name} ({tech_name})",
                    )
                    entry["ctext_language"] = ctext_code
                    entry["ctext_tech"] = tech_code
                    result[model_key] = entry
        
        # Also create combined models that support multiple technologies
        # For languages that support POS/UPOS, create a "full" model entry
        for ctext_code, (iso_1, iso_2, lang_name) in ctext_languages.items():
            supported_techs = []
            # Check POS and UPOS (prefer UPOS if available)
            for tech_code in ["upos", "pos", "tok", "sent"]:
                tech_langs = available_techs.get(tech_code, [])
                # For UPOS, also check if it's available even if not in the list
                if ctext_code in tech_langs or (tech_code == "upos" and tech_langs == []):
                    supported_techs.append(tech_code)
            
            # Also check for lemma and NER (these are separate technologies)
            for tech_code in ["lemma", "ner"]:
                tech_langs = available_techs.get(tech_code, [])
                if ctext_code in tech_langs or (tech_langs == [] and tech_code in available_techs):
                    supported_techs.append(tech_code)
            
            if supported_techs:
                # Determine features description
                has_upos = "upos" in supported_techs
                has_pos = "pos" in supported_techs
                has_lemma = "lemma" in supported_techs
                has_ner = "ner" in supported_techs
                
                features_parts = []
                if has_upos:
                    features_parts.append("UPOS")
                elif has_pos:
                    features_parts.append("POS")
                features_parts.append("tokenization")
                features_parts.append("sentence segmentation")
                if has_lemma:
                    features_parts.append("lemmatization")
                if has_ner:
                    features_parts.append("NER")
                
                features_desc = ", ".join(features_parts)
                
                model_key = f"{ctext_code}_full"
                entry = build_model_entry(
                    "ctext",
                    model_key,
                    language_code=iso_1,
                    language_name=lang_name,
                    features=features_desc,
                    name=f"{lang_name} (full pipeline)",
                )
                entry["ctext_language"] = ctext_code
                entry["ctext_techs"] = supported_techs
                result[model_key] = entry
        
        if refresh_cache:
            try:
                write_model_cache_entry(cache_key, result)
            except (OSError, PermissionError):
                pass
        
    except Exception as exc:
        if verbose:
            logging.warning(f"Failed to list CTexT models: {exc}")
        # Return empty dict on error
        return {}
    
    return result


def list_ctext_models(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
) -> int:
    """List available CTexT models."""
    try:
        entries = get_ctext_model_entries(
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            verbose=True,
        )
        
        print(f"\nAvailable CTexT models:")
        print(f"{'Model Key':<30} {'ISO':<8} {'Language':<25} {'Features':<30}")
        print("=" * 100)
        
        sorted_items = sorted(
            entries.items(),
            key=lambda x: (
                x[1].get(LANGUAGE_FIELD_ISO) or x[1].get(LANGUAGE_FIELD_NAME) or "",
                x[0]
            )
        )
        
        for model_key, entry in sorted_items:
            lang_iso = entry.get(LANGUAGE_FIELD_ISO) or ""
            lang_display = entry.get(LANGUAGE_FIELD_NAME) or ""
            features = entry.get("features", "unknown")
            print(f"{model_key:<30} {lang_iso:<8} {lang_display:<25} {features:<30}")
        
        print(f"\nTotal: {len(entries)} model(s)")
        return 0
    except Exception as e:
        print(f"Error listing CTexT models: {e}")
        import traceback
        traceback.print_exc()
        return 1


class CTexTBackend(BackendManager):
    """CTexT-based backend for South African languages."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        tech: Optional[str] = None,
        download_model: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize CTexT backend.
        
        Args:
            model_name: Model name (e.g., "zu_pos" or "af_full")
            language: Language code (will be mapped to CTexT code)
            tech: Technology to use (pos, upos, tok, sent, ner, lemma, morph, pc)
            download_model: Whether flexipipe is allowed to auto-download missing models
            verbose: Enable verbose logging
        """
        self._model_name = model_name
        self._actual_techs_used: Optional[List[str]] = None  # Track which technologies are actually used
        self._language = language
        self._tech = tech
        self._verbose = verbose
        # Respect flexipipe's install-models / download-model directive
        # (set via --download-model, config, or TEITOK settings)
        self._download_model = bool(download_model)
        
        # Check for Java before trying to import ctextcore
        import os
        import shutil
        import subprocess
        from pathlib import Path
        
        def _test_java(java_path: str) -> bool:
            """Test if a Java executable is functional."""
            try:
                result = subprocess.run(
                    [java_path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version_output = result.stderr or result.stdout
                # Check for macOS stub error message
                if "Unable to locate a Java Runtime" in version_output or "No Java runtime present" in version_output:
                    return False
                return True
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                return False
        
        # Find Java executable - check multiple locations
        java_cmd = None
        
        # 1. Check PATH first, but verify it's functional
        path_java = shutil.which("java")
        if path_java and _test_java(path_java):
            java_cmd = path_java
        
        # 2. Check JAVA_HOME
        if not java_cmd:
            java_home = os.getenv("JAVA_HOME")
            if java_home:
                java_path = Path(java_home) / "bin" / "java"
                if java_path.exists() and _test_java(str(java_path)):
                    java_cmd = str(java_path)
        
        # 3. Check common Homebrew locations (macOS)
        if not java_cmd:
            homebrew_prefix = os.getenv("HOMEBREW_PREFIX", "/opt/homebrew")
            common_java_paths = [
                Path(homebrew_prefix) / "opt" / "openjdk@17" / "bin" / "java",
                Path(homebrew_prefix) / "opt" / "openjdk@21" / "bin" / "java",
                Path(homebrew_prefix) / "opt" / "openjdk@11" / "bin" / "java",
                Path("/usr/local/opt/openjdk@17/bin/java"),
                Path("/usr/local/opt/openjdk@21/bin/java"),
                Path("/usr/local/opt/openjdk@11/bin/java"),
            ]
            for java_path in common_java_paths:
                if java_path.exists() and _test_java(str(java_path)):
                    java_cmd = str(java_path)
                    break
        
        if not java_cmd:
            raise RuntimeError(
                "CTexT backend requires Java (OpenJDK 17+) to be installed and available.\n"
                "On macOS with Homebrew, add Java to PATH:\n"
                "  echo 'export PATH=\"/opt/homebrew/opt/openjdk@17/bin:$PATH\"' >> ~/.zshrc\n"
                "Or set JAVA_HOME:\n"
                "  export JAVA_HOME=/opt/homebrew/opt/openjdk@17\n"
                "Or install from: https://openjdk.org"
            )
        
        # Lazy import and initialize CTexT core
        try:
            from ctextcore.core import CCore
        except ImportError as exc:
            raise ImportError(
                "CTexT backend requires the 'ctextcore' package. "
                "Install it with: pip install ctextcore"
            ) from exc
        
        # Set JAVA_HOME if we found Java in a non-standard location
        # This ensures ctextcore uses the correct Java
        if java_cmd and not java_cmd.startswith("/usr/bin/java"):
            java_bin_dir = Path(java_cmd).parent
            java_home = java_bin_dir.parent
            # For Homebrew Java, the structure is: .../openjdk@17/bin/java
            # JAVA_HOME should point to .../openjdk@17
            if "openjdk" in str(java_home):
                os.environ["JAVA_HOME"] = str(java_home)
            else:
                # For standard installations, JAVA_HOME is usually the parent of bin
                os.environ["JAVA_HOME"] = str(java_home)
        
        # Redirect stderr/stdout BEFORE initializing CCore() to suppress Java errors
        import io
        import contextlib
        
        devnull_fd_init = os.open(os.devnull, os.O_WRONLY)
        try:
            original_stderr_init = os.dup(2)
            original_stdout_init = os.dup(1)
            try:
                # Redirect both stderr and stdout to suppress Java errors during initialization
                os.dup2(devnull_fd_init, 2)
                os.dup2(devnull_fd_init, 1)
                
                try:
                    self._core = CCore(**kwargs)
                finally:
                    # Restore stdout/stderr after initialization
                    os.dup2(original_stdout_init, 1)
                    os.dup2(original_stderr_init, 2)
            finally:
                os.close(original_stderr_init)
                os.close(original_stdout_init)
        finally:
            os.close(devnull_fd_init)
        
        # Now handle exceptions (but stderr/stdout are restored)
        try:
            pass  # CCore() already initialized above
        except Exception as exc:
            error_msg = str(exc)
            # Check if it's a Java-related error
            if "java" in error_msg.lower() or "Java Runtime" in error_msg or "Unable to locate a Java Runtime" in error_msg:
                raise RuntimeError(
                    "CTexT backend requires Java (OpenJDK 17+) to be installed and available on PATH. "
                    "Install Java from https://openjdk.org or via your system package manager. "
                    f"Original error: {exc}"
                ) from exc
            raise RuntimeError(f"Failed to initialize CTexT core: {exc}") from exc
        
        # Resolve CTexT language code
        if model_name and "_" in model_name:
            # Extract language from model name (e.g., "zu_pos" -> "zu")
            parts = model_name.split("_", 1)
            ctext_lang = parts[0]
        elif language:
            ctext_lang = _map_language_to_ctext(language)
            if not ctext_lang:
                raise ValueError(
                    f"Language '{language}' is not supported by CTexT. "
                    f"Supported languages: {', '.join(sorted(set(CTEXT_LANGUAGE_MAP.values())))}"
                )
        else:
            raise ValueError("Either model_name or language must be provided")
        
        self._ctext_language = ctext_lang
        
        # Resolve technology
        if model_name and "_" in model_name:
            parts = model_name.split("_", 1)
            tech_from_model = parts[1]
            if tech_from_model in CTEXT_TECH_MAP.values():
                self._ctext_tech = tech_from_model
            elif tech_from_model == "full":
                # Full pipeline - will use multiple technologies
                self._ctext_tech = None
            else:
                self._ctext_tech = tech or "pos"  # Default to POS
        else:
            self._ctext_tech = tech or "pos"  # Default to POS
        
        # Check if model is available
        if self._ctext_tech:
            try:
                available_langs = self._core.get_available_languages(self._ctext_tech)
                if self._ctext_language not in available_langs:
                    raise ValueError(
                        f"CTexT technology '{self._ctext_tech}' is not available for language '{self._ctext_language}'"
                    )
            except Exception as exc:
                if verbose:
                    logging.warning(f"Could not verify model availability: {exc}")

    @property
    def model_name(self) -> str:
        if self._model_name:
            return self._model_name
        # If we've tracked which technologies were actually used, use that
        if self._actual_techs_used:
            return f"{self._ctext_language}_{'+'.join(self._actual_techs_used)}"
        # Otherwise, use the tech name or "full"
        if self._ctext_tech == "full" or self._ctext_tech is None:
            return f"{self._ctext_language}_full"
        return f"{self._ctext_language}_{self._ctext_tech}"

    @property
    def _backend_info(self) -> str:
        """Used by CLI to describe this backend."""
        return f"ctext: {self.model_name}"

    def supports_training(self) -> bool:
        return False

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        """Tag a document using CTexT."""
        del preserve_pos_tags, components
        
        start_time = time.time()
        
        # Convert document to text
        # Store original text for SpaceAfter inference (CTexT may return text without spaces)
        # CTexT character positions are relative to the full input text, not per-sentence
        original_text_by_sentence = []
        original_full_text = None
        if use_raw_text or not document.sentences:
            text_parts = []
            for sentence in document.sentences:
                if sentence.text:
                    text_parts.append(sentence.text)
                    original_text_by_sentence.append(sentence.text)
                else:
                    # Fallback: reconstruct from tokens
                    reconstructed = " ".join(token.form for token in sentence.tokens)
                    text_parts.append(reconstructed)
                    original_text_by_sentence.append(reconstructed)
            text = "\n".join(text_parts)
            original_full_text = text  # Store full text for character position lookup
            if not text.strip():
                # Fallback: reconstruct from tokens
                text = "\n".join(
                    " ".join(token.form for token in sent.tokens)
                    for sent in document.sentences
                )
                original_text_by_sentence = [
                    " ".join(token.form for token in sent.tokens)
                    for sent in document.sentences
                ]
                original_full_text = text
        else:
            # Use tokenized text
            text_parts = []
            for sentence in document.sentences:
                if sentence.text:
                    # Prefer original sentence text if available
                    reconstructed = sentence.text
                    original_text_by_sentence.append(reconstructed)
                else:
                    # Reconstruct from tokens
                    reconstructed = " ".join(token.form for token in sentence.tokens)
                    original_text_by_sentence.append(reconstructed)
                text_parts.append(reconstructed)
            text = "\n".join(text_parts)
            original_full_text = text  # Store full text for character position lookup
        
        if not text.strip():
            # Empty document
            return NeuralResult(
                document=Document(id=document.id, meta=dict(document.meta)),
                stats={"elapsed_seconds": time.time() - start_time},
            )
        
        # Determine which technology/technologies to use
        # CTexT POS/UPOS includes tokenization, so we can use it directly
        # For full pipeline, we combine multiple technologies: UPOS + lemma + NER
        techs_to_use = []
        if self._ctext_tech == "full" or self._ctext_tech is None:
            # Full pipeline: try to combine UPOS + lemma + NER
            # Get available technologies using list_available_techs()
            try:
                available_techs = self._core.list_available_techs()
            except Exception:
                available_techs = {}
            
            # For full pipeline, use both UPOS and POS if both are available
            # UPOS gives universal POS tags, POS gives language-specific tags (xpos)
            upos_langs = available_techs.get("upos", [])
            pos_langs = available_techs.get("pos", [])
            upos_available = self._ctext_language in upos_langs or (upos_langs == [] and "upos" in available_techs)
            pos_available = self._ctext_language in pos_langs or (pos_langs == [] and "pos" in available_techs)
            
            if upos_available:
                techs_to_use.append("upos")
            if pos_available:
                techs_to_use.append("pos")
            
            # If neither is available, try UPOS as fallback (will fail gracefully if not available)
            if not techs_to_use:
                techs_to_use.append("upos")
            
            # Add lemma if available
            lemma_langs = available_techs.get("lemma", [])
            if self._ctext_language in lemma_langs or (lemma_langs == [] and "lemma" in available_techs):
                techs_to_use.append("lemma")
            
            # Add NER if available
            ner_langs = available_techs.get("ner", [])
            if self._ctext_language in ner_langs or (ner_langs == [] and "ner" in available_techs):
                techs_to_use.append("ner")
        else:
            techs_to_use = [self._ctext_tech]
        
        # Respect flexipipe's download_model directive before calling CTexT.
        # When download_model is enabled, try to trigger a non-interactive download
        # via ctextcore's download_model() API (if available) so that the user
        # does not see CTexT's raw prompt / progress output.
        auto_download = getattr(self, "_download_model", False)
        if auto_download:
            download_func = getattr(self._core, "download_model", None)
            if download_func is not None:
                try:
                    import sys
                    # Download all required technologies
                    for tech in techs_to_use:
                        try:
                            # Concise, flexipipe-style status message
                            print(
                                f"[flexipipe] Ensuring CTExT model is installed for language "
                                f"'{self._ctext_language}' (tech: {tech})...",
                                file=sys.stderr,
                                flush=True,
                            )
                            # Suppress any noisy progress output from ctextcore itself
                            devnull_fd_dl = os.open(os.devnull, os.O_WRONLY)
                            try:
                                original_stderr_dl = os.dup(2)
                                try:
                                    os.dup2(devnull_fd_dl, 2)
                                    with contextlib.redirect_stderr(io.StringIO()):
                                        # Try both positional and keyword calling conventions
                                        try:
                                            download_func(self._ctext_language, tech)
                                        except TypeError:
                                            download_func(language=self._ctext_language, tech=tech)
                                finally:
                                    os.dup2(original_stderr_dl, 2)
                                    os.close(original_stderr_dl)
                            finally:
                                os.close(devnull_fd_dl)
                        except Exception:
                            # If automatic download fails for one tech, continue with others
                            # CTExT's own handling will prompt if needed
                            pass
                except Exception:
                    # If automatic download fails, we fall back to CTExT's own handling,
                    # which may still prompt the user. This is better than hard-failing.
                    pass
        
        # Process with CTexT
        # Handle download prompts and suppress Java errors from ctextcore
        import io
        import os
        import sys
        import contextlib
        
        # Check if we should auto-answer download prompts
        auto_download = getattr(self, "_download_model", False)
        
        try:
            # Suppress Java OCR errors (stderr) and Java errors (stdout)
            # Java errors are printed to stdout, not stderr, so we need to redirect both
            # The Java subprocess writes directly to file descriptor 1, bypassing sys.stdout
            # So we need to redirect at the file descriptor level BEFORE any Java process starts
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                # Save original stderr and stdout
                original_stderr = os.dup(2)  # stderr is fd 2
                original_stdout_main = os.dup(1)  # stdout is fd 1
                try:
                    # Redirect both stderr and stdout to devnull to suppress Java errors
                    # This must happen BEFORE any ctextcore calls, including process_text
                    os.dup2(devnull_fd, 2)
                    os.dup2(devnull_fd, 1)
                    # Also redirect sys.stdout to ensure Python-level redirection works
                    # This is important because some libraries might cache sys.stdout
                    original_sys_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    # Helper function to call a single technology
                    # This needs to handle download prompts
                    def call_tech(tech_name: str, allow_prompt: bool = True) -> Optional[List[Dict[str, Any]]]:
                        """Call a single CTexT technology and return its result.
                        
                        Args:
                            tech_name: Technology to call
                            allow_prompt: If False, skip technologies that need downloading instead of prompting
                        """
                        try:
                            # If auto_download is enabled, mock stdin to auto-answer "Y"
                            if auto_download:
                                # Auto-answer "Y" to download prompts
                                class AutoAnswerStdin:
                                    def __init__(self, original_stdin):
                                        self._original = original_stdin
                                        self._buffer = io.StringIO("Y\n")
                                    
                                    def read(self, size=-1):
                                        return self._buffer.read(size)
                                    
                                    def readline(self, size=-1):
                                        return self._buffer.readline(size)
                                    
                                    def __getattr__(self, name):
                                        return getattr(self._original, name)
                                
                                original_stdin = sys.stdin
                                sys.stdin = AutoAnswerStdin(original_stdin)
                                try:
                                    # stdout is already redirected at the outer level to suppress Java errors
                                    # No need to redirect again here
                                    return self._core.process_text(
                                        text_input=text,
                                        language=self._ctext_language,
                                        tech=tech_name,
                                        output_format="json",
                                    )
                                finally:
                                    sys.stdin = original_stdin
                            elif not allow_prompt:
                                # Don't allow prompts - auto-answer "N" to skip downloading
                                # This allows the pipeline to continue without non-essential technologies
                                # Suppress stdout to hide the download prompt message
                                class AutoAnswerNoStdin:
                                    def __init__(self, original_stdin):
                                        self._original = original_stdin
                                        self._buffer = io.StringIO("N\n")  # Answer "N" to skip download
                                    
                                    def read(self, size=-1):
                                        return self._buffer.read(size)
                                    
                                    def readline(self, size=-1):
                                        return self._buffer.readline(size)
                                    
                                    def __getattr__(self, name):
                                        return getattr(self._original, name)
                                
                                original_stdin = sys.stdin
                                sys.stdin = AutoAnswerNoStdin(original_stdin)
                                try:
                                    # stdout is already redirected at the outer level to suppress Java errors
                                    # No need to redirect again here
                                    # Try to call it - if it prompts, it will get "N" and skip
                                    return self._core.process_text(
                                        text_input=text,
                                        language=self._ctext_language,
                                        tech=tech_name,
                                        output_format="json",
                                    )
                                except Exception as e:
                                    # If it fails because we answered "N", that's expected - skip this technology
                                    error_str = str(e).lower()
                                    if "not installed" in error_str or "download" in error_str or "not available" in error_str:
                                        if self._verbose:
                                            logging.warning(
                                                f"CTexT technology '{tech_name}' requires download. "
                                                f"Use --download-model to auto-download. Skipping."
                                            )
                                        return None
                                    # Other errors - re-raise
                                    raise
                                finally:
                                    sys.stdin = original_stdin
                            else:
                                # Normal interactive prompt (stdin is a TTY and prompts are allowed)
                                # stdout is already redirected at the outer level to suppress Java errors
                                # But we need to ensure it stays redirected during process_text
                                # (it should already be redirected, but be explicit)
                                # Note: prompts from ctextcore will not be visible, but ctextcore will
                                # still read from stdin, so it will work (user just won't see the prompt)
                                return self._core.process_text(
                                    text_input=text,
                                    language=self._ctext_language,
                                    tech=tech_name,
                                    output_format="json",
                                )
                        except Exception as e:
                            # If technology is not available, return None
                            error_str = str(e).lower()
                            # Check if it's a download-related error
                            if "not installed" in error_str or "download" in error_str:
                                if not allow_prompt:
                                    if self._verbose:
                                        logging.warning(
                                            f"CTexT technology '{tech_name}' requires download. "
                                            f"Use --download-model to auto-download. Skipping."
                                        )
                                    return None
                                # If allow_prompt is True but it still fails, it might need download
                                # Try to continue - the error might be transient
                                if self._verbose:
                                    logging.warning(f"CTexT technology '{tech_name}' failed (may need download): {e}")
                                return None
                            # Check if it's an initialization error (e.g., "could not initialise")
                            # These are different from download errors - the model might be downloaded but not working
                            if "could not initialise" in error_str or ("initialise" in error_str and "error" in error_str):
                                # Log but don't fail completely - NER might not be available for this language
                                if self._verbose:
                                    logging.warning(f"CTexT technology '{tech_name}' could not be initialized: {e}")
                                return None
                            if self._verbose:
                                logging.warning(f"CTexT technology '{tech_name}' failed: {e}")
                            return None
                    
                    # Set up stdout suppression for download progress if auto_download is enabled
                    if auto_download:
                        # stdout is already redirected at the outer level to suppress Java errors
                        # No need to redirect again here
                        with contextlib.redirect_stderr(io.StringIO()):
                            # Call all technologies and collect results
                            # When auto_download is True, all technologies should auto-download
                            # When auto_download is False, only essential technologies (upos, pos) should prompt
                            essential_techs = ["upos", "pos"]
                            results_by_tech = {}
                            for tech in techs_to_use:
                                # When auto_download is True, allow all technologies to download
                                # When auto_download is False, only essential technologies can prompt
                                allow_prompt = auto_download or (tech in essential_techs)
                                tech_result = call_tech(tech, allow_prompt=allow_prompt)
                                if tech_result:
                                    results_by_tech[tech] = tech_result
                            
                            # Track which technologies were actually used successfully
                            if results_by_tech:
                                self._actual_techs_used = list(results_by_tech.keys())
                                
                                # Merge results if we have multiple technologies
                                if len(results_by_tech) > 1:
                                    result = _merge_ctext_results(results_by_tech)
                                elif len(results_by_tech) == 1:
                                    result = list(results_by_tech.values())[0]
                                else:
                                    # No technologies succeeded - try fallback
                                    if "upos" in techs_to_use:
                                        tech_result = call_tech("pos", allow_prompt=True)
                                        if tech_result:
                                            result = tech_result
                                        else:
                                            raise RuntimeError(
                                                "No CTExT technologies available. "
                                                "Use --download-model to auto-download missing models."
                                            )
                                    else:
                                        raise RuntimeError(
                                            "No CTExT technologies available. "
                                            "Use --download-model to auto-download missing models."
                                        )
                        # stdout is already redirected at outer level, no need to restore here
                    else:
                        # Normal processing - allow interactive prompt for essential technologies only
                        # stdout is already redirected to suppress Java errors
                        # For non-essential technologies (lemma, ner), suppress prompts
                        essential_techs = ["upos", "pos"]
                        try:
                            with contextlib.redirect_stderr(io.StringIO()):
                                # Call all technologies and collect results
                                results_by_tech = {}
                                for tech in techs_to_use:
                                    # Allow prompts only for essential technologies
                                    allow_prompt = tech in essential_techs
                                    
                                    # stdout is already redirected to devnull to suppress Java errors
                                    # call_tech will handle stdout redirection internally if needed
                                    try:
                                        tech_result = call_tech(tech, allow_prompt=allow_prompt)
                                        if tech_result:
                                            results_by_tech[tech] = tech_result
                                    finally:
                                        # Ensure stdout stays redirected (it should already be)
                                        pass
                                
                                # Track which technologies were actually used successfully
                                if results_by_tech:
                                    self._actual_techs_used = list(results_by_tech.keys())
                                
                                # Merge results if we have multiple technologies
                                if len(results_by_tech) > 1:
                                    result = _merge_ctext_results(results_by_tech)
                                elif len(results_by_tech) == 1:
                                    result = list(results_by_tech.values())[0]
                                else:
                                    # No technologies succeeded - try fallback
                                    if "upos" in techs_to_use:
                                        tech_result = call_tech("pos", allow_prompt=True)
                                        if tech_result:
                                            result = tech_result
                                        else:
                                            raise RuntimeError(
                                                "No CTexT technologies available. "
                                                "Use --download-model to auto-download missing models."
                                            )
                                    else:
                                        raise RuntimeError(
                                            "No CTexT technologies available. "
                                            "Use --download-model to auto-download missing models."
                                        )
                        finally:
                            # stdout is already redirected, no need to restore here
                            pass
                finally:
                    # Restore sys.stdout
                    sys.stdout.close()
                    sys.stdout = original_sys_stdout
                    # Restore original stderr and stdout file descriptors
                    os.dup2(original_stderr, 2)
                    os.dup2(original_stdout_main, 1)
                    os.close(original_stderr)
                    os.close(original_stdout_main)
            finally:
                os.close(devnull_fd)
            
            # Debug: print raw CTexT output if verbose
            if self._verbose:
                import json
                import sys
                print("[flexipipe] DEBUG: Raw CTexT JSON output:", file=sys.stderr)
                print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
            
            # Convert to Document
            result_doc = _ctext_json_to_document(result, original_doc=document)
            
            # Infer space_after using CTexT's character positions
            # CTexT provides start_char and end_char for each token, which are relative to the full input text
            # We use these to check the actual spacing in the original full text
            if original_full_text:
                # Use the full original text for character position lookup
                # CTexT character positions are relative to the full input text, not per-sentence
                if self._verbose:
                    import sys
                    print(f"[flexipipe] DEBUG: Inferring spaceAfter using full text (length={len(original_full_text)}), result_doc has {len(result_doc.sentences)} sentences", file=sys.stderr)
                
                # Infer space_after for all tokens using character positions from full text
                for sentence in result_doc.sentences:
                    for token_idx, token in enumerate(sentence.tokens):
                        is_last_token = (token_idx == len(sentence.tokens) - 1)
                        
                        # Use character positions from CTexT if available
                        char_end = getattr(token, 'char_end', None)
                        if char_end is not None:
                            if char_end < len(original_full_text):
                                # Check if there's whitespace after this token
                                # char_end is the position AFTER the token, so original_full_text[char_end] is the next character
                                next_char = original_full_text[char_end]
                                token.space_after = next_char.isspace()
                                if self._verbose:
                                    import sys
                                    print(f"[flexipipe] DEBUG: Token '{token.form}' (char_end={char_end}): next_char={repr(next_char)}, space_after={token.space_after}", file=sys.stderr)
                            elif is_last_token:
                                # Last token - set to None (no SpaceAfter entry in CoNLL-U)
                                token.space_after = None
                            else:
                                # char_end is beyond text length - shouldn't happen, but default to True
                                token.space_after = True
                        elif is_last_token:
                            # Last token - set to None (no SpaceAfter entry in CoNLL-U)
                            token.space_after = None
                        else:
                            # No character position available - fall back to text matching
                            if self._verbose:
                                import sys
                                print(f"[flexipipe] DEBUG: Token '{token.form}' has no char_end, will use fallback", file=sys.stderr)
                
                # Update sentence texts from original_text_by_sentence if counts match
                if original_text_by_sentence and len(result_doc.sentences) == len(original_text_by_sentence):
                    for sent_idx, sentence in enumerate(result_doc.sentences):
                        if sent_idx < len(original_text_by_sentence):
                            sentence.text = original_text_by_sentence[sent_idx]
                else:
                    # If sentence counts don't match, reconstruct sentence texts from tokens
                    # This preserves the sentence structure from CTexT
                    for sentence in result_doc.sentences:
                        if not sentence.text:
                            from ..conllu import _reconstruct_sentence_text
                            sentence.text = _reconstruct_sentence_text(sentence.tokens)
            elif original_text_by_sentence and len(result_doc.sentences) == len(original_text_by_sentence):
                # Fallback: match sentences and use standard inference
                for sent_idx, sentence in enumerate(result_doc.sentences):
                    if sent_idx < len(original_text_by_sentence):
                        sentence.text = original_text_by_sentence[sent_idx]
                from ..engine import _infer_space_after_from_text
                _infer_space_after_from_text(result_doc)
            elif document.sentences and document.sentences[0].text:
                # Fallback: use original document text with standard inference
                from ..engine import _infer_space_after_from_text
                _infer_space_after_from_text(result_doc)
            
        except Exception as exc:
            raise RuntimeError(f"CTexT processing failed: {exc}") from exc
        
        elapsed = time.time() - start_time
        token_count = sum(len(sent.tokens) for sent in result_doc.sentences)
        stats = {
            "elapsed_seconds": elapsed,
            "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
            "sentences_per_second": len(result_doc.sentences) / elapsed if elapsed > 0 else 0.0,
        }
        
        return NeuralResult(document=result_doc, stats=stats)
    
    def train(
        self,
        train_data: Union[Document, List[Document], Path],
        output_dir: Path,
        *,
        dev_data: Optional[Union[Document, List[Document], Path]] = None,
        **kwargs,
    ) -> Path:
        raise NotImplementedError("CTexT backend does not support training")


def _create_ctext_backend(
    *,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    tech: Optional[str] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> CTexTBackend:
    """Factory function to create CTexT backend."""
    from ..backend_utils import validate_backend_kwargs, resolve_model_from_language
    
    validate_backend_kwargs(kwargs, "CTexT", allowed_extra=["port", "timeout", "threads", "memory", "be_quiet", "max_char_length", "download_model", "training"])
    
    # Resolve model from language if needed
    resolved_model = None
    if language and not model_name:
        try:
            resolved_model = resolve_model_from_language(
                language=language,
                backend_name="ctext",
                model_name=model_name,
                preferred_only=True,
                use_cache=True,
            )
        except ValueError:
            # No model found in registry - that's okay, we'll try to create one from language
            pass
    
    return CTexTBackend(
        model_name=resolved_model or model_name,
        language=language,
        tech=tech,
        verbose=verbose,
        **kwargs,
    )


BACKEND_SPEC = BackendSpec(
    name="ctext",
    description="CTexT - NCHLT core technologies for South African languages",
    factory=_create_ctext_backend,
    get_model_entries=get_ctext_model_entries,
    list_models=list_ctext_models,
    supports_training=False,
    is_rest=False,
    url="https://github.com/ctextdev/ctextcore",
    install_instructions="Install via: pip install ctextcore (requires Java OpenJDK 17+ - install from https://openjdk.org)",
)
