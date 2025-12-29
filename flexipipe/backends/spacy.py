"""Backend spec for the SpaCy backend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from ..backend_spec import BackendSpec
from ..language_utils import LANGUAGE_FIELD_ISO, LANGUAGE_FIELD_NAME
from ..model_storage import get_backend_models_dir
from ..spacy_backend import (
    SPACY_DEFAULT_MODELS,
    SpacyBackend,
    get_spacy_model_entries,
    list_spacy_models,
)
def _resolve_spacy_model_name(
    model_name: Optional[str],
    language: Optional[str],
) -> Optional[str]:
    """Resolve SpaCy model name using central resolution function."""
    from ..backend_utils import resolve_model_from_language
    
    try:
        return resolve_model_from_language(
            language=language,
            backend_name="spacy",
            model_name=model_name,
            preferred_only=True,
            use_cache=True,
        )
    except ValueError:
        # If central resolution fails, try SpaCy default models as fallback
        if not language:
            return model_name
        lang_norm = language.lower()
        from ..language_mapping import normalize_language_code
        lang_iso1, lang_iso2, lang_iso3 = normalize_language_code(lang_norm)
        lang_iso_codes = {lang_iso1, lang_iso2, lang_iso3, lang_norm} - {None}
        for iso_code in lang_iso_codes:
            if iso_code in SPACY_DEFAULT_MODELS:
                return SPACY_DEFAULT_MODELS[iso_code]
        return SPACY_DEFAULT_MODELS.get(lang_norm)



def _ensure_spacy_model_available(model_name: str) -> None:
    try:
        import spacy
        from spacy.cli import download as spacy_download
    except ImportError as exc:  # pragma: no cover - import guard
        raise ValueError(
            "SpaCy is not installed. Install it with: pip install \"flexipipe[spacy]\""
        ) from exc

    # Already a path on disk
    if Path(model_name).exists():
        return

    # Already installed as a package?
    try:
        spacy.util.get_package_path(model_name)
        return
    except (OSError, IOError, ImportError, AttributeError):
        pass

    spacy_dir = get_backend_models_dir("spacy")
    target_path = spacy_dir / model_name
    
    # Check if already downloaded to flexipipe directory
    if target_path.exists() and (target_path / "meta.json").exists():
        return
    
    # Check registry for download_url (flexipipe or community models)
    try:
        from ..model_registry import get_remote_models_for_backend
        registry_models = get_remote_models_for_backend(
            "spacy",
            source=None,  # Check all sources
            use_cache=True,
            refresh_cache=False,
            verbose=False,
        )
        
        # Find model in registry
        model_entry = None
        for entry in registry_models:
            if entry.get("model") == model_name:
                model_entry = entry
                break
        
        # If found in registry with download_url, download from there
        if model_entry and model_entry.get("download_url"):
            download_url = model_entry["download_url"]
            source = model_entry.get("source", "flexipipe")
            print(f"[flexipipe] Downloading SpaCy model '{model_name}' from {source} registry...")
            
            # Handle different URL types
            if "huggingface.co" in download_url or download_url.startswith("hf://"):
                # HuggingFace model - extract model name from URL
                # URLs like: https://huggingface.co/masakhane/yoruba-pos-tagger-afroxlmr
                # or: hf://masakhane/yoruba-pos-tagger-afroxlmr
                hf_model_name = download_url.replace("hf://", "").replace("https://huggingface.co/", "").rstrip("/")
                # Remove any trailing paths (e.g., /resolve/main)
                if "/" in hf_model_name:
                    parts = hf_model_name.split("/")
                    # Take org/model parts (first two parts)
                    hf_model_name = "/".join(parts[:2])
                
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    raise ValueError(
                        f"HuggingFace model '{model_name}' requires 'huggingface_hub' package. "
                        f"Install it with: pip install huggingface_hub"
                    )
                
                import tempfile
                temp_dir = tempfile.mkdtemp()
                try:
                    downloaded_path = snapshot_download(hf_model_name, cache_dir=temp_dir)
                    shutil.copytree(downloaded_path, target_path, dirs_exist_ok=False)
                    
                    # Verify it's actually a SpaCy model (should have meta.json)
                    if not (target_path / "meta.json").exists():
                        # This is not a SpaCy model - it's likely a raw transformer model
                        # Clean up and provide helpful error
                        try:
                            shutil.rmtree(target_path)
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"[flexipipe] The model '{model_name}' downloaded from HuggingFace is not a SpaCy pipeline model. "
                            f"It appears to be a raw transformer model (config.json, model.safetensors, etc.), not a SpaCy model package. "
                            f"SpaCy models require a specific structure with meta.json and pipeline components. "
                            f"Please check if there's a SpaCy-wrapped version of this model available, or use a different model."
                        )
                    
                    print(f"[flexipipe] Model downloaded to {target_path}")
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
                return
            else:
                # Direct download (zip, tar.gz, or directory)
                import tempfile
                import zipfile
                import tarfile
                import urllib.request
                
                temp_dir = tempfile.mkdtemp()
                try:
                    # Download to temp file
                    archive_name = download_url.split("/")[-1]
                    # Remove query parameters if present
                    if "?" in archive_name:
                        archive_name = archive_name.split("?")[0]
                    archive_path = Path(temp_dir) / archive_name
                    
                    print(f"[flexipipe] Downloading from {download_url}...")
                    urllib.request.urlretrieve(download_url, archive_path)
                    
                    # Extract if it's an archive
                    extracted_dir = Path(temp_dir) / "extracted"
                    extracted_dir.mkdir()
                    
                    if archive_name.endswith(".zip"):
                        with zipfile.ZipFile(archive_path, "r") as zip_ref:
                            zip_ref.extractall(extracted_dir)
                    elif archive_name.endswith((".tar.gz", ".tgz")):
                        with tarfile.open(archive_path, "r:*") as tar:
                            tar.extractall(extracted_dir)
                    elif archive_name.endswith(".tar"):
                        with tarfile.open(archive_path, "r") as tar:
                            tar.extractall(extracted_dir)
                    else:
                        # Not an archive, assume it's a directory structure
                        # Copy directly
                        if archive_path.is_dir():
                            extracted_dir = archive_path
                        else:
                            # Single file - create a model directory structure
                            extracted_dir.mkdir()
                            shutil.copy2(archive_path, extracted_dir / archive_name)
                    
                    # Find the model directory (should contain meta.json)
                    model_dir = None
                    for candidate in [extracted_dir, *extracted_dir.rglob("*")]:
                        if (candidate / "meta.json").exists():
                            model_dir = candidate
                            break
                    
                    if model_dir is None:
                        # No meta.json found, assume the extracted directory is the model
                        model_dir = extracted_dir
                        # Check if there's a single subdirectory
                        subdirs = [d for d in extracted_dir.iterdir() if d.is_dir()]
                        if len(subdirs) == 1:
                            model_dir = subdirs[0]
                    
                    # Copy to target
                    shutil.copytree(model_dir, target_path, dirs_exist_ok=False)
                    print(f"[flexipipe] Model downloaded to {target_path}")
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
                return
    except Exception as exc:
        # Registry check failed, continue with standard download
        pass
    
    # Check if this is a HuggingFace model (contains /)
    if "/" in model_name:
        # Download from HuggingFace
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ValueError(
                f"HuggingFace model '{model_name}' requires 'huggingface_hub' package. "
                f"Install it with: pip install huggingface_hub"
            )
        
        print(f"[flexipipe] Downloading SpaCy model from HuggingFace: '{model_name}'...")
        try:
            # Download to flexipipe directory
            target_path = spacy_dir / model_name.replace("/", "_")
            if target_path.exists():
                # Already downloaded
                return
            
            # Download to a temporary location first
            import tempfile
            temp_dir = tempfile.mkdtemp()
            try:
                downloaded_path = snapshot_download(model_name, cache_dir=temp_dir)
                # Copy to flexipipe directory
                import shutil
                shutil.copytree(downloaded_path, target_path, dirs_exist_ok=False)
                print(f"[flexipipe] Model downloaded to {target_path}")
            finally:
                # Clean up temp directory
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - depends on network
            raise ValueError(
                f"Failed to download HuggingFace model '{model_name}': {exc}. "
                f"Please check the model name and try again."
            ) from exc
        return

    # Standard SpaCy model download
    print(f"[flexipipe] Downloading SpaCy model '{model_name}'...")
    try:
        spacy_download(model_name)
    except Exception as exc:  # pragma: no cover - depends on network
        raise ValueError(
            f"Failed to download SpaCy model '{model_name}': {exc}. "
            f"Please install it manually: python -m spacy download {model_name}"
        ) from exc

    # Try to copy downloaded model into flexipipe storage for consistency
    try:
        downloaded_path = spacy.util.get_package_path(model_name)
    except Exception:
        downloaded_path = None

    if downloaded_path and Path(downloaded_path).exists():
        target_path = spacy_dir / model_name
        if not target_path.exists():
            try:
                shutil.copytree(downloaded_path, target_path, dirs_exist_ok=False)
            except Exception:
                # Copy is best-effort; keep going if it fails
                pass


def _create_spacy_backend(
    *,
    model_name: str | None = None,
    model_path: str | Path | None = None,
    language: str | None = None,
    download_model: bool = False,
    training: bool = False,  # accepted for parity with create_backend signature
    verbose: bool = False,
    **kwargs: Any,
) -> SpacyBackend:
    model_name = _resolve_spacy_model_name(model_name, language)
    
    # Only download if explicitly requested
    if download_model and model_name and not model_path:
        _ensure_spacy_model_available(model_name)

    # Pass training flag to backend constructors
    kwargs["training"] = training
    if model_path:
        return SpacyBackend.from_model_path(model_path, verbose=verbose, **kwargs)
    if model_name:
        # Check if model is installed before attempting to load
        from ..model_storage import is_model_installed
        model_installed = is_model_installed("spacy", model_name)
        
        if not model_installed:
            # Model is not installed - check if it exists in registry
            try:
                from ..model_registry import get_remote_models_for_backend
                registry_models = get_remote_models_for_backend(
                    "spacy",
                    source=None,  # Check all sources
                    use_cache=True,
                    refresh_cache=False,
                    verbose=False,
                )
                
                # Find model in registry
                model_entry = None
                for entry in registry_models:
                    if entry.get("model") == model_name:
                        model_entry = entry
                        break
                
                # If found in registry with download_url, provide helpful message
                if model_entry and model_entry.get("download_url"):
                    source = model_entry.get("source", "registry")
                    raise RuntimeError(
                        f"[flexipipe] SpaCy model '{model_name}' is not installed, but is available in the {source} registry. "
                        f"Download it with: flexipipe --backend spacy --model {model_name} --download-model"
                    )
            except RuntimeError:
                # Re-raise our helpful error
                raise
            except Exception:
                # Registry check failed, continue to try loading (will fail with original error)
                pass
        
        # Try to load the model (will fail if not installed, but we've already checked registry above)
        try:
            return SpacyBackend.from_pretrained(model_name, verbose=verbose, **kwargs)
        except (ValueError, OSError) as e:
            # Model loading failed - check registry one more time in case first check missed it
            error_str = str(e)
            if "not found" in error_str.lower() or "can't find model" in error_str.lower() or "E050" in error_str:
                try:
                    from ..model_registry import get_remote_models_for_backend
                    registry_models = get_remote_models_for_backend(
                        "spacy",
                        source=None,
                        use_cache=True,
                        refresh_cache=False,
                        verbose=False,
                    )
                    
                    # Find model in registry
                    model_entry = None
                    for entry in registry_models:
                        if entry.get("model") == model_name:
                            model_entry = entry
                            break
                    
                    # If found in registry with download_url, provide helpful message
                    if model_entry and model_entry.get("download_url"):
                        source = model_entry.get("source", "registry")
                        raise RuntimeError(
                            f"[flexipipe] SpaCy model '{model_name}' is not installed, but is available in the {source} registry. "
                            f"Download it with: flexipipe --backend spacy --model {model_name} --download-model"
                        )
                except RuntimeError:
                    # Re-raise our helpful error
                    raise
                except Exception:
                    # Registry check failed, provide generic error message
                    pass
            # Re-raise original error if not in registry or registry check failed
            raise RuntimeError(
                f"[flexipipe] SpaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}, "
                f"or use --download-model if it's available in the registry."
            )
    if language:
        return SpacyBackend.blank(language, verbose=verbose, **kwargs)

    # Fall back to constructor to trigger auto-selection logic
    raise ValueError("Must provide either model_name, model_path, or language for SpaCy backend")


BACKEND_SPEC = BackendSpec(
    name="spacy",
    description="SpaCy - Fast NLP library with pre-trained models for many languages",
    factory=_create_spacy_backend,
    get_model_entries=lambda *args, **kwargs: get_spacy_model_entries(*args, **kwargs)[0],
    list_models=list_spacy_models,
    supports_training=True,
    is_rest=False,
    url="https://spacy.io",
)


