"""Backend and registry spec for ClassLA."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..backend_spec import BackendSpec
from ..doc import Document, Entity, Sentence, Token
from ..language_utils import (
    LANGUAGE_FIELD_ISO,
    LANGUAGE_FIELD_NAME,
    build_model_entry,
    cache_entries_standardized,
)
from ..model_registry import fetch_remote_registry, get_registry_url
from ..model_storage import (
    get_backend_models_dir,
    read_model_cache_entry,
    write_model_cache_entry,
)
from ..neural_backend import BackendManager, NeuralResult

MODEL_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


def _remove_language_from_stanza_constant(language: str, verbose: bool = False) -> bool:
    """
    Remove a language code from Stanza's constant.py file.
    
    Args:
        language: Language code to remove (e.g., "sna")
        verbose: Whether to print progress messages
    
    Returns:
        True if the language was removed or didn't exist, False if it couldn't be removed
    """
    try:
        # Try to find stanza path without importing (in case import fails due to AssertionError)
        stanza_path = None
        try:
            import stanza
            stanza_path = Path(stanza.__file__).parent
        except (ImportError, AssertionError):
            # If import fails, try to find stanza in site-packages
            import site
            for site_packages in site.getsitepackages():
                potential_path = Path(site_packages) / "stanza"
                if potential_path.exists() and (potential_path / "models" / "common" / "constant.py").exists():
                    stanza_path = potential_path
                    break
        
        if not stanza_path:
            if verbose:
                print(f"[flexipipe] Warning: Could not find Stanza installation")
            return False
        
        constant_py = stanza_path / "models" / "common" / "constant.py"
        
        if not constant_py.exists():
            return False
        
        if not os.access(constant_py, os.W_OK):
            return False
        
        # Read the file
        with constant_py.open("r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if language code exists
        if f'("{language}",' not in content and f"('{language}'," not in content:
            return True  # Already removed or never existed
        
        # Remove the line containing this language code
        lines = content.splitlines()
        new_lines = []
        removed = False
        
        for line in lines:
            # Check if this line contains the language code we want to remove
            # Match patterns like: ("sna", "Shona"), or ('sna', 'Shona'),
            if (f'("{language}",' in line or f"('{language}'," in line) and not removed:
                removed = True
                if verbose:
                    print(f"[flexipipe] Removed language code '{language}' from Stanza constant.py")
                continue  # Skip this line
            new_lines.append(line)
        
        if not removed:
            if verbose:
                print(f"[flexipipe] Language code '{language}' not found in Stanza constant.py (may have been already removed)")
            return True  # Not an error if it's already gone
        
        if removed:
            # Write the modified file
            new_content = "\n".join(new_lines)
            if content.endswith("\n") and not new_content.endswith("\n"):
                new_content += "\n"
            
            with constant_py.open("w", encoding="utf-8") as f:
                f.write(new_content)
        
        return True
        
    except Exception as e:
        if verbose:
            import traceback
            print(f"[flexipipe] Warning: Could not remove language code from Stanza constant.py: {e}")
            traceback.print_exc()
        return False


def _add_language_to_stanza_constant(language: str, language_name: Optional[str] = None, verbose: bool = False) -> bool:
    """
    Automatically add a language code to Stanza's constant.py file.
    
    Args:
        language: Language code to add (e.g., "sna")
        language_name: Optional language name (e.g., "Shona"). If not provided, uses the code.
        verbose: Whether to print progress messages
    
    Returns:
        True if the language was added or already exists, False if it couldn't be added
    """
    try:
        # First, try to import stanza to check if it's available
        # If import fails due to AssertionError, we may need to clean up first
        try:
            import stanza
        except AssertionError as assert_err:
            if verbose:
                print(f"[flexipipe] Warning: Stanza import failed with AssertionError (likely duplicate language mapping)")
                print(f"[flexipipe] Attempting to fix constant.py by removing conflicting entries...")
            # Try to remove the conflicting entry and retry
            # This is a bit of a chicken-and-egg problem, so we'll try to access the file directly
            import sys
            # Find stanza path from sys.modules or site-packages
            for module_name in list(sys.modules.keys()):
                if 'stanza' in module_name:
                    try:
                        module = sys.modules[module_name]
                        if hasattr(module, '__file__'):
                            stanza_path = Path(module.__file__).parent
                            constant_py = stanza_path / "models" / "common" / "constant.py"
                            if constant_py.exists():
                                # Remove the conflicting entry
                                _remove_language_from_stanza_constant(language, verbose=verbose)
                                # Try importing again
                                import importlib
                                importlib.reload(sys.modules[module_name])
                                import stanza
                                break
                    except Exception:
                        continue
            
            # If we still can't import, try to find stanza in site-packages
            import site
            for site_packages in site.getsitepackages():
                stanza_path = Path(site_packages) / "stanza"
                if stanza_path.exists():
                    constant_py = stanza_path / "models" / "common" / "constant.py"
                    if constant_py.exists():
                        _remove_language_from_stanza_constant(language, verbose=verbose)
                        break
        
        import stanza
        import importlib
        stanza_path = Path(stanza.__file__).parent
        constant_py = stanza_path / "models" / "common" / "constant.py"
        
        if not constant_py.exists():
            if verbose:
                print(f"[flexipipe] Warning: Could not find Stanza constant.py at {constant_py}")
            return False
        
        # Check if file is writable
        if not os.access(constant_py, os.W_OK):
            if verbose:
                print(f"[flexipipe] Warning: Stanza constant.py is not writable: {constant_py}")
                print(f"[flexipipe] This may be because Stanza is installed in a system location.")
                print(f"[flexipipe] Consider installing Stanza in a virtual environment or with --user flag.")
            return False
        
        # Read the file
        with constant_py.open("r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if language code already exists
        if f'("{language}",' in content or f"('{language}'," in content:
            if verbose:
                print(f"[flexipipe] Language code '{language}' already exists in Stanza constant.py")
            return True
        
        # Check if another code already maps to the same language name
        # This prevents conflicts when Stanza builds the reverse mapping
        import re
        # Extract all existing mappings to check for conflicts
        # Look specifically in the lcode2lang_raw list
        existing_mappings = {}
        in_list = False
        for line in content.splitlines():
            if "lcode2lang_raw" in line and "=" in line:
                in_list = True
                continue
            if in_list:
                if line.strip() == "]":
                    break
                # Match tuples like ("code", "Name"),
                match = re.search(r'\(["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\)', line)
                if match:
                    code, name = match.groups()
                    existing_mappings[code] = name
        
        # Find the lcode2lang_raw list
        if "lcode2lang_raw" not in content:
            if verbose:
                print(f"[flexipipe] Warning: Could not find lcode2lang_raw in constant.py")
            return False
        
        # Get language name - use provided name or capitalize the code
        if not language_name:
            # Try to get a reasonable name from the code
            language_name = language.upper() if len(language) <= 2 else language.capitalize()
        
        # Check if another code already maps to the same language name
        # If so, and if that code would be used by Stanza (e.g., "sn" for "sna"),
        # we can skip adding this code to avoid conflicts in the reverse mapping
        conflicting_code = None
        for code, name in existing_mappings.items():
            if name == language_name and code != language:
                # Check if this is a normalization case (e.g., "sn" for "sna")
                # Stanza normalizes 3-letter codes to 2-letter codes
                if len(code) == 2 and len(language) == 3 and language.startswith(code):
                    if verbose:
                        print(f"[flexipipe] Language code '{code}' already maps to '{language_name}'")
                        print(f"[flexipipe] Since Stanza normalizes '{language}' to '{code}', the normalized code already exists")
                        print(f"[flexipipe] No need to add '{language}' - Stanza will use '{code}'")
                    return True  # Return True since the normalized code already exists
                elif code != language:
                    # Different codes mapping to same name - this will cause an assertion error on reload
                    conflicting_code = code
                    if verbose:
                        print(f"[flexipipe] Warning: Language code '{code}' already maps to '{language_name}'")
                        print(f"[flexipipe] Adding '{language}' with the same name may cause a conflict on reload")
                        print(f"[flexipipe] Will add the code but skip reload to avoid assertion error")
                    # We'll still add it, but skip the reload
        
        # Parse the list to find insertion point (alphabetically sorted)
        lines = content.splitlines()
        new_lines = []
        inserted = False
        in_list = False
        list_start_idx = None
        
        for i, line in enumerate(lines):
            # Detect start of lcode2lang_raw list
            if "lcode2lang_raw" in line and "=" in line:
                in_list = True
                list_start_idx = len(new_lines)  # Track position in new_lines
                new_lines.append(line)
                continue
            
            if in_list:
                # Check if we've reached the end of the list
                if line.strip() == "]":
                    # Insert our language code before the closing bracket if not already inserted
                    if not inserted:
                        # Find the right position alphabetically by looking through entries
                        insert_pos = len(new_lines)  # Default: insert before closing bracket
                        
                        # Look through the list entries we've collected so far
                        for j in range(list_start_idx + 1, len(new_lines)):
                            list_line = new_lines[j].strip()
                            # Check if this line has a language code tuple
                            if list_line.startswith('("') or list_line.startswith("('"):
                                # Extract the code from the line
                                try:
                                    # Format: ("code", "Name"),
                                    if '("' in list_line:
                                        code_start = list_line.find('("') + 2
                                        quote_char = '"'
                                    else:
                                        code_start = list_line.find("('") + 2
                                        quote_char = "'"
                                    code_end = list_line.find(quote_char, code_start)
                                    if code_end > code_start:
                                        list_code = list_line[code_start:code_end]
                                        # If this code is greater than ours, insert before it
                                        if list_code > language:
                                            insert_pos = j
                                            break
                                except Exception:
                                    pass
                        
                        # Format the new entry with proper indentation
                        # Use the same indentation as other entries (typically 4 spaces)
                        indent = "    "  # Standard 4-space indent
                        new_entry = f'{indent}("{language}", "{language_name}"),'
                        new_lines.insert(insert_pos, new_entry)
                        inserted = True
                        if verbose:
                            print(f"[flexipipe] Added language code '{language}' to Stanza constant.py")
                    new_lines.append(line)
                    in_list = False
                    continue
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        if not inserted:
            if verbose:
                print(f"[flexipipe] Warning: Could not find insertion point for language code '{language}'")
            return False
        
        # Write the modified file
        new_content = "\n".join(new_lines)
        # Ensure file ends with newline if original did
        if content.endswith("\n") and not new_content.endswith("\n"):
            new_content += "\n"
        
        with constant_py.open("w", encoding="utf-8") as f:
            f.write(new_content)
        
        # Check if reloading would cause conflicts
        # If another code maps to the same language name, reloading will cause an AssertionError
        skip_reload = False
        if conflicting_code:
            skip_reload = True
            if verbose:
                print(f"[flexipipe] Skipping Stanza reload to avoid assertion error (conflict with '{conflicting_code}')")
                print(f"[flexipipe] The language code has been added to constant.py")
                print(f"[flexipipe] Restart Python or reimport Stanza to pick up the changes")
        
        # Reload stanza module to pick up the changes (unless it would cause conflicts)
        if not skip_reload:
            try:
                importlib.reload(stanza)
                if verbose:
                    print(f"[flexipipe] Reloaded Stanza module to recognize new language code")
            except (AssertionError, Exception) as reload_err:
                if verbose:
                    if isinstance(reload_err, AssertionError):
                        print(f"[flexipipe] Warning: Reloading Stanza caused an assertion error (likely due to duplicate language name)")
                        print(f"[flexipipe] The language code has been added to constant.py")
                        print(f"[flexipipe] Restart Python or reimport Stanza to pick up the changes")
                    else:
                        print(f"[flexipipe] Warning: Could not reload Stanza module: {reload_err}")
                # Continue anyway - the file is modified, next import will pick it up
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"[flexipipe] Warning: Could not modify Stanza constant.py: {e}")
            import traceback
            traceback.print_exc()
        return False


def _get_fallback_classla_models() -> Dict[tuple[str, str], Dict[str, Any]]:
    """Get fallback hardcoded ClassLA models (used when registry is unavailable)."""
    return {
        ("hr", "standard"): {"package": "set", "name": "Croatian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("hr", "nonstandard"): {"package": "set", "name": "Croatian (nonstandard)", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("sr", "standard"): {"package": "set", "name": "Serbian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("sr", "nonstandard"): {"package": "set", "name": "Serbian (nonstandard)", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("bg", "standard"): {"package": "btb", "name": "Bulgarian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse", "ner"]},
        ("mk", "standard"): {"package": "mk", "name": "Macedonian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats"]},  # No depparse
        ("sl", "standard"): {"package": "ssj", "name": "Slovenian", "default_features": ["tokenization", "upos", "lemma", "xpos", "feats", "depparse"]},  # Has depparse
    }


def get_classla_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Dict[str, str]]:
    del kwargs
    cache_key = "classla"
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached:
            fixed = False
            for model_key, entry in cached.items():
                if isinstance(entry, dict) and not entry.get("language_iso") and "-" in model_key:
                    entry["language_iso"] = model_key.split("-")[0].lower()
                    fixed = True
            if fixed and refresh_cache:
                try:
                    write_model_cache_entry(cache_key, cached)
                except (OSError, PermissionError):
                    pass
            if cache_entries_standardized(cached):
                return cached

    # Try to fetch from remote registry first
    known_models: Dict[tuple[str, str], Dict[str, Any]] = {}
    try:
        registry = fetch_remote_registry(
            backend="classla",
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            verbose=verbose,
        )
        if registry:
            # Extract models from registry structure
            sources = registry.get("sources", {})
            for source_type in ["official", "flexipipe", "community"]:
                if source_type in sources:
                    for model_entry in sources[source_type]:
                        model_name = model_entry.get("model")
                        if model_name and "-" in model_name:
                            # Parse model name like "hr-standard" or "mk-standard"
                            parts = model_name.split("-", 1)
                            if len(parts) == 2:
                                lang_code, variant = parts
                                if variant in ("standard", "nonstandard"):
                                    known_models[(lang_code, variant)] = {
                                        "package": model_entry.get("package", ""),
                                        "name": model_entry.get("language_name") or model_entry.get("name", lang_code.upper()),
                                        "default_features": model_entry.get("features", "").split(", ") if model_entry.get("features") else [],
                                    }
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] ClassLA registry unavailable ({exc}), using fallback models.")
    
    # Fall back to hardcoded models if registry is empty or unavailable
    if not known_models:
        known_models = _get_fallback_classla_models()

    result: Dict[str, Dict[str, str]] = {}
    installed_models: Dict[str, Dict[str, str]] = {}
    # Track available processors for each model
    model_processors: Dict[str, Set[str]] = {}
    
    # Only scan directories if explicitly requested (refresh_cache) or if cache is empty
    # This matches the behavior of other backends like SpaCy
    should_scan_directories = refresh_cache
    if not should_scan_directories:
        # Check if cache exists - if not, we need to scan at least once
        # We already checked for valid cache at the top, so if we get here, cache is expired or missing
        cached_check = read_model_cache_entry(cache_key, max_age_seconds=None)  # Check if any cache exists
        if not cached_check:
            should_scan_directories = True  # First time - need to scan
        else:
            # Cache exists but is expired - we could use it as a fallback, but for consistency
            # with other backends, we'll scan if cache is expired (unless explicitly requested not to)
            # Actually, let's use the expired cache if it exists, since scanning is expensive
            # The cache check at the top already handles valid cache, so this is for expired cache
            pass
    
    if should_scan_directories:
        classla_resources = get_backend_models_dir("classla", create=False)
        if classla_resources.exists():
            # Scan for models in two ways:
            # 1. Direct structure: lang/processor/package.pt (official models)
            # 2. Subdirectory structure: model-name/lang/processor/package.pt (trained models)
            for item in classla_resources.iterdir():
                if not item.is_dir():
                    continue
                
                # Check if this is a direct lang directory (e.g., "hr", "sr") or a model subdirectory (e.g., "sna-masakhane")
                # Look for lang/processor structure inside
                lang_dirs_to_scan = []
                
                # Check if this item itself is a lang directory (has processor subdirectories)
                has_processors = any(
                    subitem.is_dir() and subitem.name in ("pos", "lemma", "depparse", "tokenize", "ner")
                    for subitem in item.iterdir()
                )
                
                if has_processors:
                    # This is a direct lang directory
                    lang_dirs_to_scan.append((item, item.name))
                else:
                    # This might be a model subdirectory - check for lang subdirectories inside
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            # Check if this subdirectory has processor subdirectories
                            has_sub_processors = any(
                                proc_item.is_dir() and proc_item.name in ("pos", "lemma", "depparse", "tokenize", "ner")
                                for proc_item in subitem.iterdir()
                            )
                            if has_sub_processors:
                                lang_dirs_to_scan.append((subitem, subitem.name))
                
                # Scan each lang directory found
                for lang_dir, lang_code in lang_dirs_to_scan:
                    for processor_dir in lang_dir.iterdir():
                        if not processor_dir.is_dir():
                            continue
                        processor_name = processor_dir.name
                        # Check for .pt files (including .pt.zip files)
                        has_model_file = False
                        model_file = None
                        # First check for regular .pt files
                        for f in processor_dir.glob("*.pt"):
                            if not f.name.endswith(".zip"):
                                model_file = f
                                has_model_file = True
                                break
                        # If no regular .pt file, check for .pt.zip
                        if not model_file:
                            for f in processor_dir.glob("*.pt.zip"):
                                model_file = f
                                has_model_file = True
                                break
                        
                        if has_model_file and model_file:
                            package = model_file.stem.replace(".zip", "")
                            variant = "nonstandard" if "nonstandard" in processor_dir.parts else "standard"
                            model_key = f"{lang_code}-{variant}"
                            installed_models.setdefault(
                                model_key,
                                {"lang": lang_code, "package": package, "variant": variant},
                            )
                            # Track which processors are available for this model
                            if model_key not in model_processors:
                                model_processors[model_key] = set()
                            model_processors[model_key].add(processor_name)
    else:
        # Not scanning directories - try to get installed models from cache
        # The cache should contain information about installed models
        cached = read_model_cache_entry(cache_key, max_age_seconds=None)
        if cached:
            # Extract installed models from cached entries
            for model_key, entry in cached.items():
                if isinstance(entry, dict) and entry.get("status") == "installed":
                    lang_code = entry.get("language_iso") or model_key.split("-")[0]
                    package = entry.get("package", lang_code)
                    variant = entry.get("variant", "standard")
                    installed_models[model_key] = {
                        "lang": lang_code,
                        "package": package,
                        "variant": variant,
                    }
                    # Try to reconstruct processors from features
                    features_str = entry.get("features", "")
                    if features_str:
                        features = [f.strip() for f in features_str.split(",")]
                        processor_to_feature = {
                            "tokenize": "tokenization",
                            "pos": "upos",
                            "lemma": "lemma",
                            "depparse": "depparse",
                            "ner": "ner",
                        }
                        feature_to_processor = {v: k for k, v in processor_to_feature.items()}
                        processors = set()
                        for feat in features:
                            proc = feature_to_processor.get(feat)
                            if proc:
                                processors.add(proc)
                        if processors:
                            model_processors[model_key] = processors

    # Check which models are actually installed
    from ..model_storage import is_model_installed
    
    for (lang_code, variant), model_info in known_models.items():
        package = model_info["package"]
        model_key = f"{lang_code}-{variant}"
        model_entry = installed_models.get(
            model_key,
            {"lang": lang_code, "package": package, "variant": variant},
        )
        lang_name = model_info.get("name", lang_code.upper())
        entry = build_model_entry(
            backend="classla",
            model_id=model_key,
            model_name=model_key,
            language_code=model_entry.get("lang", lang_code),
            language_name=lang_name,
            package=model_entry.get("package", package),
            description=f"ClassLA model for {lang_name} ({variant})",
        )
        entry["language_iso"] = entry.get("language_iso") or lang_code.lower()
        entry["package"] = model_entry.get("package", package)
        entry["variant"] = model_entry.get("variant", variant)
        
        # Set status based on whether model is actually installed
        try:
            if is_model_installed("classla", model_key):
                entry["status"] = "installed"
            else:
                entry["status"] = "available"
        except Exception:
            # If check fails, default to available
            entry["status"] = "available"
        
        # Set features based on available processors (if installed) or default features (if not installed)
        available_processors = model_processors.get(model_key, set())
        features_list = []
        
        if available_processors:
            # Model is installed - detect features from actual processors
            # Map processor names to feature names
            processor_to_feature = {
                "tokenize": "tokenization",
                "pos": "upos",
                "lemma": "lemma",
                "depparse": "depparse",
                "ner": "ner",
            }
            for proc, feat in processor_to_feature.items():
                if proc in available_processors:
                    features_list.append(feat)
            # Also include xpos and feats if pos is available (POS tagger provides these)
            if "pos" in available_processors:
                if "xpos" not in features_list:
                    features_list.append("xpos")
                if "feats" not in features_list:
                    features_list.append("feats")
        else:
            # Model is not installed - use default features from known_models
            default_features = model_info.get("default_features", [])
            features_list = default_features.copy()
        
        if features_list:
            entry["features"] = ", ".join(features_list)
        
        result[model_key] = entry
    
    # Also add any discovered trained models that aren't in known_models
    # These are models trained via flexipipe that may not be in the registry
    for model_key, model_info in installed_models.items():
        if model_key not in result:
            # This is a trained model not in the registry
            lang_code = model_info.get("lang", model_key.split("-")[0])
            package = model_info.get("package", lang_code)
            variant = model_info.get("variant", "standard")
            
            # Get language name
            from ..language_utils import standardize_language_metadata
            metadata = standardize_language_metadata(lang_code, None)
            lang_name = metadata.get(LANGUAGE_FIELD_NAME) or (lang_code.upper() if len(lang_code) <= 2 else lang_code.capitalize())
            
            # Get available processors for this model
            available_processors = model_processors.get(model_key, set())
            features_list = []
            processor_to_feature = {
                "tokenize": "tokenization",
                "pos": "upos",
                "lemma": "lemma",
                "depparse": "depparse",
                "ner": "ner",
            }
            for proc, feat in processor_to_feature.items():
                if proc in available_processors:
                    features_list.append(feat)
            if "pos" in available_processors:
                if "xpos" not in features_list:
                    features_list.append("xpos")
                if "feats" not in features_list:
                    features_list.append("feats")
            
            entry = build_model_entry(
                backend="classla",
                model_id=model_key,
                model_name=model_key,
                language_code=lang_code,
                language_name=lang_name,
                package=package,
                description=f"ClassLA trained model for {lang_name} ({variant})",
            )
            entry["language_iso"] = lang_code.lower()
            entry["package"] = package
            entry["variant"] = variant
            entry["status"] = "installed"
            entry["source"] = "trained"
            if features_list:
                entry["features"] = ", ".join(features_list)
            
            result[model_key] = entry

    # Always write to cache if we scanned directories or if refresh_cache was requested
    # This ensures the cache is updated with discovered models
    if should_scan_directories or refresh_cache:
        try:
            write_model_cache_entry(cache_key, result)
        except (OSError, PermissionError):
            pass
    return result


def _classla_doc_to_document(
    classla_doc,
    original_doc: Optional[Document] = None,
) -> Document:
    doc = Document(id="")
    if original_doc:
        doc.id = original_doc.id
        doc.meta = original_doc.meta.copy()
        doc.attrs = original_doc.attrs.copy()

    for stanza_sent in classla_doc.sentences:
        sent_id = getattr(stanza_sent, "sent_id", None) or ""
        sentence = Sentence(id=sent_id, sent_id=sent_id, text=stanza_sent.text, tokens=[])

        if hasattr(stanza_sent, "ents") and stanza_sent.ents:
            for ent in stanza_sent.ents:
                try:
                    token_start = getattr(ent, "start", None)
                    token_end = getattr(ent, "end", None)
                    if hasattr(ent, "tokens") and ent.tokens:
                        token_start = ent.tokens[0].id
                        token_end = ent.tokens[-1].id
                    if token_start is None or token_end is None:
                        continue
                    start_idx = int(str(token_start).split("-")[0])
                    end_idx = int(str(token_end).split("-")[-1])
                    entity = Entity(
                        start=start_idx + 1,
                        end=end_idx,
                        label=getattr(ent, "type", ""),
                        text=getattr(ent, "text", ""),
                    )
                    sentence.entities.append(entity)
                except Exception:
                    continue

        for token_idx, stanza_token in enumerate(stanza_sent.tokens):
            def _to_int_id(token_id):
                if token_id is None:
                    return token_idx + 1
                token_str = str(token_id)
                if "-" in token_str:
                    token_str = token_str.split("-")[0]
                try:
                    return int(token_str)
                except (ValueError, TypeError):
                    return token_idx + 1

            if hasattr(stanza_token, "words") and len(stanza_token.words) > 1:
                subtokens = []
                for word in stanza_token.words:
                    subtokens.append(
                        Token(
                            id=_to_int_id(word.id),
                            form=word.text,
                            lemma=word.lemma or "",
                            upos=word.upos or "",
                            xpos=word.xpos or "",
                            feats=word.feats or "",
                            head=word.head if word.head else 0,
                            deprel=word.deprel or "",
                            space_after=("SpaceAfter=No" not in (word.misc or "")) if word.misc else True,
                        )
                    )
                token = Token(
                    id=_to_int_id(stanza_token.id),
                    form=stanza_token.text,
                    lemma=stanza_token.words[0].lemma if stanza_token.words else "",
                    upos=stanza_token.words[0].upos if stanza_token.words else "",
                    xpos=stanza_token.words[0].xpos if stanza_token.words else "",
                    feats=stanza_token.words[0].feats if stanza_token.words else "",
                    head=stanza_token.words[0].head if stanza_token.words and stanza_token.words[0].head else 0,
                    deprel=stanza_token.words[0].deprel if stanza_token.words else "",
                    is_mwt=True,
                    subtokens=subtokens,
                    space_after=("SpaceAfter=No" not in (stanza_token.misc or "")) if stanza_token.misc else True,
                )
                token.parts = [st.form for st in subtokens]
            else:
                word = stanza_token.words[0] if hasattr(stanza_token, "words") and stanza_token.words else stanza_token
                token = Token(
                    id=_to_int_id(stanza_token.id),
                    form=stanza_token.text,
                    lemma=getattr(word, "lemma", "") or "",
                    upos=getattr(word, "upos", "") or "",
                    xpos=getattr(word, "xpos", "") or "",
                    feats=getattr(word, "feats", "") or "",
                    head=getattr(word, "head", 0) or 0,
                    deprel=getattr(word, "deprel", "") or "",
                    space_after=("SpaceAfter=No" not in (stanza_token.misc or "")) if getattr(stanza_token, "misc", None) else True,
                )
            sentence.tokens.append(token)

        if sentence.tokens:
            sentence.tokens[-1].space_after = None
        doc.sentences.append(sentence)
    return doc


def _try_download_fasttext_embeddings(
    language: str,
    language_normalized: Optional[str],
    output_dir: Path,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Try to download fastText embeddings for a language.
    
    Args:
        language: Original language code (e.g., "sna")
        language_normalized: Normalized language code (e.g., "sn")
        output_dir: Directory to save the embeddings
        verbose: Whether to print progress messages
    
    Returns:
        Path to downloaded .vec file if successful, None otherwise
    """
    # Try both the original and normalized language codes
    codes_to_try = [language]
    if language_normalized and language_normalized not in codes_to_try:
        codes_to_try.append(language_normalized)
    
    for lang_code in codes_to_try:
        fasttext_url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{lang_code}.vec"
        output_file = output_dir / f"wiki.{lang_code}.vec"
        
        if output_file.exists():
            if verbose:
                print(f"[flexipipe] Found existing fastText embeddings: {output_file}")
            return output_file
        
        try:
            if verbose:
                print(f"[flexipipe] Downloading fastText embeddings from: {fasttext_url}")
            
            import urllib.request
            import urllib.error
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            try:
                urllib.request.urlretrieve(fasttext_url, output_file)
                if output_file.exists() and output_file.stat().st_size > 0:
                    if verbose:
                        print(f"[flexipipe] Successfully downloaded fastText embeddings: {output_file}")
                    return output_file
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    if verbose:
                        print(f"[flexipipe] fastText embeddings not available for language code '{lang_code}' (404)")
                    continue
                else:
                    if verbose:
                        print(f"[flexipipe] Error downloading fastText embeddings: {e}")
                    continue
            except Exception as e:
                if verbose:
                    print(f"[flexipipe] Error downloading fastText embeddings: {e}")
                continue
        except Exception as e:
            if verbose:
                print(f"[flexipipe] Could not download fastText embeddings: {e}")
            continue
    
    return None


def _convert_fasttext_to_stanza_pretrain(
    fasttext_vec: Path,
    pretrain_dir: Path,
    language: str,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Try to convert fastText .vec file to Stanza .pt pretrain format.
    
    Args:
        fasttext_vec: Path to fastText .vec file
        pretrain_dir: Directory to save the .pt file
        language: Language code
        verbose: Whether to print progress messages
    
    Returns:
        Path to converted .pt file if successful, None otherwise
    """
    if not fasttext_vec.exists():
        return None
    
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    output_pt = pretrain_dir / f"{language}.pretrain.pt"
    
    # Check if conversion is already done
    if output_pt.exists():
        if verbose:
            print(f"[flexipipe] Found existing pretrain file: {output_pt}")
        return output_pt
    
    try:
        # Use subprocess to convert fastText embeddings to Stanza format
        # This avoids importing stanza directly, which may fail due to protobuf issues
        import subprocess
        import sys
        
        if verbose:
            print(f"[flexipipe] Converting fastText embeddings to Stanza format...")
        
        # Create a small Python script to do the conversion
        # This runs in a fresh subprocess, avoiding import issues
        conversion_script = f"""
import sys
try:
    from stanza.models.common.pretrain import Pretrain
    pretrain = Pretrain(r"{output_pt}", r"{fasttext_vec}")
    pretrain.load()
    sys.exit(0)
except Exception as e:
    print(f"Conversion error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        
        # Run the conversion script in a subprocess
        result = subprocess.run(
            [sys.executable, "-c", conversion_script],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0 and output_pt.exists():
            if verbose:
                print(f"[flexipipe] Successfully converted to: {output_pt}")
            return output_pt
        else:
            if verbose:
                if result.stderr:
                    print(f"[flexipipe] Warning: Could not convert fastText embeddings: {result.stderr.strip()}")
                else:
                    print(f"[flexipipe] Warning: Could not convert fastText embeddings (conversion script failed)")
    except Exception as e:
        if verbose:
            print(f"[flexipipe] Warning: Error during fastText conversion: {e}")
    except Exception as e:
        if verbose:
            print(f"[flexipipe] Warning: Error during fastText conversion: {e}")
    
    return None


def _try_download_xlmroberta_embeddings(
    pretrain_dir: Path,
    language: str,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Try to extract XLM-RoBERTa embeddings and save in Stanza format.
    
    Args:
        pretrain_dir: Directory to save the .pt file
        language: Language code
        verbose: Whether to print progress messages
    
    Returns:
        Path to saved .pt file if successful, None otherwise
    """
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    output_pt = pretrain_dir / f"{language}.pretrain.pt"
    
    # Check if already exists and is valid
    if output_pt.exists():
        try:
            import torch
            # Verify it's in the correct format (dict with 'emb' and 'vocab')
            data = torch.load(str(output_pt), map_location='cpu')
            if isinstance(data, dict) and 'emb' in data and 'vocab' in data:
                if verbose:
                    print(f"[flexipipe] Found existing XLM-RoBERTa pretrain file: {output_pt}")
                return output_pt
            else:
                # File exists but is in wrong format - delete it and regenerate
                if verbose:
                    print(f"[flexipipe] Existing pretrain file has incorrect format, regenerating...")
                output_pt.unlink()
        except Exception:
            # If we can't verify, delete and regenerate to be safe
            if verbose:
                print(f"[flexipipe] Could not verify existing pretrain file, regenerating...")
            try:
                output_pt.unlink()
            except Exception:
                pass
    
    try:
        if verbose:
            print(f"[flexipipe] Attempting to extract XLM-RoBERTa embeddings...")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            import subprocess
            import sys
            
            # Load XLM-RoBERTa model and tokenizer
            if verbose:
                print(f"[flexipipe] Loading XLM-RoBERTa-base model and tokenizer...")
            model = AutoModel.from_pretrained('xlm-roberta-base')
            tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            
            # Extract word embeddings and vocabulary
            embeddings = model.embeddings.word_embeddings.weight.data.cpu()
            vocab_dict = tokenizer.get_vocab()
            
            # Create a temporary .vec file in fastText format
            # Stanza's Pretrain class can then convert it to the proper .pt format
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.vec', delete=False, encoding='utf-8') as tmp_vec:
                # Write header: vocab_size embedding_dim
                vocab_size = len(vocab_dict)
                embedding_dim = embeddings.shape[1]
                tmp_vec.write(f"{vocab_size} {embedding_dim}\n")
                
                # Write embeddings in fastText format: token embedding_vector
                # Sort by index to ensure consistent ordering
                sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
                for token, idx in sorted_vocab:
                    # Get embedding for this token index
                    if idx < embeddings.shape[0]:
                        embedding = embeddings[idx].numpy()
                        # Write token and space-separated embedding values
                        embedding_str = ' '.join(str(float(x)) for x in embedding)
                        tmp_vec.write(f"{token} {embedding_str}\n")
                
                tmp_vec_path = tmp_vec.name
            
            # Use subprocess to convert the .vec file to .pt format
            # This avoids importing stanza directly, which may fail due to protobuf issues
            if verbose:
                print(f"[flexipipe] Converting XLM-RoBERTa embeddings to Stanza format...")
            
            conversion_script = f"""
import sys
try:
    from stanza.models.common.pretrain import Pretrain
    pretrain = Pretrain(r"{output_pt}", r"{tmp_vec_path}")
    pretrain.load()
    sys.exit(0)
except Exception as e:
    print(f"Conversion error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            
            result = subprocess.run(
                [sys.executable, "-c", conversion_script],
                capture_output=True,
                text=True,
            )
            
            # Clean up temporary .vec file
            try:
                Path(tmp_vec_path).unlink()
            except Exception:
                pass
            
            if result.returncode == 0 and output_pt.exists():
                if verbose:
                    print(f"[flexipipe] Successfully extracted XLM-RoBERTa embeddings: {output_pt}")
                    print(f"[flexipipe] Note: XLM-RoBERTa embeddings are multilingual and may work")
                    print(f"[flexipipe] for languages not covered by fastText, but may require")
                    print(f"[flexipipe] additional configuration in Stanza training.")
                return output_pt
            else:
                if verbose:
                    if result.stderr:
                        print(f"[flexipipe] Warning: Could not convert XLM-RoBERTa embeddings: {result.stderr.strip()}")
                    else:
                        print(f"[flexipipe] Warning: Could not convert XLM-RoBERTa embeddings (conversion script failed)")
        except ImportError:
            if verbose:
                print(f"[flexipipe] Warning: transformers library not available. Install with: pip install transformers")
        except Exception as e:
            if verbose:
                print(f"[flexipipe] Warning: Could not extract XLM-RoBERTa embeddings: {e}")
                import traceback
                if verbose:
                    traceback.print_exc()
    except Exception as e:
        if verbose:
            print(f"[flexipipe] Warning: Error during XLM-RoBERTa extraction: {e}")
    
    return None


class ClassLABackend(BackendManager):
    """ClassLA-based neural backend."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        package: Optional[str] = None,
        processors: Optional[str] = None,
        use_gpu: bool = False,
        download_model: bool = False,
        verbose: bool = False,
        type: Optional[str] = None,
    ):
        from ..model_storage import setup_backend_environment

        setup_backend_environment("classla")

        if not verbose:
            logging.getLogger("classla").setLevel(logging.WARNING)

        try:
            import classla
        except ImportError as exc:
            raise ImportError("ClassLA backend requires the 'classla' package. Install it with: pip install classla") from exc
        try:
            from classla.pipeline.core import ResourcesFileNotFound  # type: ignore
        except ImportError:
            ResourcesFileNotFound = Exception

        self.classla = classla
        self._resources_error = ResourcesFileNotFound
        self._language = language or (model_name.split("-")[0] if model_name and "-" in model_name else model_name) or "hr"
        default_package = {"hr": "set", "sr": "set", "bg": "btb", "mk": "mk", "sl": "ssj"}
        self._package = package or default_package.get(self._language)
        
        # Determine default processors based on model capabilities
        default_processors = ["tokenize", "pos", "lemma"]
        # Check model registry to see what features are available
        if processors is None and model_name:
            try:
                from .classla import get_classla_model_entries  # circular-safe import (self-file)
                entries = get_classla_model_entries(use_cache=True, refresh_cache=False)
                entry = entries.get(model_name)
                if entry:
                    features_str = entry.get("features", "")
                    if features_str:
                        features = [f.strip() for f in features_str.split(",")]
                        # Map features to processors
                        feature_to_processor = {
                            "tokenization": "tokenize",
                            "upos": "pos",
                            "xpos": "pos",  # xpos comes from pos processor
                            "feats": "pos",  # feats comes from pos processor
                            "lemma": "lemma",
                            "depparse": "depparse",
                            "ner": "ner",
                        }
                        available_processors = []
                        # Always include tokenize (it's always available for ClassLA models)
                        available_processors.append("tokenize")
                        for feat in features:
                            proc = feature_to_processor.get(feat)
                            if proc and proc not in available_processors:
                                available_processors.append(proc)
                        if available_processors:
                            default_processors = available_processors
            except Exception:
                # If registry lookup fails, use default
                pass
        
        if processors:
            self._processors = processors
        else:
            self._processors = ",".join(default_processors)
        self._use_gpu = use_gpu
        self._download = download_model
        self._verbose = verbose
        if model_name and "-" in model_name and not type:
            parts = model_name.split("-", 1)
            if len(parts) == 2 and parts[1] in ("standard", "nonstandard"):
                self._type = parts[1]
            else:
                self._type = type or "standard"
        else:
            self._type = type or "standard"
        self._pipelines: Dict[bool, classla.Pipeline] = {}

    def _check_for_trained_model(self, language: str) -> bool:
        """
        Check if a trained model exists for the given language.
        
        Checks both the original language code and its normalized form (e.g., "sna" and "sn").
        
        Args:
            language: Language code to check
            
        Returns:
            True if a trained model is found, False otherwise
        """
        from ..model_storage import get_backend_models_dir
        from ..language_utils import standardize_language_metadata
        
        classla_dir = get_backend_models_dir("classla", create=False)
        if not classla_dir.exists():
            return False
        
        # Get normalized language code
        lang_codes_to_check = [language]
        if len(language) == 3:
            metadata = standardize_language_metadata(language, None)
            iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
            if iso_1 and len(iso_1) == 2 and iso_1 != language:
                lang_codes_to_check.append(iso_1)
        
        # Check for trained models in subdirectories (e.g., model-name/lang/processor/package.pt)
        for item in classla_dir.iterdir():
            if not item.is_dir():
                continue
            
            # Check if this item itself is a lang directory
            for lang_code in lang_codes_to_check:
                lang_dir = item if (item.name == lang_code and any(
                    subitem.is_dir() and subitem.name in ("pos", "lemma", "depparse", "tokenize", "ner")
                    for subitem in item.iterdir()
                )) else None
                
                # Or check if it's a model subdirectory containing a lang directory
                if not lang_dir:
                    for subitem in item.iterdir():
                        if subitem.is_dir() and subitem.name == lang_code:
                            # Check if this subdirectory has processor subdirectories
                            has_processors = any(
                                proc_item.is_dir() and proc_item.name in ("pos", "lemma", "depparse", "tokenize", "ner")
                                for proc_item in subitem.iterdir()
                            )
                            if has_processors:
                                lang_dir = subitem
                                break
                
                if lang_dir:
                    # Check if there are actual model files (.pt files)
                    for processor_dir in lang_dir.iterdir():
                        if processor_dir.is_dir() and processor_dir.name in ("pos", "lemma", "depparse", "tokenize", "ner"):
                            model_files = list(processor_dir.glob("*.pt"))
                            if model_files and any(not f.name.endswith(".zip") for f in model_files):
                                return True
        
        return False

    def _build_pipeline(self, pretokenized: bool):
        if not self._verbose:
            classla_logger = logging.getLogger("classla")
            classla_logger.setLevel(logging.WARNING)
            for handler in classla_logger.handlers:
                handler.setLevel(logging.WARNING)
            classla_logger.propagate = False

        from ..model_storage import get_backend_models_dir

        classla_dir = get_backend_models_dir("classla", create=False)
        
        # Check if we have a trained model for this language
        # If so, automatically add the language to Stanza's constant.py if needed
        has_trained_model = self._check_for_trained_model(self._language)
        if has_trained_model:
            # Try to add language to Stanza's constant.py automatically
            # This ensures ClassLA can load the trained model
            try:
                from ..language_utils import standardize_language_metadata
                metadata = standardize_language_metadata(self._language, None)
                language_name = metadata.get(LANGUAGE_FIELD_NAME) or (self._language.upper() if len(self._language) <= 2 else self._language.capitalize())
                
                # Check if language is already supported
                try:
                    import stanza
                    stanza_languages = []
                    if hasattr(stanza.Pipeline, 'supported_processors'):
                        stanza_languages = list(stanza.Pipeline.supported_processors.keys())
                    
                    # Also check normalized code
                    lang_normalized = self._language
                    if len(self._language) == 3:
                        iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                        if iso_1 and len(iso_1) == 2:
                            lang_normalized = iso_1
                    
                    # If language is not supported, add it
                    if not stanza_languages or (lang_normalized not in stanza_languages and self._language not in stanza_languages):
                        if self._verbose:
                            print(f"[flexipipe] Detected trained model for language '{self._language}'. Adding language to Stanza's constant.py...")
                        _add_language_to_stanza_constant(self._language, language_name, verbose=self._verbose)
                        if lang_normalized != self._language and lang_normalized not in stanza_languages:
                            _add_language_to_stanza_constant(lang_normalized, language_name, verbose=self._verbose)
                except (ImportError, AttributeError, Exception):
                    # If we can't check, try to add it anyway
                    if self._verbose:
                        print(f"[flexipipe] Detected trained model for language '{self._language}'. Adding language to Stanza's constant.py...")
                    _add_language_to_stanza_constant(self._language, language_name, verbose=self._verbose)
            except Exception as e:
                if self._verbose:
                    print(f"[flexipipe] Warning: Could not automatically add language to Stanza's constant.py: {e}")

        # Determine which language code to use for loading
        # ClassLA may normalize 3-letter codes to 2-letter codes
        # If we have a trained model, check if ClassLA expects the normalized code
        lang_to_use = self._language
        if has_trained_model:
            # If we have a trained model, check if ClassLA normalizes the language
            from ..language_utils import standardize_language_metadata
            metadata = standardize_language_metadata(self._language, None)
            if len(self._language) == 3:
                iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                if iso_1 and len(iso_1) == 2:
                    # ClassLA will normalize 'sna' to 'sn'
                    # Since training now saves models directly to the root level using normalized code,
                    # we should use the normalized code to load
                    # Check if model exists under normalized code in resources.json
                    resources_file = classla_dir / "resources.json"
                    if resources_file.exists():
                        try:
                            import json
                            with resources_file.open("r", encoding="utf-8") as f:
                                resources = json.load(f)
                            # Check if model exists under normalized code
                            if isinstance(resources, dict) and iso_1 in resources:
                                # Model registered under normalized code - use that
                                lang_to_use = iso_1
                                if self._verbose:
                                    print(f"[flexipipe] Using normalized language code '{iso_1}' (from '{self._language}') to load trained model")
                        except Exception:
                            # If we can't check, use original code
                            pass
                    # Also check if the directory exists at root level with normalized code
                    normalized_lang_dir = classla_dir / iso_1
                    if normalized_lang_dir.exists() and any(normalized_lang_dir.glob("**/*.pt")):
                        lang_to_use = iso_1
                        if self._verbose:
                            print(f"[flexipipe] Found trained model at root level with normalized code '{iso_1}', using that")
        
        config: Dict[str, Union[str, bool]] = {
            "lang": lang_to_use,
            "processors": self._processors,
            "use_gpu": self._use_gpu,
            "type": self._type,
            "dir": str(classla_dir),
        }
        if pretokenized:
            config["tokenize_pretokenized"] = True

        try:
            return self.classla.Pipeline(**config)
        except Exception as e:
            error_str = str(e)
            
            # Check for "No processor to load" error first - this is a common error that should be handled gracefully
            is_no_processor_error = (
                "No processor to load" in error_str or
                "no processor to load" in error_str.lower()
            )
            
            if is_no_processor_error:
                # Check if we have a trained model that might not be properly registered
                if has_trained_model:
                    raise RuntimeError(
                        f"ClassLA trained model found for language '{self._language}' (package: {self._package}), "
                        f"but no processors are configured. "
                        f"This may indicate that the model was not properly registered in resources.json. "
                        f"Check if processors are defined for '{self._language}' in: {classla_dir / 'resources.json'}"
                    ) from e
                else:
                    # No trained model - check if language is supported by ClassLA
                    # Check if language is in the ClassLA registry's supported_languages field
                    is_supported_language = False
                    try:
                        # Fetch registry to check supported_languages field
                        from ..model_registry import fetch_remote_registry
                        registry = fetch_remote_registry(
                            backend="classla",
                            use_cache=True,
                            refresh_cache=False,
                            verbose=False,
                        )
                        if registry:
                            supported_languages = registry.get("supported_languages", [])
                            # Check both original and normalized language codes
                            lang_normalized = self._language
                            if len(self._language) == 3:
                                from ..language_utils import standardize_language_metadata
                                metadata = standardize_language_metadata(self._language, None)
                                iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                                if iso_1 and len(iso_1) == 2:
                                    lang_normalized = iso_1
                            
                            # Check if language is in supported_languages list
                            is_supported_language = (
                                self._language in supported_languages or
                                lang_normalized in supported_languages
                            )
                    except Exception:
                        # If we can't check, assume it might be supported
                        pass
                    
                    package_info = f" (package: {self._package})" if self._package else ""
                    
                    if is_supported_language:
                        # Language is supported but model is not installed
                        raise RuntimeError(
                            f"ClassLA found no processors to load for language '{self._language}'{package_info} (type: {self._type}). "
                            f"The model is not installed. "
                            f"Use --download-model to install, or run: classla.download('{self._language}', type='{self._type}')"
                        ) from e
                    else:
                        # Language is not supported by ClassLA - check if there are any downloadable models in registry
                        raise RuntimeError(
                            f"ClassLA found no processors to load for language '{self._language}'{package_info} (type: {self._type}). "
                            f"There are currently no known downloadable models for this language. "
                            f"To use ClassLA with this language, you need to train a model first using: "
                            f"flexipipe train --backend classla --language {self._language} --name <model-name>"
                        ) from e
            
            # If we have a trained model and the original language code failed,
            # try the normalized code as a fallback (ClassLA may normalize 3-letter to 2-letter codes)
            if has_trained_model and lang_to_use == self._language:
                from ..language_utils import standardize_language_metadata
                metadata = standardize_language_metadata(self._language, None)
                if len(self._language) == 3:
                    iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                    if iso_1 and len(iso_1) == 2 and iso_1 != self._language:
                        # Try with normalized code
                        config_normalized = dict(config)
                        config_normalized["lang"] = iso_1
                        try:
                            if self._verbose:
                                print(f"[flexipipe] Retrying with normalized language code '{iso_1}' (from '{self._language}')...")
                            return self.classla.Pipeline(**config_normalized)
                        except Exception:
                            # If normalized code also fails, continue with original error handling
                            pass
            # Re-raise original exception to continue with normal error handling
            raise
        except (ValueError, TypeError) as e:
            # Check if this is a JSON parsing error from ClassLA's resource loading
            error_str = str(e)
            if "Expecting value" in error_str or "JSONDecodeError" in error_str:
                # Check if resources.json is empty or corrupted and try to fix it
                resources_file = classla_dir / "resources.json"
                if resources_file.exists():
                    try:
                        import json
                        with open(resources_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if not content:
                                # Empty file - delete it and try to rebuild
                                if self._verbose:
                                    print(f"[flexipipe] Detected empty resources.json, attempting to rebuild...", file=sys.stderr)
                                resources_file.unlink()
                            else:
                                # Try to parse - if it fails, it's corrupted
                                json.loads(content)
                                # If we get here, JSON is valid but ClassLA still failed
                                # This might be a different issue
                                raise RuntimeError(
                                    f"ClassLA failed to load resources despite valid JSON file. "
                                    f"This may indicate a ClassLA version mismatch or corrupted model files. "
                                    f"Try re-downloading: classla.download('{self._language}', type='{self._type}')"
                                ) from e
                    except (json.JSONDecodeError, ValueError):
                        # Corrupted JSON - delete it and try to rebuild
                        if self._verbose:
                            print(f"[flexipipe] Detected corrupted resources.json, attempting to rebuild...", file=sys.stderr)
                        resources_file.unlink()
                
                # Always try to rebuild when we detect corrupted/empty resources.json
                # This is a recovery mechanism, not a download request
                if self._verbose:
                    print(f"[flexipipe] Rebuilding ClassLA resources for {self._language} (type: {self._type})...", file=sys.stderr)
                try:
                    self.classla.download(self._language, type=self._type, verbose=self._verbose)
                    
                    # Invalidate caches so the new model appears immediately
                    try:
                        from ..model_catalog import invalidate_unified_catalog_cache
                        invalidate_unified_catalog_cache()
                        # Also invalidate ClassLA's local cache
                        from ..model_storage import get_cache_dir
                        cache_dir = get_cache_dir()
                        classla_cache = cache_dir / "classla.json"
                        if classla_cache.exists():
                            classla_cache.unlink()
                        # Refresh ClassLA model entries to include the new model
                        get_classla_model_entries(refresh_cache=True, verbose=self._verbose)
                    except Exception:
                        pass  # Best effort - don't fail if cache invalidation fails
                    
                    # Retry pipeline creation after download
                    return self.classla.Pipeline(**config)
                except Exception as download_error:
                    raise RuntimeError(
                        f"Failed to rebuild ClassLA resources: {download_error}. "
                        f"Try manually: classla.download('{self._language}', type='{self._type}')"
                    ) from download_error
            
            # Check for "No processor to load" error - this happens when ClassLA can't find any processors
            # for the given language/package combination
            error_str = str(e)
            is_no_processor_error = (
                "No processor to load" in error_str or
                "no processor to load" in error_str.lower()
            )
            
            if is_no_processor_error:
                # Check if we have a trained model that might not be properly registered
                if has_trained_model:
                    raise RuntimeError(
                        f"ClassLA trained model found for language '{self._language}' (package: {self._package}), "
                        f"but no processors are configured. "
                        f"This may indicate that the model was not properly registered in resources.json. "
                        f"Check if processors are defined for '{self._language}' in: {classla_dir / 'resources.json'}"
                    ) from e
                else:
                    # No trained model - check if language is supported by ClassLA
                    # Check if language is in the ClassLA registry's supported_languages field
                    is_supported_language = False
                    try:
                        # Fetch registry to check supported_languages field
                        from ..model_registry import fetch_remote_registry
                        registry = fetch_remote_registry(
                            backend="classla",
                            use_cache=True,
                            refresh_cache=False,
                            verbose=False,
                        )
                        if registry:
                            supported_languages = registry.get("supported_languages", [])
                            # Check both original and normalized language codes
                            lang_normalized = self._language
                            if len(self._language) == 3:
                                from ..language_utils import standardize_language_metadata
                                metadata = standardize_language_metadata(self._language, None)
                                iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                                if iso_1 and len(iso_1) == 2:
                                    lang_normalized = iso_1
                            
                            # Check if language is in supported_languages list
                            is_supported_language = (
                                self._language in supported_languages or
                                lang_normalized in supported_languages
                            )
                    except Exception:
                        # If we can't check, assume it might be supported
                        pass
                    
                    package_info = f" (package: {self._package})" if self._package else ""
                    
                    if is_supported_language:
                        # Language is supported but model is not installed
                        raise RuntimeError(
                            f"ClassLA found no processors to load for language '{self._language}'{package_info} (type: {self._type}). "
                            f"The model is not installed. "
                            f"Use --download-model to install, or run: classla.download('{self._language}', type='{self._type}')"
                        ) from e
                    else:
                        # Language is not supported by ClassLA - check if there are any downloadable models in registry
                        raise RuntimeError(
                            f"ClassLA found no processors to load for language '{self._language}'{package_info} (type: {self._type}). "
                            f"There are currently no known downloadable models for this language. "
                            f"To use ClassLA with this language, you need to train a model first using: "
                            f"flexipipe train --backend classla --language {self._language} --name <model-name>"
                        ) from e
            
            error_str = str(e)
            if "not enough values to unpack" in error_str or "expected 2" in error_str:
                config_minimal = {
                    "lang": self._language,
                    "processors": self._processors,
                    "use_gpu": self._use_gpu,
                    "type": self._type,
                }
                pipeline = self.classla.Pipeline(**config_minimal)
                if pretokenized and hasattr(pipeline, "tokenize_pretokenized"):
                    pipeline.tokenize_pretokenized = True
                return pipeline
            raise
        except Exception as e:
            # Catch any Exception (including ResourcesFileNotFound) to check if it's a resources error
            error_str = str(e)
            
            # Check for "No processor to load" error - this happens when ClassLA can't find any processors
            # for the given language/package combination
            is_no_processor_error = (
                "No processor to load" in error_str or
                "no processor to load" in error_str.lower()
            )
            
            if is_no_processor_error:
                # Check if we have a trained model that might not be properly registered
                if has_trained_model:
                    raise RuntimeError(
                        f"ClassLA trained model found for language '{self._language}' (package: {self._package}), "
                        f"but no processors are configured. "
                        f"This may indicate that the model was not properly registered in resources.json. "
                        f"Check if processors are defined for '{self._language}' in: {classla_dir / 'resources.json'}"
                    ) from e
                else:
                    # No trained model - check if language is supported by ClassLA
                    # Check if language is in the ClassLA registry's supported_languages field
                    is_supported_language = False
                    try:
                        # Fetch registry to check supported_languages field
                        from ..model_registry import fetch_remote_registry
                        registry = fetch_remote_registry(
                            backend="classla",
                            use_cache=True,
                            refresh_cache=False,
                            verbose=False,
                        )
                        if registry:
                            supported_languages = registry.get("supported_languages", [])
                            # Check both original and normalized language codes
                            lang_normalized = self._language
                            if len(self._language) == 3:
                                from ..language_utils import standardize_language_metadata
                                metadata = standardize_language_metadata(self._language, None)
                                iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                                if iso_1 and len(iso_1) == 2:
                                    lang_normalized = iso_1
                            
                            # Check if language is in supported_languages list
                            is_supported_language = (
                                self._language in supported_languages or
                                lang_normalized in supported_languages
                            )
                    except Exception:
                        # If we can't check, assume it might be supported
                        pass
                    
                    package_info = f" (package: {self._package})" if self._package else ""
                    
                    if is_supported_language:
                        # Language is supported but model is not installed
                        raise RuntimeError(
                            f"ClassLA found no processors to load for language '{self._language}'{package_info} (type: {self._type}). "
                            f"The model is not installed. "
                            f"Use --download-model to install, or run: classla.download('{self._language}', type='{self._type}')"
                        ) from e
                    else:
                        # Language is not supported by ClassLA - check if there are any downloadable models in registry
                        raise RuntimeError(
                            f"ClassLA found no processors to load for language '{self._language}'{package_info} (type: {self._type}). "
                            f"There are currently no known downloadable models for this language. "
                            f"To use ClassLA with this language, you need to train a model first using: "
                            f"flexipipe train --backend classla --language {self._language} --name <model-name>"
                        ) from e
            
            is_resources_error = (
                "Resources file not found" in error_str or
                "resources" in error_str.lower() or
                isinstance(e, self._resources_error)
            )
            
            # Check if this is a missing pretrain/vector file error
            is_missing_pretrain = (
                "vector file is not provided" in error_str.lower() or
                ("pretrain" in error_str.lower() and ("not found" in error_str.lower() or "missing" in error_str.lower()))
            )
            
            if is_missing_pretrain:
                # If download is explicitly requested, try to download the complete model
                if self._download:
                    if self._verbose:
                        print(f"[flexipipe] Missing pretrain/vector file detected. Downloading complete ClassLA model for {self._language} (type: {self._type})...", flush=True)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    try:
                        # Download with all required processors, including pretrain
                        processors_list = [p.strip() for p in self._processors.split(",") if p.strip()]
                        # Always include pretrain if pos is requested (POS tagger requires pretrain vectors)
                        if "pos" in processors_list and "pretrain" not in processors_list:
                            processors_list.append("pretrain")
                        processors_dict = {proc: True for proc in processors_list}
                        self.classla.download(
                            self._language, 
                            type=self._type, 
                            processors=processors_dict,
                            verbose=self._verbose
                        )
                        if self._verbose:
                            sys.stdout.flush()
                            sys.stderr.flush()
                            print(f"[flexipipe] ClassLA model download completed", flush=True)
                        
                        # Invalidate caches so the new model appears immediately
                        try:
                            from ..model_catalog import invalidate_unified_catalog_cache
                            invalidate_unified_catalog_cache()
                            # Also invalidate ClassLA's local cache
                            from ..model_storage import get_cache_dir
                            cache_dir = get_cache_dir()
                            classla_cache = cache_dir / "classla.json"
                            if classla_cache.exists():
                                classla_cache.unlink()
                            # Cache will be rebuilt on next access
                        except Exception:
                            pass  # Best effort - don't fail if cache invalidation fails
                        
                        # Retry pipeline creation after download
                        return self.classla.Pipeline(**config)
                    except Exception as download_error:
                        raise RuntimeError(
                            f"Failed to download ClassLA model: {download_error}. "
                            f"Try manually: classla.download('{self._language}', type='{self._type}')"
                        ) from download_error
                # Check if we have a trained model before suggesting download
                if has_trained_model:
                    raise RuntimeError(
                        f"ClassLA trained model found for language '{self._language}', but the pretrain/vector file is missing. "
                        f"This is required for the POS tagger. "
                        f"Check if pretrain files exist in: {classla_dir / self._language / 'pretrain'}"
                    ) from e
                else:
                    # If download is not requested, raise error with instructions
                    raise RuntimeError(
                        f"ClassLA model for language '{self._language}' (type: {self._type}) is incomplete. "
                        f"The pretrain/vector file is missing, which is required for the POS tagger. "
                        f"Use --download-model to download the complete model, or run: "
                        f"classla.download('{self._language}', type='{self._type}')"
                    ) from e
            
            if is_resources_error:
                # Check if resources.json is missing or empty - if so, try to rebuild automatically
                resources_file = classla_dir / "resources.json"
                should_auto_rebuild = False
                
                if not resources_file.exists():
                    should_auto_rebuild = True
                    if self._verbose:
                        print(f"[flexipipe] Resources file not found, attempting to rebuild...", file=sys.stderr)
                elif resources_file.exists():
                    # Check if file is empty
                    try:
                        if resources_file.stat().st_size == 0:
                            should_auto_rebuild = True
                            if self._verbose:
                                print(f"[flexipipe] Resources file is empty, attempting to rebuild...", file=sys.stderr)
                            resources_file.unlink()
                    except OSError:
                        pass
                
                # Auto-rebuild only for corrupted/empty resources.json (recovery mechanism)
                # For missing models, require explicit --download-model flag
                if should_auto_rebuild:
                    if self._verbose:
                        print(f"[flexipipe] Rebuilding ClassLA resources for {self._language} (type: {self._type})...", file=sys.stderr)
                    try:
                        # Only download the processors we actually need, not all available processors
                        # This prevents downloading unnecessary models (like Ukrainian lemmatizer)
                        # Processors should be a dict mapping processor names to True/False or package names
                        processors_list = [p.strip() for p in self._processors.split(",") if p.strip()]
                        # Always include pretrain if pos is requested (POS tagger requires pretrain vectors)
                        if "pos" in processors_list and "pretrain" not in processors_list:
                            processors_list.append("pretrain")
                        processors_dict = {proc: True for proc in processors_list}
                        self.classla.download(
                            self._language, 
                            type=self._type, 
                            processors=processors_dict,
                            verbose=self._verbose
                        )
                        
                        # Invalidate caches so the new model appears immediately
                        try:
                            from ..model_catalog import invalidate_unified_catalog_cache
                            invalidate_unified_catalog_cache()
                            # Also invalidate ClassLA's local cache
                            from ..model_storage import get_cache_dir
                            cache_dir = get_cache_dir()
                            classla_cache = cache_dir / "classla.json"
                            if classla_cache.exists():
                                classla_cache.unlink()
                            # Cache will be rebuilt on next access
                        except Exception:
                            pass  # Best effort - don't fail if cache invalidation fails
                        
                        # Retry pipeline creation after download
                        return self.classla.Pipeline(**config)
                    except Exception as download_error:
                        raise RuntimeError(
                            f"Failed to rebuild ClassLA resources: {download_error}. "
                            f"Try manually: classla.download('{self._language}', type='{self._type}')"
                        ) from download_error
                # Check if we have a trained model before suggesting download
                if has_trained_model:
                    # We have a trained model but ClassLA can't load it
                    # This might be because the language isn't in SUPPORTED_LANGUAGES
                    # We already tried to add it above, but if it still fails, provide a helpful error
                    raise RuntimeError(
                        f"ClassLA trained model found for language '{self._language}', but it cannot be loaded. "
                        f"This may be because the language code is not recognized by ClassLA. "
                        f"The language has been automatically added to Stanza's constant.py. "
                        f"If the error persists, try restarting Python or check the model files at: {classla_dir}"
                    ) from e
                else:
                    # No trained model found - suggest download
                    raise RuntimeError(
                        f"ClassLA model not found for language '{self._language}' "
                        f"(package: {self._package}, type: {self._type}). "
                        f"Use --download-model to install, or run: classla.download('{self._language}', type='{self._type}')"
                    ) from e
            
            # Check if it's a file/OS error for depparse fallback
            error_str_lower = error_str.lower()
            if "depparse" in error_str_lower or "parser" in error_str_lower or "no such file" in error_str_lower:
                processors_list = [p.strip() for p in self._processors.split(",") if p.strip()]
                if "depparse" in processors_list:
                    processors_list.remove("depparse")
                    config_fallback = dict(config)
                    config_fallback["processors"] = ",".join(processors_list)
                    return self.classla.Pipeline(**config_fallback)
            
            # Not handled - re-raise
            raise

    def _get_pipeline(self, pretokenized: bool):
        if pretokenized not in self._pipelines:
            self._pipelines[pretokenized] = self._build_pipeline(pretokenized)
        return self._pipelines[pretokenized]

    def _run_raw(self, document: Document):
        pipeline = self._get_pipeline(pretokenized=False)
        text = "\n".join(sent.text for sent in document.sentences if sent.text)
        return pipeline(text)

    def _run_pretokenized(self, document: Document):
        from ..doc_utils import get_effective_form
        pipeline = self._get_pipeline(pretokenized=True)
        pretokenized = [[get_effective_form(token) for token in sentence.tokens] for sentence in document.sentences]
        return pipeline(pretokenized)

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[List[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        del overrides, preserve_pos_tags, components

        start_time = time.time()
        if use_raw_text or not document.sentences:
            classla_doc = self._run_raw(document)
            result_doc = _classla_doc_to_document(classla_doc)
        else:
            classla_doc = self._run_pretokenized(document)
            result_doc = _classla_doc_to_document(classla_doc, original_doc=document)

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
        """
        Train a ClassLA model from CoNLL-U data.
        
        Args:
            train_data: Training data (Document, list of Documents, or Path to CoNLL-U file/dir)
            output_dir: Directory to save the trained model
            dev_data: Optional development data
            **kwargs: Additional training parameters:
                - language: Language code (required)
                - package: Package name (optional, defaults based on language)
                - type: Model type ("standard" or "nonstandard", default: "standard")
                - unimorph_vocab: Path to UniMorph vocabulary file (optional)
                - tagset_file: Path to tagset.xml for XPOS mapping (optional, for UniMorph)
                - verbose: Whether to print progress
                - force: Whether to overwrite existing model
        
        Returns:
            Path to the trained model directory
        """
        import shutil
        import tempfile
        from datetime import datetime
        from ..conllu import conllu_to_document, document_to_conllu
        
        language = kwargs.get("language") or self._language
        if not language:
            raise ValueError("ClassLA training requires a language code. Provide --language.")
        
        # Use model name (from --name during training) as package if provided, allowing multiple models per language
        # This enables dialects and different training variants (e.g., "sna-masakhane", "sna-other")
        # Note: --name is used for training (naming the model being created), --model is used for tagging (selecting existing model)
        model_name = kwargs.get("model_name") or kwargs.get("name")  # Support both for backward compatibility
        if model_name:
            # Use the model name as the package to distinguish between different models
            package = model_name
        else:
            # Fall back to explicit package, then default based on language
            package = kwargs.get("package") or self._package
            if not package:
                default_package = {"hr": "set", "sr": "set", "bg": "btb", "mk": "mk", "sl": "ssj"}
                package = default_package.get(language, language)
        
        model_type = kwargs.get("type") or self._type or "standard"
        verbose = bool(kwargs.get("verbose", False) or self._verbose)
        force = bool(kwargs.get("force", False))
        unimorph_vocab = kwargs.get("unimorph_vocab")
        tagset_file = kwargs.get("tagset_file")
        
        output_dir = Path(output_dir).resolve()
        
        # Handle force flag
        if output_dir.exists():
            if any(output_dir.iterdir()):
                if force:
                    if verbose:
                        print(f"[flexipipe] Warning: Model already exists at {output_dir}. Emptying directory (--force specified).", file=sys.stderr)
                    shutil.rmtree(output_dir)
                else:
                    raise ValueError(
                        f"Output directory {output_dir} must be empty before training. "
                        f"Use --force to overwrite existing model."
                    )
            else:
                shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect training data
        train_files = self._collect_conllu_files(train_data, split="train")
        if dev_data:
            dev_files = self._collect_conllu_files(dev_data, split="dev")
        else:
            dev_files = []
            if isinstance(train_data, (str, Path)):
                base_candidate = Path(train_data)
                if base_candidate.is_dir():
                    try:
                        dev_files = self._collect_conllu_files(base_candidate, split="dev")
                    except ValueError:
                        dev_files = []
        
        # Collect test data if available (for Stanza's automatic test evaluation)
        test_files = []
        if isinstance(train_data, (str, Path)):
            base_candidate = Path(train_data)
            if base_candidate.is_dir():
                try:
                    test_files = self._collect_conllu_files(base_candidate, split="test")
                except ValueError:
                    test_files = []
        
        if not train_files:
            raise ValueError("No training data files found.")
        
        if verbose:
            print(f"[flexipipe] Training ClassLA model for language '{language}' (package: {package}, type: {model_type})")
            print(f"[flexipipe] Training files: {len(train_files)}")
            if dev_files:
                print(f"[flexipipe] Dev files: {len(dev_files)}")
            if test_files:
                print(f"[flexipipe] Test files: {len(test_files)}")
        
        # Prepare training data in ClassLA's expected format
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            
            # Load all training data to analyze annotation coverage
            from ..conllu import conllu_to_document, document_to_conllu
            from ..train import _has_annotation
            
            # Load all sentences from training files
            all_train_sentences = []
            for train_file in train_files:
                doc = conllu_to_document(train_file.read_text(encoding="utf-8", errors="replace"))
                all_train_sentences.extend(doc.sentences)
            
            # Dynamically split sentences by annotation coverage
            # This allows using larger datasets for tagging than for parsing
            # (similar to how flexitag handles incomplete sentences)
            tagging_sentences = []  # Sentences with UPOS/XPOS/FEATS
            parsing_sentences = []  # Sentences with HEAD/DEPREL
            lemma_sentences = []    # Sentences with LEMMA
            
            for sent in all_train_sentences:
                has_tagging = _has_annotation(sent, "upos") or _has_annotation(sent, "xpos")
                has_parsing = _has_annotation(sent, "head") and _has_annotation(sent, "deprel")
                has_lemma = _has_annotation(sent, "lemma")
                
                if has_tagging:
                    tagging_sentences.append(sent)
                if has_parsing:
                    parsing_sentences.append(sent)
                if has_lemma:
                    lemma_sentences.append(sent)
            
            if verbose:
                print(f"[flexipipe] Annotation coverage analysis:")
                print(f"  Total sentences: {len(all_train_sentences)}")
                print(f"  Sentences with tagging (UPOS/XPOS): {len(tagging_sentences)}")
                print(f"  Sentences with parsing (HEAD/DEPREL): {len(parsing_sentences)}")
                print(f"  Sentences with lemma: {len(lemma_sentences)}")
            
            # Create separate training files for different components
            # This allows CLASSLA to use larger datasets for tagging than parsing
            train_tagging_conllu = tmp_dir / "train_tagging.conllu"
            train_parsing_conllu = tmp_dir / "train_parsing.conllu"
            train_lemma_conllu = tmp_dir / "train_lemma.conllu"
            
            if tagging_sentences:
                tagging_doc = Document(id="tagging_train", sentences=tagging_sentences)
                train_tagging_conllu.write_text(document_to_conllu(tagging_doc), encoding="utf-8")
            
            if parsing_sentences:
                parsing_doc = Document(id="parsing_train", sentences=parsing_sentences)
                train_parsing_conllu.write_text(document_to_conllu(parsing_doc), encoding="utf-8")
            
            if lemma_sentences:
                lemma_doc = Document(id="lemma_train", sentences=lemma_sentences)
                train_lemma_conllu.write_text(document_to_conllu(lemma_doc), encoding="utf-8")
            
            # Also create a combined file for components that need all annotations
            train_conllu = tmp_dir / "train.conllu"
            self._merge_conllu_files(train_files, train_conllu)
            
            dev_conllu = None
            if dev_files:
                dev_conllu = tmp_dir / "dev.conllu"
                self._merge_conllu_files(dev_files, dev_conllu)
            elif train_files:
                # Use a portion of training data as dev if no dev set provided
                # This is a simple approach - in practice, you'd want proper splitting
                dev_conllu = tmp_dir / "dev.conllu"
                shutil.copy(train_conllu, dev_conllu)
            
            # Create test file if test data is available
            test_conllu = None
            if test_files:
                test_conllu = tmp_dir / "test.conllu"
                self._merge_conllu_files(test_files, test_conllu)
            
            # Process UniMorph vocabulary if provided
            vocab_file = None
            if unimorph_vocab:
                vocab_file = self._prepare_unimorph_vocab(
                    Path(unimorph_vocab),
                    train_conllu,
                    tmp_dir,
                    tagset_file=Path(tagset_file) if tagset_file else None,
                    verbose=verbose,
                )
            
            # ClassLA models are stored in: lang/processor/package.pt
            # IMPORTANT: ClassLA expects models at the root level of the ClassLA models directory,
            # not in a nested structure. So we need to save directly to classla_dir/language/processor/package.pt
            # However, we still use output_dir for training metadata and temporary files
            from ..model_storage import get_backend_models_dir
            classla_root = get_backend_models_dir("classla", create=True)
            
            # Determine the final location for the model (use normalized code if ClassLA normalizes)
            lang_normalized = language
            if len(language) == 3:
                from ..language_utils import standardize_language_metadata
                metadata = standardize_language_metadata(language, None)
                iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                if iso_1 and len(iso_1) == 2:
                    lang_normalized = iso_1
            
            # Save model files directly to root level (like official models: mk/pos/..., bg/pos/...)
            # Use normalized code since ClassLA will normalize it anyway
            model_lang_dir = classla_root / lang_normalized
            model_lang_dir.mkdir(parents=True, exist_ok=True)
            
            # Also keep a reference to the original output_dir for metadata
            metadata_dir = output_dir / language
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # For now, we'll save the training data and metadata
            # Actual model training would require ClassLA's training tools
            # This implementation prepares everything for training
            
            # Save training data (including component-specific splits)
            training_data_dir = model_lang_dir / "training_data"
            training_data_dir.mkdir(exist_ok=True)
            shutil.copy(train_conllu, training_data_dir / "train.conllu")
            if dev_conllu:
                shutil.copy(dev_conllu, training_data_dir / "dev.conllu")
            if test_conllu:
                shutil.copy(test_conllu, training_data_dir / "test.conllu")
            
            # Save component-specific training files
            if tagging_sentences and train_tagging_conllu.exists():
                shutil.copy(train_tagging_conllu, training_data_dir / "train_tagging.conllu")
            if parsing_sentences and train_parsing_conllu.exists():
                shutil.copy(train_parsing_conllu, training_data_dir / "train_parsing.conllu")
            if lemma_sentences and train_lemma_conllu.exists():
                shutil.copy(train_lemma_conllu, training_data_dir / "train_lemma.conllu")
            
            # Save vocabulary if available
            if vocab_file:
                vocab_dir = model_lang_dir / "vocab"
                vocab_dir.mkdir(exist_ok=True)
                shutil.copy(vocab_file, vocab_dir / "vocab.json")
            
            # Create metadata
            meta = {
                "name": f"{language}-{model_type}",
                "language": language,
                "package": package,
                "type": model_type,
                "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "training_files": [str(f) for f in train_files],
                "dev_files": [str(f) for f in dev_files] if dev_files else [],
                "unimorph_vocab": str(unimorph_vocab) if unimorph_vocab else None,
                "flexipipe": {
                    "original_language": language,
                    "source": str(train_data) if isinstance(train_data, (str, Path)) else "in-memory",
                }
            }
            
            meta_path = output_dir / "meta.json"
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            
            # Create a resources.json entry for ClassLA
            resources_entry = {
                "default_processors": {
                    "tokenize": None,
                    "pos": None,
                    "lemma": None,
                },
                "lang": language,
                "package": package,
                "type": model_type,
            }
            
            resources_path = model_lang_dir / "resources.json"
            with resources_path.open("w", encoding="utf-8") as f:
                json.dump(resources_entry, f, indent=2, ensure_ascii=False)
            
            # Run CLASSLA training using Stanza's training infrastructure
            # Since CLASSLA is based on Stanza, we use Stanza's training via subprocess
            if verbose:
                print(f"[flexipipe] Starting CLASSLA model training...")
                print(f"[flexipipe] Component-specific training files:")
                if tagging_sentences:
                    print(f"  - train_tagging.conllu: {len(tagging_sentences)} sentences (for POS tagging)")
                if parsing_sentences:
                    print(f"  - train_parsing.conllu: {len(parsing_sentences)} sentences (for dependency parsing)")
                if lemma_sentences:
                    print(f"  - train_lemma.conllu: {len(lemma_sentences)} sentences (for lemmatization)")
            
            training_successful = False
            components_trained = []
            
            # Check if pretrain file is provided
            wordvec_pretrain_file = kwargs.get("wordvec_pretrain_file")
            if wordvec_pretrain_file:
                wordvec_path = Path(wordvec_pretrain_file).expanduser().resolve()
                if not wordvec_path.exists():
                    raise ValueError(f"Pretrain file not found: {wordvec_path}")
                if verbose:
                    print(f"[flexipipe] Using pretrain embeddings from: {wordvec_path}")
            
            try:
                import subprocess
                import os
                
                # Train POS tagger if we have tagging data
                if tagging_sentences and train_tagging_conllu.exists():
                    if verbose:
                        print(f"[flexipipe] Training POS tagger...")
                    
                    pos_train_file = training_data_dir / "train_tagging.conllu"
                    pos_dev_file = training_data_dir / "dev.conllu" if dev_conllu and (training_data_dir / "dev.conllu").exists() else pos_train_file
                    
                    # Create processor directory structure: lang/processor/package.pt
                    pos_output_dir = model_lang_dir / "pos"
                    pos_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Use Stanza's training infrastructure (CLASSLA is based on Stanza)
                    # Stanza training expects treebank names, so we create a temporary UD treebank structure
                    try:
                        # Try to import stanza, handling various import errors
                        # Note: We don't strictly need to import stanza here - the subprocess call will use a fresh Python process
                        # But we try to import it to check language support and add languages to constant.py if needed
                        stanza_imported = False
                        stanza = None
                        try:
                            import stanza
                            stanza_imported = True
                        except (AssertionError, AttributeError) as import_err:
                            if isinstance(import_err, AssertionError):
                                if verbose:
                                    print(f"[flexipipe] Warning: Stanza import failed due to duplicate language mapping in constant.py")
                                    print(f"[flexipipe] Attempting to fix by removing conflicting 'sna' entry (since 'sn' already exists)...")
                                # Remove "sna" if it exists, since "sn" already covers it
                                _remove_language_from_stanza_constant("sna", verbose=verbose)
                                # Try importing again - need to clear from cache first
                                import importlib
                                # Clear stanza from cache if present (sys is already imported at module level)
                                for key in list(sys.modules.keys()):
                                    if key.startswith('stanza'):
                                        del sys.modules[key]
                                try:
                                    import stanza
                                    stanza_imported = True
                                except Exception as retry_err:
                                    if verbose:
                                        print(f"[flexipipe] Warning: Could not import Stanza after fixing constant.py: {retry_err}")
                                        print(f"[flexipipe] This may be due to a corrupted Stanza installation or protobuf issues.")
                                        print(f"[flexipipe] Training will proceed using subprocess (fresh Python process will import Stanza correctly)")
                                    # Don't raise - we can still proceed with subprocess
                                    stanza_imported = False
                            else:
                                # AttributeError (e.g., protobuf issues)
                                if verbose:
                                    print(f"[flexipipe] Warning: Stanza import failed due to internal error: {import_err}")
                                    print(f"[flexipipe] This may be due to a corrupted Stanza installation or protobuf issues.")
                                    print(f"[flexipipe] Training will proceed using subprocess (fresh Python process will import Stanza correctly)")
                                    print(f"[flexipipe] If training fails, try: pip install --upgrade --force-reinstall stanza")
                                # Don't raise - we can still proceed with subprocess
                                stanza_imported = False
                        
                        # CRITICAL: Add language to constant.py BEFORE attempting training
                        # This prevents UnknownLanguageError during argument parsing (find_wordvec_pretrain)
                        # Even if we provide --wordvec_pretrain_file, Stanza still tries to auto-detect during argument parsing
                        # Try multiple methods to check if language is supported
                        stanza_languages = []
                        lang_normalized_for_check = language
                        if len(language) == 3:
                            from ..language_utils import standardize_language_metadata
                            metadata = standardize_language_metadata(language, None)
                            iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                            if iso_1 and len(iso_1) == 2:
                                lang_normalized_for_check = iso_1
                        
                        # Method 1: Try Pipeline.supported_processors (newer Stanza versions)
                        if stanza_imported:
                            try:
                                if hasattr(stanza.Pipeline, 'supported_processors'):
                                    stanza_languages = list(stanza.Pipeline.supported_processors.keys())
                            except (AttributeError, Exception):
                                pass
                            
                            # Method 2: Try resources.common.get_resources_json (fallback)
                            if not stanza_languages:
                                try:
                                    from stanza.resources.common import get_resources_json
                                    resources = get_resources_json()
                                    if resources and "resources" in resources:
                                        stanza_languages = list(set(
                                            res.get("lang", "") for res in resources.get("resources", [])
                                            if res.get("lang")
                                        ))
                                except (ImportError, Exception):
                                    pass
                        
                        # If we still can't determine supported languages, proactively add the language
                        # This is safer than assuming it's supported
                        if not stanza_languages or (lang_normalized_for_check not in stanza_languages and language not in stanza_languages):
                            if verbose:
                                if stanza_languages:
                                    print(f"[flexipipe] Language '{language}' (normalized: '{lang_normalized_for_check}') is not in Stanza's supported languages.")
                                else:
                                    print(f"[flexipipe] Could not determine Stanza's supported languages, proactively adding '{language}' to prevent errors.")
                                print(f"[flexipipe] Adding language code to Stanza's constant.py to prevent UnknownLanguageError during argument parsing...")
                            
                            # Try to add the language code automatically
                            language_name = kwargs.get("language_name")
                            if not language_name:
                                from ..language_utils import standardize_language_metadata
                                metadata = standardize_language_metadata(language, None)
                                language_name = metadata.get(LANGUAGE_FIELD_NAME) or (language.upper() if len(language) <= 2 else language.capitalize())
                            
                            # Add both original and normalized codes if needed
                            codes_to_add = [language]
                            if lang_normalized_for_check != language and lang_normalized_for_check not in codes_to_add:
                                codes_to_add.append(lang_normalized_for_check)
                            
                            all_added = True
                            for code_to_add in codes_to_add:
                                added = _add_language_to_stanza_constant(code_to_add, language_name, verbose=verbose)
                                if not added:
                                    all_added = False
                            
                            if all_added:
                                # Try to reload stanza to pick up the new language, but don't fail if it doesn't work
                                # The subprocess call will use a fresh Python process that will import stanza correctly
                                try:
                                    for key in list(sys.modules.keys()):
                                        if key.startswith('stanza'):
                                            del sys.modules[key]
                                    import stanza
                                    
                                    # Try to verify the language was added (but don't fail if we can't check)
                                    try:
                                        if hasattr(stanza.Pipeline, 'supported_processors'):
                                            updated_languages = list(stanza.Pipeline.supported_processors.keys())
                                            if lang_normalized_for_check in updated_languages or language in updated_languages:
                                                if verbose:
                                                    print(f"[flexipipe] Successfully added language code to Stanza's constant.py.")
                                            else:
                                                if verbose:
                                                    print(f"[flexipipe] Warning: Language still not recognized after adding to constant.py.")
                                                    print(f"[flexipipe] This may cause UnknownLanguageError during argument parsing.")
                                    except (AttributeError, Exception):
                                        # Can't verify, but proceed anyway
                                        if verbose:
                                            print(f"[flexipipe] Added language code to Stanza's constant.py (verification unavailable).")
                                except Exception as reload_err:
                                    # If reload fails (e.g., protobuf issues), that's okay
                                    # The subprocess call will use a fresh Python process
                                    if verbose:
                                        print(f"[flexipipe] Note: Could not reload Stanza module to verify language code: {reload_err}")
                                        print(f"[flexipipe] The language code has been added to constant.py")
                                        print(f"[flexipipe] Training will proceed using subprocess (fresh Python process will pick up the changes)")
                            else:
                                if verbose:
                                    print(f"[flexipipe] Warning: Could not automatically add language code to Stanza.")
                                    print(f"[flexipipe] This may cause UnknownLanguageError during argument parsing.")
                        
                        # Check if language is recognized by Stanza's resource registry
                        # If not, try to get pretrain embeddings proactively before training starts
                        if not wordvec_pretrain_file and stanza_imported:
                            try:
                                # Try to check if language is in Stanza's resource registry
                                from stanza.resources.common import get_resources_json, UnknownLanguageError
                                resources = get_resources_json()
                                # Check if language is in the resources
                                lang_normalized = language
                                if len(language) == 3:
                                    from ..language_utils import standardize_language_metadata
                                    metadata = standardize_language_metadata(language, None)
                                    iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                                    if iso_1 and len(iso_1) == 2:
                                        lang_normalized = iso_1
                                
                                # Check if language has pretrains in resources
                                has_pretrains = False
                                if resources and "resources" in resources:
                                    for res in resources.get("resources", []):
                                        if res.get("lang") == lang_normalized or res.get("lang") == language:
                                            if "pretrain" in res.get("alias", "").lower() or "pretrain" in str(res).lower():
                                                has_pretrains = True
                                                break
                                
                                # If no pretrains available, try to get them proactively
                                if not has_pretrains:
                                    if verbose:
                                        print(f"[flexipipe] Warning: Language '{language}' does not have pretrain embeddings in Stanza's resource registry.")
                                        print(f"[flexipipe] Attempting to obtain pretrain embeddings proactively...")
                                    
                                    # Try fastText first
                                    fasttext_embeddings = _try_download_fasttext_embeddings(
                                        language, lang_normalized, model_lang_dir, verbose=verbose
                                    )
                                    
                                    if fasttext_embeddings:
                                        pretrain_pt = _convert_fasttext_to_stanza_pretrain(
                                            fasttext_embeddings, model_lang_dir / "pretrain", language, verbose=verbose
                                        )
                                        if pretrain_pt:
                                            wordvec_pretrain_file = str(pretrain_pt)
                                            wordvec_path = Path(pretrain_pt)
                                            if verbose:
                                                print(f"[flexipipe] Using fastText embeddings: {pretrain_pt}")
                                    
                                    # If fastText failed, try XLM-RoBERTa
                                    if not wordvec_pretrain_file:
                                        pretrain_pt = _try_download_xlmroberta_embeddings(
                                            model_lang_dir / "pretrain", language, verbose=verbose
                                        )
                                        if pretrain_pt:
                                            wordvec_pretrain_file = str(pretrain_pt)
                                            wordvec_path = Path(pretrain_pt)
                                            if verbose:
                                                print(f"[flexipipe] Using XLM-RoBERTa embeddings: {pretrain_pt}")
                                    
                                    if not wordvec_pretrain_file and verbose:
                                        print(f"[flexipipe] Could not obtain pretrain embeddings proactively.")
                                        print(f"[flexipipe] Training will attempt to proceed, but may fail if pretrains are required.")
                            except Exception as e:
                                # If we can't check, just proceed - the error will be caught later
                                if verbose:
                                    print(f"[flexipipe] Could not check pretrain availability: {e}")
                                pass
                        elif not wordvec_pretrain_file and not stanza_imported:
                            # If stanza import failed, still try to get pretrain embeddings proactively
                            if verbose:
                                print(f"[flexipipe] Warning: Could not import Stanza to check pretrain availability.")
                                print(f"[flexipipe] Attempting to obtain pretrain embeddings proactively...")
                            
                            # Try fastText first
                            lang_normalized = language
                            if len(language) == 3:
                                from ..language_utils import standardize_language_metadata
                                metadata = standardize_language_metadata(language, None)
                                iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                                if iso_1 and len(iso_1) == 2:
                                    lang_normalized = iso_1
                            
                            fasttext_embeddings = _try_download_fasttext_embeddings(
                                language, lang_normalized, model_lang_dir, verbose=verbose
                            )
                            
                            if fasttext_embeddings:
                                pretrain_pt = _convert_fasttext_to_stanza_pretrain(
                                    fasttext_embeddings, model_lang_dir / "pretrain", language, verbose=verbose
                                )
                                if pretrain_pt:
                                    wordvec_pretrain_file = str(pretrain_pt)
                                    wordvec_path = Path(pretrain_pt)
                                    if verbose:
                                        print(f"[flexipipe] Using fastText embeddings: {pretrain_pt}")
                            
                            # If fastText failed, try XLM-RoBERTa
                            if not wordvec_pretrain_file:
                                pretrain_pt = _try_download_xlmroberta_embeddings(
                                    model_lang_dir / "pretrain", language, verbose=verbose
                                )
                                if pretrain_pt:
                                    wordvec_pretrain_file = str(pretrain_pt)
                                    wordvec_path = Path(pretrain_pt)
                                    if verbose:
                                        print(f"[flexipipe] Using XLM-RoBERTa embeddings: {pretrain_pt}")
                        
                        if verbose:
                            print(f"[flexipipe] Using Stanza training infrastructure for POS tagger...")
                        
                        # Create a temporary directory structure that Stanza expects
                        # Stanza looks for data in: data/pos/{lang}_{package}.train.in.conllu
                        import tempfile
                        with tempfile.TemporaryDirectory() as stanza_temp:
                            stanza_temp_path = Path(stanza_temp)
                            
                            # Create the directory structure Stanza expects
                            # Format: data/pos/{lang}_{package}.train.in.conllu
                            data_dir = stanza_temp_path / "data"
                            pos_data_dir = data_dir / "pos"
                            pos_data_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Determine the normalized language code (Stanza may normalize 3-letter to 2-letter)
                            lang_normalized_for_file = language
                            if len(language) == 3:
                                from ..language_utils import standardize_language_metadata
                                metadata = standardize_language_metadata(language, None)
                                iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                                if iso_1 and len(iso_1) == 2:
                                    lang_normalized_for_file = iso_1
                            
                            # Create training files with Stanza's expected naming convention
                            # Format: {lang_normalized}_{package}.train.in.conllu
                            train_stanza_file = pos_data_dir / f"{lang_normalized_for_file}_{package}.train.in.conllu"
                            dev_stanza_file = pos_data_dir / f"{lang_normalized_for_file}_{package}.dev.in.conllu"
                            test_stanza_file = pos_data_dir / f"{lang_normalized_for_file}_{package}.test.in.conllu"
                            
                            # Determine if depparse is disabled (no parsing sentences means depparse is disabled)
                            depparse_disabled = not parsing_sentences
                            
                            # Clean CoNLL-U files before copying (remove empty sentences, normalize blank lines, fix HEAD field)
                            def clean_conllu_file(input_path: Path, output_path: Path, depparse_disabled: bool = False) -> None:
                                """Clean a CoNLL-U file by removing empty sentences, normalizing blank lines, and fixing HEAD field.
                                
                                If depparse_disabled is True, skips dependency structure fixes (HEAD/DEPREL remain as '_').
                                Otherwise, ensures exactly one root per sentence to satisfy Stanza's evaluation requirements.
                                """
                                HEAD_COL = 6  # HEAD is column 7 (0-indexed: 6)
                                DEPREL_COL = 7  # DEPREL is column 8 (0-indexed: 7)
                                
                                with input_path.open("r", encoding="utf-8", errors="replace") as infile:
                                    with output_path.open("w", encoding="utf-8") as outfile:
                                        current_sentence_comments = []
                                        current_sentence_tokens = []  # List of (line, parts) tuples
                                        
                                        def process_sentence():
                                            """Process and write a complete sentence."""
                                            if not current_sentence_tokens:
                                                return
                                            
                                            if depparse_disabled:
                                                # If depparse is disabled, we still need to fix HEAD to be numeric
                                                # (Stanza's evaluation requires numeric HEAD, even for POS-only training)
                                                # But we can use a minimal structure: all tokens point to root (0)
                                                first_token_id = None
                                                for idx, (line, parts) in enumerate(current_sentence_tokens):
                                                    if len(parts) >= 8:
                                                        # Try to parse token ID
                                                        token_id_str = parts[0].strip()
                                                        try:
                                                            if '-' not in token_id_str and '.' not in token_id_str:
                                                                token_id = int(token_id_str)
                                                                if first_token_id is None:
                                                                    first_token_id = token_id
                                                        except (ValueError, TypeError):
                                                            if first_token_id is None:
                                                                first_token_id = 1
                                                
                                                if first_token_id is None:
                                                    first_token_id = 1
                                                
                                                # Fix HEAD and DEPREL for all tokens (minimal structure: all point to root)
                                                for idx, (line, parts) in enumerate(current_sentence_tokens):
                                                    # Ensure we have at least HEAD and DEPREL columns (need at least 8 columns)
                                                    # If not, pad with empty fields
                                                    while len(parts) < 8:
                                                        parts.append('_')
                                                    
                                                    # Set HEAD to '0' for first token, or to first_token_id for others
                                                    if idx == 0:
                                                        parts[HEAD_COL] = '0'
                                                        if len(parts) > DEPREL_COL and parts[DEPREL_COL] == '_':
                                                            parts[DEPREL_COL] = 'root'
                                                    else:
                                                        parts[HEAD_COL] = str(first_token_id)
                                                        if len(parts) > DEPREL_COL and parts[DEPREL_COL] == '_':
                                                            parts[DEPREL_COL] = 'dep'
                                                    
                                                    # Ensure we have all 10 columns for CoNLL-U format
                                                    while len(parts) < 10:
                                                        parts.append('_')
                                                    
                                                    # Update the line
                                                    current_sentence_tokens[idx] = ('\t'.join(parts), parts)
                                                
                                                # Write the sentence
                                                for comment in current_sentence_comments:
                                                    outfile.write(comment)
                                                    outfile.write("\n")
                                                
                                                for line, parts in current_sentence_tokens:
                                                    outfile.write(line)
                                                    outfile.write("\n")
                                                
                                                outfile.write("\n")
                                                return
                                            
                                            # If depparse is enabled, ensure exactly one root per sentence
                                            # First, normalize HEAD and DEPREL fields
                                            # Convert '_' to '0' for HEAD, and identify roots
                                            roots = []
                                            first_token_id = None
                                            
                                            for idx, (line, parts) in enumerate(current_sentence_tokens):
                                                if len(parts) >= 8:
                                                    # Convert '_' in HEAD to '0'
                                                    if parts[HEAD_COL] == '_':
                                                        parts[HEAD_COL] = '0'
                                                    
                                                    # Try to parse token ID (may be numeric or range like "1-2")
                                                    token_id_str = parts[0].strip()
                                                    try:
                                                        # If it's a simple number, use it
                                                        if '-' not in token_id_str and '.' not in token_id_str:
                                                            token_id = int(token_id_str)
                                                            if first_token_id is None:
                                                                first_token_id = token_id
                                                            # Check if this is a root
                                                            if parts[HEAD_COL] == '0':
                                                                roots.append((idx, token_id))
                                                    except (ValueError, TypeError):
                                                        # Not a simple numeric ID, use index as fallback
                                                        if first_token_id is None:
                                                            first_token_id = 1  # Default to 1
                                                        if parts[HEAD_COL] == '0':
                                                            roots.append((idx, first_token_id))
                                            
                                            # Ensure exactly one root
                                            if len(roots) == 0:
                                                # No root found - make the first token the root
                                                if current_sentence_tokens:
                                                    first_line, first_parts = current_sentence_tokens[0]
                                                    if len(first_parts) >= 8:
                                                        first_parts[HEAD_COL] = '0'
                                                        if first_parts[DEPREL_COL] == '_':
                                                            first_parts[DEPREL_COL] = 'root'
                                                        roots = [(0, first_token_id or 1)]
                                            elif len(roots) > 1:
                                                # Multiple roots - keep only the first one, make others point to it
                                                root_idx, root_token_id = roots[0]
                                                for idx, token_id in roots[1:]:
                                                    line, parts = current_sentence_tokens[idx]
                                                    if len(parts) >= 8:
                                                        parts[HEAD_COL] = str(root_token_id)
                                                        if parts[DEPREL_COL] == '_':
                                                            parts[DEPREL_COL] = 'dep'
                                            
                                            # Now fix all other tokens to point to the root if they don't have a valid HEAD
                                            root_idx, root_token_id = roots[0]
                                            for idx, (line, parts) in enumerate(current_sentence_tokens):
                                                if len(parts) >= 8:
                                                    # If HEAD is '_' or invalid, point to root
                                                    if parts[HEAD_COL] == '_' or parts[HEAD_COL].strip() == '':
                                                        if idx == root_idx:
                                                            parts[HEAD_COL] = '0'
                                                        else:
                                                            parts[HEAD_COL] = str(root_token_id)
                                                    
                                                    # Fix DEPREL if needed
                                                    if parts[DEPREL_COL] == '_':
                                                        if parts[HEAD_COL] == '0':
                                                            parts[DEPREL_COL] = 'root'
                                                        else:
                                                            parts[DEPREL_COL] = 'dep'
                                                    
                                                    # Update the line with fixed parts
                                                    current_sentence_tokens[idx] = ('\t'.join(parts), parts)
                                            
                                            # Write the sentence
                                            for comment in current_sentence_comments:
                                                outfile.write(comment)
                                                outfile.write("\n")
                                            
                                            for line, parts in current_sentence_tokens:
                                                outfile.write(line)
                                                outfile.write("\n")
                                            
                                            outfile.write("\n")
                                        
                                        for line in infile:
                                            line = line.rstrip()
                                            
                                            # Empty line = sentence boundary
                                            if not line:
                                                process_sentence()
                                                # Reset for next sentence
                                                current_sentence_comments = []
                                                current_sentence_tokens = []
                                                continue
                                            
                                            # Comment lines - keep them
                                            if line.startswith("#"):
                                                current_sentence_comments.append(line)
                                                continue
                                            
                                            # Regular token line - check if it's valid
                                            parts = line.split("\t")
                                            if len(parts) >= 2:
                                                # Valid token line (has at least ID and FORM)
                                                token_id = parts[0].strip()
                                                # Skip if token ID is empty or invalid
                                                if token_id and not token_id.startswith("#"):
                                                    current_sentence_tokens.append((line, parts))
                                                # If invalid, skip this line
                                            # else: Invalid line format - skip it
                                        
                                        # Write any remaining sentence at end of file
                                        process_sentence()
                            
                            # Clean and copy training file
                            if depparse_disabled and verbose:
                                print(f"[flexipipe] Note: Dependency parsing is disabled, using minimal dependency structure (all tokens point to root).")
                            clean_conllu_file(pos_train_file, train_stanza_file, depparse_disabled=depparse_disabled)
                            
                            # Clean and copy dev file (if it exists and is different from train)
                            if pos_dev_file.exists() and pos_dev_file != pos_train_file:
                                clean_conllu_file(pos_dev_file, dev_stanza_file, depparse_disabled=depparse_disabled)
                            else:
                                # If no separate dev file, use train file but warn
                                if verbose:
                                    print(f"[flexipipe] Warning: No separate dev file found, using training file for evaluation")
                                clean_conllu_file(pos_train_file, dev_stanza_file, depparse_disabled=depparse_disabled)
                            
                            # Stanza automatically tries to evaluate on test data after training
                            # If test data doesn't exist, create it from dev data to avoid FileNotFoundError
                            # test_conllu is created earlier and copied to training_data_dir
                            pos_test_file = training_data_dir / "test.conllu" if (training_data_dir / "test.conllu").exists() else None
                            
                            # Create test file: use test data if available, otherwise copy dev file
                            if pos_test_file and pos_test_file.exists() and pos_test_file != pos_train_file:
                                clean_conllu_file(pos_test_file, test_stanza_file, depparse_disabled=depparse_disabled)
                            else:
                                # No test data available - copy dev file to avoid FileNotFoundError
                                # Stanza will evaluate on dev data again, which is fine for training
                                if verbose:
                                    print(f"[flexipipe] Note: No test data found, using dev data for test evaluation")
                                clean_conllu_file(dev_stanza_file, test_stanza_file, depparse_disabled=depparse_disabled)
                            
                            # If we have a pretrain file, also copy it to where Stanza expects it
                            # This prevents Stanza from trying to auto-download during argument parsing
                            # Stanza looks for pretrains in: {lang}/pretrain/*.pt (any .pt file)
                            pretrain_dir = stanza_temp_path / lang_normalized_for_file / "pretrain"
                            pretrain_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Always ensure pretrain file is in temp directory if we have one
                            # This must be done BEFORE running the command, as Stanza checks during argument parsing
                            if wordvec_pretrain_file and wordvec_path.exists():
                                # Copy the pretrain file - Stanza will look for any .pt file in this directory
                                # Use multiple names to ensure Stanza finds it
                                pretrain_dest = pretrain_dir / f"{lang_normalized_for_file}.pretrain.pt"
                                pretrain_dest_generic = pretrain_dir / "pretrain.pt"
                                
                                # Copy with normalized language code
                                shutil.copy(wordvec_path, pretrain_dest)
                                
                                # Also copy with a generic name
                                shutil.copy(wordvec_path, pretrain_dest_generic)
                                
                                # Also try with original language code if different
                                if lang_normalized_for_file != language:
                                    pretrain_dest_orig = pretrain_dir / f"{language}.pretrain.pt"
                                    shutil.copy(wordvec_path, pretrain_dest_orig)
                                
                                # Verify ALL files were copied successfully - this is critical
                                copied_files = []
                                for dest_file in [pretrain_dest, pretrain_dest_generic]:
                                    if dest_file.exists():
                                        copied_files.append(dest_file.name)
                                    else:
                                        if verbose:
                                            print(f"[flexipipe] ERROR: Failed to copy pretrain file to {dest_file}")
                                
                                if lang_normalized_for_file != language:
                                    orig_dest = pretrain_dir / f"{language}.pretrain.pt"
                                    if orig_dest.exists():
                                        copied_files.append(orig_dest.name)
                                
                                if not copied_files:
                                    raise RuntimeError(
                                        f"Failed to copy pretrain file to temp directory. "
                                        f"Source: {wordvec_path}, Dest dir: {pretrain_dir}"
                                    )
                                
                                if verbose:
                                    print(f"[flexipipe] Copied pretrain file to temp directory:")
                                    for fname in copied_files:
                                        print(f"[flexipipe]   - {pretrain_dir / fname}")
                                    # List all .pt files in the pretrain directory for debugging
                                    pretrain_files = list(pretrain_dir.glob("*.pt"))
                                    if pretrain_files:
                                        print(f"[flexipipe] All pretrain files in temp directory: {[str(f.name) for f in pretrain_files]}")
                                
                                # Final verification: ensure at least one .pt file exists before proceeding
                                final_check = list(pretrain_dir.glob("*.pt"))
                                if not final_check:
                                    raise RuntimeError(
                                        f"No pretrain files found in {pretrain_dir} after copying. "
                                        f"This will cause Stanza to fail during argument parsing."
                                    )
                            else:
                                # No pretrain file provided - this will cause Stanza to try auto-download
                                # which will fail for unknown languages
                                if verbose:
                                    print(f"[flexipipe] Warning: No pretrain file provided. Stanza will attempt auto-detection.")
                                    print(f"[flexipipe] This may fail for unknown languages. Consider providing --wordvec-pretrain-file.")
                            
                            # Stanza training uses treebank shorthand (e.g., "sn_sna" for normalized lang + package)
                            shorthand = f"{lang_normalized_for_file}_{package}"
                            
                            # Use Stanza's training via subprocess
                            # Set STANZA_RESOURCES_DIR to our temp directory so Stanza finds the treebank
                            env = os.environ.copy()
                            # Stanza also looks in ~/stanza_resources by default
                            # We'll set it to our temp directory temporarily
                            original_stanza_resources = env.get("STANZA_RESOURCES_DIR")
                            env["STANZA_RESOURCES_DIR"] = str(stanza_temp_path)
                            
                            train_cmd = [
                                sys.executable, "-m", "stanza.utils.training.run_pos",
                                shorthand,
                                "--train",
                                "--save_dir", str(pos_output_dir),
                                "--save_name", package,
                                "--no_charlm",  # Don't use character-level models
                            ]
                            
                            # Add training duration and early stopping arguments
                            # Default: 50000 max steps (Stanza's default), but allow override
                            max_steps = kwargs.get("max_steps", 50000)
                            if max_steps:
                                train_cmd.extend(["--max_steps", str(max_steps)])
                            
                            # Early stopping: max_steps_before_stop (Stanza's argument name)
                            # This is the number of evaluation steps without improvement before stopping
                            # Default: 3 evaluations (if patience is provided, convert to steps)
                            # Note: Stanza uses --max_steps_before_stop, not --patience
                            patience = kwargs.get("patience", 3)
                            if patience is not None and patience > 0:
                                # Convert patience (number of evaluations) to steps
                                # If eval_interval is 100, patience=3 means 300 steps
                                eval_interval_for_steps = kwargs.get("eval_interval", 100)
                                max_steps_before_stop = patience * eval_interval_for_steps
                                train_cmd.extend(["--max_steps_before_stop", str(max_steps_before_stop)])
                            
                            # Evaluation interval (default: 100 steps, as seen in logs)
                            eval_interval = kwargs.get("eval_interval", 100)
                            if eval_interval:
                                train_cmd.extend(["--eval_interval", str(eval_interval)])
                            
                            # Add --wordvec_pretrain_file if provided (bypasses automatic download)
                            # wordvec_pretrain_file and wordvec_path are already checked and resolved above
                            if wordvec_pretrain_file:
                                train_cmd.extend(["--wordvec_pretrain_file", str(wordvec_path)])
                            
                            # CRITICAL: Verify pretrain file exists in temp directory BEFORE running command
                            # Stanza checks for pretrains during argument parsing, so it must be there
                            if wordvec_pretrain_file:
                                pretrain_files_final = list(pretrain_dir.glob("*.pt"))
                                if not pretrain_files_final:
                                    # This is a critical error - Stanza will fail during argument parsing
                                    raise RuntimeError(
                                        f"Pretrain file not found in temp directory before running Stanza command. "
                                        f"Expected location: {pretrain_dir}/*.pt\n"
                                        f"Source file: {wordvec_path}\n"
                                        f"This will cause Stanza to fail during argument parsing. "
                                        f"Please check file permissions and disk space."
                                    )
                                if verbose:
                                    print(f"[flexipipe] Verified pretrain files exist in temp directory: {[f.name for f in pretrain_files_final]}")
                            
                            if verbose:
                                print(f"[flexipipe] Training POS tagger with {len(tagging_sentences)} sentences...")
                                print(f"[flexipipe] Running: {' '.join(train_cmd)}")
                            
                            # Always capture output so we can parse errors, but print it if verbose
                            result = subprocess.run(
                                train_cmd,
                                capture_output=True,  # Always capture to check for errors
                                text=True,
                                check=False,
                                env=env,
                                cwd=str(stanza_temp_path),
                            )
                            
                            # Filter out non-fatal pretrain warnings/tracebacks when we're providing --wordvec_pretrain_file
                            # These occur during argument parsing but don't affect training since we provide the file explicitly
                            # The error happens in find_wordvec_pretrain -> download() which tries to download resources
                            # even when --wordvec_pretrain_file is provided, because it's called during build_model_filename
                            is_pretrain_auto_detect_error = False
                            if wordvec_pretrain_file and result.stderr:
                                # Check if the error is about pretrain auto-detection (non-fatal when we provide --wordvec_pretrain_file)
                                stderr_lower = result.stderr.lower()
                                error_combined = (result.stderr + "\n" + (result.stdout or "")).lower()
                                if (
                                    "cannot find any pretrains" in stderr_lower or
                                    "no pretrains in the system" in stderr_lower or
                                    ("cannot figure out which pretrain to use" in stderr_lower and "unknownlanguageerror" in stderr_lower) or
                                    ("unknownlanguageerror" in error_combined and "unknown language requested" in error_combined and "find_wordvec_pretrain" in error_combined) or
                                    ("file not found" in error_combined and "pretrain" in error_combined and "find_wordvec_pretrain" in error_combined)
                                ):
                                    is_pretrain_auto_detect_error = True
                                    if verbose:
                                        print(f"[flexipipe] Note: Ignoring expected pretrain auto-detection error (non-fatal when using --wordvec-pretrain-file)")
                                        print(f"[flexipipe] Stanza's find_wordvec_pretrain tries to auto-detect pretrains during argument parsing (in build_model_filename),")
                                        print(f"[flexipipe] but training will proceed with the provided file: {wordvec_path}")
                                    
                                    # If this is the only error and return code is non-zero, check if training actually proceeded
                                    # Sometimes Stanza continues despite this error if --wordvec_pretrain_file is provided
                                    if result.returncode != 0:
                                        # Check if there are other errors besides the pretrain auto-detect
                                        # Look for actual training errors (not just pretrain detection)
                                        has_other_errors = False
                                        if result.stdout:
                                            stdout_lower = result.stdout.lower()
                                            # Check for actual training errors
                                            if any(phrase in stdout_lower for phrase in [
                                                "train file not found",
                                                "error",
                                                "failed",
                                                "exception",
                                                "traceback"
                                            ]):
                                                # But exclude pretrain-related errors
                                                if not all(phrase in stdout_lower for phrase in ["pretrain", "unknownlanguageerror"]):
                                                    has_other_errors = True
                                        
                                        if not has_other_errors:
                                            # Only pretrain auto-detect error - this is non-fatal
                                            # Training should proceed, but let's check if it actually did
                                            if verbose:
                                                print(f"[flexipipe] Only pretrain auto-detection error detected. Checking if training actually proceeded...")
                            
                            # Print output if verbose (skip stderr if it's just the expected pretrain auto-detect error)
                            if verbose:
                                if result.stdout:
                                    print(result.stdout)
                                if result.stderr and not is_pretrain_auto_detect_error:
                                    print(result.stderr, file=sys.stderr)
                            
                            # Restore original STANZA_RESOURCES_DIR
                            if original_stanza_resources:
                                env["STANZA_RESOURCES_DIR"] = original_stanza_resources
                            elif "STANZA_RESOURCES_DIR" in env:
                                del env["STANZA_RESOURCES_DIR"]
                            
                            # Check if training was actually skipped (file not found)
                            error_output_for_check = ""
                            if result.stderr:
                                error_output_for_check += result.stderr
                            if result.stdout:
                                error_output_for_check += "\n" + result.stdout
                            
                            is_skipped = (
                                "TRAIN FILE NOT FOUND" in error_output_for_check or
                                "... skipping" in error_output_for_check or
                                "skipping" in error_output_for_check.lower()
                            )
                            
                            if result.returncode == 0 and not is_skipped:
                                # Verify that the model file was actually created
                                pos_model_file = pos_output_dir / f"{package}.pt"
                                if pos_model_file.exists():
                                    components_trained.append("pos")
                                    training_successful = True
                                    if verbose:
                                        print(f"[flexipipe] POS tagger training completed successfully")
                                else:
                                    if verbose:
                                        print(f"[flexipipe] Warning: Training reported success but model file not found: {pos_model_file}")
                            elif is_skipped:
                                if verbose:
                                    print(f"[flexipipe] Error: Training was skipped - training file not found")
                                    print(f"[flexipipe] Expected file: {train_stanza_file}")
                                    print(f"[flexipipe] This indicates a problem with the data directory structure")
                            else:
                                # Extract error message from stderr or stdout
                                # Combine both stderr and stdout to catch all error information
                                error_output = ""
                                if result.stderr:
                                    error_output += result.stderr
                                if result.stdout:
                                    error_output += "\n" + result.stdout
                                if not error_output:
                                    error_output = "Unknown error"
                                
                                error_msg = error_output[:5000] if len(error_output) > 5000 else error_output  # Truncate for display
                                
                                # Extract normalized language code from error message if present
                                # Stanza may normalize language codes (e.g., "sna" -> "sn")
                                language_normalized = None
                                import re
                                # Look for pattern: "Unknown language requested: <code>"
                                match = re.search(r'Unknown language requested:\s*(\w+)', error_output)
                                if match:
                                    detected_code = match.group(1)
                                    if detected_code != language:
                                        language_normalized = detected_code
                                        if verbose:
                                            print(f"[flexipipe] Detected language normalization: '{language}' -> '{language_normalized}'")
                                
                                # Also try to get 2-letter code from language metadata
                                if not language_normalized and len(language) == 3:
                                    from ..language_utils import standardize_language_metadata
                                    metadata = standardize_language_metadata(language, None)
                                    iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                                    if iso_1 and len(iso_1) == 2 and iso_1 != language:
                                        language_normalized = iso_1
                                
                                # Check error types - prioritize pretrain errors since they're more specific
                                # Check for pretrain errors first (these are fatal and shouldn't trigger retry)
                                is_pretrain_error = (
                                    "Cannot find any pretrains" in error_output or 
                                    "No pretrains" in error_output or
                                    "wordvec_pretrain_file" in error_output
                                )
                                
                                # Check if it's a language recognition error
                                # Note: Stanza may convert UnknownLanguageError to FileNotFoundError
                                # So we check for both the error type and the error message pattern in the full output
                                is_language_error = (
                                    not is_pretrain_error and (  # Only check if not a pretrain error
                                        "UnknownLanguageError" in error_output or 
                                        "Unknown language" in error_output or
                                        (language_normalized and f"Unknown language requested: {language_normalized}" in error_output) or
                                        f"Unknown language requested: {language}" in error_output
                                    )
                                )
                                
                                if is_pretrain_error:
                                    # Pretrain embeddings are required but missing - try multiple fallback options
                                    if verbose:
                                        normalized_msg = f" (normalized to '{language_normalized}')" if language_normalized else ""
                                        print(f"[flexipipe] Error: No pretrain embeddings found for language '{language}'{normalized_msg}.")
                                        print(f"[flexipipe] Stanza training requires pretrain word embeddings (.pt files).")
                                        print(f"[flexipipe]")
                                    
                                    pretrain_pt = None
                                    
                                    # Fallback 1: Try fastText embeddings
                                    if verbose:
                                        print(f"[flexipipe] Attempting fallback 1: Downloading fastText embeddings...")
                                    fasttext_embeddings = _try_download_fasttext_embeddings(
                                        language, language_normalized, model_lang_dir, verbose=verbose
                                    )
                                    
                                    if fasttext_embeddings:
                                        # Try to convert fastText embeddings to Stanza format
                                        pretrain_pt = _convert_fasttext_to_stanza_pretrain(
                                            fasttext_embeddings, model_lang_dir / "pretrain", language, verbose=verbose
                                        )
                                    
                                    # Fallback 2: Try XLM-RoBERTa embeddings if fastText failed
                                    if not pretrain_pt:
                                        if verbose:
                                            print(f"[flexipipe] Attempting fallback 2: Extracting XLM-RoBERTa embeddings...")
                                        pretrain_pt = _try_download_xlmroberta_embeddings(
                                            model_lang_dir / "pretrain", language, verbose=verbose
                                        )
                                    
                                    if pretrain_pt:
                                        # Retry training with the extracted embeddings
                                        if verbose:
                                            print(f"[flexipipe] Using extracted embeddings: {pretrain_pt}")
                                        
                                        # Update train_cmd to use the pretrain file
                                        # Remove existing --wordvec_pretrain_file if present
                                        new_train_cmd = []
                                        skip_next = False
                                        for i, arg in enumerate(train_cmd):
                                            if skip_next:
                                                skip_next = False
                                                continue
                                            if arg == "--wordvec_pretrain_file":
                                                skip_next = True
                                                continue
                                            new_train_cmd.append(arg)
                                        
                                        new_train_cmd.extend(["--wordvec_pretrain_file", str(pretrain_pt)])
                                        
                                        if verbose:
                                            print(f"[flexipipe] Retrying training with extracted embeddings...")
                                            print(f"[flexipipe] Running: {' '.join(new_train_cmd)}")
                                        
                                        result = subprocess.run(
                                            new_train_cmd,
                                            capture_output=True,
                                            text=True,
                                            check=False,
                                            env=env,
                                            cwd=str(stanza_temp_path),
                                        )
                                        
                                        if verbose:
                                            if result.stdout:
                                                print(result.stdout)
                                            if result.stderr:
                                                print(result.stderr, file=sys.stderr)
                                        
                                        if result.returncode == 0:
                                            components_trained.append("pos")
                                            training_successful = True
                                            if verbose:
                                                print(f"[flexipipe] POS tagger training completed successfully using extracted embeddings")
                                        else:
                                            if verbose:
                                                print(f"[flexipipe] Warning: Training with extracted embeddings still failed.")
                                                print(f"[flexipipe] Error: {result.stderr[:500] if result.stderr else 'Unknown error'}")
                                    else:
                                        if verbose:
                                            print(f"[flexipipe] Warning: Could not obtain embeddings from fastText or XLM-RoBERTa.")
                                    
                                    # If we still don't have pretrain embeddings, show manual options
                                    if not training_successful:
                                        if verbose:
                                            print(f"[flexipipe]")
                                            print(f"[flexipipe] Manual options to proceed:")
                                            print(f"[flexipipe]   1. Provide pretrain embeddings manually:")
                                            print(f"[flexipipe]      - Download fastText embeddings: wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{language}.vec")
                                            print(f"[flexipipe]      - Convert them to Stanza's .pt format (see Stanza documentation)")
                                            print(f"[flexipipe]      - Then run training with: --wordvec-pretrain-file /path/to/embeddings.pt")
                                            print(f"[flexipipe]")
                                            print(f"[flexipipe]   2. Use a different backend that doesn't require pretrains:")
                                            print(f"[flexipipe]      - fastText: flexipipe train --backend fasttext --language {language} ...")
                                            print(f"[flexipipe]      - SpaCy: flexipipe train --backend spacy --language {language} ...")
                                            print(f"[flexipipe]      - flexitag: flexipipe train --backend flexitag --language {language} ...")
                                            print(f"[flexipipe]")
                                            print(f"[flexipipe] Training data has been prepared at: {training_data_dir}")
                                            print(f"[flexipipe] You can use the prepared data with Stanza's training scripts once you have pretrain embeddings.")
                                elif is_language_error:
                                    # Determine which language code(s) to add
                                    # Stanza may normalize 3-letter codes to 2-letter codes
                                    codes_to_add = [language]
                                    if language_normalized and language_normalized not in codes_to_add:
                                        codes_to_add.append(language_normalized)
                                    
                                    if verbose:
                                        if language_normalized:
                                            print(f"[flexipipe] Language code '{language}' is normalized to '{language_normalized}' by Stanza.")
                                            print(f"[flexipipe] Attempting to add both codes to Stanza's constant.py automatically...")
                                        else:
                                            print(f"[flexipipe] Language code '{language}' is not recognized by Stanza.")
                                            print(f"[flexipipe] Attempting to add it to Stanza's constant.py automatically...")
                                    
                                    # Try to automatically add the language code(s) to Stanza's constant.py
                                    # Get language name from kwargs or try to look it up
                                    language_name = kwargs.get("language_name")
                                    if not language_name:
                                        from ..language_utils import standardize_language_metadata
                                        metadata = standardize_language_metadata(language, None)
                                        language_name = metadata.get(LANGUAGE_FIELD_NAME) or (language.upper() if len(language) <= 2 else language.capitalize())
                                    
                                    # Add all necessary language codes
                                    all_added = True
                                    for code_to_add in codes_to_add:
                                        added = _add_language_to_stanza_constant(code_to_add, language_name, verbose=verbose)
                                        if not added:
                                            all_added = False
                                    added = all_added
                                    
                                    if added:
                                        if verbose:
                                            print(f"[flexipipe] Retrying training with updated language code...")
                                        
                                        # Retry training after adding the language code
                                        result = subprocess.run(
                                            train_cmd,
                                            capture_output=True,  # Always capture to check for errors
                                            text=True,
                                            check=False,
                                            env=env,
                                            cwd=str(stanza_temp_path),
                                        )
                                        
                                        # Print output if verbose
                                        if verbose:
                                            if result.stdout:
                                                print(result.stdout)
                                            if result.stderr:
                                                print(result.stderr, file=sys.stderr)
                                        
                                        # Restore original STANZA_RESOURCES_DIR
                                        if original_stanza_resources:
                                            env["STANZA_RESOURCES_DIR"] = original_stanza_resources
                                        elif "STANZA_RESOURCES_DIR" in env:
                                            del env["STANZA_RESOURCES_DIR"]
                                        
                                        if result.returncode == 0:
                                            components_trained.append("pos")
                                            training_successful = True
                                            if verbose:
                                                print(f"[flexipipe] POS tagger training completed successfully after adding language code")
                                        else:
                                            # Still failed after adding language code
                                            # Combine both stderr and stdout
                                            error_output = ""
                                            if result.stderr:
                                                error_output += result.stderr
                                            if result.stdout:
                                                error_output += "\n" + result.stdout
                                            if not error_output:
                                                error_output = "Unknown error"
                                            error_msg = error_output[:1000] if len(error_output) > 1000 else error_output
                                            if verbose:
                                                print(f"[flexipipe] Warning: Training still failed after adding language code.")
                                                print(f"[flexipipe] Error: {error_msg[:500]}")
                                                print(f"[flexipipe] This may be due to missing pretrain embeddings.")
                                                print(f"[flexipipe] Training data has been prepared at: {training_data_dir}")
                                    else:
                                        # Could not add language code automatically
                                        if verbose:
                                            import stanza
                                            stanza_path = Path(stanza.__file__).parent
                                            print(f"[flexipipe] Warning: Could not automatically add language code to Stanza.")
                                            print(f"[flexipipe] To add support for this language manually:")
                                            print(f"[flexipipe]   1. Edit: {stanza_path / 'models' / 'common' / 'constant.py'}")
                                            print(f"[flexipipe]   2. Add: (\"{language}\", \"{language_name}\"), to the lcode2lang_raw list")
                                            print(f"[flexipipe]   3. Retry training")
                                            print(f"[flexipipe] Alternatively, you can:")
                                            print(f"[flexipipe]   - Prepare pretrain embeddings (.pt file) and use --wordvec_pretrain_file")
                                            print(f"[flexipipe]   - Use CLASSLA's training tools directly with the prepared data")
                                            print(f"[flexipipe]   - Train using a different backend (e.g., fastText, SpaCy) that supports this language")
                                            print(f"[flexipipe] Training data has been prepared at: {training_data_dir}")
                                else:
                                    if verbose:
                                        print(f"[flexipipe] Warning: POS training returned exit code {result.returncode}")
                                        if error_msg and error_msg != "Unknown error":
                                            # Show more of the error if available
                                            print(f"[flexipipe] Error: {error_msg}")
                                    # Don't fail completely - training data is prepared
                                    if verbose:
                                        print(f"[flexipipe] Training data prepared. To complete training manually, use:")
                                        print(f"[flexipipe]   python -m stanza.utils.training.run_pos {shorthand} --train --save_dir {pos_output_dir} --save_name {package}")
                                    
                    except ImportError as import_err:
                        if verbose:
                            print(f"[flexipipe] Warning: Stanza not available: {import_err}")
                        raise RuntimeError("Stanza training requires Stanza to be installed. Install with: pip install stanza") from import_err
                    except subprocess.CalledProcessError as proc_err:
                        # This shouldn't happen since we use check=False, but handle it just in case
                        if verbose:
                            print(f"[flexipipe] Warning: Training subprocess failed: {proc_err}")
                            if proc_err.stderr:
                                print(f"[flexipipe] Error output: {proc_err.stderr[:500]}")
                        if verbose:
                            print(f"[flexipipe] Training data prepared at: {training_data_dir}")
                    except Exception as e:
                        if verbose:
                            print(f"[flexipipe] Warning: Could not train POS tagger: {e}")
                            import traceback
                            if verbose:
                                traceback.print_exc()
                        # Don't fail completely - at least data is prepared
                        if verbose:
                            print(f"[flexipipe] Training data prepared at: {training_data_dir}")
                
                # Train dependency parser if we have parsing data
                if parsing_sentences and train_parsing_conllu.exists():
                    if verbose:
                        print(f"[flexipipe] Training dependency parser...")
                    # Similar approach for parser - would use run_depparse
                    # For now, we'll prepare the data
                    if verbose:
                        print(f"[flexipipe] Parser training data prepared at: {training_data_dir / 'train_parsing.conllu'}")
                
                # Train lemmatizer if we have lemma data
                if lemma_sentences and train_lemma_conllu.exists():
                    if verbose:
                        print(f"[flexipipe] Training lemmatizer...")
                    # Similar approach for lemma - would use run_lemma
                    # For now, we'll prepare the data
                    if verbose:
                        print(f"[flexipipe] Lemma training data prepared at: {training_data_dir / 'train_lemma.conllu'}")
                
                # Create processor configuration file after training
                # This allows ClassLA to properly load the trained models
                processors_config = {}
                
                # Add tokenizer if available (ClassLA may have a default tokenizer)
                # For now, we'll set it to None and let ClassLA use its default
                processors_config["tokenize"] = None
                processors_config["mwt"] = None  # Multi-word token expansion
                
                # Add POS tagger if trained
                if "pos" in components_trained:
                    pos_model_file = pos_output_dir / f"{package}.pt"
                    if pos_model_file.exists():
                        processors_config["pos"] = {
                            "type": "classla.models.tagger",
                            "model_path": f"pos/{package}.pt",
                        }
                        # Add pretrain path if available
                        pretrain_dir = model_lang_dir / "pretrain"
                        if pretrain_dir.exists():
                            # Look for pretrain files
                            pretrain_files = list(pretrain_dir.glob("*.pt"))
                            if pretrain_files:
                                pretrain_file = pretrain_files[0]
                                processors_config["pos"]["pretrain_path"] = f"pretrain/{pretrain_file.name}"
                
                # Add lemmatizer if trained
                if "lemma" in components_trained:
                    lemma_output_dir = model_lang_dir / "lemma"
                    lemma_model_file = lemma_output_dir / f"{package}.pt"
                    if lemma_model_file.exists():
                        processors_config["lemma"] = {
                            "type": "classla.models.lemmatizer",
                            "model_path": f"lemma/{package}.pt",
                        }
                
                # Add dependency parser if trained
                if "depparse" in components_trained:
                    depparse_output_dir = model_lang_dir / "depparse"
                    depparse_model_file = depparse_output_dir / f"{package}.pt"
                    if depparse_model_file.exists():
                        processors_config["depparse"] = {
                            "type": "classla.models.parser",
                            "model_path": f"depparse/{package}.pt",
                        }
                        # Add pretrain path if available
                        pretrain_dir = model_lang_dir / "pretrain"
                        if pretrain_dir.exists():
                            pretrain_files = list(pretrain_dir.glob("*.pt"))
                            if pretrain_files:
                                pretrain_file = pretrain_files[0]
                                processors_config["depparse"]["pretrain_path"] = f"pretrain/{pretrain_file.name}"
                
                # NER is typically not trained via Stanza, so set to None
                processors_config["ner"] = None
                
                # Save processor configuration
                processors_config_path = model_lang_dir / "processors.json"
                with processors_config_path.open("w", encoding="utf-8") as f:
                    json.dump(processors_config, f, indent=2, ensure_ascii=False)
                
                # Update resources.json with processor configuration
                resources_entry["processors"] = processors_config
                resources_entry["default_processors"] = {
                    "tokenize": "tokenize" if processors_config.get("tokenize") else None,
                    "pos": "pos" if processors_config.get("pos") else None,
                    "lemma": "lemma" if processors_config.get("lemma") else None,
                    "depparse": "depparse" if processors_config.get("depparse") else None,
                }
                
                # Write updated resources.json in model directory
                with resources_path.open("w", encoding="utf-8") as f:
                    json.dump(resources_entry, f, indent=2, ensure_ascii=False)
                
                # Also update the main resources.json at the root of ClassLA models directory
                # This is what ClassLA/Stanza uses to discover models
                if training_successful:
                    try:
                        from ..model_storage import get_backend_models_dir
                        from ..language_utils import standardize_language_metadata
                        
                        classla_root = get_backend_models_dir("classla", create=False)
                        main_resources_path = classla_root / "resources.json"
                        
                        # Read existing resources.json or create new structure
                        main_resources = {}
                        if main_resources_path.exists():
                            try:
                                with main_resources_path.open("r", encoding="utf-8") as f:
                                    main_resources = json.load(f)
                            except Exception:
                                # If reading fails, start fresh
                                main_resources = {}
                        
                        # Ensure main_resources is a dict keyed by language
                        if not isinstance(main_resources, dict):
                            main_resources = {}
                        
                        # Get normalized language code (ClassLA may normalize 3-letter codes to 2-letter)
                        lang_normalized = language
                        if len(language) == 3:
                            metadata = standardize_language_metadata(language, None)
                            iso_1 = metadata.get(LANGUAGE_FIELD_ISO)
                            if iso_1 and len(iso_1) == 2:
                                lang_normalized = iso_1
                        
                        # Register model using ClassLA's structure: language -> processor -> package/type
                        # Files are stored as: lang/processor/package.pt (e.g., sn/pos/sna-masakhane.pt)
                        # Resources.json structure: "sn": { "pos": { "sna-masakhane": { ... } } }
                        
                        # First, ensure the normalized language code entry exists
                        if lang_normalized not in main_resources:
                            main_resources[lang_normalized] = {}
                        
                        if not isinstance(main_resources[lang_normalized], dict):
                            main_resources[lang_normalized] = {}
                        
                        # Register each processor that was trained
                        # Structure: language -> processor -> package/type -> config
                        # Files: lang/processor/package.pt (e.g., sn/pos/sna-masakhane.pt)
                        for processor_name, processor_config in processors_config.items():
                            # Skip special keys
                            if processor_name in ("default_processors", "mwt"):
                                continue
                            
                            # Only register processors that are actually configured (not None)
                            if processor_config is None:
                                continue
                            
                            # Initialize processor entry
                            if processor_name not in main_resources[lang_normalized]:
                                main_resources[lang_normalized][processor_name] = {}
                            
                            if not isinstance(main_resources[lang_normalized][processor_name], dict):
                                main_resources[lang_normalized][processor_name] = {}
                            
                            # Use package as the key (this is the .pt filename without extension)
                            # This allows multiple models per language (e.g., "sna-masakhane", "standard", "nonstandard")
                            package_key = package  # e.g., "sna-masakhane" or "standard"
                            
                            # Store processor-specific configuration
                            # The structure matches ClassLA's format where each processor has its own entry
                            processor_entry = {}
                            
                            # Add processor-specific metadata from processor_config if it's a dict
                            if isinstance(processor_config, dict):
                                # Copy relevant fields (pretrain_path, dependencies, etc.)
                                for key, value in processor_config.items():
                                    if key not in ("pretrain_path", "dependencies", "library", "md5"):
                                        continue
                                    processor_entry[key] = value
                            
                            # Add standard metadata
                            processor_entry["lang"] = lang_normalized
                            processor_entry["package"] = package
                            processor_entry["type"] = model_type
                            
                            main_resources[lang_normalized][processor_name][package_key] = processor_entry
                            
                            if verbose:
                                print(f"[flexipipe] Registered {processor_name} processor: {lang_normalized}/{processor_name}/{package_key}")
                        
                        # Now add aliases to support multiple models per language
                        # 1. Alias from original language code to normalized code (if different)
                        if language != lang_normalized:
                            if language not in main_resources:
                                main_resources[language] = {}
                            
                            # Check if this is already an alias entry
                            if "alias" in main_resources[language]:
                                # Already an alias - make sure it points to the right place
                                if main_resources[language]["alias"] != lang_normalized:
                                    if verbose:
                                        print(f"[flexipipe] Warning: Language '{language}' already has alias '{main_resources[language]['alias']}', updating to '{lang_normalized}'")
                                    main_resources[language]["alias"] = lang_normalized
                            else:
                                # Create alias entry
                                main_resources[language] = {
                                    "alias": lang_normalized
                                }
                                if verbose:
                                    print(f"[flexipipe] Created alias '{language}' -> '{lang_normalized}' in resources.json")
                        
                        # 2. Alias from model name to language/package combination (if model name is different from package)
                        # This allows using the model name directly (e.g., "sna-masakhane") to load the model
                        # Note: ClassLA's alias system only supports language codes, so we create an alias
                        # that points to the normalized language code, and then specify the package when loading
                        if model_name and model_name != package:
                            # Store the package mapping in a special structure
                            # We'll use the model name as a language code alias that points to the normalized code
                            # The actual package will be specified when loading
                            if model_name not in main_resources:
                                main_resources[model_name] = {
                                    "alias": lang_normalized,
                                    # Store package info in a comment-like structure (ClassLA may ignore this)
                                    "_package": package,
                                    "_type": model_type,
                                }
                                if verbose:
                                    print(f"[flexipipe] Created model name alias '{model_name}' -> '{lang_normalized}' (package: {package}, type: {model_type})")
                            elif "alias" in main_resources[model_name]:
                                # Update existing alias
                                main_resources[model_name]["alias"] = lang_normalized
                                main_resources[model_name]["_package"] = package
                                main_resources[model_name]["_type"] = model_type
                                if verbose:
                                    print(f"[flexipipe] Updated model name alias '{model_name}' -> '{lang_normalized}' (package: {package}, type: {model_type})")
                        
                        # Write updated main resources.json
                        with main_resources_path.open("w", encoding="utf-8") as f:
                            json.dump(main_resources, f, indent=2, ensure_ascii=False)
                        
                        if verbose:
                            processors_registered = [p for p in processors_config.keys() if p not in ("default_processors", "mwt") and processors_config.get(p) is not None]
                            if lang_normalized != language:
                                print(f"[flexipipe] Registered model in main resources.json: '{lang_normalized}' -> processors: {', '.join(processors_registered)} -> package: '{package}' (type: {model_type})")
                                print(f"[flexipipe] Created alias '{language}' -> '{lang_normalized}'")
                            else:
                                print(f"[flexipipe] Registered model in main resources.json: '{lang_normalized}' -> processors: {', '.join(processors_registered)} -> package: '{package}' (type: {model_type})")
                    except Exception as e:
                        if verbose:
                            print(f"[flexipipe] Warning: Could not update main resources.json: {e}")
                        # Don't fail training if this fails
                
                if training_successful:
                    if verbose:
                        if components_trained:
                            print(f"[flexipipe] CLASSLA training completed. Components trained: {', '.join(components_trained)}")
                        else:
                            print(f"[flexipipe] CLASSLA training data preparation completed.")
                        print(f"[flexipipe] Model directory structure created at: {output_dir}")
                        print(f"[flexipipe] Processor configuration saved to: {processors_config_path}")
                        print(f"[flexipipe]")
                        # Language has been automatically added to Stanza's constant.py during training
                        # No additional action needed
                    
                    # Invalidate caches so the new model appears immediately
                    try:
                        from ..model_catalog import invalidate_unified_catalog_cache
                        invalidate_unified_catalog_cache()
                        # Also invalidate ClassLA's local cache
                        from ..model_storage import get_cache_dir
                        cache_dir = get_cache_dir()
                        classla_cache = cache_dir / "classla.json"
                        if classla_cache.exists():
                            classla_cache.unlink()
                        # Cache will be rebuilt on next access (with --refresh-cache or when cache expires)
                    except Exception:
                        pass  # Best effort - don't fail training if cache invalidation fails
                else:
                    if verbose:
                        print(f"[flexipipe] Training data prepared at {output_dir}")
                        print(f"[flexipipe] Note: Training attempted but may require additional setup.")
                        
            except Exception as e:
                if verbose:
                    print(f"[flexipipe] Warning: Could not complete CLASSLA training: {e}")
                    import traceback
                    if verbose:
                        traceback.print_exc()
                    print(f"[flexipipe] Training data prepared at {output_dir}")
                    print(f"[flexipipe] Use CLASSLA's training tools with the prepared data files.")
        
        return output_dir
    
    def _collect_conllu_files(
        self,
        data: Union[Document, List[Document], Path],
        *,
        split: Optional[str] = None,
    ) -> List[Path]:
        """Collect CoNLL-U files from various input types."""
        from ..conllu import conllu_to_document, document_to_conllu
        import tempfile
        
        if isinstance(data, Path):
            path = data
            if path.is_file():
                if path.suffix.lower() not in (".conllu", ".conll"):
                    raise ValueError(f"Unsupported training data format: {path.suffix}")
                return [path]
            if path.is_dir():
                files: List[Path] = []
                if split:
                    patterns = [
                        f"*ud-{split}.conllu",
                        f"*_{split}.conllu",
                        f"*{split}.conllu",
                    ]
                    for pattern in patterns:
                        matched = sorted(path.glob(pattern))
                        if matched:
                            files = matched
                            break
                if not files:
                    files = sorted(path.glob("*.conllu")) + sorted(path.glob("*.conll"))
                if not files:
                    raise ValueError(f"No .conllu files found in directory {path}")
                return files
            raise ValueError(f"Training data path does not exist: {path}")
        
        # For Document objects, write to temporary file
        if isinstance(data, Document):
            docs = [data]
        elif isinstance(data, list):
            docs = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conllu', delete=False, encoding='utf-8') as f:
            for doc in docs:
                conllu_text = document_to_conllu(doc)
                f.write(conllu_text)
                f.write("\n\n")
            temp_path = Path(f.name)
        
        return [temp_path]
    
    def _merge_conllu_files(self, files: List[Path], output: Path) -> None:
        """Merge multiple CoNLL-U files into one."""
        with output.open("w", encoding="utf-8") as out_f:
            for file_path in files:
                with file_path.open("r", encoding="utf-8", errors="replace") as in_f:
                    content = in_f.read().strip()
                    if content:
                        out_f.write(content)
                        if not content.endswith("\n\n"):
                            out_f.write("\n\n")
    
    def _prepare_unimorph_vocab(
        self,
        unimorph_file: Path,
        corpus_file: Path,
        output_dir: Path,
        *,
        tagset_file: Optional[Path] = None,
        verbose: bool = False,
    ) -> Path:
        """Prepare UniMorph vocabulary for ClassLA training."""
        from ..lexicon import load_unimorph_lexicon
        from ..lexicon import _extract_xpos_tags  # type: ignore
        
        if not unimorph_file.exists():
            raise ValueError(f"UniMorph vocabulary file not found: {unimorph_file}")
        
        if verbose:
            print(f"[flexipipe] Loading UniMorph vocabulary from {unimorph_file}")
        
        # Extract XPOS tags from corpus for matching
        corpus_xpos_tags = None
        if corpus_file.exists():
            corpus_xpos_tags = _extract_xpos_tags(corpus_file)
            if verbose:
                print(f"[flexipipe] Extracted {len(corpus_xpos_tags)} XPOS tags from corpus")
        
        # Load and convert UniMorph lexicon
        vocab = load_unimorph_lexicon(
            unimorph_file,
            tagset_file=tagset_file,
            corpus_xpos_tags=corpus_xpos_tags,
            default_count=1,
        )
        
        # Save vocabulary in JSON format
        vocab_file = output_dir / "vocab.json"
        vocab_data = {
            "vocab": vocab,
            "metadata": {
                "source": "unimorph",
                "unimorph_file": str(unimorph_file),
                "corpus_file": str(corpus_file),
                "tagset_file": str(tagset_file) if tagset_file else None,
            }
        }
        
        with vocab_file.open("w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"[flexipipe] UniMorph vocabulary converted: {len(vocab)} entries")
        
        return vocab_file

    def supports_training(self) -> bool:
        return True


def _list_classla_models(*args, **kwargs) -> int:
    entries = get_classla_model_entries(*args, **kwargs)
    print(f"\nAvailable ClassLA models:")
    print(f"{'Model ID':<20} {'ISO':<6} {'Language':<20} {'Package':<15} {'Variant':<15} {'Status':<25}")
    print("=" * 101)
    for key in sorted(entries.keys()):
        model_info = entries[key]
        model_id = key
        iso = (model_info.get("language_iso") or "")[:6]
        lang = model_info.get("language_name", "")
        pkg = model_info.get("package", "")
        variant = model_info.get("variant", "standard")
        status = model_info.get("status", "Available")
        print(f"{model_id:<20} {iso:<6} {lang:<20} {pkg:<15} {variant:<15} {status:<25}")
    print(f"\nTotal: {len(entries)} model(s)")
    print("\nClassLA models are downloaded automatically on first use")
    print("Features: tokenization, lemma, upos, xpos, feats, depparse, NER")
    print("\nUsage: --backend classla --model <Model ID>")
    return 0


def _create_classla_backend(
    *,
    model_name: str | None = None,
    language: str | None = None,
    package: str | None = None,
    processors: str | None = None,
    use_gpu: bool = False,
    download_model: bool = False,
    verbose: bool = False,
    type: str | None = None,
    training: bool = False,
    **kwargs: Any,
) -> ClassLABackend:
    _ = training
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unexpected ClassLA backend arguments: {unexpected}")
    return ClassLABackend(
        model_name=model_name,
        language=language,
        package=package,
        processors=processors,
        use_gpu=use_gpu,
        download_model=download_model,
        verbose=verbose,
        type=type,
    )


BACKEND_SPEC = BackendSpec(
    name="classla",
    description="ClassLA - Fork of Stanza for South Slavic languages",
    factory=_create_classla_backend,
    get_model_entries=get_classla_model_entries,
    list_models=_list_classla_models,
    supports_training=True,
    is_rest=False,
    url="https://github.com/clarinsi/classla",
)

