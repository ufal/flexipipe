#!/usr/bin/env python3
"""
Script to harvest all installable SpaCy models and update the registry JSON.

This script:
1. Fetches all models from GitHub releases
2. Verifies which models are actually installable (checks PyPI)
3. Updates the registry JSON, preserving curated models and adding all installable ones
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

from flexipipe.spacy_backend import SPACY_LANGUAGE_NAMES
from flexipipe.language_utils import build_model_entry


def _fetch_spacy_remote_models() -> Dict[str, Dict[str, str]]:
    """
    Fetch the latest SpaCy models from GitHub releases.

    Returns a dict keyed by model name containing language, description, and latest_version.
    """
    github_models: Dict[str, Dict[str, str]] = {}
    page = 1
    per_page = 100

    while True:
        github_url = f"https://api.github.com/repos/explosion/spacy-models/releases?page={page}&per_page={per_page}"
        try:
            response = requests.get(github_url, timeout=10.0)
            response.raise_for_status()
            releases = response.json()
        except Exception:
            break

        if not releases:
            break

        for release in releases:
            tag_name = release.get("tag_name", "")
            match = re.match(r"^(.+?)-(\d+\.\d+\.\d+(?:\.\d+|post\d+)?)$", tag_name)
            if not match:
                continue
            model_name = match.group(1)
            version_str = match.group(2)

            # Filter out deprecated/non-downloadable models
            # These patterns indicate deprecated models that are no longer available
            deprecated_patterns = [
                r".*_pytt_.*",  # PyTorch transformer models (deprecated)
                r".*_vectors_.*",  # Vector models (deprecated, replaced by transformer models)
            ]
            if any(re.match(pattern, model_name) for pattern in deprecated_patterns):
                continue

            parts = model_name.split("_")
            lang = parts[0] if parts else ""

            desc_parts: List[str] = []
            if "core" in model_name:
                if "web" in model_name:
                    desc_parts.append("Web")
                elif "news" in model_name:
                    desc_parts.append("News")
                else:
                    desc_parts.append("Core")
            elif "ent" in model_name:
                desc_parts.append("Entity")

            if model_name.endswith("_trf") or "_trf_" in model_name:
                desc_parts.append("Transformer")
            elif model_name.endswith("_lg") or "_lg" in model_name:
                desc_parts.append("Large")
            elif model_name.endswith("_md") or "_md" in model_name:
                desc_parts.append("Medium")
            elif model_name.endswith("_sm") or "_sm" in model_name:
                desc_parts.append("Small")

            lang_name = SPACY_LANGUAGE_NAMES.get(lang, lang.upper() if lang else "Unknown")
            size_desc = " ".join(desc_parts) if desc_parts else "Model"
            desc = f"{lang_name} ({size_desc.lower()})"

            existing = github_models.get(model_name)
            if existing and existing.get("latest_version", "") >= version_str:
                continue

            github_models[model_name] = build_model_entry(
                "spacy",
                model_name,
                language_code=lang,
                language_name=lang_name,
                description=desc,
                latest_version=version_str,
            )

        if len(releases) < per_page:
            break
        page += 1

    return github_models


def check_model_on_pypi(model_name: str, version: str) -> bool:
    """
    Check if a model package exists on PyPI.
    
    SpaCy models are published as packages like 'en-core-web-sm' (with hyphens).
    """
    # Convert model name to PyPI package name: en_core_web_sm -> en-core-web-sm
    pypi_name = model_name.replace("_", "-")
    
    try:
        # Check PyPI JSON API
        url = f"https://pypi.org/pypi/{pypi_name}/{version}/json"
        response = requests.get(url, timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def load_existing_registry(registry_path: Path) -> Dict[str, Any]:
    """Load existing registry, creating empty structure if it doesn't exist."""
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return {
        "version": "1.0",
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "backend_version": ">=3.0.0",
        "sources": {
            "official": []
        }
    }


def get_curated_model_names(registry: Dict[str, Any]) -> Set[str]:
    """
    Extract model names from curated entries.
    
    Curated models are identified by:
    - Having `curated: true` flag
    - Having `preferred: true` flag (manually selected)
    - Having `size_mb` field (manually added metadata)
    """
    curated = set()
    official = registry.get("sources", {}).get("official", [])
    for entry in official:
        # Consider models curated if they have any of these indicators
        if entry.get("curated") or entry.get("preferred") or "size_mb" in entry:
            model = entry.get("model")
            if model:
                curated.add(model)
    return curated


def build_model_entry_from_github(
    model_name: str,
    version: str,
    github_entry: Dict[str, str],
    curated: bool = False,
) -> Dict[str, Any]:
    """Build a registry entry from GitHub model data."""
    lang_iso = github_entry.get("language_iso", "")
    lang_name = github_entry.get("language_name", "")
    description = github_entry.get("description", "")
    
    entry: Dict[str, Any] = {
        "model": model_name,
        "language_iso": lang_iso,
        "language_name": lang_name,
        "backend_version": ">=3.0.0",
        "download_command": f"python -m spacy download {model_name}",
        "description": description,
    }
    
    # Mark curated models
    if curated:
        entry["curated"] = True
    
    # Determine preferred status (curated small models are often preferred)
    if curated and model_name.endswith("_sm"):
        entry["preferred"] = True
    elif curated:
        entry["preferred"] = False
    
    return entry


def get_installed_models() -> Dict[str, Dict[str, str]]:
    """Get models from installed SpaCy models directory and standard location."""
    from flexipipe.model_storage import get_backend_models_dir
    
    installed_models = {}
    
    # Scan flexipipe models directory
    spacy_dir = get_backend_models_dir("spacy", create=False)
    if spacy_dir.exists():
        for model_dir in spacy_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            meta_json = model_dir / "meta.json"
            if not meta_json.exists():
                continue
            
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                model_name = model_dir.name
                lang = meta.get("lang", "")
                version = meta.get("version", "")
                description = meta.get("description", "")
                
                lang_name = SPACY_LANGUAGE_NAMES.get(lang, lang.upper() if lang else "Unknown")
                
                installed_models[model_name] = {
                    "model": model_name,
                    "language_iso": lang,
                    "language_name": lang_name,
                    "description": description or f"{lang_name} model",
                    "latest_version": version,
                }
            except Exception:
                continue
    
    # Also scan standard SpaCy location (site-packages)
    try:
        import importlib.util
        import pkg_resources
        
        for dist in pkg_resources.working_set:
            name = dist.metadata.get("Name", "")
            # SpaCy models use hyphens in package names: en-core-web-sm
            if name and ("-core-" in name or "-ent-" in name):
                model_name = name.replace("-", "_")
                if model_name not in installed_models:
                    # Try to get metadata from the package
                    try:
                        spec = importlib.util.find_spec(name)
                        if spec and spec.origin:
                            meta_path = Path(spec.origin).parent / "meta.json"
                            if meta_path.exists():
                                with open(meta_path, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                
                                lang = meta.get("lang", "")
                                version = meta.get("version", dist.version)
                                description = meta.get("description", "")
                                lang_name = SPACY_LANGUAGE_NAMES.get(lang, lang.upper() if lang else "Unknown")
                                
                                installed_models[model_name] = {
                                    "model": model_name,
                                    "language_iso": lang,
                                    "language_name": lang_name,
                                    "description": description or f"{lang_name} model",
                                    "latest_version": version,
                                }
                    except Exception:
                        # If we can't get metadata, still add the model with basic info
                        parts = model_name.split("_")
                        lang = parts[0] if parts else ""
                        lang_name = SPACY_LANGUAGE_NAMES.get(lang, lang.upper() if lang else "Unknown")
                        installed_models[model_name] = {
                            "model": model_name,
                            "language_iso": lang,
                            "language_name": lang_name,
                            "description": f"{lang_name} model",
                            "latest_version": dist.version,
                        }
    except Exception:
        pass
    
    return installed_models


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Harvest installable SpaCy models and update registry"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent.parent.parent / "docs" / "registries" / "spacy.json",
        help="Path to registry JSON file",
    )
    parser.add_argument(
        "--verify-pypi",
        action="store_true",
        help="Verify each model exists on PyPI (slower but more accurate)",
    )
    parser.add_argument(
        "--use-installed",
        action="store_true",
        help="Include models from installed directory as a source",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    args = parser.parse_args()
    
    print(f"[flexipipe] Harvesting SpaCy models from GitHub...")
    github_models = _fetch_spacy_remote_models()
    print(f"[flexipipe] Found {len(github_models)} models from GitHub releases")
    
    # Also get installed models if requested
    installed_models = {}
    if args.use_installed:
        print(f"[flexipipe] Scanning installed models...")
        installed_models = get_installed_models()
        print(f"[flexipipe] Found {len(installed_models)} installed models")
        
        # Merge installed models into github_models (installed takes precedence for version info)
        for model_name, info in installed_models.items():
            if model_name not in github_models:
                github_models[model_name] = info
            else:
                # Use installed version if it's newer or if github doesn't have version
                installed_version = info.get("latest_version", "")
                github_version = github_models[model_name].get("latest_version", "")
                if installed_version and (not github_version or installed_version > github_version):
                    github_models[model_name]["latest_version"] = installed_version
    
    # Load existing registry
    registry = load_existing_registry(args.registry)
    curated_models = get_curated_model_names(registry)
    print(f"[flexipipe] Found {len(curated_models)} curated models in existing registry")
    
    # Build new registry entries
    new_entries: Dict[str, Dict[str, Any]] = {}
    
    # First, preserve curated models with their existing metadata
    # Only preserve entries that were explicitly curated (had curated=true or were in original list)
    official = registry.get("sources", {}).get("official", [])
    for entry in official:
        model_name = entry.get("model")
        if model_name and model_name in curated_models:
            # Preserve existing entry, mark as curated
            entry_copy = dict(entry)
            entry_copy["curated"] = True
            new_entries[model_name] = entry_copy
    
    # Then, add all GitHub models (curated ones will be overwritten with preserved versions above)
    verified_count = 0
    skipped_count = 0
    
    for model_name, github_entry in github_models.items():
        version = github_entry.get("latest_version", "")
        
        if not version:
            if args.verbose:
                print(f"[flexipipe] Skipping {model_name}: no version")
            skipped_count += 1
            continue
        
        # If verifying, check PyPI
        if args.verify_pypi:
            if not check_model_on_pypi(model_name, version):
                if args.verbose:
                    print(f"[flexipipe] Skipping {model_name}: not found on PyPI")
                skipped_count += 1
                continue
            verified_count += 1
        
        # Skip if already in curated (preserved above)
        if model_name in new_entries:
            continue
        
        # Add new entry
        is_curated = model_name in curated_models
        new_entries[model_name] = build_model_entry_from_github(
            model_name, version, github_entry, curated=is_curated
        )
    
    # Update registry
    registry["sources"]["official"] = sorted(
        list(new_entries.values()),
        key=lambda x: (x.get("language_iso", ""), x.get("model", ""))
    )
    registry["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    # Write updated registry
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    with open(args.registry, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    curated_count = sum(1 for e in registry["sources"]["official"] if e.get("curated"))
    total_count = len(registry["sources"]["official"])
    
    print(f"\n[flexipipe] Registry updated:")
    print(f"  Total models: {total_count}")
    print(f"  Curated models: {curated_count}")
    print(f"  Additional models: {total_count - curated_count}")
    if args.verify_pypi:
        print(f"  Verified on PyPI: {verified_count}")
        print(f"  Skipped (not on PyPI): {skipped_count}")
    print(f"  Registry saved to: {args.registry}")


if __name__ == "__main__":
    main()

