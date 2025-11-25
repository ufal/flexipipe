#!/usr/bin/env python3
"""
Script to add a flexipipe-trained model to the registry.

This script helps add models trained with flexipipe to the "flexipipe" source
category in the registry, making them discoverable by users.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_registry(registry_path: Path) -> Dict[str, Any]:
    """Load existing registry."""
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return {
        "version": "1.0",
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "backend_version": ">=3.0.0",
        "sources": {
            "official": [],
            "flexipipe": [],
            "community": []
        }
    }


def add_flexipipe_model(
    registry_path: Path,
    backend: str,
    model_name: str,
    language_iso: str,
    language_name: str,
    download_url: str,
    description: str,
    *,
    backend_version: str = ">=3.0.0",
    preferred: bool = False,
    size_mb: Optional[int] = None,
    source: str = "flexipipe",  # "flexipipe" or "community"
    **kwargs: Any,
) -> None:
    """
    Add a flexipipe-trained or community model to the registry.
    
    Args:
        registry_path: Path to registry JSON file
        backend: Backend name (e.g., "spacy", "flexitag")
        model_name: Model name/identifier
        language_iso: ISO 639-1 language code
        language_name: Human-readable language name
        download_url: URL where the model can be downloaded
        description: Model description
        backend_version: Required backend version (default: ">=3.0.0")
        preferred: Whether this is a preferred model for the language
        size_mb: Model size in MB (optional)
        source: Source category - "flexipipe" or "community" (default: "flexipipe")
        **kwargs: Additional fields to include in the model entry
    """
    registry = load_registry(registry_path)
    
    # Ensure sources structure exists
    if "sources" not in registry:
        registry["sources"] = {}
    if source not in registry["sources"]:
        registry["sources"][source] = []
    
    # Build model entry
    model_entry: Dict[str, Any] = {
        "model": model_name,
        "language_iso": language_iso,
        "language_name": language_name,
        "backend_version": backend_version,
        "download_url": download_url,
        "description": description,
        "source": source,  # Mark source
    }
    
    if preferred:
        model_entry["preferred"] = True
    
    if size_mb is not None:
        model_entry["size_mb"] = size_mb
    
    # Add any additional fields
    model_entry.update(kwargs)
    
    # Check if model already exists
    existing_models = registry["sources"][source]
    for i, existing in enumerate(existing_models):
        if existing.get("model") == model_name:
            print(f"[flexipipe] Updating existing model entry: {model_name}")
            existing_models[i] = model_entry
            break
    else:
        print(f"[flexipipe] Adding new model entry: {model_name}")
        existing_models.append(model_entry)
    
    # Update timestamp
    registry["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    # Write updated registry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print(f"[flexipipe] Registry updated: {len(existing_models)} {source} model(s) for {backend}")
    print(f"[flexipipe] Saved to: {registry_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add a flexipipe-trained model to the registry"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent.parent.parent / "docs" / "registries",
        help="Path to registries directory (default: docs/registries)",
    )
    parser.add_argument(
        "--backend",
        required=True,
        help="Backend name (e.g., spacy, flexitag)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name/identifier",
    )
    parser.add_argument(
        "--language-iso",
        required=True,
        help="ISO 639-1 language code (e.g., en, de)",
    )
    parser.add_argument(
        "--language-name",
        required=True,
        help="Human-readable language name (e.g., English, German)",
    )
    parser.add_argument(
        "--download-url",
        required=True,
        help="URL where the model can be downloaded",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="Model description",
    )
    parser.add_argument(
        "--backend-version",
        default=">=3.0.0",
        help="Required backend version (default: >=3.0.0)",
    )
    parser.add_argument(
        "--preferred",
        action="store_true",
        help="Mark as preferred model for this language",
    )
    parser.add_argument(
        "--size-mb",
        type=int,
        help="Model size in MB",
    )
    parser.add_argument(
        "--source",
        choices=["flexipipe", "community"],
        default="flexipipe",
        help="Source category: 'flexipipe' for flexipipe-trained models, 'community' for community-provided models",
    )
    
    args = parser.parse_args()
    
    registry_file = args.registry / f"{args.backend}.json"
    
    add_flexipipe_model(
        registry_path=registry_file,
        backend=args.backend,
        model_name=args.model,
        language_iso=args.language_iso,
        language_name=args.language_name,
        download_url=args.download_url,
        description=args.description,
        backend_version=args.backend_version,
        preferred=args.preferred,
        size_mb=args.size_mb,
        source=args.source,
    )


if __name__ == "__main__":
    main()

