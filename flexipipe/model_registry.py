"""
Remote model registry for flexipipe.

This module provides a centralized system for:
1. Serving standard lists of available models from remote sources (GitHub, HuggingFace, Lindat, etc.)
2. Hosting flexipipe-trained models
3. Allowing users to share their trained models
4. Harvesting and curating model lists
5. Tracking backend version compatibility
6. Unified model discovery (remote + local)

Architecture:
- Remote model lists are served as JSON from a central repository (e.g., GitHub)
- Each backend can have multiple model list sources (official, community, flexipipe-hosted)
- Model entries include: backend version compatibility, preferred flags, download URLs, etc.
- Local models are merged with remote lists for unified discovery
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
except ImportError:
    requests = None

from .model_storage import get_cache_dir, read_model_cache_entry, write_model_cache_entry

# Default remote model registry base URL (can be overridden via config or env var)
DEFAULT_REGISTRY_BASE_URL = "https://raw.githubusercontent.com/flexipipe/flexipipe-models/main/registries"

# Cache settings
REGISTRY_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
REGISTRY_CACHE_KEY = "remote_model_registry"


def get_registry_url(backend: Optional[str] = None) -> str:
    """
    Get the remote model registry URL for a backend.
    
    Args:
        backend: Backend name (e.g., "flexitag", "spacy"). If None, returns base URL.
        
    Returns:
        Registry URL for the backend, or base URL if backend is None
        Can be a file:// URL for local directories or http(s):// for remote
    """
    from .model_storage import read_config
    from .backend_registry import get_backend_info
    import os
    
    # First, check if backend has a specific registry URL in its spec
    if backend:
        backend_info = get_backend_info(backend)
        if backend_info and backend_info.model_registry_url:
            return backend_info.model_registry_url
    
    # Check config for backend-specific URL
    config = read_config()
    if backend:
        backend_key = f"model_registry_url_{backend}"
        registry_url = config.get(backend_key)
        if registry_url:
            return registry_url
    
    # Check config for local registry directory (for development only - explicit opt-in)
    # This is checked AFTER backend-specific URLs but BEFORE base URLs
    # so developers can override per-backend if needed
    local_registry_dir = config.get("model_registry_local_dir")
    if local_registry_dir:
        local_path = Path(local_registry_dir).expanduser().resolve()
        if backend:
            registry_file = local_path / "registries" / f"{backend}.json"
            return f"file://{registry_file}"
        return f"file://{local_path}"
    
    # Check environment variable for local registry directory (development only)
    local_registry_dir = os.environ.get("FLEXIPIPE_MODEL_REGISTRY_LOCAL_DIR")
    if local_registry_dir:
        local_path = Path(local_registry_dir).expanduser().resolve()
        if backend:
            registry_file = local_path / "registries" / f"{backend}.json"
            return f"file://{registry_file}"
        return f"file://{local_path}"
    
    # Check config for base URL
    registry_url = config.get("model_registry_base_url")
    if registry_url:
        if backend:
            # Append backend name to base URL
            from urllib.parse import urljoin
            return urljoin(registry_url.rstrip("/") + "/", f"{backend}.json")
        return registry_url
    
    # Check environment variable for base URL
    registry_url = os.environ.get("FLEXIPIPE_MODEL_REGISTRY_BASE_URL")
    if registry_url:
        if backend:
            from urllib.parse import urljoin
            return urljoin(registry_url.rstrip("/") + "/", f"{backend}.json")
        return registry_url
    
    # Legacy: check for old single registry URL
    registry_url = config.get("model_registry_url")
    if registry_url:
        return registry_url
    
    registry_url = os.environ.get("FLEXIPIPE_MODEL_REGISTRY_URL")
    if registry_url:
        return registry_url
    
    # Default: use base URL pattern
    if backend:
        from urllib.parse import urljoin
        return urljoin(DEFAULT_REGISTRY_BASE_URL.rstrip("/") + "/", f"{backend}.json")
    return DEFAULT_REGISTRY_BASE_URL


def fetch_remote_registry(
    backend: Optional[str] = None,
    url: Optional[str] = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_ttl_seconds: int = REGISTRY_CACHE_TTL_SECONDS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fetch the remote model registry for a backend.
    
    The registry is a JSON file with the following structure:
    {
        "version": "1.0",
        "last_updated": "2024-01-01T00:00:00Z",
        "sources": {
            "official": [...],  # Official models from upstream
            "flexipipe": [...],  # Models trained/hosted by flexipipe
            "community": [...]   # Community-contributed models
        },
        "backend_version": ">=3.0.0"  # Optional: required backend version
    }
    
    Args:
        backend: Backend name (e.g., "flexitag", "spacy"). Used to determine URL if url is None.
        url: Registry URL (defaults to backend-specific URL)
        use_cache: If True, use cached registry if available
        refresh_cache: If True, force refresh from remote
        cache_ttl_seconds: Cache TTL in seconds
        verbose: If True, print progress messages
        
    Returns:
        Registry dictionary
    """
    if url is None:
        url = get_registry_url(backend)
    
    cache_key = f"{REGISTRY_CACHE_KEY}:{url}"
    
    # Check cache
    if use_cache and not refresh_cache:
        cached = read_model_cache_entry(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached:
            if verbose:
                print(f"[flexipipe] Using cached remote model registry from {url}")
            return cached
    
    if not requests:
        if verbose:
            print("[flexipipe] Warning: 'requests' package not available, cannot fetch remote registry")
        return {}
    
    # Handle local file paths
    if url.startswith("file://"):
        file_path = Path(url[7:])  # Remove "file://" prefix
        if not file_path.exists():
            if verbose:
                print(f"[flexipipe] Warning: registry file not found: {file_path}")
            return {}
        if verbose:
            print(f"[flexipipe] Reading registry from local file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            # Cache the registry
            try:
                write_model_cache_entry(cache_key, registry)
            except (OSError, PermissionError):
                pass
            return registry
        except Exception as exc:
            if verbose:
                print(f"[flexipipe] Warning: failed to read registry file: {exc}")
            return {}
    
    # Prepare request headers (for GitHub token authentication)
    headers = {}
    import os
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("FLEXIPIPE_GITHUB_TOKEN")
    is_github_url = "github.com" in url or "githubusercontent.com" in url
    
    # For private repos, raw.githubusercontent.com doesn't work even with tokens
    # Use GitHub API instead when we have a token and it's a raw URL
    if github_token and "raw.githubusercontent.com" in url:
        # Convert raw URL to GitHub API URL
        # https://raw.githubusercontent.com/owner/repo/branch/path
        # -> https://api.github.com/repos/owner/repo/contents/path?ref=branch
        import re
        match = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url)
        if match:
            owner, repo, branch, path = match.groups()
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
            if verbose:
                print(f"[flexipipe] Fetching from GitHub API (private repo): {api_url}")
            api_headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            try:
                api_response = requests.get(api_url, headers=api_headers, timeout=10.0)
                api_response.raise_for_status()
                api_data = api_response.json()
                # GitHub API returns base64-encoded content
                import base64
                content = base64.b64decode(api_data.get("content", "")).decode("utf-8")
                registry = json.loads(content)
                # Cache the registry
                try:
                    write_model_cache_entry(cache_key, registry)
                except (OSError, PermissionError):
                    pass
                return registry
            except Exception as exc:
                if verbose:
                    print(f"[flexipipe] Warning: GitHub API fetch failed: {exc}")
                # Fall through to try raw URL (might work for public repos)
    
    if verbose:
        if is_github_url:
            if github_token:
                print(f"[flexipipe] Fetching remote model registry from {url} (using GitHub token authentication)...")
            else:
                print(f"[flexipipe] Fetching remote model registry from {url} (no GitHub token found - may fail for private repos)...")
        else:
            print(f"[flexipipe] Fetching remote model registry from {url}...")
    
    if github_token and is_github_url:
        headers["Authorization"] = f"token {github_token}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        registry = response.json()
        
        # Validate registry structure
        if not isinstance(registry, dict):
            raise ValueError("Registry must be a JSON object")
        
        # Cache the registry
        if refresh_cache or not use_cache:
            try:
                write_model_cache_entry(cache_key, registry)
            except (OSError, PermissionError):
                pass  # Cache write is best-effort
        
        return registry
    except Exception as exc:
        if verbose:
            print(f"[flexipipe] Warning: failed to fetch remote registry: {exc}")
        # Return cached version if available (even if stale)
        cached = read_model_cache_entry(cache_key, max_age_seconds=None)
        if cached:
            if verbose:
                print("[flexipipe] Using stale cached registry")
            return cached
        return {}


def get_remote_models_for_backend(
    backend: str,
    *,
    source: Optional[str] = None,  # "official", "flexipipe", "community", or None for all
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get remote models for a specific backend.
    
    Args:
        backend: Backend name (e.g., "spacy", "stanza", "flexitag")
        source: Model source filter ("official", "flexipipe", "community", or None for all)
        use_cache: If True, use cached registry
        refresh_cache: If True, force refresh
        verbose: If True, print progress messages
        
    Returns:
        List of model entry dictionaries with source field set
    """
    registry = fetch_remote_registry(
        backend=backend,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    
    # Per-backend registries have sources at the top level
    sources = registry.get("sources", {})
    
    if source:
        models = sources.get(source, [])
        # Mark source
        for model in models:
            model["source"] = source
        return models
    
    # Combine all sources
    all_models: List[Dict[str, Any]] = []
    for source_name in ["official", "flexipipe", "community"]:
        models = sources.get(source_name, [])
        # Mark source for each model
        for model in models:
            model["source"] = source_name
        all_models.extend(models)
    
    return all_models


def merge_remote_and_local_models(
    backend: str,
    local_models: Dict[str, Dict[str, Any]],
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Merge remote and local models for a backend.
    
    Local models take precedence (they override remote entries with the same model name).
    
    Args:
        backend: Backend name
        local_models: Dictionary of local model entries (from get_*_model_entries)
        use_cache: If True, use cached registry
        refresh_cache: If True, force refresh
        verbose: If True, print progress messages
        
    Returns:
        Merged dictionary of model entries
    """
    # Mark all local models with source="local"
    merged = {}
    for model_name, entry in local_models.items():
        entry_copy = dict(entry)
        entry_copy["source"] = "local"
        merged[model_name] = entry_copy
    
    # Add remote models (only if not already present locally)
    remote_models = get_remote_models_for_backend(
        backend,
        source=None,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )
    
    for model_entry in remote_models:
        model_name = model_entry.get("model")
        if not model_name:
            continue
        
        # Only add if not already present locally
        if model_name not in merged:
            # Mark as remote and determine source type
            source_type = model_entry.get("source", "remote")
            if source_type not in ["official", "flexipipe", "community"]:
                # Determine from registry structure if not specified
                source_type = "remote"
            model_entry["source"] = source_type
            model_entry["available_without_download"] = False
            merged[model_name] = model_entry
        else:
            # Merge metadata from remote (but keep local status)
            local_entry = merged[model_name]
            # Update metadata fields that might be missing locally
            for key in ["backend_version", "download_url", "preferred", "description"]:
                if key not in local_entry and key in model_entry:
                    local_entry[key] = model_entry[key]
            # Mark source as local (local takes precedence)
            local_entry["source"] = "local"
    
    return merged


def check_backend_version_compatibility(
    backend: str,
    model_entry: Dict[str, Any],
    current_backend_version: Optional[str] = None,
) -> bool:
    """
    Check if a model is compatible with the current backend version.
    
    Args:
        backend: Backend name
        model_entry: Model entry dictionary (should have "backend_version" field)
        current_backend_version: Current backend version (if None, will try to detect)
        
    Returns:
        True if compatible, False otherwise
    """
    required_version = model_entry.get("backend_version")
    if not required_version:
        # No version requirement - assume compatible
        return True
    
    if current_backend_version is None:
        # Try to detect current version
        try:
            if backend == "spacy":
                import spacy
                current_backend_version = spacy.__version__
            elif backend == "stanza":
                import stanza
                current_backend_version = stanza.__version__
            elif backend == "flair":
                import flair
                current_backend_version = flair.__version__
            else:
                # Unknown backend - assume compatible
                return True
        except Exception:
            # Can't detect version - assume compatible
            return True
    
    # Simple version comparison (can be enhanced)
    # For now, just check if versions match exactly or if required is a range
    if isinstance(required_version, str):
        if ">=" in required_version or "<=" in required_version or "==" in required_version:
            # TODO: Implement proper version range parsing
            return True
        return required_version == current_backend_version
    
    return True


def get_backend_version_from_registry(
    backend: str,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Optional[str]:
    """
    Get required backend version from the backend's registry.
    
    Args:
        backend: Backend name
        use_cache: If True, use cached registry
        refresh_cache: If True, force refresh
        
    Returns:
        Required backend version string (e.g., ">=3.0.0") or None
    """
    registry = fetch_remote_registry(
        backend=backend,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        verbose=False,
    )
    
    return registry.get("backend_version")

