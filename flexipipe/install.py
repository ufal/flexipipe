"""Install optional backend dependencies."""

from __future__ import annotations

import argparse
import sys

from .dependency_utils import _run_pip_install, _module_available
from .backend_registry import get_backend_choices, get_backend_info


def run_install(args: argparse.Namespace) -> int:
    """Install optional backend dependencies."""
    # Map backend names to their extra names (for pip install flexipipe[extra])
    BACKEND_TO_EXTRA = {
        "fasttext": "fasttext",
        "spacy": "spacy",
        "stanza": "stanza",
        "classla": "classla",
        "flair": "flair",
        "transformers": "transformers",
        "nametag": "nametag",
        "udpipe": "udpipe",
        "udmorph": "udmorph",
        "hunspell": "phunspell",  # Backend name is "hunspell", but package is "phunspell"
    }
    
    all_backends = set(get_backend_choices())
    backends_to_install = []
    invalid_backends = []
    no_install_needed = []
    
    # Process backend names
    for backend in args.backends:
        backend_lower = backend.lower()
        
        if backend_lower == "all":
            # Install all backends that have extras
            backends_to_install.extend(BACKEND_TO_EXTRA.keys())
            continue
        
        if backend_lower not in all_backends:
            invalid_backends.append(backend)
            continue
        
        # Get backend info to check install instructions
        backend_info = get_backend_info(backend_lower)
        if backend_info and backend_info.install_instructions:
            # Backend has custom install instructions (e.g., REST service, CLI tool)
            no_install_needed.append((backend_lower, backend_info.install_instructions))
            continue
        
        if backend_lower in BACKEND_TO_EXTRA:
            backends_to_install.append(backend_lower)
        else:
            # Backend doesn't have an extra - use default message or backend's instructions
            if backend_info and backend_info.install_instructions:
                no_install_needed.append((backend_lower, backend_info.install_instructions))
            elif args.verbose:
                print(f"[flexipipe] Backend '{backend}' does not have an installable extra")
    
    if invalid_backends:
        print(f"[flexipipe] Error: Unknown backend(s): {', '.join(invalid_backends)}", file=sys.stderr)
        print(f"[flexipipe] Available backends: {', '.join(sorted(all_backends))}", file=sys.stderr)
        return 1
    
    # Show messages for backends that don't need installation
    for backend_name, instructions in no_install_needed:
        print(f"[flexipipe] {instructions}")
    
    if not backends_to_install:
        if not no_install_needed and not args.verbose:
            print("[flexipipe] No backends require installation (all are built-in or require CLI tools)")
        return 0
    
    # Deduplicate while preserving order
    seen = set()
    unique_backends = []
    for backend in backends_to_install:
        if backend not in seen:
            seen.add(backend)
            unique_backends.append(backend)
    
    # Install each backend's extra
    success_count = 0
    failed_backends = []
    
    for backend in unique_backends:
        extra_name = BACKEND_TO_EXTRA[backend]
        
        # Check if already installed
        # For REST backends (nametag, udpipe, udmorph), requests is already in base,
        # but we still allow installation of the extra (it's idempotent and may include other deps)
        module_map = {
            "fasttext": "fasttext",
            "spacy": "spacy",
            "stanza": "stanza",
            "classla": "classla",
            "flair": "flair",
            "transformers": "transformers",
            "hunspell": "phunspell",  # Backend name is "hunspell", but module is "phunspell"
            # REST backends: nametag, udpipe, udmorph don't have specific modules to check
            # They use requests which is already in base, but we still install the extra
        }
        
        module_to_check = module_map.get(backend)
        if module_to_check and _module_available(module_to_check):
            if args.verbose:
                print(f"[flexipipe] Backend '{backend}' dependencies already installed")
            success_count += 1
            continue
        
        if args.verbose:
            print(f"[flexipipe] Installing dependencies for backend '{backend}'...")
        
        if _run_pip_install(extra_name):
            # Verify installation for backends with specific modules
            if module_to_check:
                if _module_available(module_to_check):
                    print(f"[flexipipe] ✓ Successfully installed dependencies for '{backend}'")
                    success_count += 1
                else:
                    print(f"[flexipipe] ⚠ Installation completed but module '{module_to_check}' is not importable", file=sys.stderr)
                    failed_backends.append(backend)
            else:
                # REST backends (nametag, udpipe, udmorph) - just report success
                print(f"[flexipipe] ✓ Successfully installed dependencies for '{backend}'")
                success_count += 1
        else:
            print(f"[flexipipe] ✗ Failed to install dependencies for '{backend}'", file=sys.stderr)
            failed_backends.append(backend)
    
    if failed_backends:
        print(f"\n[flexipipe] Failed to install: {', '.join(failed_backends)}", file=sys.stderr)
        return 1
    
    if success_count > 0:
        print(f"\n[flexipipe] Successfully installed dependencies for {success_count} backend(s)")
    
    return 0

