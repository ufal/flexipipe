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
    
    # Map backend names to direct pip packages (for backends not in flexipipe extras)
    BACKEND_TO_PACKAGE = {
        "ctext": "ctextcore",
        "udkanbun": "udkanbun",
        "udapi": "udapi",
    }
    
    all_backends = set(get_backend_choices())
    backends_to_install = []
    backends_to_install_direct = []  # Backends that install packages directly via pip
    invalid_backends = []
    no_install_needed = []
    
    # Process backend names
    for backend in args.backends:
        backend_lower = backend.lower()
        
        if backend_lower == "all":
            # Install all backends that have extras
            backends_to_install.extend(BACKEND_TO_EXTRA.keys())
            backends_to_install_direct.extend(BACKEND_TO_PACKAGE.keys())
            continue
        
        # Check if it's a direct package (like udapi) that's not a backend
        # This check must come before the all_backends check
        if backend_lower in BACKEND_TO_PACKAGE:
            backends_to_install_direct.append(backend_lower)
            continue
        
        if backend_lower not in all_backends:
            invalid_backends.append(backend)
            continue
        
        # Get backend info to check install instructions
        backend_info = get_backend_info(backend_lower)
        
        if backend_info and backend_info.install_instructions:
            # Check if install_instructions contains "pip install" - if so, we can extract and install
            instructions = backend_info.install_instructions
            if "pip install" in instructions.lower():
                # Try to extract package name from instructions like "Install via: pip install ctextcore"
                import re
                match = re.search(r'pip install\s+([^\s]+)', instructions, re.IGNORECASE)
                if match:
                    package_name = match.group(1).strip()
                    # Store as (backend_name, package_name) tuple
                    backends_to_install_direct.append((backend_lower, package_name))
                    continue
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
    
    if not backends_to_install and not backends_to_install_direct:
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
    
    # Initialize counters
    success_count = 0
    failed_backends = []
    
    # Install backends that install packages directly via pip
    for item in backends_to_install_direct:
        if isinstance(item, tuple):
            backend_name, package_name = item
        else:
            backend_name = item
            package_name = BACKEND_TO_PACKAGE[backend_name]
        
        # Check if already installed
        # Some packages have different module names than package names
        module_name_map = {
            "ctextcore": "ctextcore",
            "udkanbun": "udkanbun",
            "udapi": "udapi",
        }
        module_name = module_name_map.get(package_name, package_name.replace("-", "_"))
        if _module_available(module_name):
            if args.verbose:
                print(f"[flexipipe] Backend '{backend_name}' dependencies already installed")
            success_count += 1
            continue
        
        if args.verbose:
            print(f"[flexipipe] Installing dependencies for backend '{backend_name}' (pip install {package_name})...")
        
        # Special handling for ctext (requires Java)
        if backend_name == "ctext":
            import shutil
            if shutil.which("java") is None:
                print(f"[flexipipe] ⚠ Warning: Java (OpenJDK 17+) is required for CTexT backend but not found on PATH.", file=sys.stderr)
                print(f"[flexipipe]    Install Java from https://openjdk.org or via your system package manager.", file=sys.stderr)
                print(f"[flexipipe]    After installing Java, ensure 'java' is available in your PATH.", file=sys.stderr)
            else:
                # Check Java version
                try:
                    import subprocess
                    result = subprocess.run(
                        ["java", "-version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    version_output = result.stderr or result.stdout
                    import re
                    version_match = re.search(r'version "(\d+)', version_output)
                    if version_match:
                        java_version = int(version_match.group(1))
                        if java_version < 17:
                            print(f"[flexipipe] ⚠ Warning: Java version {java_version} found, but CTexT requires Java 17+.", file=sys.stderr)
                            print(f"[flexipipe]    Please upgrade to Java OpenJDK 17+ from https://openjdk.org", file=sys.stderr)
                except Exception:
                    pass  # If we can't check version, continue anyway
        
        # Install package directly via pip
        if _run_pip_install(package_name, direct=True):
            if _module_available(module_name):
                print(f"[flexipipe] ✓ Successfully installed dependencies for '{backend_name}'")
                success_count += 1
            else:
                print(f"[flexipipe] ⚠ Installation completed but module '{module_name}' is not importable", file=sys.stderr)
                failed_backends.append(backend_name)
        else:
            print(f"[flexipipe] ✗ Failed to install dependencies for '{backend_name}'", file=sys.stderr)
            failed_backends.append(backend_name)
    
    # Install each backend's extra
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

