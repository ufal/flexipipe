"""Install optional backend dependencies."""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from pathlib import Path

from .dependency_utils import _run_pip_install, _module_available
from .backend_registry import get_backend_choices, get_backend_info


def _build_c_launcher(repo_scripts: Path) -> bool:
    """If scripts/flexipipe_launcher.c exists, try to build the binary. Return True if binary exists."""
    launcher_c = repo_scripts / "flexipipe_launcher.c"
    launcher_bin = repo_scripts / "flexipipe_launcher"
    if not launcher_c.is_file():
        return launcher_bin.is_file()
    makefile = repo_scripts / "Makefile"
    if makefile.is_file():
        r = subprocess.run(
            ["make", "flexipipe_launcher"],
            cwd=repo_scripts,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("[flexipipe] Could not build C launcher (make failed). Using shell wrapper.", file=sys.stderr)
            if r.stderr:
                print(r.stderr, file=sys.stderr)
            return False
        print("[flexipipe] Built C launcher for faster startup.")
    else:
        r = subprocess.run(
            ["cc", "-O2", "-o", "flexipipe_launcher", "flexipipe_launcher.c"],
            cwd=repo_scripts,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("[flexipipe] Could not build C launcher (cc failed). Using shell wrapper.", file=sys.stderr)
            if r.stderr:
                print(r.stderr, file=sys.stderr)
            return False
        print("[flexipipe] Built C launcher for faster startup.")
    return launcher_bin.is_file() and launcher_bin.stat().st_size > 0


def _install_wrapper_script(args: argparse.Namespace) -> int:
    """Install the flexipipe wrapper so you can run 'flexipipe' instead of 'python -m flexipipe'.
    Builds the C launcher from scripts/flexipipe_launcher.c when present (faster startup); else uses the shell script."""
    repo_scripts = Path(__file__).parent.parent / "scripts"
    launcher_bin = repo_scripts / "flexipipe_launcher"
    script_path = Path(__file__).parent / "data" / "flexipipe_wrapper.sh"
    # Try to build C launcher if source exists (e.g. development repo)
    if (repo_scripts / "flexipipe_launcher.c").is_file():
        _build_c_launcher(repo_scripts)
    use_launcher_bin = launcher_bin.is_file() and launcher_bin.stat().st_size > 0
    if not use_launcher_bin and not script_path.exists():
        print("[flexipipe] Wrapper not found (expected data/flexipipe_wrapper.sh or scripts/flexipipe_launcher).", file=sys.stderr)
        return 1
    if use_launcher_bin:
        script_content = None
    else:
        script_content = script_path.read_text()

    install_path = getattr(args, "wrapper_path", None)
    use_sudo = False
    if install_path:
        install_path = Path(install_path).expanduser().resolve()
        if install_path.is_dir():
            install_path = install_path / "flexipipe"
        install_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Prompt interactively
        print("\nInstall the wrapper script so you can run 'flexipipe' instead of 'python -m flexipipe'.")
        print("  1. /usr/local/bin (system-wide, may require sudo)")
        print("  2. ~/bin (user-local; ensure ~/bin is on your PATH)")
        print("  3. Custom path")
        print("  4. Cancel")
        try:
            choice = input("Choice [1-4] (default: 2): ").strip() or "2"
        except EOFError:
            print("[flexipipe] No input; skipping wrapper install.", file=sys.stderr)
            return 0
        if choice == "4":
            return 0
        if choice == "1":
            install_path = Path("/usr/local/bin/flexipipe")
            use_sudo = True
        elif choice == "2":
            install_path = Path.home() / "bin" / "flexipipe"
            install_path.parent.mkdir(parents=True, exist_ok=True)
            use_sudo = False
        elif choice == "3":
            try:
                custom = input("Enter directory or full path: ").strip()
            except EOFError:
                return 0
            if not custom:
                return 0
            install_path = Path(custom).expanduser().resolve()
            if install_path.is_dir():
                install_path = install_path / "flexipipe"
            install_path.parent.mkdir(parents=True, exist_ok=True)
            use_sudo = False
        else:
            print("[flexipipe] Invalid choice.", file=sys.stderr)
            return 1

    try:
        if use_sudo:
            import tempfile
            if use_launcher_bin:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                    tmp.write(launcher_bin.read_bytes())
                    tmp_path = tmp.name
            else:
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh") as tmp:
                    tmp.write(script_content)
                    tmp_path = tmp.name
            result = subprocess.run(["sudo", "cp", tmp_path, str(install_path)], capture_output=True, text=True)
            Path(tmp_path).unlink(missing_ok=True)
            if result.returncode != 0:
                print(f"[flexipipe] Error: {result.stderr or result.stdout}", file=sys.stderr)
                return 1
            subprocess.run(["sudo", "chmod", "+x", str(install_path)], check=True)
        else:
            if use_launcher_bin:
                install_path.write_bytes(launcher_bin.read_bytes())
            else:
                install_path.write_text(script_content)
            install_path.chmod(install_path.stat().st_mode | stat.S_IEXEC)
        kind = "C launcher" if use_launcher_bin else "Wrapper script"
        print(f"[flexipipe] ✓ {kind} installed to: {install_path}")
        if install_path.parent == Path.home() / "bin":
            print("[flexipipe] Ensure ~/bin is on your PATH (e.g. export PATH=\"$HOME/bin:$PATH\" in ~/.bashrc or ~/.zshrc).")
        return 0
    except Exception as e:
        print(f"[flexipipe] Error installing wrapper: {e}", file=sys.stderr)
        return 1


def _run_self_update(args: argparse.Namespace) -> int:
    """Upgrade flexipipe to the latest version from PyPI (or reinstall if editable)."""
    try:
        from importlib.metadata import distribution, PackageNotFoundError
        dist = distribution("flexipipe")
        files = list(dist.files) if dist.files else []
        is_editable = False
        if files:
            try:
                first_file = files[0]
                if hasattr(first_file, "locate"):
                    path_str = str(Path(first_file.locate()).resolve())
                    if "site-packages" not in path_str and "dist-packages" not in path_str:
                        is_editable = True
            except (OSError, ValueError, AttributeError):
                pass
    except (ImportError, PackageNotFoundError):
        is_editable = False

    if is_editable:
        print("[flexipipe] You appear to be using an editable (development) install.")
        print("[flexipipe] To update: pull the latest changes, then run 'pip install -e .' in the repo.")
        return 0

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "flexipipe"]
    if getattr(args, "verbose", False):
        cmd.append("-v")
    print(f"[flexipipe] Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        import importlib
        importlib.invalidate_caches()
        print("[flexipipe] ✓ flexipipe updated. Restart the shell or run 'flexipipe --version' to confirm.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[flexipipe] Upgrade failed: {e}", file=sys.stderr)
        return 1


def run_install(args: argparse.Namespace) -> int:
    """Install optional backend dependencies or the wrapper script."""
    # Special case: install wrapper script (flexipipe install wrapper [--path DIR])
    if len(args.backends) == 1 and args.backends[0].lower() == "wrapper":
        return _install_wrapper_script(args)
    # Special case: upgrade flexipipe (flexipipe install update)
    if len(args.backends) == 1 and args.backends[0].lower() == "update":
        return _run_self_update(args)

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

