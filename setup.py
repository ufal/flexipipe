#!/usr/bin/env python3
"""
Setup script for FlexiPipe
"""

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

try:
    import pybind11
except ImportError as exc:
    raise RuntimeError(
        "pybind11 is required to build flexipipe. "
        "Ensure you're using pip>=21 and that pyproject.toml build requirements are respected, "
        "or install pybind11 manually (pip install pybind11)."
    ) from exc
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Read version from flexipipe/__init__.py
version = "1.0.0"
init_file = Path(__file__).parent / "flexipipe" / "__init__.py"
if init_file.exists():
    for line in init_file.read_text(encoding='utf-8').split('\n'):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            break

BASE_REQUIREMENTS = [
    "pybind11>=2.10",
    "langcodes>=3.3.0",
    "language-data>=1.1.0",
    "pycountry>=23.12.0",
    "requests>=2.31.0",
    "tabulate>=0.9.0",
]

# xmltokenizer is a separate repo (sibling checkout for local dev).
_XT_SIBLING = (Path(__file__).resolve().parent.parent / "xmltokenizer")
if (_XT_SIBLING / "pyproject.toml").is_file():
    _XMLTOKENIZER_DEP = f"xmltokenizer @ file://{_XT_SIBLING}"
else:
    _XMLTOKENIZER_DEP = "xmltokenizer>=0.0.1"

EXTRAS = {
    "fasttext": ["fasttext-numpy2>=0.9.2.post2"],
    "spacy": ["spacy>=3.7.0"],
    "stanza": ["stanza>=1.8.0"],
    "classla": ["classla>=2.1.0"],
    "flair": ["flair>=0.13.0", "torch>=2.6.0"],
    "transformers": [
        "torch>=2.6.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "accelerate>=0.20.0",
    ],
    "nametag": ["requests>=2.31.0"],
    "udpipe": ["requests>=2.31.0"],
    "udmorph": ["requests>=2.31.0"],
    "heliport": ["heliport>=0.5.0"],
    "langdetect": ["langdetect>=1.0.9"],
    "langid": ["langid>=1.1.6"],
    "phunspell": ["phunspell>=0.1.0"],
    # TEITOK writeback: xmltokenizer as reader/writer; flexipipe runs NLP on nlp_plaintext.
    "xmltokenizer": [_XMLTOKENIZER_DEP],
}

all_extras = sorted({dep for deps in EXTRAS.values() for dep in deps})
EXTRAS["all"] = all_extras
EXTRAS["dev"] = sorted(
    set(
        all_extras
        + [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    )
)

DEPS_BASE = Path(__file__).parent


def _download_and_extract(url: str, target_dir: Path) -> Path:
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive_path = tmp_path / url.split("/")[-1]
        with urllib.request.urlopen(url) as response:
            data = response.read()
        archive_path.write_bytes(data)
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(tmp_path)
        extracted_roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if not extracted_roots:
            raise RuntimeError(f"Failed to extract dependency from {url}")
        root = extracted_roots[0]
        # Copy contents into target_dir
        for child in root.iterdir():
            dest = target_dir / child.name
            if dest.exists():
                continue
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
    return target_dir


class FlexiBuildExt(build_ext):
    RAPIDJSON_URL = "https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz"
    PUGIXML_URL = "https://github.com/zeux/pugixml/archive/refs/tags/v1.14.tar.gz"

    def run(self):
        rapidjson_include, pugixml_src_dir, pugixml_source = self._prepare_third_party()
        for ext in self.extensions:
            if ext.name == "flexipipe.pipeline_cpp":
                include_dirs = list(ext.include_dirs or [])
                include_dirs.extend([rapidjson_include, pugixml_src_dir])
                ext.include_dirs = include_dirs
                sources = list(ext.sources or [])
                if pugixml_source not in sources:
                    sources.append(pugixml_source)
                ext.sources = sources
        super().run()
        # Try to build flexitag_py extension after building other extensions
        self._build_flexitag_py()

    def _build_flexitag_py(self):
        """Build flexitag_py C++ extension using CMake if possible."""
        import os
        import subprocess

        # Default: soft-fail so CoNLL-U / non-flexitag CLI installs stay easy.
        # TEITOK quiet installs hard-require flexitag_py unless explicitly overridden.
        require_env = os.environ.get("FLEXIPIPE_REQUIRE_FLEXITAG_PY", "").strip().lower()
        quiet_install = os.environ.get("FLEXIPIPE_QUIET_INSTALL", "").lower() in (
            "1",
            "true",
            "yes",
        )
        noninteractive = os.environ.get("FLEXIPIPE_NONINTERACTIVE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if require_env in ("0", "false", "no"):
            require_flexitag = False
        elif require_env in ("1", "true", "yes"):
            require_flexitag = True
        else:
            require_flexitag = bool(quiet_install or noninteractive)
        def _log(msg: str) -> None:
            # Always emit to stderr so pip -q / quiet mode cannot hide failures.
            print(msg, file=sys.stderr)

        def _fail_or_skip(reason: str) -> None:
            _log(f"[flexipipe] {reason}")
            if require_flexitag:
                raise RuntimeError(
                    f"{reason}\n"
                    "flexitag_py is required for TEITOK/flexitag usage. "
                    "Install build deps (cmake, g++, libicu-dev) and rebuild, "
                    "or unset FLEXIPIPE_QUIET_INSTALL / set FLEXIPIPE_REQUIRE_FLEXITAG_PY=0 "
                    "to allow a Python-only install."
                )
            _log("[flexipipe] Continuing without flexitag_py (Python fallback will be used)")

        flexitag_dir = DEPS_BASE / "flexitag"
        if not flexitag_dir.exists() or not (flexitag_dir / "CMakeLists.txt").exists():
            _fail_or_skip("flexitag directory not found, cannot build flexitag_py")
            return

        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            _fail_or_skip("CMake not found, cannot build flexitag_py")
            return

        # Force the *build* interpreter (e.g. venv Python 3.10), not a stale
        # system pybind11 that may still point at /usr/include/python3.6m.
        import sysconfig

        python_include = sysconfig.get_path("include")
        cmake_cmd = [
            "cmake",
            "..",
            "-DFLEXITAG_BUILD_PYTHON=ON",
            # Shared libflexitag + copy it next to flexitag_py (with $ORIGIN RPATH).
            # Static linking previously dropped the module from some wheels when
            # -fPIC was missing; shipping both files is more reliable for pip.
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        if python_include:
            cmake_cmd.extend(
                [
                    f"-DPython_INCLUDE_DIR={python_include}",
                    f"-DPython3_INCLUDE_DIR={python_include}",
                    f"-DPYTHON_INCLUDE_DIR={python_include}",
                ]
            )
        try:
            import pybind11

            pybind11_dir = pybind11.get_cmake_dir()
            if pybind11_dir:
                cmake_cmd.append(f"-Dpybind11_DIR={pybind11_dir}")
                _log(f"[flexipipe] Using pybind11 CMake package at {pybind11_dir}")
                _log(f"[flexipipe] Using Python include dir {python_include}")
        except Exception as exc:
            _log(f"[flexipipe] Warning: could not resolve pybind11 CMake dir: {exc}")

        build_dir = flexitag_dir / "build"
        # Drop stale cmake cache that may still point at an old Python.
        cache_file = build_dir / "CMakeCache.txt"
        if cache_file.exists():
            try:
                cache_file.unlink()
            except OSError:
                pass
        build_dir.mkdir(exist_ok=True)

        try:
            _log("[flexipipe] Building flexitag_py C++ extension...")
            result = subprocess.run(
                cmake_cmd,
                cwd=build_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "").strip()
                hint = ""
                if "ICU" in details:
                    hint = (
                        "\nHint: install ICU development headers "
                        "(Ubuntu/Debian: sudo apt-get install libicu-dev)."
                    )
                elif "python3." in details and "include" in details:
                    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
                    hint = (
                        f"\nHint: CMake picked the wrong Python headers. "
                        f"Install matching headers (Ubuntu/Debian: sudo apt-get install "
                        f"python{pyver}-dev) and ensure pip's pybind11 is used "
                        f"(not a system pybind11 built for another Python)."
                    )
                _fail_or_skip(f"CMake configuration failed for flexitag_py:{hint}\n{details}")
                return

            result = subprocess.run(
                ["cmake", "--build", ".", "--target", "flexitag_py", "-j"],
                cwd=build_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "").strip()
                _fail_or_skip(f"flexitag_py build failed:\n{details}")
                return

            so_files = list(build_dir.rglob("flexitag_py*.so")) + list(
                build_dir.rglob("flexitag_py*.pyd")
            )
            if not so_files:
                _fail_or_skip(
                    "flexitag_py module not found after build "
                    f"(searched under {build_dir})"
                )
                return

            built_module = so_files[0]
            _log(f"[flexipipe] Successfully built flexitag_py: {built_module}")

            # If a shared libflexitag was still produced, ship it next to the module
            # and rely on $ORIGIN RPATH (see flexitag/CMakeLists.txt).
            shared_libs = (
                list(build_dir.rglob("libflexitag.so*"))
                + list(build_dir.rglob("libflexitag*.dylib"))
                + list(build_dir.rglob("flexitag*.dll"))
            )

            if hasattr(self, "build_lib") and self.build_lib:
                target_dir = Path(self.build_lib) / "flexipipe"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / built_module.name
                shutil.copy2(built_module, target_path)
                _log(f"[flexipipe] Copied flexitag_py to {target_path}")
                for lib in shared_libs:
                    if lib.is_symlink():
                        # Preserve soname symlinks when possible
                        dest = target_dir / lib.name
                        if dest.exists() or dest.is_symlink():
                            dest.unlink()
                        dest.symlink_to(os.readlink(lib))
                    elif lib.is_file():
                        shutil.copy2(lib, target_dir / lib.name)
                        _log(f"[flexipipe] Copied {lib.name} next to flexitag_py")

                flexitag_build_dir = Path(self.build_lib) / "flexitag" / "build"
                flexitag_build_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_module, flexitag_build_dir / built_module.name)
                for lib in shared_libs:
                    if lib.is_file() and not lib.is_symlink():
                        shutil.copy2(lib, flexitag_build_dir / lib.name)

                # Ensure the wheel will contain the extension (setuptools 80+
                # can omit unexpected non-.py files unless present in build_lib).
                if not target_path.is_file():
                    _fail_or_skip(
                        f"flexitag_py was built but not copied into build_lib: {target_path}"
                    )
                    return
                shipped = list(target_dir.glob("flexitag_py*")) + list(
                    target_dir.glob("libflexitag*")
                )
                _log(f"[flexipipe] Wheel payload in {target_dir}: {[p.name for p in shipped]}")
                if not any(p.name.startswith("libflexitag") for p in shipped):
                    _log(
                        "[flexipipe] Warning: libflexitag.* not found next to flexitag_py; "
                        "import may fail unless flexitag was linked statically"
                    )
        except RuntimeError:
            raise
        except Exception as e:
            _fail_or_skip(f"Error building flexitag_py: {e}")

    def _prepare_third_party(self):
        build_temp = Path(self.build_temp or "build")
        deps_dir = build_temp / "_deps"
        deps_dir.mkdir(parents=True, exist_ok=True)

        rapidjson_dir = _download_and_extract(self.RAPIDJSON_URL, deps_dir / "rapidjson")
        pugixml_dir = _download_and_extract(self.PUGIXML_URL, deps_dir / "pugixml")

        rapidjson_include = str((rapidjson_dir / "include").resolve())
        pugixml_src_dir = str((pugixml_dir / "src").resolve())
        pugixml_source = str((pugixml_dir / "src" / "pugixml.cpp").resolve())
        return rapidjson_include, pugixml_src_dir, pugixml_source


def install_wrapper_script():
    """Install the flexipipe wrapper script (interactive or via env vars)."""
    import os
    import shutil
    import stat
    import subprocess
    
    # Check for non-interactive mode (for automated installs like from PHP)
    noninteractive = os.environ.get("FLEXIPIPE_NONINTERACTIVE", "").lower() in ("1", "true", "yes")
    quiet_install = os.environ.get("FLEXIPIPE_QUIET_INSTALL", "").lower() in ("1", "true", "yes")
    install_wrapper = os.environ.get("FLEXIPIPE_INSTALL_WRAPPER", "").lower() in ("1", "true", "yes")
    wrapper_dir = os.environ.get("FLEXIPIPE_WRAPPER_DIR", "").strip()
    venv_from_env = os.environ.get("FLEXIPIPE_VENV_PATH", "").strip() or None

    # Non-interactive opt-in: install wrapper to FLEXIPIPE_WRAPPER_DIR or ~/bin
    if install_wrapper:
        script_source = Path(__file__).parent / "flexipipe" / "data" / "flexipipe_wrapper.sh"
        if not script_source.exists():
            script_source = Path(__file__).parent / "scripts" / "flexipipe"
        if not script_source.exists():
            print(f"[flexipipe] Wrapper script not found at {script_source}", file=sys.stderr)
            return
        install_dir = Path(wrapper_dir).expanduser().resolve() if wrapper_dir else Path.home() / "bin"
        install_path = install_dir / "flexipipe"
        install_path.parent.mkdir(parents=True, exist_ok=True)
        script_content = script_source.read_text()
        if venv_from_env:
            lines = script_content.split("\n")
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith("# FLEXIPIPE_REPO_PATH") or line.startswith("# Optional: Set path"):
                    insert_pos = i
                    break
            lines.insert(insert_pos, f'VENV_PATH="{venv_from_env}"')
            script_content = "\n".join(lines)
        try:
            install_path.write_text(script_content)
            install_path.chmod(install_path.stat().st_mode | stat.S_IEXEC)
            print(f"[flexipipe] Wrapper script installed to: {install_path}", file=sys.stderr)
        except Exception as e:
            print(f"[flexipipe] Error installing wrapper script: {e}", file=sys.stderr)
        return

    if noninteractive or quiet_install:
        # Skip interactive wrapper script installation; suggest installing later
        print("\n[flexipipe] To add the 'flexipipe' command to your PATH, run: python -m flexipipe install wrapper", file=sys.stderr)
        return

    print("\n" + "="*70)
    print("Flexipipe Wrapper Script Installation")
    print("="*70)
    print("\nThis will install a wrapper script that allows you to run")
    print("'flexipipe' directly instead of 'python -m flexipipe'.")
    print()
    
    # Ask about virtual environment
    use_venv = input("Do you want to use a virtual environment for flexipipe? [y/N]: ").strip().lower()
    venv_path = None
    if use_venv in ('y', 'yes'):
        venv_path = input("Enter the path to your virtual environment (or press Enter to skip): ").strip()
        if not venv_path:
            venv_path = None
        elif not Path(venv_path).exists():
            print(f"Warning: Virtual environment path does not exist: {venv_path}")
            use_venv = input("Continue anyway? [y/N]: ").strip().lower()
            if use_venv not in ('y', 'yes'):
                venv_path = None
    
    # Ask where to install the script
    print("\nWhere would you like to install the wrapper script?")
    print("  1. /usr/local/bin (system-wide, requires sudo)")
    print("  2. ~/bin (user-local, add to PATH)")
    print("  3. Custom location")
    print("  4. Skip installation")
    
    choice = input("Enter choice [1-4] (default: 4): ").strip() or "4"
    
    if choice == "4":
        print("Skipping wrapper script installation.")
        print("You can install it later with: python -m flexipipe install wrapper")
        return
    
    # Prefer script bundled in package (single source of truth); fall back to repo scripts/
    script_source = Path(__file__).parent / "flexipipe" / "data" / "flexipipe_wrapper.sh"
    if not script_source.exists():
        script_source = Path(__file__).parent / "scripts" / "flexipipe"
    if not script_source.exists():
        print(f"Error: Wrapper script not found (tried flexipipe/data/flexipipe_wrapper.sh and scripts/flexipipe)")
        return
    
    if choice == "1":
        install_path = Path("/usr/local/bin/flexipipe")
        use_sudo = True
    elif choice == "2":
        install_path = Path.home() / "bin" / "flexipipe"
        install_path.parent.mkdir(parents=True, exist_ok=True)
        use_sudo = False
    elif choice == "3":
        custom_path = input("Enter installation path: ").strip()
        if not custom_path:
            print("No path provided, skipping installation.")
            return
        install_path = Path(custom_path).expanduser().resolve()
        install_path.parent.mkdir(parents=True, exist_ok=True)
        use_sudo = False
    else:
        print("Invalid choice, skipping installation.")
        print("You can install it later with: python -m flexipipe install wrapper")
        return
    
    # Read the script and customize it
    script_content = script_source.read_text()
    
    # Add configuration at the top if venv_path is set
    if venv_path:
        venv_line = f'VENV_PATH="{venv_path}"'
        # Insert after the configuration comments
        lines = script_content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('# FLEXIPIPE_REPO_PATH') or line.startswith('# Optional: Set path'):
                insert_pos = i
                break
        lines.insert(insert_pos, venv_line)
        script_content = '\n'.join(lines)
    
    # Write to installation location
    try:
        if use_sudo:
            # Write to temp file first, then copy with sudo
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as tmp:
                tmp.write(script_content)
                tmp_path = tmp.name
            result = subprocess.run(
                ['sudo', 'cp', tmp_path, str(install_path)],
                capture_output=True,
                text=True
            )
            Path(tmp_path).unlink()
            if result.returncode != 0:
                print(f"Error installing script: {result.stderr}")
                return
            subprocess.run(['sudo', 'chmod', '+x', str(install_path)])
        else:
            install_path.write_text(script_content)
            install_path.chmod(install_path.stat().st_mode | stat.S_IEXEC)
        
        print(f"\n✓ Wrapper script installed to: {install_path}")
        if choice == "2":
            print(f"\nNote: Make sure ~/bin is in your PATH.")
            print("Add this to your ~/.bashrc or ~/.zshrc:")
            print("  export PATH=\"$HOME/bin:$PATH\"")
        print()
    except Exception as e:
        print(f"Error installing wrapper script: {e}")


class FlexiInstall(install):
    """Custom install command that prompts for wrapper script installation."""
    
    def run(self):
        # Run the standard install
        install.run(self)
        
        # After installation, optionally install wrapper script
        # - If FLEXIPIPE_INSTALL_WRAPPER=1: install non-interactively to FLEXIPIPE_WRAPPER_DIR or ~/bin
        # - Else if running interactively: prompt (unless FLEXIPIPE_NONINTERACTIVE or FLEXIPIPE_QUIET_INSTALL)
        import os
        install_wrapper = os.environ.get("FLEXIPIPE_INSTALL_WRAPPER", "").lower() in ("1", "true", "yes")
        noninteractive = os.environ.get("FLEXIPIPE_NONINTERACTIVE", "").lower() in ("1", "true", "yes")
        quiet_install = os.environ.get("FLEXIPIPE_QUIET_INSTALL", "").lower() in ("1", "true", "yes")
        prompt_for_wrapper = sys.stdin.isatty() and not noninteractive and not quiet_install

        if install_wrapper or prompt_for_wrapper:
            try:
                install_wrapper_script()
            except KeyboardInterrupt:
                print("\n\nWrapper script installation cancelled.")
            except Exception as e:
                print(f"\nError during wrapper script installation: {e}")
                print("You can install it later with: python -m flexipipe install wrapper")
        else:
            # No TTY (e.g. pip install from git in some environments): suggest wrapper
            print("\n[flexipipe] To add the 'flexipipe' command to your PATH, run: python -m flexipipe install wrapper", file=sys.stderr)


setup(
    name="flexipipe",
    version=version,
    description="Flexible transformer-based NLP pipeline for tagging, parsing, and normalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/flexipipe",
    packages=find_packages(),
    package_data={
        "flexipipe": [
            "flexitag_py*.so",
            "flexitag_py*.pyd",
            "libflexitag.so*",
            "libflexitag*.dylib",
            "data/flexipipe_wrapper.sh",
            "data/flexipipe_launcher.c",
        ],
        "": [
            "flexitag/build/flexitag_py*.so",
            "flexitag/build/flexitag_py*.pyd",
            "flexitag/build/libflexitag.so*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=BASE_REQUIREMENTS,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "flexipipe=flexipipe.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, universal-dependencies, tagging, parsing, bert, transformers, normalization",
    ext_modules=[
        Extension(
            "flexipipe.viterbi_cpp",
            [
                "src/viterbi_cpp.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++17",
                "-O3",  # Optimize for speed
                "-Wall",
            ] if sys.platform != "win32" else [
                "/std:c++17",
                "/O2",  # Optimize for speed on Windows
            ],
        ),
        Extension(
            "flexipipe.pipeline_cpp",
            [
                "src/pipeline_pybind.cpp",
                "src/vocab_loader.cpp",
                "src/tokenizer.cpp",
                "src/normalizer.cpp",
                "src/contractions.cpp",
                "src/viterbi_optimized.cpp",
                "src/io_conllu.cpp",
                "src/io_teitok.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
                "src",
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++17",
                "-O3",
                "-Wall",
            ] if sys.platform != "win32" else [
                "/std:c++17",
                "/O2",
            ],
        ),
    ],
    cmdclass={
        "build_ext": FlexiBuildExt,
        "install": FlexiInstall,
    },
    zip_safe=False,
)

