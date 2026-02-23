# Installation

## Requirements

1. **Python 3.8+** (Python 3.11 recommended)
2. Install:
   ```bash
   python -m pip install -r requirements.txt
   ```
   Note: `torch>=2.6.0` is required for the transformers backend (security). The transformers backend will check this at runtime.
3. Install backend extras when needed (keeps the default install small):
   ```bash
   pip install "flexipipe[spacy]"
   pip install "flexipipe[stanza]"
   pip install "flexipipe[flair]"
   pip install "flexipipe[transformers]"
   pip install "flexipipe[udkanbun]"
   pip install "flexipipe[all]"
   # Or via flexipipe:
   flexipipe install udkanbun
   flexipipe install udapi   # For HTML and LaTeX output
   ```
4. Build the native flexitag modules if needed (see `README_CPP.md`). C++ deps (pugixml, rapidjson) are fetched via CMake.

After installation, run:

```bash
python -m flexipipe config --wizard
```

---

## Non-interactive installation

For CI/CD or scripts, set `FLEXIPIPE_NONINTERACTIVE` or `FLEXIPIPE_QUIET_INSTALL` to skip prompts:

```bash
FLEXIPIPE_NONINTERACTIVE=1 pip install git+https://github.com/ufal/flexipipe.git
```

---

## Optional: Install the `flexipipe` wrapper script

Run the CLI as `flexipipe` (instead of `python -m flexipipe`) by installing a launcher.

* **During pip install**: If you run `pip install` interactively (without the non-interactive env vars), you will be asked whether to install the wrapper and where.
* **Anytime after install**: Run `python -m flexipipe install wrapper` (or `flexipipe install wrapper` if already on PATH). You will be prompted for the install location, or pass a path:
  ```bash
  flexipipe install wrapper --path ~/bin
  flexipipe install wrapper --path /usr/local/bin   # may need sudo
  ```
  **C launcher (faster startup):** The C launcher source is in the package. When you run `flexipipe install wrapper`, the command builds it (using `cc`) and installs the binary when possible; otherwise the shell script is installed. The launcher respects `VENV_PATH`, `VIRTUAL_ENV`, and `FLEXIPIPE_REPO_PATH` (for development installs).
* **Non-interactive**: Set `FLEXIPIPE_INSTALL_WRAPPER=1` (and optionally `FLEXIPIPE_WRAPPER_DIR` or `FLEXIPIPE_VENV_PATH`) when running `pip install`.

Upgrade flexipipe: **`flexipipe install update`**. Ensure the wrapper install directory is on your `PATH`.

---

## Optional extras behaviour

* Auto-install extras on first use: `flexipipe config --set-auto-install-extras true`
* Disable prompts (batch): `flexipipe config --set-prompt-install-extras false`

---

## Language detection

Pluggable language identification. Default detector: `fasttext`. Switch or disable:

```bash
flexipipe config --set-language-detector fasttext
flexipipe config --set-language-detector none
```

The wizard (`flexipipe config --wizard`) asks which detector to use. Download fastText model: `flexipipe config --download-language-model`.
