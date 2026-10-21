"""Installing a model from the registry, and where flexipipe-hub is (for publishing).

Models are had from where they are hosted — the LINDAT repository, the sender's own server, Hugging Face,
or flexipipe-hub when the sender cannot host it — always by the plain URL the registry gives. The hub's
address is needed only to publish (TEITOK's publish page links to its deposit form): the config's
``model_hub_url``, else ``FLEXIPIPE_MODEL_HUB_URL``, else ``hub.json`` at the root of the registry repository
(``{"url": "https://…/flexipipe-hub"}``, next to ``registries/``) — so the hub can move without a new flexipipe.

``install_model`` downloads an entry's package, checks its sha256 (when the entry has one), unpacks it
safely into the backend's model folder (``<models>/<backend>/<model>/``) and notes where it came from
(``installed-from.json``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin

HUB_FILE = "hub.json"
INSTALLED_FROM = "installed-from.json"
_HUB_CACHE_KEY = "model_hub_url"
_HUB_CACHE_SECONDS = 60 * 60 * 24

# <backend>/<model>[@v<N>], as `flexipipe install model` takes it
_SPEC = re.compile(r"^([a-z0-9_-]+)/([A-Za-z0-9][A-Za-z0-9._-]*)(?:@v(\d+))?$")


def parse_spec(spec: str) -> Optional[Tuple[str, str, Optional[int]]]:
    """``<backend>/<model>[@v<N>]`` → (backend, model, version or None); None for anything else."""
    m = _SPEC.match((spec or "").strip())
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)) if m.group(3) else None


def _hub_file_location() -> Optional[str]:
    """Where the registry repository's hub.json is: next to its registries/ folder."""
    from .model_registry import get_registry_url

    base = get_registry_url(None)
    if not base:
        return None
    if base.startswith("file://"):
        root = Path(base[7:])
        if root.name == "registries":
            root = root.parent
        return f"file://{root / HUB_FILE}"
    base = base.rstrip("/")
    if base.endswith("/registries"):
        base = base[: -len("/registries")]
    return urljoin(base + "/", HUB_FILE)


def _read_json_location(location: str) -> Optional[dict]:
    try:
        if location.startswith("file://"):
            return json.loads(Path(location[7:]).read_text(encoding="utf-8"))
        import requests

        r = requests.get(location, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def hub_url(*, refresh: bool = False) -> Optional[str]:
    """The hub's address (no trailing slash), or None when nothing says where it is: for publishing."""
    from .model_storage import read_config, read_model_cache_entry, write_model_cache_entry

    given = read_config().get("model_hub_url") or os.environ.get("FLEXIPIPE_MODEL_HUB_URL")
    if given:
        return str(given).rstrip("/")
    location = _hub_file_location()
    if not location:
        return None
    key = f"{_HUB_CACHE_KEY}:{location}"
    if not refresh and not location.startswith("file://"):
        cached = read_model_cache_entry(key, max_age_seconds=_HUB_CACHE_SECONDS)
        if cached and cached.get("url"):
            return str(cached["url"]).rstrip("/")
    data = _read_json_location(location)
    url = (data or {}).get("url")
    if not url:
        return None
    if not location.startswith("file://"):
        write_model_cache_entry(key, {"url": url})
    return str(url).rstrip("/")


# --- the registry entry ------------------------------------------------------------------------

def find_entry(backend: str, model: str, *, refresh: bool = False) -> Optional[Dict[str, Any]]:
    """The registry entry of backend/model (any source), or None."""
    from .model_registry import get_remote_models_for_backend

    for entry in get_remote_models_for_backend(backend, use_cache=True, refresh_cache=refresh):
        if entry.get("model") == model:
            return entry
    return None


def version_fields(entry: Dict[str, Any], version: Optional[int]) -> Dict[str, Any]:
    """The fields of one version of an entry (the current one when version is None or current)."""
    current = int(entry.get("version") or 1)
    if version is None or version == current:
        return entry
    for old in entry.get("versions") or []:
        if int(old.get("version") or 0) == version:
            return old
    known = [current] + [int(v.get("version") or 0) for v in entry.get("versions") or []]
    raise ValueError(f"{entry.get('model')} has no version {version} (versions: {', '.join(map(str, sorted(known)))})")


def package_address(fields: Dict[str, Any]) -> Optional[str]:
    """Where a version's files are: its download_url, else its url (a self-hosted model)."""
    return fields.get("download_url") or fields.get("url") or None


# --- downloading and unpacking -----------------------------------------------------------------

def _download(url: str, target: Path, *, verbose: bool = False) -> str:
    """Download url into target; returns the file's sha256."""
    digest = hashlib.sha256()
    if url.startswith("file://"):
        with open(url[7:], "rb") as src, open(target, "wb") as out:
            for block in iter(lambda: src.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
        return digest.hexdigest()
    import requests

    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{url}: the server answered {r.status_code}")
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, 0.0
        with open(target, "wb") as out:
            for block in r.iter_content(1 << 20):
                digest.update(block)
                out.write(block)
                done += len(block)
                if verbose and total > 20 * 1048576 and time.time() - shown > 2:  # large files only
                    print(f"[flexipipe] … {done / 1048576:.1f} of {total / 1048576:.1f} MB")
                    shown = time.time()
    return digest.hexdigest()


def _safe_member_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"the package has a file outside its folder: {name}")
    return path


def _unpack(archive: Path, target: Path) -> None:
    """Unpack a .tar(.gz) or .zip into target; refuses absolute paths, '..', links and devices."""
    root = target.resolve()
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for m in members:
                _safe_member_path(root, m.name)
                if not (m.isfile() or m.isdir()):
                    raise RuntimeError(f"the package has something that is not a file or folder: {m.name}")
            if hasattr(tarfile, "data_filter"):
                tar.extractall(root, members=members, filter="data")
            else:  # Python without extraction filters: the members were checked above
                tar.extractall(root, members=members)
        return
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                _safe_member_path(root, name)
            z.extractall(root)
        return
    raise RuntimeError("the download is not a .tar.gz or .zip package")


def _model_root(folder: Path, backend: str) -> Path:
    """The folder that holds the model: past a single top folder (a folder deposited as such); for
    flexitag the shallowest one with model_vocab.json."""
    if backend == "flexitag":
        found = sorted(folder.rglob("model_vocab.json"), key=lambda p: len(p.parts))
        if not found:
            raise RuntimeError("the package has no model_vocab.json: it is not a flexitag model")
        return found[0].parent
    entries = [p for p in folder.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return folder


def _current_version() -> str:
    from .__main__ import _VERSION

    return _VERSION


def _made_with_newer(folder: Path) -> Optional[str]:
    """The flexipipe version the model's manifest names, when it is newer than this one."""
    try:
        made = json.loads((folder / "flexipipe-model.json").read_text(encoding="utf-8")).get("flexipipe_version")
    except Exception:
        return None
    as_tuple = lambda v: tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])
    try:
        return made if made and as_tuple(made) > as_tuple(_current_version()) else None
    except Exception:
        return None


def installed_from(backend: str, model: str) -> Optional[Dict[str, Any]]:
    """What installed-from.json says about an installed model (None when not installed this way)."""
    from .model_storage import get_backend_models_dir

    f = get_backend_models_dir(backend, create=False) / model / INSTALLED_FROM
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _into_place(work: Path, root: Path, target: Path, record: Dict[str, Any], verbose: bool) -> None:
    """The unpacked model (root) to target, with installed-from.json: the old one aside first, back if the
    move fails."""
    newer = _made_with_newer(root)
    if newer and verbose:
        print(f"[flexipipe] Note: the model was made with flexipipe {newer}, this is {_current_version()}: "
              "if it does not load, update flexipipe ('flexipipe install update')")
    (root / INSTALLED_FROM).write_text(json.dumps(dict(record, installed=time.strftime("%Y-%m-%dT%H:%M:%S")),
                                                  indent=2) + "\n", encoding="utf-8")
    old = None
    if target.exists():
        old = work / "previous"
        target.rename(old)
    try:
        root.rename(target)
    except Exception:
        if old is not None:
            old.rename(target)
        raise


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_ARCHIVES = (".tar.gz", ".tgz", ".tar", ".zip")


def looks_like_package(spec: str) -> bool:
    """A package file or a model folder on disk, rather than <backend>/<model> from the registry."""
    return spec.lower().endswith(_ARCHIVES) or Path(spec).expanduser().exists()


def _package_manifest(folder: Path) -> Tuple[Optional[Path], dict]:
    found = sorted(folder.rglob("flexipipe-model.json"), key=lambda p: len(p.parts))
    if not found:
        return None, {}
    try:
        return found[0], json.loads(found[0].read_text(encoding="utf-8"))
    except Exception:
        return found[0], {}


def install_package(
    path,
    *,
    name: Optional[str] = None,
    backend: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Install a model from a package file (.tar.gz / .zip, e.g. made by `flexipipe publish --pack`) or a
    model folder — one someone sent, or from another server: no registry, no hub. Returns its folder."""
    from .model_pack import public_name
    from .model_storage import get_backend_models_dir, get_flexipipe_models_dir

    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise ValueError(f"{path}: there is no such file or folder")
    work = Path(tempfile.mkdtemp(prefix=".install-package-", dir=get_flexipipe_models_dir(create=True)))
    try:
        unpacked = work / "unpacked"
        digest = None
        if src.is_dir():
            shutil.copytree(src, unpacked, symlinks=False, ignore=shutil.ignore_patterns(INSTALLED_FROM, ".*"))
        else:
            unpacked.mkdir()
            _unpack(src, unpacked)
            digest = hashlib.sha256(src.read_bytes()).hexdigest() if src.stat().st_size < (1 << 31) else None
        manifest_file, manifest = _package_manifest(unpacked)
        backend = (backend or manifest.get("backend") or "").lower()
        if not backend:
            if next(unpacked.rglob("model_vocab.json"), None) is not None:
                backend = "flexitag"
            else:
                raise ValueError("the package does not say which backend it is for (no flexipipe-model.json): give --backend")
        root = manifest_file.parent if manifest_file is not None else _model_root(unpacked, backend)
        if backend == "flexitag":
            root = _model_root(root, backend)
        if not name:
            # the corpus it was trained on and when: "ode-2026-10-19"
            corpus = public_name(str(manifest.get("trained_on") or "")) or ""
            when = (manifest.get("created") or "")[:10]
            folder = root.name if root != unpacked else src.name.split(".")[0]
            name = "-".join(x for x in (corpus, when or folder) if x) or folder
            name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
        if not _NAME.match(name or ""):
            raise ValueError(f"{name!r} cannot be a model name (letters, digits, . _ -): give --name")
        target = get_backend_models_dir(backend, create=True) / name
        if target.exists() and not force:
            raise FileExistsError(f"{target} exists: use --force to replace it, or --name for another name")
        final_work = target.parent / f".install-{name}"
        shutil.rmtree(final_work, ignore_errors=True)
        root.rename(final_work)  # the same disk: the models folder
        _into_place(work, final_work, target, {"backend": backend, "model": name, "source": "package",
                                               "file": src.name, "sha256": digest}, verbose)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if verbose:
        print(f"[flexipipe] ✓ Installed the {backend} model {name} from {src.name} in {target}")
        print(f"[flexipipe] Use it with: flexipipe process --backend {backend} --model {name} …")
    return target


def install_model(
    backend: str,
    model: str,
    *,
    version: Optional[int] = None,
    force: bool = False,
    refresh: bool = False,
    verbose: bool = True,
) -> Path:
    """Install backend/model from its registry entry. Returns the model's folder."""
    from .model_storage import get_backend_models_dir

    # installing reads the registry fresh (an address may have changed: e.g. a model moved from the hub to
    # the repository); the cached copy only when that fails (offline)
    entry = None
    try:
        entry = find_entry(backend, model, refresh=True)
    except Exception:
        pass
    if entry is None:
        entry = find_entry(backend, model, refresh=False)
    if entry is None:
        raise ValueError(f"there is no {backend} model {model!r} in the registry")
    fields = version_fields(entry, version)
    number = int(fields.get("version") or entry.get("version") or 1)
    address = package_address(fields)
    if not address:
        raise ValueError(f"the registry has no address for {backend}/{model}'s files")
    url = address
    expected = (fields.get("sha256") or "").lower() or None

    models_dir = get_backend_models_dir(backend, create=True)
    target = models_dir / model
    before = installed_from(backend, model)
    if target.exists() and not force:
        if before and before.get("version") == number and before.get("sha256") == expected:
            if verbose:
                print(f"[flexipipe] {backend}/{model} version {number} is already installed ({target})")
            return target
        raise FileExistsError(f"{target} exists: use --force to replace it")

    work = Path(tempfile.mkdtemp(prefix=f".install-{model}-", dir=models_dir))
    try:
        archive = work / "package"
        if verbose:
            print(f"[flexipipe] Downloading {backend}/{model} version {number} from {url}")
        got = _download(url, archive, verbose=verbose)
        if expected and got != expected:
            raise RuntimeError(f"the download's sha256 is {got}, the registry says {expected}: not installed")
        unpacked = work / "unpacked"
        unpacked.mkdir()
        _unpack(archive, unpacked)
        root = _model_root(unpacked, backend)
        _into_place(work, root, target, {
            "backend": backend, "model": model, "version": number,
            "address": address, "url": url, "sha256": got, "checked": bool(expected),
        }, verbose)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if verbose:
        note = "" if expected else " (the registry gives no sha256: not checked)"
        print(f"[flexipipe] ✓ Installed {backend}/{model} version {number} in {target}{note}")
    return target


def run_install_models(specs: List[str], *, force: bool = False, name: Optional[str] = None,
                       backend: Optional[str] = None) -> int:
    """``flexipipe install model <backend>/<model>[@v<N>] …`` (from the registry) or
    ``flexipipe install model FILE.tar.gz|FOLDER [--name N] [--backend B]`` (a package)."""
    if not specs:
        print("[flexipipe] Name the model: flexipipe install model <backend>/<model> (a version: <backend>/<model>@v<N>), "
              "or a package: flexipipe install model FILE.tar.gz")
        return 1
    if name and len(specs) > 1:
        print("[flexipipe] --name goes with one package at a time")
        return 1
    status = 0
    for spec in specs:
        try:
            if looks_like_package(spec):
                install_package(spec, name=name, backend=backend, force=force)
                continue
            parsed = parse_spec(spec)
            if not parsed:
                print(f"[flexipipe] {spec}: expected <backend>/<model>, <backend>/<model>@v<N> (e.g. flexitag/ode-osp@v2), "
                      "or a package file / model folder")
                status = 1
                continue
            install_model(parsed[0], parsed[1], version=parsed[2], force=force)
        except Exception as e:  # one model's problem does not stop the others
            print(f"[flexipipe] {spec}: {e}")
            status = 1
    return status
