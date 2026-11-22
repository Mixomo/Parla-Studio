"""Projects: the whole state of a tab, saved non-destructively in one `.parla` file.

A project is light and referential: a small ZIP with `project.json` (settings, edits, transcripts, analysis...) and
`arrays/` (small numeric analysis only). The original audio files are never copied nor modified: the project keeps
their paths (absolute and relative to the project), so it opens again as long as they stay where they were. When they
moved, `resolve_files` finds them next to the project, or the user relinks them and `remap_paths` updates every
reference inside the state.

Derived audio (separated voice, processed results...) is NOT inside the project: it goes to the sidecar folder
`<name>.parla-media/` next to it, as 24-bit FLAC files named by their content (saving again does not rewrite what did
not change, and files no longer used are removed). If that folder is deleted, the project still opens; only that
derived audio is missing and has to be generated again (separate / process).

Serialization is explicit and safe (no pickle): only JSON types, numpy arrays (`allow_pickle=False`), paths and the
app's own dataclasses (from a fixed list of modules) can be stored.
"""
from __future__ import annotations

import dataclasses
import datetime
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from .leveler import Ctx

EXT = ".parla"
FORMAT = 2              # 2: derived audio in the sidecar media folder (1: inside the ZIP, still readable)
MEDIA_SUFFIX = "-media"
KINDS = ("leveler", "dataset", "speakers", "censor")
_SAFE_MODULES = ("parla_studio.core.dataset", "parla_studio.core.speakers", "parla_studio.core.censor",
                 "parla_studio.core.models", "parla_studio.core.leveler", "parla_studio.core.smart",
                 "parla_studio.core.aligner")
_FLAC_MIN = 1_000_000   # long float audio is stored as 24-bit FLAC (lossless enough, ~3x smaller)


class ProjectError(Exception):
    """The file is not a valid project for this app."""


# ---------------------------------------------------------------- encoding
def _encode(obj, arrays: dict):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = obj
        return {"__nd__": key}
    if isinstance(obj, Path):
        return {"__path__": str(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        if cls.__module__ not in _SAFE_MODULES:
            raise TypeError(f"Cannot save {cls.__module__}.{cls.__name__}")
        return {"__dc__": f"{cls.__module__}:{cls.__qualname__}", "f": _encode(vars(obj), arrays)}
    if isinstance(obj, dict):
        if all(isinstance(k, str) and not k.startswith("__") for k in obj):
            return {k: _encode(v, arrays) for k, v in obj.items()}
        return {"__map__": [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, (list, set)):
        return [_encode(v, arrays) for v in obj]
    raise TypeError(f"Cannot save a value of type {type(obj).__name__}")


def _decode(obj, arrays: dict):
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if "__nd__" in obj:
        return arrays[obj["__nd__"]]
    if "__path__" in obj:
        return Path(obj["__path__"])
    if "__tuple__" in obj:
        return tuple(_decode(v, arrays) for v in obj["__tuple__"])
    if "__map__" in obj:
        return {_hashable(_decode(k, arrays)): _decode(v, arrays) for k, v in obj["__map__"]}
    if "__dc__" in obj:
        import importlib
        mod, name = obj["__dc__"].split(":")
        if mod not in _SAFE_MODULES:
            raise ProjectError(f"Unknown data type in the project: {obj['__dc__']}")
        cls = getattr(importlib.import_module(mod), name, None)
        if cls is None or not dataclasses.is_dataclass(cls):
            raise ProjectError(f"Unknown data type in the project: {obj['__dc__']}")
        inst = object.__new__(cls)  # the exact saved state, without re-running __post_init__
        inst.__dict__.update(_decode(obj["f"], arrays))
        return inst
    return {k: _decode(v, arrays) for k, v in obj.items()}


def _hashable(k):
    return tuple(k) if isinstance(k, list) else k


def _is_audio(a: np.ndarray) -> bool:
    """Long mono/stereo float audio (derived audio: goes to the media folder). Analysis arrays stay in the ZIP."""
    if a.dtype.kind != "f" or a.size < _FLAC_MIN or a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] > 2):
        return False
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    return np.isfinite(peak) and peak <= 1.0


# ---------------------------------------------------------------- files and relinking
def file_entry(path: str, project_path: Path) -> dict:
    p = Path(path)
    try:
        rel = os.path.relpath(p, project_path.parent)
    except ValueError:  # another drive
        rel = None
    try:
        size = p.stat().st_size
    except OSError:
        size = None
    return {"path": str(path), "rel": rel, "name": p.name, "size": size}


def resolve_files(entries: list[dict], project_path: Path) -> tuple[dict[str, str], list[dict]]:
    """Original path → where it is now. Tries the saved path, the path relative to the project and the project
    folder. Returns (found, missing)."""
    found, missing = {}, []
    base = project_path.parent
    for e in entries:
        cands = [Path(e["path"])]
        if e.get("rel"):
            cands.append(base / e["rel"])
        cands.append(base / e["name"])
        hit = next((c for c in cands if c.is_file()), None)
        if hit is None:
            missing.append(e)
        else:
            found[e["path"]] = str(hit) if hit != Path(e["path"]) else e["path"]
    return found, missing


def search_folder(folder: Path, names: set[str]) -> dict[str, str]:
    """Look for files by name under `folder` (recursive). name → first path found."""
    out: dict[str, str] = {}
    low = {n.lower(): n for n in names}
    for root, _dirs, files in os.walk(folder):
        for f in files:
            n = low.get(f.lower())
            if n and n not in out:
                out[n] = str(Path(root) / f)
        if len(out) == len(names):
            break
    return out


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


def remap_paths(tree, mapping: dict[str, str]):
    """Replace every reference to the old paths (strings, saved Path objects and map keys) in the raw state."""
    if not mapping:
        return tree
    norm = {_norm(k): v for k, v in mapping.items() if k != v}
    if not norm:
        return tree

    def fix_str(s: str) -> str:
        if ("/" in s or "\\" in s) and len(s) < 1024:
            return norm.get(_norm(s), s)
        return s

    def walk(o):
        if isinstance(o, str):
            return fix_str(o)
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, dict):
            return {(fix_str(k) if isinstance(k, str) else k): walk(v) for k, v in o.items()}
        return o

    return walk(tree)


def media_dir(project_path: Path) -> Path:
    """Sidecar folder with the derived audio of a project: `name.parla` → `name.parla-media/`."""
    p = Path(project_path)
    return p.with_name(p.name + MEDIA_SUFFIX)


def _digest(a: np.ndarray) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=12)
    h.update(f"{a.shape}{a.dtype}".encode())
    h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


# ---------------------------------------------------------------- reading and writing
def save(path: Path, kind: str, state: dict, files: list[str], ctx: Ctx | None = None) -> Path:
    """Write the project (atomically: a failed save never breaks the previous file). Derived audio goes to the
    sidecar media folder, the .parla file keeps only the state."""
    ctx = ctx or Ctx()
    path = Path(path)
    if path.suffix.lower() != EXT:
        path = path.with_suffix(EXT)
    arrays: dict[str, np.ndarray] = {}
    doc = {"app": "Parla Studio", "format": FORMAT, "kind": kind,
           "saved": datetime.datetime.now().isoformat(timespec="seconds"),
           "files": [file_entry(f, path) for f in files], "state": _encode(state, arrays)}
    from . import audio_io
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    media = media_dir(path)
    used: set[str] = set()
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            meta = {}
            for k, (key, a) in enumerate(arrays.items()):
                ctx.check()
                if _is_audio(a):
                    # derived audio: outside the project, named by content (unchanged audio is not rewritten)
                    name = f"{_digest(a)}.flac"
                    used.add(name)
                    f = media / name
                    if not f.is_file():
                        media.mkdir(parents=True, exist_ok=True)
                        part = f.with_name(name + ".part")
                        x = a if a.ndim == 2 else a[:, None]
                        sf.write(str(part), audio_io.quantize24(x), 48000, format="FLAC", subtype="PCM_24")
                        part.replace(f)
                    meta[key] = {"media": name, "shape": list(a.shape), "dtype": str(a.dtype)}
                else:
                    buf = io.BytesIO()
                    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
                    z.writestr(f"arrays/{key}.npy", buf.getvalue())
                ctx.progress(0.95 * (k + 1) / max(1, len(arrays)))
            doc["arrays"] = meta
            doc["media"] = media.name if used else None
            z.writestr("project.json", json.dumps(doc, ensure_ascii=False, indent=1))
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    # derived audio no longer used by this project
    if media.is_dir():
        for f in media.glob("*.flac"):
            if f.name not in used:
                try:
                    f.unlink()
                except OSError:
                    pass
        try:
            media.rmdir()   # only if empty
        except OSError:
            pass
    ctx.progress(1.0)
    return path


def read_header(path: Path) -> dict:
    """project.json without the arrays (kind, files, date): fast, for the project manager."""
    try:
        with zipfile.ZipFile(path) as z:
            doc = json.loads(z.read("project.json").decode("utf-8"))
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as e:
        raise ProjectError(f"Not a Parla Studio project: {Path(path).name}") from e
    if doc.get("app") != "Parla Studio" or doc.get("kind") not in KINDS:
        raise ProjectError(f"Not a Parla Studio project: {Path(path).name}")
    if int(doc.get("format", 0)) > FORMAT:
        raise ProjectError("The project was saved by a newer version of the app")
    return doc


def load(path: Path, mapping: dict[str, str] | None = None, ctx: Ctx | None = None) -> tuple[str, dict]:
    """(kind, state) with the arrays loaded and the file references remapped to where the files are now.

    Derived audio missing from the media folder loads as None (that part has to be generated again); the
    names of what was missing are in state["__missing_media__"]."""
    import soundfile as sf
    ctx = ctx or Ctx()
    path = Path(path)
    doc = read_header(path)
    arrays: dict = {}
    meta = doc.get("arrays", {})
    media = media_dir(path)
    missing = []
    external = {k: m for k, m in meta.items() if m.get("media")}
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.startswith("arrays/")]
        total = max(1, len(names) + len(external))
        for k, n in enumerate(names):
            ctx.check()
            key = Path(n).stem
            data = z.read(n)
            if n.endswith(".flac"):   # format 1: derived audio inside the ZIP
                x, _sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                m = meta.get(key, {})
                arrays[key] = np.ascontiguousarray(x.reshape(m["shape"]) if "shape" in m else x)
            else:
                arrays[key] = np.load(io.BytesIO(data), allow_pickle=False)
            ctx.progress(0.9 * (k + 1) / total)
    for k, (key, m) in enumerate(external.items()):
        ctx.check()
        f = media / m["media"]
        if f.is_file():
            x, _sr = sf.read(str(f), dtype="float32", always_2d=True)
            arrays[key] = np.ascontiguousarray(x.reshape(m["shape"]))
        else:
            arrays[key] = None
            missing.append(m["media"])
        ctx.progress(0.9 * (len(names) + k + 1) / total)
    state = remap_paths(doc["state"], mapping or {})
    out = _decode(state, arrays)
    if isinstance(out, dict) and missing:
        out["__missing_media__"] = missing
    ctx.progress(1.0)
    return doc["kind"], out


# ---------------------------------------------------------------- recent projects
_HEADERS: dict[str, tuple[float, str | None]] = {}   # path -> (mtime, kind or None if not a project)


def kind_folder(root: Path, kind: str) -> Path:
    """Subfolder of the projects folder for one tab: projects/leveler, projects/dataset, ..."""
    return Path(root) / kind


def _kind_of(p: Path) -> str | None:
    key = str(p.resolve())
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    hit = _HEADERS.get(key)
    if hit is None or hit[0] != mt:
        try:
            k = read_header(p)["kind"]
        except (ProjectError, OSError):
            k = None
        hit = _HEADERS[key] = (mt, k)
    return hit[1]


def organize(root: Path) -> list[tuple[Path, Path]]:
    """Move every project of the projects folder into the subfolder of its tab (keeping any inner subfolder,
    e.g. projects/presets/x.parla of the Leveler → projects/leveler/presets/x.parla), together with its media
    folder. Never overwrites: a name already taken gets " (2)". Returns [(old, new)]."""
    root = Path(root)
    if not root.is_dir():
        return []
    moved = []
    for p in sorted(root.rglob(f"*{EXT}")):
        if not p.is_file() or any(part.endswith(MEDIA_SUFFIX) for part in p.parts):
            continue
        kind = _kind_of(p)
        if kind is None:
            continue
        rel = p.relative_to(root)
        if rel.parts[0] == kind and len(rel.parts) > 1:
            continue                              # already in its tab's folder
        if rel.parts[0] in KINDS and len(rel.parts) > 1:
            rel = Path(*rel.parts[1:])            # in another tab's folder
        dst = kind_folder(root, kind) / rel
        n = 2
        while dst.exists() or media_dir(dst).exists():
            dst = dst.with_name(f"{p.stem} ({n}){EXT}")
            n += 1
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            p.replace(dst)
            if media_dir(p).is_dir():
                media_dir(p).replace(media_dir(dst))
        except OSError:
            continue
        moved.append((p, dst))
    return moved


def library(folder: Path | None, kind: str | None = None) -> list[dict]:
    """Projects inside `folder` (subfolders included), newest first: [{path, kind, name, opened, library}].
    Headers are cached by modification time, so listing the folder on every menu opening is cheap."""
    if folder is None or not Path(folder).is_dir():
        return []
    out = []
    for p in Path(folder).rglob(f"*{EXT}"):
        if not p.is_file() or any(part.endswith(MEDIA_SUFFIX) for part in p.parts):
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        key = str(p.resolve())
        k = _kind_of(p)
        if k is None or (kind is not None and k != kind):
            continue
        rel = p.relative_to(folder)
        if rel.parts[0] == k and len(rel.parts) > 1:
            rel = Path(*rel.parts[1:])          # inside its tab's folder: the name starts there
        name = rel.with_suffix("").as_posix()   # subfolders show as "group/name": folders work as preset groups
        out.append({"path": key, "kind": k, "name": name, "library": True,
                    "opened": datetime.datetime.fromtimestamp(mt).isoformat(timespec="seconds")})
    out.sort(key=lambda it: it["opened"], reverse=True)
    return out


class RecentProjects:
    """data/projects.json: projects opened or saved, most recent first; plus the projects folder (`folder`)."""

    def __init__(self, path: Path, limit: int = 50, folder: Path | None = None) -> None:
        self.path, self.limit, self.folder = path, limit, folder
        try:
            self.items: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.items = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass

    def touch(self, path: Path, kind: str) -> None:
        p = str(Path(path).resolve())
        self.items = [it for it in self.items if _norm(it["path"]) != _norm(p)]
        self.items.insert(0, {"path": p, "kind": kind, "name": Path(p).stem,
                              "opened": datetime.datetime.now().isoformat(timespec="seconds")})
        del self.items[self.limit:]
        self.save()

    def remove(self, path: str) -> None:
        self.items = [it for it in self.items if _norm(it["path"]) != _norm(path)]
        self.save()

    def rename(self, old: Path, new: Path) -> None:
        for it in self.items:
            if _norm(it["path"]) == _norm(str(old)):
                it["path"], it["name"] = str(Path(new).resolve()), Path(new).stem
        self.save()

    def of(self, kind: str | None = None) -> list[dict]:
        return [it for it in self.items if kind is None or it["kind"] == kind]

    def library(self, kind: str | None = None) -> list[dict]:
        return library(self.folder, kind)

    def everything(self, kind: str | None = None) -> list[dict]:
        """Recent projects and every project in the folder (once each), most recent first."""
        lib = {_norm(it["path"]): it for it in self.library(kind)}
        out = []
        for it in self.of(kind):
            n = _norm(it["path"])
            out.append({**it, "library": n in lib, **({"name": lib[n]["name"]} if n in lib else {})})
            lib.pop(n, None)
        out += lib.values()
        out.sort(key=lambda it: it.get("opened", ""), reverse=True)
        return out


def relink(path: Path, mapping: dict[str, str]) -> None:
    """Point a saved project at the files' new locations without opening it (rewrites project.json only)."""
    path = Path(path)
    doc = read_header(path)
    doc["state"] = remap_paths(doc["state"], mapping)
    doc["files"] = [file_entry(mapping.get(e["path"], e["path"]), path) for e in doc["files"]]
    tmp = path.with_name(path.name + ".part")
    try:
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as dst:
            for info in src.infolist():
                if info.filename != "project.json":
                    dst.writestr(info, src.read(info.filename))
            dst.writestr("project.json", json.dumps(doc, ensure_ascii=False, indent=1))
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
