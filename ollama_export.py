#!/usr/bin/env python3
"""ollama_export.py - pack an Ollama model into a portable archive.

Interactively lists the models installed locally, lets you pick one, and
writes a compressed archive (tar.gz or zip) containing the model's
manifest and blobs, laid out exactly as Ollama stores them under
~/.ollama/models. On the target machine, extract the archive into the
Ollama models directory and the model shows up in `ollama list`.

Requires only the Python standard library (Python 3.9+).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MANIFESTS_DIRNAME = "manifests"
BLOBS_DIRNAME = "blobs"
README_NAME = "IMPORT-README.txt"
CHUNK = 1024 * 1024


class ExportError(Exception):
    """Fatal, user-facing failure; reported by both CLI and GUI front-ends."""


@dataclass
class Model:
    ref: str  # e.g. registry.ollama.ai/library/llama3.2/latest
    manifest_path: Path
    manifest: dict
    layers: list[dict] = field(default_factory=list)  # dicts: digest, blob_path, size

    @property
    def display(self) -> str:
        """Short reference shown to the user, e.g. llama3.2:latest."""
        parts = self.ref.split("/")
        if parts and parts[0] == "registry.ollama.ai":
            parts = parts[1:]
        if len(parts) < 3:
            return self.ref
        name = "/".join(parts[:-1]) + ":" + parts[-1]
        return name[len("library/"):] if name.startswith("library/") else name

    @property
    def total_size(self) -> int:
        return sum(l["size"] for l in self.layers)


def find_models_dir(cli_value: str | None) -> Path:
    if cli_value:
        d = Path(cli_value).expanduser()
    elif os.environ.get("OLLAMA_MODELS"):
        d = Path(os.environ["OLLAMA_MODELS"]).expanduser()
    else:
        d = Path.home() / ".ollama" / "models"
    # Accept either the models directory itself (contains manifests/) or the
    # Ollama root one level up (contains models/manifests/).
    if not (d / MANIFESTS_DIRNAME).is_dir() and (d / "models" / MANIFESTS_DIRNAME).is_dir():
        d = d / "models"
    if not (d / MANIFESTS_DIRNAME).is_dir():
        raise ExportError(
            f"no Ollama model store found at {d}\n"
            f"       (expected a '{MANIFESTS_DIRNAME}' directory).\n"
            f"       Set OLLAMA_MODELS or pass --models-dir."
        )
    return d


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{n} B"


def load_models(models_dir: Path) -> list[Model]:
    manifests_root = models_dir / MANIFESTS_DIRNAME
    blobs_root = models_dir / BLOBS_DIRNAME
    models: list[Model] = []
    for path in sorted(manifests_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        layers: list[dict] = []
        entries = []
        cfg = manifest.get("config")
        if isinstance(cfg, dict) and cfg.get("digest"):
            entries.append(cfg)
        entries.extend(manifest.get("layers") or [])
        for entry in entries:
            digest = entry.get("digest")
            if not digest:
                continue
            algo, _, hexd = digest.partition(":")
            blob_path = blobs_root / f"{algo}-{hexd}"
            size = entry.get("size") or 0
            if not size and blob_path.is_file():
                size = blob_path.stat().st_size
            layers.append({"digest": digest, "blob_path": blob_path, "size": size})
        if layers:
            models.append(
                Model(
                    ref=path.relative_to(manifests_root).as_posix(),
                    manifest_path=path,
                    manifest=manifest,
                    layers=layers,
                )
            )
    return models


def choose_model(models: list[Model], wanted: str | None) -> Model:
    if wanted:
        for m in models:
            candidates = (m.display, m.ref, m.ref.split("/", 1)[-1])
            if wanted in candidates:
                return m
        sub = [m for m in models if wanted in m.display]
        if len(sub) == 1:
            return sub[0]
        sys.exit(f"error: model '{wanted}' not found (or ambiguous)")
    print("\nInstalled models:\n")
    for i, m in enumerate(models, 1):
        print(
            f"  {i:2}. {m.display:<45} "
            f"{human_size(m.total_size):>10}  ({len(m.layers)} blobs)"
        )
    while True:
        choice = input(f"\nSelect a model [1-{len(models)}], or 'q' to quit: ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        matches = [m for m in models if choice in (m.display, m.ref)]
        if len(matches) == 1:
            return matches[0]
        print("  invalid selection, try again")


def confirm(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{prompt} {hint} ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  please answer y or n")


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def resolve_format(fmt: str | None, out_path: Path) -> str:
    if fmt:
        return fmt
    suffix = "".join(out_path.suffixes).lower()
    if suffix == ".zip":
        return "zip"
    if suffix in (".tar.gz", ".tgz"):
        return "tar"
    while True:
        ans = input("Archive format - 'tar.gz' or 'zip' [tar.gz]: ").strip().lower()
        if not ans or ans in ("tar", "tar.gz", "tgz", "gz"):
            return "tar"
        if ans == "zip":
            return "zip"
        print("  please answer 'tar.gz' or 'zip'")


class Progress:
    """Tracks packing progress; optionally draws the console bar and/or
    reports updates through an on_progress(done, total, label) callback."""

    BAR_WIDTH = 28

    def __init__(self, total: int, console: bool = True, on_progress=None) -> None:
        self.total = total
        self.done = 0
        self.label = ""
        self._last_pct = -1
        self._console = console and sys.stdout is not None
        self._on_progress = on_progress

    def update(self, n: int) -> None:
        self.done += n
        if self._on_progress:
            self._on_progress(self.done, self.total, self.label)
        if not self._console:
            return
        pct = (self.done * 100 // self.total) if self.total else 100
        if pct == self._last_pct:
            return
        self._last_pct = pct
        filled = pct * self.BAR_WIDTH // 100
        bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
        line = (
            f"\r  [{bar}] {pct:3d}%  "
            f"{human_size(self.done):>9} / {human_size(self.total):<9}  "
            f"{self.label[:36]:<36}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self) -> None:
        self.update(self.total - self.done if self.total else 0)
        if self._console:
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()


class ProgressReader:
    """File-like reader that feeds the shared progress bar while copying."""

    def __init__(self, path: Path, progress: Progress) -> None:
        self._f = open(path, "rb")
        self._p = progress

    def read(self, n: int = -1) -> bytes:
        data = self._f.read(n)
        self._p.update(len(data))
        return data

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "ProgressReader":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def import_instructions(model: Model, archive_name: str, fmt: str) -> str:
    target = "~/.ollama/models"
    if fmt == "zip":
        unix = f"unzip {archive_name} -d {target}"
        win = f"tar -xf {archive_name} -C %USERPROFILE%\\.ollama\\models"
    else:
        unix = f"tar -xzf {archive_name} -C {target}"
        win = f"tar -xzf {archive_name} -C %USERPROFILE%\\.ollama\\models"
    return f"""\
This archive contains the Ollama model '{model.display}' exported with
ollama_export.py. The files inside mirror Ollama's storage layout.

Import on the target machine:

  1. Copy this archive to the target machine.

  2. Extract it into the Ollama models directory - by default
     ~/.ollama/models (Windows: %USERPROFILE%\\.ollama\\models). If the
     target machine sets the OLLAMA_MODELS environment variable, extract
     into that directory instead:

       Linux / macOS : {unix}
       Windows       : {win}

  3. Verify the model is registered:

         ollama list

  4. Run it:

         ollama run {model.display}
"""


def write_archive(
    model: Model,
    models_dir: Path,
    out_path: Path,
    fmt: str,
    console: bool = True,
    on_progress=None,
) -> None:
    blobs: list[tuple[Path, str]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for layer in model.layers:
        digest = layer["digest"]
        if digest in seen:
            continue
        seen.add(digest)
        src: Path = layer["blob_path"]
        arc = f"{BLOBS_DIRNAME}/{src.name}"
        if src.is_file():
            blobs.append((src, arc))
        else:
            missing.append(digest)
    if missing:
        raise ExportError(
            f"{len(missing)} referenced blob(s) are missing from "
            f"{models_dir / BLOBS_DIRNAME}:\n  " + "\n  ".join(missing)
        )

    arc_manifest = (
        f"{MANIFESTS_DIRNAME}/"
        f"{model.manifest_path.relative_to(models_dir / MANIFESTS_DIRNAME).as_posix()}"
    )
    readme = import_instructions(model, out_path.name, fmt).encode("utf-8")
    total = sum(src.stat().st_size for src, _ in blobs)
    progress = Progress(total, console=console, on_progress=on_progress)

    if console:
        print(f"Packing {len(blobs)} blob(s), {human_size(total)} total:")
    if fmt == "tar":
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(model.manifest_path, arcname=arc_manifest)
            info = tarfile.TarInfo(README_NAME)
            info.size = len(readme)
            tar.addfile(info, io.BytesIO(readme))
            for src, arc in blobs:
                progress.label = src.name
                info = tar.gettarinfo(str(src), arcname=arc)
                with ProgressReader(src, progress) as reader:
                    tar.addfile(info, reader)
    else:
        with zipfile.ZipFile(out_path, "w", allowZip64=True) as zf:
            zf.write(model.manifest_path, arc_manifest)
            zf.writestr(README_NAME, readme)
            for src, arc in blobs:
                progress.label = src.name
                st = src.stat()
                zinfo = zipfile.ZipInfo(arc, time.localtime(st.st_mtime)[:6])
                zinfo.file_size = st.st_size
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                zinfo.external_attr = (st.st_mode & 0xFFFF) << 16
                with zf.open(zinfo, "w", force_zip64=True) as dest:
                    with ProgressReader(src, progress) as reader:
                        shutil.copyfileobj(reader, dest, CHUNK)
    progress.finish()


def main() -> None:
    # Never crash on unprintable characters (e.g. non-ASCII model names on a
    # non-UTF-8 Windows console).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="Export a locally installed Ollama model to a tar.gz/zip "
        "archive that can be unpacked into ~/.ollama/models on another machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
        "  python ollama_export.py                     # interactive\n"
        "  python ollama_export.py -m llama3.2 -y      # export without prompting\n"
        "  python ollama_export.py --list              # list installed models\n",
    )
    parser.add_argument(
        "--models-dir",
        help="Ollama models directory (default: OLLAMA_MODELS env var or ~/.ollama/models)",
    )
    parser.add_argument("-m", "--model", help="model to export, e.g. library/llama3.2:latest")
    parser.add_argument("-o", "--output", help="output archive path (default: ./ollama_<model>.tar.gz)")
    parser.add_argument("--format", choices=["tar", "zip"], help="archive format")
    parser.add_argument("-y", "--yes", action="store_true", help="skip confirmation prompts")
    parser.add_argument("--list", action="store_true", help="list installed models and exit")
    args = parser.parse_args()

    try:
        models_dir = find_models_dir(args.models_dir)
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    models = load_models(models_dir)
    if not models:
        sys.exit(f"error: no models found under {models_dir / MANIFESTS_DIRNAME}")

    if args.list:
        for m in models:
            print(f"{m.display}\t{human_size(m.total_size)}\t{len(m.layers)} blobs")
        return

    model = choose_model(models, args.model)
    print(f"\nSelected: {model.display}")
    print(f"  manifest : {model.manifest_path}")
    print(f"  blobs    : {len(model.layers)}")
    print(f"  size     : {human_size(model.total_size)}")

    default_ext = ".zip" if args.format == "zip" else ".tar.gz"
    default_path = Path.cwd() / f"ollama_{safe_stem(model.display)}{default_ext}"
    raw = args.output
    if not raw:
        entered = input(f"\nExport to [{default_path}]: ").strip()
        raw = entered or str(default_path)
    out_path = Path(raw).expanduser()
    if out_path.is_dir():
        out_path = out_path / default_path.name

    fmt = resolve_format(args.format, out_path)
    if fmt == "tar" and not str(out_path).lower().endswith((".tar.gz", ".tgz")):
        out_path = Path(str(out_path) + ".tar.gz")
    if fmt == "zip" and out_path.suffix.lower() != ".zip":
        out_path = out_path.with_suffix(".zip")

    needed = model.total_size
    free = shutil.disk_usage(out_path.parent if out_path.parent.exists() else Path.cwd()).free
    if free < needed:
        sys.exit(
            f"error: not enough free space at {out_path.parent}\n"
            f"       need ~{human_size(needed)}, have {human_size(free)}"
        )

    fmt_label = "tar.gz" if fmt == "tar" else "zip"
    print(f"\nReady to export:")
    print(f"  model   : {model.display}")
    print(f"  archive : {out_path}")
    print(f"  format  : {fmt_label}")
    print(f"  payload : {human_size(needed)} (uncompressed)")
    if not (args.yes or confirm("Proceed?")):
        sys.exit("aborted")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_archive(model, models_dir, out_path, fmt)
    except ExportError as exc:
        sys.exit(f"error: {exc}")
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        sys.exit(f"error: export failed: {exc}")

    on_disk = out_path.stat().st_size
    print(f"Done: {out_path} ({human_size(on_disk)} on disk)")
    print()
    print("On the target machine, extract into the Ollama models directory")
    print(f"(see {README_NAME} inside the archive for step-by-step commands):")
    print(f"  Linux/macOS : tar -xzf {out_path.name} -C ~/.ollama/models" if fmt == "tar"
          else f"  Linux/macOS : unzip {out_path.name} -d ~/.ollama/models")
    print(f"  Windows     : tar -xzf {out_path.name} -C %USERPROFILE%\\.ollama\\models" if fmt == "tar"
          else f"  Windows     : tar -xf {out_path.name} -C %USERPROFILE%\\.ollama\\models")
    print(f"Then: ollama list   (model '{model.display}' should appear)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\naborted")
