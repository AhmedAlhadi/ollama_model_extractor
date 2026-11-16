#!/usr/bin/env python3
"""ollama_export_gui.py - Tkinter front-end for ollama_export.

Lists the models installed in the local Ollama store, lets you pick one,
and packs its manifest + blobs into a .tar.gz or .zip archive (the
packing logic lives in ollama_export.py, which must sit next to this
file).

Usage:
    python ollama_export_gui.py [--models-dir PATH]

On Windows you can double-click this file, or run it with pythonw.exe to
avoid a console window:

    pythonw ollama_export_gui.py

Linux note: Tkinter ships with most distributions; on Debian/Ubuntu
install it with `sudo apt install python3-tk`.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from ollama_export import (
        ExportError,
        find_models_dir,
        human_size,
        import_instructions,
        load_models,
        safe_stem,
        write_archive,
    )
except ImportError as exc:
    raise SystemExit(
        f"{exc}\n"
        "ollama_export_gui.py must sit next to the CURRENT ollama_export.py -\n"
        "the core module that defines ExportError. Re-copy both files from the\n"
        "same version together, then run again."
    )

POLL_MS = 100


class ExporterApp(tk.Tk):
    def __init__(self, models_dir_arg: str | None = None) -> None:
        super().__init__()
        self.title("Ollama Model Exporter")
        self.minsize(660, 480)

        self.models = []
        self.loaded_dir: Path | None = None
        self.models_dir = tk.StringVar()
        self.details_var = tk.StringVar(value="Select a model to see details.")
        self.status_var = tk.StringVar(value="")
        self.format_var = tk.StringVar(value="tar")
        self._queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_widgets()
        self._initial_dir(models_dir_arg)

    # ---------- UI construction ----------

    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Models dir:").pack(side="left")
        dir_entry = ttk.Entry(top, textvariable=self.models_dir, width=60)
        dir_entry.pack(side="left", fill="x", expand=True, padx=4)
        dir_entry.bind("<Return>", lambda _e: self._rescan())
        ttk.Button(top, text="Browse...", command=self._browse).pack(side="left")
        ttk.Button(top, text="Refresh", command=self._rescan).pack(side="left", padx=(4, 0))

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, **pad)
        self.tree = ttk.Treeview(
            mid, columns=("size", "blobs"), show="tree headings", selectmode="browse"
        )
        self.tree.heading("#0", text="Model")
        self.tree.heading("size", text="Size")
        self.tree.heading("blobs", text="Blobs")
        self.tree.column("#0", width=340, anchor="w")
        self.tree.column("size", width=110, anchor="e")
        self.tree.column("blobs", width=56, anchor="e")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda _e: self._start_export())
        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)

        ttk.Label(self, textvariable=self.details_var, justify="left").pack(
            fill="x", padx=12
        )

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)
        ttk.Label(bottom, text="Format:").pack(side="left")
        ttk.Radiobutton(
            bottom, text="tar.gz", variable=self.format_var, value="tar"
        ).pack(side="left", padx=(4, 2))
        ttk.Radiobutton(
            bottom, text="zip", variable=self.format_var, value="zip"
        ).pack(side="left")
        self.export_btn = ttk.Button(bottom, text="Export...", command=self._start_export)
        self.export_btn.pack(side="right")

        prog = ttk.Frame(self)
        prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=1000)
        self.progress.pack(fill="x")
        ttk.Label(prog, textvariable=self.status_var).pack(anchor="w", pady=(2, 0))

    # ---------- model store handling ----------

    def _initial_dir(self, arg: str | None) -> None:
        try:
            d = find_models_dir(arg)
        except ExportError:
            d = (
                Path(arg).expanduser()
                if arg
                else Path.home() / ".ollama" / "models"
            )
        self.models_dir.set(str(d))
        self._reload()

    def _browse(self) -> None:
        chosen = filedialog.askdirectory(parent=self, initialdir=self.models_dir.get())
        if chosen:
            self.models_dir.set(chosen)
            self._rescan()

    @staticmethod
    def _normalize(raw: str) -> str:
        d = Path(raw.strip()).expanduser()
        if not (d / "manifests").is_dir() and (d / "models" / "manifests").is_dir():
            d = d / "models"
        return str(d)

    def _rescan(self) -> None:
        self.models_dir.set(self._normalize(self.models_dir.get()))
        self._reload()

    def _reload(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.models = []
        self.loaded_dir = None
        d = Path(self.models_dir.get())
        if not (d / "manifests").is_dir():
            self.status_var.set(f"No 'manifests' directory under {d}")
            return
        self.loaded_dir = d
        self.models = load_models(d)
        for i, m in enumerate(self.models):
            self.tree.insert(
                "", "end", iid=str(i), text=m.display,
                values=(human_size(m.total_size), len(m.layers)),
            )
        if self.models:
            self.tree.selection_set("0")
            self.status_var.set(f"{len(self.models)} model(s) found in {d}")
        else:
            self.status_var.set(f"No models found in {d}")

    def _selected(self):
        sel = self.tree.selection()
        return self.models[int(sel[0])] if sel else None

    def _on_select(self, _event=None) -> None:
        m = self._selected()
        if not m:
            self.details_var.set("Select a model to see details.")
            return
        self.details_var.set(
            f"{m.display}    {human_size(m.total_size)}    {len(m.layers)} blob(s)\n"
            f"Manifest: {m.manifest_path}"
        )

    # ---------- export ----------

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        self.export_btn.config(state="disabled" if busy else "normal")
        if status:
            self.status_var.set(status)

    def _start_export(self) -> None:
        if self._busy:
            return
        m = self._selected()
        if not m:
            messagebox.showinfo("No model selected", "Select a model in the list first.")
            return
        fmt = self.format_var.get()
        ext = ".zip" if fmt == "zip" else ".tar.gz"
        out_str = filedialog.asksaveasfilename(
            parent=self,
            title="Export model as",
            initialdir=str(Path.home()),
            initialfile=f"ollama_{safe_stem(m.display)}{ext}",
            defaultextension=ext,
            filetypes=[
                ("tar.gz archive", "*.tar.gz"),
                ("zip archive", "*.zip"),
                ("All files", "*.*"),
            ],
        )
        if not out_str:
            return
        out_path = Path(out_str).expanduser()
        if out_path.exists() and not messagebox.askyesno(
            "Overwrite?", f"{out_path} already exists.\nOverwrite it?"
        ):
            return
        try:
            parent = out_path.parent if out_path.parent.exists() else Path.home()
            free = shutil.disk_usage(parent).free
            if free < m.total_size:
                messagebox.showerror(
                    "Not enough free space",
                    f"The export needs ~{human_size(m.total_size)} but only "
                    f"{human_size(free)} is free on the destination drive.",
                )
                return
        except OSError:
            pass

        self._set_busy(True, f"Exporting {m.display} ...")
        self.progress.config(value=0)
        threading.Thread(
            target=self._worker, args=(m, out_path, fmt), daemon=True
        ).start()
        self.after(POLL_MS, self._poll)

    def _worker(self, m, out_path: Path, fmt: str) -> None:
        q = self._queue
        try:
            write_archive(
                m,
                self.loaded_dir,
                out_path,
                fmt,
                console=False,
                on_progress=lambda done, total, label: q.put(("p", done, total, label)),
            )
            q.put(("ok", out_path))
        except ExportError as exc:
            q.put(("err", str(exc)))
        except OSError as exc:
            q.put(("err", f"Export failed: {exc}"))

    def _poll(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "p":
                    _, done, total, label = msg
                    self.progress.config(value=(done * 1000 // total) if total else 1000)
                    self.status_var.set(
                        f"Packing {label} - {human_size(done)} / {human_size(total)}"
                    )
                elif msg[0] == "ok":
                    self._finish_ok(msg[1])
                elif msg[0] == "err":
                    self._finish_err(msg[1])
        except queue.Empty:
            pass
        if self._busy:
            self.after(POLL_MS, self._poll)

    def _finish_ok(self, out_path: Path) -> None:
        self.progress.config(value=1000)
        self._set_busy(False)
        size = out_path.stat().st_size
        self.status_var.set(f"Done: {out_path} ({human_size(size)} on disk)")
        if messagebox.askyesno(
            "Export complete",
            f"Created:\n{out_path}\n({human_size(size)} on disk)\n\n"
            f"Show import instructions for the target machine?",
        ):
            m = self._selected()
            if m:
                self._show_import_help(m, out_path)

    def _finish_err(self, msg: str) -> None:
        self.progress.config(value=0)
        self._set_busy(False, "Export failed.")
        messagebox.showerror("Export failed", msg)

    def _show_import_help(self, m, out_path: Path) -> None:
        win = tk.Toplevel(self)
        win.title("Import on the target machine")
        win.transient(self)
        text = import_instructions(m, out_path.name, self.format_var.get())
        box = tk.Text(win, width=74, height=20, wrap="word")
        box.pack(fill="both", expand=True, padx=8, pady=8)
        box.insert("1.0", text)
        box.config(state="disabled")

        def copy() -> None:
            win.clipboard_clear()
            win.clipboard_append(text)
            self.status_var.set("Import instructions copied to the clipboard.")

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btns, text="Copy to clipboard", command=copy).pack(side="left")
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI exporter for Ollama models.")
    parser.add_argument(
        "--models-dir",
        help="Ollama models directory (default: OLLAMA_MODELS or ~/.ollama/models)",
    )
    args = parser.parse_args()
    app = ExporterApp(args.models_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
