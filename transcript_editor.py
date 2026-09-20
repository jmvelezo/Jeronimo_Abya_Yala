from __future__ import annotations

import copy
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
from tkinter import ttk

from visual_theme import asset_path
from subprocess_utils import hidden_process_kwargs

try:
    from core_transcriber import (
        HAS_OPENAI,
        OpenAI,
        TranscriberConfig,
        STT_ENGINE_LOCAL_FAST_WHISPER,
        STT_ENGINE_OPENAI,
        transcribe_audio,
        transcribe_audio_local_faster_whisper,
    )
except Exception:  # permite abrir el editor como herramienta de texto aunque el core falle
    HAS_OPENAI = False
    OpenAI = None  # type: ignore
    TranscriberConfig = object  # type: ignore
    STT_ENGINE_LOCAL_FAST_WHISPER = "local_faster_whisper"
    STT_ENGINE_OPENAI = "openai"
    transcribe_audio = None  # type: ignore
    transcribe_audio_local_faster_whisper = None  # type: ignore


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".mkv", ".ogg", ".flac"}
TIME_RE = re.compile(r"(?P<s>\d{2}:\d{2}:\d{2})(?:[,\.]\d{1,3})?\s*[-–→]+\s*(?P<e>\d{2}:\d{2}:\d{2})(?:[,\.]\d{1,3})?")
SPEAKER_RE = re.compile(r"^[^:\n]{1,90}:\s+")


# Territorio Vivo — paleta del editor Tk/ttk.
# Esta tabla es exclusivamente visual; no interviene en lógica ni datos.
C = {
    "bg": "#171D1B",
    "panel": "#202824",
    "panel2": "#26312B",
    "panel3": "#2B3831",
    "text": "#F4F0E6",
    "muted": "#AAB7AF",
    "accent": "#B85F40",
    "accent_dark": "#8F4934",
    "green": "#3F765E",
    "orange": "#D4A84F",
    "red": "#A8493D",
    "water": "#477C82",
    "border": "#34433C",
    "canvas": "#121816",
    "wave": "#477C82",
    "track": "#466057",
    "button": "#2A352F",
}


def _seconds_to_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hms_to_seconds(value: str) -> float:
    h, m, s = value.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)




def _new_temp_wav(prefix: str) -> Path:
    """Crea un temporal de forma segura y devuelve su ruta cerrada."""
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".wav")
    os.close(fd)
    return Path(name)

def _safe_preview(text: str, max_chars: int = 140) -> str:
    one = re.sub(r"\s+", " ", text or "").strip()
    if len(one) > max_chars:
        return one[: max_chars - 1] + "…"
    return one


def _get_media_duration(path: Path) -> float:
    if not path or not path.exists():
        return 0.0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            **hidden_process_kwargs(),
        )
        import json

        data = json.loads(r.stdout or "{}")
        return float(data.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        return 0.0


def _extract_ab_segment(audio_path: Path, start: float, end: float) -> Path:
    duration = max(0.1, float(end) - float(start))
    tmp = _new_temp_wav("jeronimo_ab_")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, start)),
        "-i",
        str(audio_path),
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, **hidden_process_kwargs())
        return tmp
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _parse_blocks(text: str, duration: float = 0.0) -> list[dict]:
    """Divide texto de entrevista en bloques editables, sin asumir formato SRT."""
    raw = (text or "").replace("\r\n", "\n")
    blocks: list[dict] = []
    current: list[str] = []

    def flush():
        nonlocal current
        joined = "\n".join(current).strip()
        current = []
        if not joined:
            return
        label = ""
        first = joined.splitlines()[0]
        if SPEAKER_RE.match(first):
            label = first.split(":", 1)[0].strip()
        start = end = None
        m = TIME_RE.search(joined)
        if m:
            try:
                start = _hms_to_seconds(m.group("s"))
                end = _hms_to_seconds(m.group("e"))
            except Exception:
                start = end = None
        blocks.append({"text": joined, "label": label, "start": start, "end": end})

    for line in raw.splitlines():
        if not line.strip():
            flush()
            continue
        if current and SPEAKER_RE.match(line):
            flush()
        current.append(line.rstrip())
    flush()

    if not blocks and raw.strip():
        blocks = [{"text": raw.strip(), "label": "", "start": None, "end": None}]

    # Si no hay timecodes, asignación aproximada solo para navegación visual.
    if blocks and duration > 0:
        n = len(blocks)
        approx = duration / max(n, 1)
        for i, b in enumerate(blocks):
            if b.get("start") is None or b.get("end") is None:
                b["start"] = round(i * approx, 3)
                b["end"] = round(min(duration, (i + 1) * approx), 3)
                b["approx_time"] = True
            else:
                b["approx_time"] = False
    return blocks


def _blocks_to_text(blocks: list[dict]) -> str:
    return "\n\n".join((b.get("text") or "").strip() for b in blocks if (b.get("text") or "").strip()) + "\n"


class TranscriptEditor(tk.Toplevel):
    """Editor mínimo de revisión para entrevistas.

    No trabaja con SRT. Abre una transcripción TXT, permite corregir bloques,
    escuchar con ffplay si está disponible y re-transcribir un tramo A→B.
    """

    def __init__(self, master=None, audio_path: str | None = None, transcript_path: str | None = None, cfg=None):
        super().__init__(master)
        self.title("Jerónimo Abya Yala — Editor de revisión")
        self.geometry("1180x760")
        self.minsize(900, 600)
        self.configure(bg=C["bg"])
        try:
            self.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass

        self.audio_path = Path(audio_path) if audio_path else None
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self.cfg = cfg
        self.duration = _get_media_duration(self.audio_path) if self.audio_path else 0.0
        self.blocks: list[dict] = []
        self.selected_i = -1
        self.undo_stack: list[list[dict]] = []
        self.redo_stack: list[list[dict]] = []
        self.dirty = False

        self.cursor = 0.0
        self.marker_a = 0.0
        self.marker_b = min(30.0, self.duration or 30.0)
        self._play_proc: subprocess.Popen | None = None
        self._play_started_at = 0.0
        self._play_origin = 0.0
        self._waveform: list[float] = []
        self._waveform_thread: threading.Thread | None = None

        self._build_ui()
        self._load_initial_files()
        self._tick()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # VIS-5: composición y estilos Territorio Vivo. Los comandos y objetos
        # funcionales conservan exactamente los mismos callbacks/nombres.
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Treeview",
            background=C["panel"],
            foreground=C["text"],
            fieldbackground=C["panel"],
            borderwidth=0,
            relief="flat",
            rowheight=32,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background=C["panel2"],
            foreground=C["muted"],
            borderwidth=0,
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Treeview",
            background=[("selected", C["panel3"])],
            foreground=[("selected", C["text"])],
        )
        style.map("Treeview.Heading", background=[("active", C["panel3"])])
        style.configure(
            "Vertical.TScrollbar",
            background=C["panel2"],
            troughcolor=C["bg"],
            bordercolor=C["bg"],
            arrowcolor=C["muted"],
            lightcolor=C["panel2"],
            darkcolor=C["panel2"],
        )

        def action_button(parent, text, command, *, tone="neutral", width=None):
            palette = {
                "neutral": (C["button"], C["text"]),
                "primary": (C["accent"], "#FFF9F2"),
                "success": (C["green"], "#FFF9F2"),
                "danger": (C["red"], "#FFF9F2"),
                "maize": (C["orange"], "#241F16"),
            }
            bg, fg = palette.get(tone, palette["neutral"])
            options = {
                "text": text,
                "command": command,
                "bg": bg,
                "fg": fg,
                "activebackground": bg,
                "activeforeground": fg,
                "relief": "flat",
                "bd": 0,
                "highlightthickness": 0,
                "padx": 11,
                "pady": 7,
                "cursor": "hand2",
                "font": ("Segoe UI Semibold", 9),
            }
            if width is not None:
                options["width"] = width
            return tk.Button(parent, **options)

        # Cabecera: identidad del editor + fuente activa + acciones de archivo.
        header = tk.Frame(self, bg=C["panel"], padx=14, pady=11)
        header.pack(fill="x")
        brand = tk.Frame(header, bg=C["panel"])
        brand.pack(side="left", fill="y")
        tk.Label(
            brand,
            text="JERÓNIMO ABYA YALA  /  EDITOR DE REVISIÓN",
            bg=C["panel"], fg=C["text"], anchor="w",
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="Territorio de trabajo local · transcripción y escucha",
            bg=C["panel"], fg=C["muted"], anchor="w",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        file_actions = tk.Frame(header, bg=C["panel"])
        file_actions.pack(side="right")
        action_button(file_actions, "Rehacer", self._redo).pack(side="right", padx=(5, 0))
        action_button(file_actions, "Deshacer", self._undo).pack(side="right", padx=(5, 0))
        action_button(file_actions, "Guardar  Ctrl+S", self._save, tone="primary").pack(side="right", padx=(9, 0))
        action_button(file_actions, "Abrir texto", self._pick_transcript).pack(side="right", padx=(5, 0))
        action_button(file_actions, "Abrir audio", self._pick_audio).pack(side="right", padx=(5, 0))

        source_bar = tk.Frame(self, bg=C["panel2"], padx=14, pady=7)
        source_bar.pack(fill="x")
        tk.Label(
            source_bar, text="FUENTE", bg=C["panel2"], fg=C["orange"],
            font=("Segoe UI Semibold", 8),
        ).pack(side="left", padx=(0, 10))
        self.lbl_audio = tk.Label(source_bar, text="Audio: —", bg=C["panel2"], fg=C["muted"], anchor="w")
        self.lbl_audio.pack(side="left", fill="x", expand=True)
        tk.Label(
            source_bar, text="● LOCAL", bg=C["panel2"], fg=C["green"],
            font=("Segoe UI Semibold", 8),
        ).pack(side="right")

        # Transporte y selección A/B. Mismos comandos que la versión base.
        playback = tk.Frame(self, bg=C["bg"], padx=12, pady=9)
        playback.pack(fill="x")
        self.btn_play = action_button(playback, "▶  Reproducir", self._toggle_play, tone="success", width=13)
        self.btn_play.pack(side="left", padx=(0, 5))
        action_button(playback, "−5 s", lambda: self._skip(-5)).pack(side="left", padx=2)
        action_button(playback, "+5 s", lambda: self._skip(5)).pack(side="left", padx=2)

        tk.Frame(playback, bg=C["border"], width=1, height=28).pack(side="left", padx=10)
        action_button(playback, "A  Marcar inicio", self._set_a_here, tone="success").pack(side="left", padx=2)
        action_button(playback, "B  Marcar fin", self._set_b_here, tone="maize").pack(side="left", padx=2)
        action_button(playback, "Re-transcribir A→B", self._retranscribe_ab, tone="primary").pack(side="left", padx=(10, 2))
        action_button(playback, "Generar waveform", self._generate_waveform).pack(side="left", padx=2)
        self.lbl_time = tk.Label(
            playback,
            text="00:00:00 / 00:00:00   A:00:00:00  B:00:00:00",
            bg=C["bg"], fg=C["text"], font=("Consolas", 9),
        )
        self.lbl_time.pack(side="right", padx=(12, 0))

        timeline_card = tk.Frame(self, bg=C["panel"], padx=7, pady=7)
        timeline_card.pack(fill="x", padx=12, pady=(0, 9))
        self.canvas = tk.Canvas(self, height=96, bg=C["canvas"], highlightthickness=0)
        self.canvas.pack(in_=timeline_card, fill="x")
        self.canvas.bind("<Button-1>", self._on_timeline_click)
        self.canvas.bind("<Configure>", lambda _e: self._draw_timeline())

        center = tk.PanedWindow(
            self,
            orient="horizontal",
            bg=C["border"],
            sashwidth=6,
            sashrelief="flat",
            bd=0,
            relief="flat",
        )
        center.pack(fill="both", expand=True, padx=12, pady=(0, 9))

        left = tk.Frame(center, bg=C["panel"], padx=0, pady=0)
        center.add(left, minsize=360, stretch="always")
        left_head = tk.Frame(left, bg=C["panel2"], padx=12, pady=8)
        left_head.pack(fill="x")
        tk.Label(
            left_head, text="TRANSCRIPCIÓN", bg=C["panel2"], fg=C["orange"],
            font=("Segoe UI Semibold", 8),
        ).pack(side="left")
        tk.Label(
            left_head, text="Bloques y hablantes", bg=C["panel2"], fg=C["muted"],
            font=("Segoe UI", 9),
        ).pack(side="right")

        tree_wrap = tk.Frame(left, bg=C["panel"])
        tree_wrap.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_wrap, columns=("#", "label", "preview"), show="headings", selectmode="browse")
        self.tree.heading("#", text="#")
        self.tree.heading("label", text="Hablante / bloque")
        self.tree.heading("preview", text="Texto")
        self.tree.column("#", width=48, stretch=False, anchor="center")
        self.tree.column("label", width=160, stretch=False)
        self.tree.column("preview", width=500, stretch=True)
        ysb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ysb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        right = tk.Frame(center, bg=C["panel"], padx=0, pady=0)
        center.add(right, minsize=380, stretch="always")
        right_head = tk.Frame(right, bg=C["panel2"], padx=12, pady=8)
        right_head.pack(fill="x")
        tk.Label(
            right_head, text="EDICIÓN", bg=C["panel2"], fg=C["orange"],
            font=("Segoe UI Semibold", 8),
        ).pack(side="left")
        self.lbl_block = tk.Label(right_head, text="Bloque seleccionado", bg=C["panel2"], fg=C["muted"], anchor="e")
        self.lbl_block.pack(side="right")

        editor_wrap = tk.Frame(right, bg=C["panel"], padx=10, pady=10)
        editor_wrap.pack(fill="both", expand=True)
        self.txt_block = tk.Text(
            editor_wrap,
            wrap="word",
            undo=False,
            bg=C["canvas"],
            fg=C["text"],
            insertbackground=C["orange"],
            selectbackground=C["green"],
            selectforeground=C["text"],
            relief="flat",
            bd=0,
            padx=14,
            pady=12,
            font=("Segoe UI", 11),
        )
        self.txt_block.pack(fill="both", expand=True)

        btns = tk.Frame(right, bg=C["panel"], padx=7, pady=8)
        btns.pack(fill="x")
        action_button(btns, "Aplicar al bloque", self._apply_block_edit, tone="success").pack(side="right", padx=3)
        action_button(btns, "Nuevo bloque", self._new_block).pack(side="left", padx=3)
        action_button(btns, "Eliminar", self._delete_block, tone="danger").pack(side="left", padx=3)
        action_button(btns, "Dividir", self._split_block).pack(side="left", padx=3)

        status = tk.Frame(self, bg=C["panel"], padx=12, pady=6)
        status.pack(fill="x", side="bottom")
        tk.Label(
            status, text="ESTADO", bg=C["panel"], fg=C["orange"],
            font=("Segoe UI Semibold", 8),
        ).pack(side="left", padx=(0, 10))
        self.lbl_status = tk.Label(status, text="Listo", bg=C["panel"], fg=C["muted"], anchor="w")
        self.lbl_status.pack(side="left", fill="x", expand=True)
        tk.Label(
            status, text="A/B · CTRL+S · SPACE", bg=C["panel"], fg=C["muted"],
            font=("Segoe UI", 8),
        ).pack(side="right")

        self.bind("<Control-s>", lambda _e: self._save())
        self.bind("<Control-z>", lambda _e: self._undo())
        self.bind("<Control-y>", lambda _e: self._redo())
        self.bind("<space>", lambda _e: self._toggle_play())
        self.bind("<Left>", lambda _e: self._skip(-1))
        self.bind("<Right>", lambda _e: self._skip(1))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_initial_files(self):
        if self.audio_path and self.audio_path.exists():
            self.lbl_audio.config(text=f"Audio: {self.audio_path.name}")
        if self.transcript_path and self.transcript_path.exists():
            self._load_transcript(self.transcript_path)
        else:
            self._refresh_tree()
        self._update_time_label()
        self._draw_timeline()

    # -------------------------------------------------------------- file I/O
    def _pick_audio(self):
        path = filedialog.askopenfilename(
            title="Seleccionar audio/video",
            filetypes=[("Audio/video", "*.wav *.mp3 *.m4a *.mp4 *.mov *.mkv *.ogg *.flac"), ("Todos", "*.*")],
        )
        if not path:
            return
        self.audio_path = Path(path)
        self.duration = _get_media_duration(self.audio_path)
        self.marker_b = min(max(self.marker_a + 1.0, self.marker_a + 30.0), self.duration or self.marker_a + 30.0)
        self.lbl_audio.config(text=f"Audio: {self.audio_path.name}")
        self._draw_timeline()

    def _pick_transcript(self):
        path = filedialog.askopenfilename(title="Seleccionar transcripción", filetypes=[("Texto", "*.txt"), ("Todos", "*.*")])
        if path:
            self._load_transcript(Path(path))

    def _load_transcript(self, path: Path):
        text = path.read_text(encoding="utf-8", errors="ignore")
        self.transcript_path = path
        self.blocks = _parse_blocks(text, self.duration)
        self.undo_stack = [copy.deepcopy(self.blocks)]
        self.redo_stack = []
        self.dirty = False
        self._refresh_tree()
        if self.blocks:
            self._select_block(0)
        self._set_status(f"Transcripción cargada: {path.name} ({len(self.blocks)} bloques)")

    def _save(self):
        if not self.transcript_path:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texto", "*.txt"), ("Todos", "*.*")])
            if not path:
                return
            self.transcript_path = Path(path)
        try:
            if self.transcript_path.exists():
                bak = self.transcript_path.with_suffix(self.transcript_path.suffix + ".bak")
                shutil.copy2(self.transcript_path, bak)
            self.transcript_path.write_text(_blocks_to_text(self.blocks), encoding="utf-8")
            self.dirty = False
            self._set_status(f"Guardado: {self.transcript_path.name} (backup .bak si existía archivo previo)")
        except Exception as exc:
            messagebox.showerror("Error al guardar", str(exc))

    # --------------------------------------------------------------- blocks
    def _push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.blocks))
        if len(self.undo_stack) > 80:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.dirty = True

    def _undo(self):
        if len(self.undo_stack) <= 1:
            return
        self.redo_stack.append(copy.deepcopy(self.blocks))
        self.undo_stack.pop()
        self.blocks = copy.deepcopy(self.undo_stack[-1])
        self._refresh_tree()
        self._select_block(min(max(self.selected_i, 0), len(self.blocks) - 1))
        self.dirty = True

    def _redo(self):
        if not self.redo_stack:
            return
        state = self.redo_stack.pop()
        self.undo_stack.append(copy.deepcopy(state))
        self.blocks = copy.deepcopy(state)
        self._refresh_tree()
        self._select_block(min(max(self.selected_i, 0), len(self.blocks) - 1))
        self.dirty = True

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, b in enumerate(self.blocks):
            label = b.get("label") or f"Bloque {i+1}"
            self.tree.insert("", "end", iid=f"b{i}", values=(str(i + 1), label, _safe_preview(b.get("text", ""))))
        self._draw_timeline()

    def _select_block(self, i: int):
        if not self.blocks:
            self.selected_i = -1
            self.txt_block.delete("1.0", "end")
            return
        i = max(0, min(i, len(self.blocks) - 1))
        self.selected_i = i
        iid = f"b{i}"
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)
        b = self.blocks[i]
        self.txt_block.delete("1.0", "end")
        self.txt_block.insert("1.0", b.get("text", ""))
        self.lbl_block.config(text=f"Bloque {i+1}/{len(self.blocks)}")
        if b.get("start") is not None:
            self.cursor = float(b.get("start") or 0.0)
        self._draw_timeline()

    def _on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        try:
            i = int(sel[0][1:])
        except Exception:
            return
        if i != self.selected_i:
            self._select_block(i)

    def _apply_block_edit(self):
        if self.selected_i < 0 or self.selected_i >= len(self.blocks):
            return
        text = self.txt_block.get("1.0", "end").strip()
        self._push_undo()
        self.blocks[self.selected_i]["text"] = text
        first = text.splitlines()[0] if text else ""
        self.blocks[self.selected_i]["label"] = first.split(":", 1)[0].strip() if SPEAKER_RE.match(first) else ""
        self._refresh_tree()
        self._select_block(self.selected_i)
        self._set_status("Bloque actualizado")

    def _new_block(self):
        self._push_undo()
        insert_at = self.selected_i + 1 if self.selected_i >= 0 else len(self.blocks)
        self.blocks.insert(insert_at, {"text": "", "label": "", "start": self.cursor, "end": min(self.cursor + 30, self.duration or self.cursor + 30), "approx_time": True})
        self._refresh_tree()
        self._select_block(insert_at)
        self.txt_block.focus_set()

    def _delete_block(self):
        if self.selected_i < 0 or self.selected_i >= len(self.blocks):
            return
        if not messagebox.askyesno("Eliminar", f"¿Eliminar bloque {self.selected_i + 1}?"):
            return
        self._push_undo()
        self.blocks.pop(self.selected_i)
        self._refresh_tree()
        self._select_block(min(self.selected_i, len(self.blocks) - 1))

    def _split_block(self):
        if self.selected_i < 0 or self.selected_i >= len(self.blocks):
            return
        full = self.txt_block.get("1.0", "end").strip()
        if not full:
            return
        idx = self.txt_block.index("insert")
        try:
            before = self.txt_block.get("1.0", idx).strip()
            after = self.txt_block.get(idx, "end").strip()
        except Exception:
            words = full.split()
            mid = len(words) // 2
            before = " ".join(words[:mid])
            after = " ".join(words[mid:])
        if not before or not after:
            words = full.split()
            if len(words) < 2:
                return
            mid = len(words) // 2
            before = " ".join(words[:mid])
            after = " ".join(words[mid:])
        self._push_undo()
        old = self.blocks[self.selected_i]
        start = float(old.get("start") or self.cursor)
        end = float(old.get("end") or (start + 30))
        mid_t = (start + end) / 2
        self.blocks[self.selected_i] = {**old, "text": before, "end": mid_t}
        self.blocks.insert(self.selected_i + 1, {**old, "text": after, "start": mid_t})
        self._refresh_tree()
        self._select_block(self.selected_i + 1)

    # -------------------------------------------------------------- playback
    def _ffplay_available(self) -> bool:
        return bool(shutil.which("ffplay")) and self.audio_path and self.audio_path.exists()

    def _toggle_play(self):
        if self._play_proc and self._play_proc.poll() is None:
            self._pause_playback()
        else:
            self._start_playback(self.cursor)

    def _start_playback(self, start: float):
        if not self._ffplay_available():
            messagebox.showwarning("Reproducción", "No se encontró ffplay o no hay audio cargado. La edición de texto funciona igual.")
            return
        self._stop_playback(update_cursor=False)
        self.cursor = max(0.0, min(float(start), self.duration or float(start)))
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", str(self.cursor), str(self.audio_path)]
        try:
            self._play_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden_process_kwargs())
            self._play_origin = self.cursor
            self._play_started_at = time.monotonic()
            self.btn_play.config(text="⏸ Pausar")
        except Exception as exc:
            messagebox.showerror("Reproducción", str(exc))

    def _pause_playback(self):
        if self._play_proc and self._play_proc.poll() is None:
            self.cursor = self._current_play_position()
        self._stop_playback(update_cursor=False)

    def _stop_playback(self, update_cursor: bool = True):
        if update_cursor and self._play_proc and self._play_proc.poll() is None:
            self.cursor = self._current_play_position()
        if self._play_proc:
            try:
                self._play_proc.terminate()
            except Exception:
                pass
        self._play_proc = None
        self.btn_play.config(text="▶ Reproducir")

    def _current_play_position(self) -> float:
        if self._play_proc and self._play_proc.poll() is None:
            pos = self._play_origin + (time.monotonic() - self._play_started_at)
            return min(pos, self.duration or pos)
        return self.cursor

    def _skip(self, delta: float):
        was_playing = self._play_proc and self._play_proc.poll() is None
        new_pos = max(0.0, min(self._current_play_position() + delta, self.duration or 10**9))
        self.cursor = new_pos
        if was_playing:
            self._start_playback(self.cursor)
        self._select_nearest_block()
        self._update_time_label()
        self._draw_timeline()

    def _set_a_here(self):
        self.marker_a = max(0.0, min(self._current_play_position(), self.duration or 10**9))
        if self.marker_b <= self.marker_a:
            self.marker_b = min(self.marker_a + 30, self.duration or self.marker_a + 30)
        self._update_time_label()
        self._draw_timeline()

    def _set_b_here(self):
        self.marker_b = max(self.marker_a + 0.1, min(self._current_play_position(), self.duration or self._current_play_position()))
        self._update_time_label()
        self._draw_timeline()

    def _select_nearest_block(self):
        if not self.blocks:
            return
        for i, b in enumerate(self.blocks):
            s = b.get("start")
            e = b.get("end")
            if s is not None and e is not None and float(s) <= self.cursor <= float(e):
                self._select_block(i)
                return

    # ------------------------------------------------------------- timeline
    def _on_timeline_click(self, event):
        if self.duration <= 0:
            return
        w = max(1, self.canvas.winfo_width())
        self.cursor = max(0.0, min(self.duration, event.x / w * self.duration))
        self._select_nearest_block()
        self._update_time_label()
        self._draw_timeline()

    def _draw_timeline(self):
        c = self.canvas
        if not c:
            return
        c.delete("all")
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        c.create_rectangle(0, 0, w, h, fill=C["canvas"], outline="")
        if self.duration <= 0:
            c.create_text(w // 2, h // 2, text="Sin audio/duración. La edición textual está disponible.", fill=C["muted"])
            return
        mid = h // 2
        if self._waveform:
            for x in range(w):
                idx = min(len(self._waveform) - 1, int(x / w * len(self._waveform)))
                amp = self._waveform[idx]
                bh = int(amp * (h * 0.30))
                c.create_line(x, mid - bh, x, mid + bh, fill=C["wave"])
        else:
            c.create_text(w // 2, mid, text="Waveform no generada", fill=C["wave"])
        # bloques aproximados
        for i, b in enumerate(self.blocks):
            s, e = b.get("start"), b.get("end")
            if s is None or e is None:
                continue
            x1 = max(0, min(w, float(s) / self.duration * w))
            x2 = max(0, min(w, float(e) / self.duration * w))
            fill = C["accent"] if i == self.selected_i else C["track"]
            c.create_rectangle(x1, h - 22, max(x1 + 2, x2), h - 6, fill=fill, outline="")
        for pos, color, label in [(self.marker_a, C["green"], "A"), (self.marker_b, C["orange"], "B"), (self._current_play_position(), C["red"], "")]:
            x = max(0, min(w, float(pos) / self.duration * w))
            c.create_line(x, 0, x, h, fill=color, width=2)
            if label:
                c.create_text(x + 4, 10, text=label, fill=color, anchor="w")

    def _generate_waveform(self):
        if not self.audio_path or not self.audio_path.exists():
            messagebox.showwarning("Waveform", "Primero cargá un audio.")
            return
        if self._waveform_thread and self._waveform_thread.is_alive():
            return
        self._set_status("Generando waveform...")

        def worker():
            tmp = _new_temp_wav("jeronimo_wave_")
            try:
                subprocess.run(["ffmpeg", "-y", "-i", str(self.audio_path), "-vn", "-acodec", "pcm_s16le", "-ar", "4000", "-ac", "1", str(tmp)], capture_output=True, text=True, check=True, **hidden_process_kwargs())
                with wave.open(str(tmp), "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                n = len(raw) // 2
                if n <= 0:
                    raise RuntimeError("Audio sin muestras")
                samples = struct.unpack(f"{n}h", raw[: n * 2])
                step = max(1, len(samples) // 1800)
                data = []
                for i in range(0, len(samples), step):
                    chunk = samples[i : i + step]
                    rms = math.sqrt(sum(s * s for s in chunk) / max(1, len(chunk)))
                    data.append(rms)
                mx = max(data) or 1.0
                data = [v / mx for v in data]
                self.after(0, lambda: self._apply_waveform(data))
            except Exception as exc:
                self.after(0, lambda: self._set_status(f"No se pudo generar waveform: {exc}"))
            finally:
                try:
                    tmp.unlink()
                except Exception:
                    pass

        self._waveform_thread = threading.Thread(target=worker, daemon=True)
        self._waveform_thread.start()

    def _apply_waveform(self, data: list[float]):
        self._waveform = data
        self._draw_timeline()
        self._set_status(f"Waveform generada ({len(data)} muestras)")

    # ---------------------------------------------------------- retranscribe
    def _retranscribe_ab(self):
        if not self.audio_path or not self.audio_path.exists():
            messagebox.showwarning("Re-transcripción", "Primero cargá un audio.")
            return
        a, b = float(self.marker_a), float(self.marker_b)
        if b <= a + 0.1:
            messagebox.showwarning("Re-transcripción", "La zona A→B debe durar al menos 0.1 segundos.")
            return
        if b - a > 600 and not messagebox.askyesno("Zona larga", f"La zona dura {(b-a)/60:.1f} minutos. ¿Continuar?"):
            return
        self._set_status(f"Re-transcribiendo A→B {_seconds_to_hms(a)}-{_seconds_to_hms(b)}...")

        def worker():
            tmp = None
            try:
                tmp = _extract_ab_segment(self.audio_path, a, b)
                text = self._transcribe_segment(tmp)
                self.after(0, lambda: self._show_retrans_preview(text, a, b))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Error de re-transcripción", str(exc)))
                self.after(0, lambda: self._set_status(f"Error de re-transcripción: {exc}"))
            finally:
                try:
                    if tmp:
                        tmp.unlink()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _transcribe_segment(self, segment_path: Path) -> str:
        cfg = self.cfg
        if cfg is None:
            raise RuntimeError("No hay configuración activa de transcripción. Abrí el editor desde la app principal para re-transcribir A→B.")
        engine = getattr(cfg, "stt_engine", STT_ENGINE_OPENAI) or STT_ENGINE_OPENAI

        def log(msg: str):
            self.after(0, lambda m=msg: self._set_status(m))

        if engine == STT_ENGINE_LOCAL_FAST_WHISPER:
            if transcribe_audio_local_faster_whisper is None:
                raise RuntimeError("No está disponible el motor local faster-whisper.")
            return transcribe_audio_local_faster_whisper(cfg, segment_path, log).strip()

        if not getattr(cfg, "api_key", ""):
            raise RuntimeError("El motor activo es OpenAI, pero no hay API key en la configuración.")
        if not HAS_OPENAI or OpenAI is None or transcribe_audio is None:
            raise RuntimeError("Falta el paquete openai para re-transcribir con OpenAI.")
        client = OpenAI(api_key=cfg.api_key)
        return transcribe_audio(client, cfg, segment_path, log).strip()

    def _show_retrans_preview(self, text: str, a: float, b: float):
        win = tk.Toplevel(self)
        win.title("Preview re-transcripción A→B")
        win.geometry("720x500")
        win.minsize(620, 420)
        win.configure(bg=C["bg"])
        try:
            win.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass

        head = tk.Frame(win, bg=C["panel"], padx=14, pady=12)
        head.pack(fill="x")
        tk.Label(
            head,
            text="RE-TRANSCRIPCIÓN  A→B",
            bg=C["panel"], fg=C["orange"],
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w")
        tk.Label(
            head,
            text=f"Zona {_seconds_to_hms(a)} → {_seconds_to_hms(b)}",
            bg=C["panel"], fg=C["text"], anchor="w",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor="w", pady=(3, 0))
        tk.Label(
            head,
            text="Revisá el texto antes de incorporarlo a la transcripción.",
            bg=C["panel"], fg=C["muted"], anchor="w",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(win, bg=C["bg"], padx=14, pady=12)
        body.pack(fill="both", expand=True)
        box = tk.Text(
            body,
            wrap="word",
            bg=C["canvas"], fg=C["text"], insertbackground=C["orange"],
            selectbackground=C["green"], selectforeground=C["text"],
            relief="flat", bd=0, padx=14, pady=12, font=("Segoe UI", 11),
        )
        box.pack(fill="both", expand=True)
        box.insert("1.0", text.strip())
        row = tk.Frame(win, bg=C["panel"], padx=12, pady=9)
        row.pack(fill="x", side="bottom")

        def apply_replace():
            new_text = box.get("1.0", "end").strip()
            if self.selected_i < 0 or self.selected_i >= len(self.blocks):
                messagebox.showwarning("Aplicar", "Seleccioná un bloque para reemplazar.")
                return
            self._push_undo()
            self.blocks[self.selected_i]["text"] = new_text
            self.blocks[self.selected_i]["start"] = a
            self.blocks[self.selected_i]["end"] = b
            self._refresh_tree()
            self._select_block(self.selected_i)
            win.destroy()
            self._set_status("Re-transcripción aplicada como reemplazo")

        def apply_insert():
            new_text = box.get("1.0", "end").strip()
            self._push_undo()
            pos = self.selected_i + 1 if self.selected_i >= 0 else len(self.blocks)
            self.blocks.insert(pos, {"text": new_text, "label": "", "start": a, "end": b, "approx_time": False})
            self._refresh_tree()
            self._select_block(pos)
            win.destroy()
            self._set_status("Re-transcripción insertada")

        tk.Button(
            row, text="Reemplazar bloque seleccionado", command=apply_replace,
            bg=C["accent"], fg="#FFF9F2", activebackground=C["accent"], activeforeground="#FFF9F2",
            relief="flat", bd=0, padx=12, pady=7, cursor="hand2",
            font=("Segoe UI Semibold", 9),
        ).pack(side="left", padx=3)
        tk.Button(
            row, text="Insertar como bloque nuevo", command=apply_insert,
            bg=C["green"], fg="#FFF9F2", activebackground=C["green"], activeforeground="#FFF9F2",
            relief="flat", bd=0, padx=12, pady=7, cursor="hand2",
            font=("Segoe UI Semibold", 9),
        ).pack(side="left", padx=3)
        tk.Button(
            row, text="Descartar", command=win.destroy,
            bg=C["button"], fg=C["text"], activebackground=C["button"], activeforeground=C["text"],
            relief="flat", bd=0, padx=12, pady=7, cursor="hand2",
            font=("Segoe UI Semibold", 9),
        ).pack(side="right", padx=3)

    # ------------------------------------------------------------ utilities
    def _update_time_label(self):
        self.lbl_time.config(
            text=(
                f"{_seconds_to_hms(self._current_play_position())} / {_seconds_to_hms(self.duration)}   "
                f"A:{_seconds_to_hms(self.marker_a)}  B:{_seconds_to_hms(self.marker_b)}"
            )
        )

    def _set_status(self, msg: str):
        self.lbl_status.config(text=msg)

    def _tick(self):
        self.cursor = self._current_play_position()
        self._update_time_label()
        self._draw_timeline()
        if self._play_proc and self._play_proc.poll() is not None:
            self._play_proc = None
            self.btn_play.config(text="▶ Reproducir")
        self.after(350, self._tick)

    def _on_close(self):
        if self.dirty:
            resp = messagebox.askyesnocancel("Cambios sin guardar", "Hay cambios sin guardar. ¿Guardar antes de salir?")
            if resp is None:
                return
            if resp:
                self._save()
        self._stop_playback(update_cursor=False)
        self.destroy()


def open_transcript_editor(master=None, audio_path: str | None = None, transcript_path: str | None = None, cfg=None):
    return TranscriptEditor(master, audio_path=audio_path, transcript_path=transcript_path, cfg=cfg)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    audio = filedialog.askopenfilename(title="Audio/video", filetypes=[("Audio/video", "*.wav *.mp3 *.m4a *.mp4 *.mov *.mkv *.ogg *.flac"), ("Todos", "*.*")])
    transcript = filedialog.askopenfilename(title="Transcripción", filetypes=[("Texto", "*.txt"), ("Todos", "*.*")])
    ed = open_transcript_editor(root, audio or None, transcript or None, None)
    ed.protocol("WM_DELETE_WINDOW", lambda: (ed.destroy(), root.destroy()))
    root.mainloop()
