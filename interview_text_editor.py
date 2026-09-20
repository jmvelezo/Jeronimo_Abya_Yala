from __future__ import annotations

import codecs
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from subprocess_utils import hidden_process_kwargs
from visual_theme import COLORS, asset_path


TXT_FILETYPES = [
    ("Transcripciones de texto", "*.txt"),
    ("Todos los archivos", "*.*"),
]

AUDIO_FILETYPES = [
    ("Audio y video", "*.wav *.mp3 *.m4a *.aac *.ogg *.flac *.wma *.mp4 *.mov *.mkv *.webm"),
    ("Todos los archivos", "*.*"),
]


_TIMECODE_RANGE_RE = re.compile(
    r"\[(?P<meta>[^]\r\n]*?)(?P<start>\d{1,2}:\d{2}:\d{2})\s*-\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2})(?P<tail>[^]\r\n]*)\]"
)


# ---------------------------------------------------------------------------
# Utilidades puras del editor/reproductor


def _appearance_color(value):
    """Resuelve los colores duales de Territorio Vivo para widgets Tk nativos."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[1] if ctk.get_appearance_mode().lower() == "dark" else value[0]
    return value


def _format_media_time(seconds: float, unknown: str = "--:--:--") -> str:
    """Formatea una posición multimedia sin introducir fracciones en la UI."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return unknown
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _clamp_position(value: float, duration: float = 0.0) -> float:
    """Limita una posición al rango reproducible conocido."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    value = max(0.0, value)
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        value = min(value, duration)
    return value


def _timecode_to_seconds(value: str) -> float | None:
    """Convierte HH:MM:SS a segundos; None indica una marca inválida."""
    try:
        hours, minutes, seconds = (int(part) for part in str(value).split(":"))
    except Exception:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return float(hours * 3600 + minutes * 60 + seconds)


def _parse_timed_turns(text: str) -> list[dict]:
    """Extrae turnos temporizados del TXT visible sin depender del pipeline.

    Cada bloque comienza en una línea que contiene un rango HH:MM:SS-HH:MM:SS
    dentro de corchetes. El bloque textual se extiende hasta el comienzo del
    siguiente encabezado temporizado. Los offsets son relativos al texto que
    ve y edita el usuario, por lo que pueden reconstruirse tras cada cambio.
    """
    text = text or ""
    starts: list[dict] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _TIMECODE_RANGE_RE.search(line)
        if match:
            start = _timecode_to_seconds(match.group("start"))
            end = _timecode_to_seconds(match.group("end"))
            if start is not None and end is not None and end >= start:
                starts.append(
                    {
                        "start": start,
                        "end": end,
                        "char_start": offset,
                        "header_end": offset + len(line.rstrip("\r\n")),
                    }
                )
        offset += len(line)

    # splitlines() devuelve [] para texto vacío y no conserva una última línea
    # sin salto; el offset total relevante siempre es len(text).
    for index, turn in enumerate(starts):
        next_start = starts[index + 1]["char_start"] if index + 1 < len(starts) else len(text)
        turn["char_end"] = max(turn["char_start"], next_start)
        # Los timecodes del RAW se escriben con resolución de segundos. Un turno
        # muy corto puede quedar 00:00:10-00:00:10: se le concede un segundo
        # visual mínimo sin alterar el archivo ni el audio.
        if turn["end"] <= turn["start"]:
            turn["effective_end"] = turn["start"] + 1.0
        else:
            turn["effective_end"] = turn["end"]

    return starts


def _find_timed_turn(turns: list[dict], position: float) -> dict | None:
    """Devuelve el turno activo para una posición de audio o None en silencios."""
    try:
        position = float(position)
    except (TypeError, ValueError):
        return None
    candidates = []
    for turn in turns or []:
        start = float(turn.get("start", 0.0) or 0.0)
        end = float(turn.get("effective_end", turn.get("end", 0.0)) or 0.0)
        if start <= position < end:
            candidates.append(turn)
    if not candidates:
        return None
    # Si el redondeo a segundos produce solapamientos, gana el turno que
    # comienza más tarde; a igualdad, el último en el documento.
    return max(candidates, key=lambda item: float(item.get("start", 0.0) or 0.0))


def _probe_media_duration(path: Path) -> float:
    """Obtiene duración con ffprobe. Devuelve 0 si el runtime no está disponible."""
    path = Path(path)
    if not path.is_file():
        return 0.0
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
            **hidden_process_kwargs(),
        )
        if result.returncode != 0:
            return 0.0
        payload = json.loads(result.stdout or "{}")
        duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
        return max(0.0, duration)
    except Exception:
        return 0.0


def _read_text_file(path: Path) -> tuple[str, str, str]:
    """Lee una transcripción conservando codificación y salto de línea cuando es posible."""
    raw = Path(path).read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"

    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig", newline

    try:
        return raw.decode("utf-8"), "utf-8", newline
    except UnicodeDecodeError:
        # Compatibilidad prudente con TXT antiguos creados en Windows.
        return raw.decode("cp1252"), "cp1252", newline


def _write_text_file(path: Path, text: str, encoding: str, newline: str) -> None:
    """Escritura explícita: no altera otros archivos ni normaliza silenciosamente el formato."""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if newline == "\r\n":
        normalized = normalized.replace("\n", "\r\n")
    Path(path).write_bytes(normalized.encode(encoding))


class FFplayTransport:
    """Transporte pequeño sobre ffplay, sin dependencia del pipeline de transcripción.

    ffplay no expone en esta integración un IPC persistente. Pausar conserva la
    posición estimada por reloj monotónico y detiene el proceso; reanudar abre
    ffplay desde esa posición. Un seek en reproducción hace exactamente un
    reinicio al confirmar la nueva posición.
    """

    def __init__(self, clock=None, popen_factory=None):
        self._clock = clock or time.monotonic
        self._popen_factory = popen_factory or subprocess.Popen
        self.audio_path: Path | None = None
        self.duration = 0.0
        self.cursor = 0.0
        self._proc = None
        self._play_origin = 0.0
        self._play_started_at = 0.0

    def load(self, path: Path, duration: float = 0.0) -> None:
        self.stop(update_cursor=False)
        self.audio_path = Path(path).resolve()
        self.duration = max(0.0, float(duration or 0.0))
        self.cursor = 0.0
        self._play_origin = 0.0
        self._play_started_at = 0.0

    def clear(self) -> None:
        self.stop(update_cursor=False)
        self.audio_path = None
        self.duration = 0.0
        self.cursor = 0.0

    def _finalize_if_exited(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            running = proc.poll() is None
        except Exception:
            running = False
        if running:
            return

        estimated = self._play_origin + max(0.0, self._clock() - self._play_started_at)
        if self.duration > 0 and estimated >= self.duration - 0.75:
            estimated = self.duration
        self.cursor = _clamp_position(estimated, self.duration)
        self._proc = None

    def is_playing(self) -> bool:
        self._finalize_if_exited()
        return self._proc is not None

    def current_position(self) -> float:
        self._finalize_if_exited()
        if self._proc is None:
            return _clamp_position(self.cursor, self.duration)
        estimated = self._play_origin + max(0.0, self._clock() - self._play_started_at)
        return _clamp_position(estimated, self.duration)

    def _terminate_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.35)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self._proc = None

    def stop(self, update_cursor: bool = True) -> None:
        if update_cursor and self._proc is not None:
            self.cursor = self.current_position()
        self._terminate_process()

    def pause(self) -> float:
        self.cursor = self.current_position()
        self._terminate_process()
        return self.cursor

    def play(self, start: float | None = None) -> float:
        if self.audio_path is None or not self.audio_path.is_file():
            raise RuntimeError("No hay un audio cargado.")
        ffplay = shutil.which("ffplay")
        if not ffplay:
            raise RuntimeError("No se encontró ffplay en el runtime de Jerónimo.")

        self.stop(update_cursor=True)
        if start is not None:
            self.cursor = _clamp_position(start, self.duration)
        if self.duration > 0 and self.cursor >= self.duration - 0.05:
            self.cursor = 0.0

        cmd = [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-ss",
            f"{self.cursor:.3f}",
            str(self.audio_path),
        ]
        try:
            self._proc = self._popen_factory(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **hidden_process_kwargs(),
            )
        except Exception as exc:
            self._proc = None
            raise RuntimeError(f"No se pudo iniciar la reproducción: {exc}") from exc

        self._play_origin = self.cursor
        self._play_started_at = self._clock()
        return self.cursor

    def seek(self, position: float, resume: bool | None = None) -> float:
        was_playing = self.is_playing() if resume is None else bool(resume)
        if self._proc is not None:
            self._terminate_process()
        self.cursor = _clamp_position(position, self.duration)
        if was_playing:
            self.play(self.cursor)
        return self.cursor

    def skip(self, delta: float) -> float:
        was_playing = self.is_playing()
        target = self.current_position() + float(delta)
        return self.seek(target, resume=was_playing)

    def close(self) -> None:
        self.stop(update_cursor=False)


class InterviewTextEditor(ctk.CTkToplevel):
    """Editor documental local para corregir entrevistas mientras se escucha audio.

    EDIT-1 mantiene apertura/edición/guardado de TXT con respaldo .bak.
    EDIT-2 agrega un transporte multimedia simple con FFplay/FFprobe.
    EDIT-3 sincroniza el documento por los timecodes ya presentes en el TXT,
    sin modificar ni consultar el pipeline de transcripción.
    """

    PLAYBACK_TICK_MS = 120
    SYNC_REBUILD_DELAY_MS = 320
    ACTIVE_TURN_TAG = "audio_active_turn"

    def __init__(
        self,
        master=None,
        transcript_path: str | Path | None = None,
        audio_path: str | Path | None = None,
    ):
        super().__init__(master)
        # HOTFIX 2 ventana: NO retirar (withdraw) un CTkToplevel durante su
        # inicialización. CustomTkinter en Windows puede volver a dejar la ventana
        # oculta al aplicar internamente el color de la barra de título. 1180x840
        # se conserva como geometría de restauración; la maximización se solicita
        # de forma diferida una vez que la Toplevel ya está creada.
        self.title("Jerónimo Abya Yala — Editor de entrevistas")
        self.geometry("1180x840")
        self.minsize(860, 650)
        self.resizable(True, True)
        self.configure(fg_color=COLORS["canvas"])
        try:
            self.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass

        self.transcript_path: Path | None = None
        self._encoding = "utf-8"
        self._newline = "\n"
        self._dirty = False
        self._loading_document = False

        self._transport = FFplayTransport()
        self._dragging_seek = False
        self._drag_was_playing = False
        self._seek_preview = 0.0
        self._tick_job = None
        self._sync_rebuild_job = None
        self._timed_turns: list[dict] = []
        self._active_turn_signature = None
        self._follow_audio_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if transcript_path:
            self._load_path(Path(transcript_path))
        else:
            self._set_document_text("")
            self._refresh_document_meta()
            self.after(80, self.text.focus_set)

        if audio_path:
            self._load_audio_path(Path(audio_path), report_errors=False)
        else:
            self._refresh_audio_meta()

        self._schedule_playback_tick()

        # HOTFIX 2 ventana: no usar ``transient(master)`` y no usar ``withdraw``.
        # La aplicación principal ya tiene su mainloop activo; programar ``zoomed``
        # unas decenas de ms después evita competir con la inicialización interna
        # de CTkToplevel en Windows. El segundo intento es único y sólo cubre el
        # caso en que el primer cambio de estado se solicite demasiado pronto.
        self.after(25, self._maximize_editor_window)
        self.after(180, self._maximize_editor_window)
        # Presentación frontal no modal. Un pulso breve de topmost evita que
        # Windows deje el editor maximizado detrás de la ventana principal; se
        # libera enseguida para conservar Alt+Tab y el orden Z normal.
        self.after(260, self._bring_editor_to_front)

    def _bring_editor_to_front(self) -> None:
        """Trae el editor al frente una sola vez sin volverlo siempre-visible."""
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass
        try:
            self.lift()
            self.focus_force()
        except Exception:
            pass
        self.after(180, self._release_editor_topmost)

    def _release_editor_topmost(self) -> None:
        try:
            if self.winfo_exists():
                self.attributes("-topmost", False)
                self.lift()
        except Exception:
            pass

    def _maximize_editor_window(self) -> bool:
        """Maximiza la ventana sin ocultarla ni alterar su geometría restaurada."""
        try:
            if not self.winfo_exists():
                return False
        except Exception:
            return False
        try:
            self.wm_state("zoomed")
            return True
        except Exception:
            pass
        try:
            self.state("zoomed")
            return True
        except Exception:
            pass
        try:
            self.attributes("-zoomed", True)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=26, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            title_box,
            text="EDITOR DE ENTREVISTAS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["terracotta"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="Corrección documental",
            font=ctk.CTkFont(size=25, weight="bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w", pady=(1, 0))

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(
            actions,
            text="Abrir",
            width=92,
            height=36,
            fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"],
            text_color=COLORS["text"],
            command=self._open_dialog,
        ).pack(side="left", padx=(0, 7))
        ctk.CTkButton(
            actions,
            text="Guardar como",
            width=118,
            height=36,
            fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"],
            text_color=COLORS["text"],
            command=self._save_as,
        ).pack(side="left", padx=(0, 7))
        self.btn_save = ctk.CTkButton(
            actions,
            text="Guardar",
            width=104,
            height=36,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["on_accent"],
            command=self._save,
        )
        self.btn_save.pack(side="left")

        workspace = ctk.CTkFrame(
            self,
            fg_color=COLORS["card_alt"],
            corner_radius=16,
            border_width=1,
            border_color=COLORS["border"],
        )
        workspace.grid(row=1, column=0, sticky="nsew", padx=26, pady=(0, 10))
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(1, weight=1)

        meta = ctk.CTkFrame(workspace, fg_color="transparent")
        meta.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 9))
        meta.grid_columnconfigure(0, weight=1)
        self.lbl_file = ctk.CTkLabel(
            meta,
            text="Documento sin guardar",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.lbl_file.grid(row=0, column=0, sticky="ew")
        self.lbl_state = ctk.CTkLabel(
            meta,
            text="Sin cambios",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
            anchor="e",
        )
        self.lbl_state.grid(row=0, column=1, sticky="e", padx=(12, 0))

        document_wrap = ctk.CTkFrame(workspace, fg_color="transparent")
        document_wrap.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 18))
        document_wrap.grid_columnconfigure(0, weight=1)
        document_wrap.grid_rowconfigure(0, weight=1)

        paper = ctk.CTkFrame(
            document_wrap,
            fg_color=COLORS["card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"],
        )
        paper.grid(row=0, column=0, sticky="nsew")
        paper.grid_columnconfigure(0, weight=1)
        paper.grid_rowconfigure(0, weight=1)

        text_bg = _appearance_color(COLORS["card"])
        text_fg = _appearance_color(COLORS["text"])
        select_bg = _appearance_color(COLORS["accent_soft"])
        insert_color = _appearance_color(COLORS["terracotta"])

        self.text = tk.Text(
            paper,
            wrap="word",
            undo=True,
            autoseparators=True,
            maxundo=-1,
            background=text_bg,
            foreground=text_fg,
            insertbackground=insert_color,
            selectbackground=select_bg,
            selectforeground=text_fg,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=42,
            pady=34,
            spacing1=2,
            spacing3=7,
            font=("Segoe UI", 12),
        )
        self.text.tag_configure(
            self.ACTIVE_TURN_TAG,
            background=_appearance_color(COLORS["accent_soft"]),
        )
        scrollbar = ctk.CTkScrollbar(paper, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.grid(row=0, column=0, sticky="nsew", padx=(1, 0), pady=1)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 4), pady=8)

        self.text.bind("<<Modified>>", self._on_text_modified)
        self.text.bind("<Control-s>", self._shortcut_save)
        self.text.bind("<Control-S>", self._shortcut_save_as)
        self.text.bind("<Control-o>", self._shortcut_open)
        # No se enlaza Space: mientras el usuario corrige, la barra espaciadora
        # siempre debe pertenecer al documento y nunca al reproductor.

        self._build_player()

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 14))
        footer.grid_columnconfigure(0, weight=1)
        self.lbl_status = ctk.CTkLabel(
            footer,
            text="Edición local · el documento sólo cambia al guardar",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.lbl_status.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            footer,
            text="Ctrl+S guardar  ·  Ctrl+Z deshacer  ·  Ctrl+Y rehacer",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=10),
            anchor="e",
        ).grid(row=0, column=1, sticky="e", padx=(16, 0))

        # Tk maneja undo/redo nativamente; sólo añadimos Ctrl+Y para Windows.
        self.text.bind("<Control-y>", self._shortcut_redo)
        self.text.bind("<Control-Y>", self._shortcut_redo)

    def _build_player(self) -> None:
        player = ctk.CTkFrame(
            self,
            fg_color=COLORS["card_alt"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        player.grid(row=2, column=0, sticky="ew", padx=26, pady=(0, 10))
        player.grid_columnconfigure(0, weight=1)

        source = ctk.CTkFrame(player, fg_color="transparent")
        source.grid(row=0, column=0, sticky="ew", padx=18, pady=(11, 3))
        source.grid_columnconfigure(0, weight=1)
        self.lbl_audio = ctk.CTkLabel(
            source,
            text="Audio: ninguno",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.lbl_audio.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            source,
            text="Abrir audio",
            width=104,
            height=30,
            fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"],
            text_color=COLORS["text"],
            command=self._pick_audio,
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.lbl_sync = ctk.CTkLabel(
            source,
            text="Sin marcas temporales · seguimiento no disponible",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=10),
            anchor="w",
        )
        self.lbl_sync.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.switch_follow = ctk.CTkSwitch(
            source,
            text="Seguir audio",
            variable=self._follow_audio_var,
            onvalue=True,
            offvalue=False,
            command=self._on_follow_audio_toggle,
            fg_color=COLORS["border"],
            progress_color=COLORS["accent"],
            button_color=COLORS["terracotta"],
            button_hover_color=COLORS["terracotta_hover"],
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=10),
        )
        self.switch_follow.grid(row=1, column=1, sticky="e", padx=(12, 0), pady=(2, 0))

        controls = ctk.CTkFrame(player, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=18, pady=(2, 2))
        controls.grid_columnconfigure(3, weight=1)

        ctk.CTkButton(
            controls,
            text="−5 s",
            width=64,
            height=32,
            fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"],
            text_color=COLORS["text"],
            command=lambda: self._skip_audio(-5.0),
        ).grid(row=0, column=0, padx=(0, 6))

        self.btn_play = ctk.CTkButton(
            controls,
            text="▶ Reproducir",
            width=112,
            height=34,
            fg_color=COLORS["terracotta"],
            hover_color=COLORS["terracotta_hover"],
            text_color=COLORS["on_accent"],
            command=self._toggle_playback,
        )
        self.btn_play.grid(row=0, column=1, padx=6)

        ctk.CTkButton(
            controls,
            text="+5 s",
            width=64,
            height=32,
            fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"],
            text_color=COLORS["text"],
            command=lambda: self._skip_audio(5.0),
        ).grid(row=0, column=2, padx=6)

        self.lbl_time = ctk.CTkLabel(
            controls,
            text="00:00:00 / --:--:--",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="e",
        )
        self.lbl_time.grid(row=0, column=4, sticky="e")

        self.seek_slider = ctk.CTkSlider(
            player,
            from_=0,
            to=1,
            number_of_steps=None,
            height=18,
            button_length=16,
            progress_color=COLORS["water"],
            button_color=COLORS["terracotta"],
            button_hover_color=COLORS["terracotta_hover"],
            fg_color=COLORS["border"],
            command=self._on_seek_slider,
        )
        self.seek_slider.grid(row=2, column=0, sticky="ew", padx=20, pady=(3, 12))
        self.seek_slider.set(0)
        self.seek_slider.configure(state="disabled")
        self.seek_slider.bind("<ButtonPress-1>", self._on_seek_press, add="+")
        self.seek_slider.bind("<ButtonRelease-1>", self._on_seek_release, add="+")

    # --------------------------------------------------------------- document
    def _set_document_text(self, text: str) -> None:
        self._loading_document = True
        try:
            self.text.delete("1.0", "end")
            self.text.insert("1.0", text)
            self.text.edit_reset()
            self.text.edit_modified(False)
            self._dirty = False
        finally:
            self._loading_document = False
        self._refresh_document_meta()
        self._rebuild_timed_turns()

    def _document_text(self) -> str:
        # "end-1c" evita añadir un salto artificial de Tk al archivo.
        return self.text.get("1.0", "end-1c")

    def _load_path(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists() or not path.is_file():
            messagebox.showerror("Editor de entrevistas", f"No se encontró el archivo:\n\n{path}", parent=self)
            return False
        try:
            text, encoding, newline = _read_text_file(path)
        except Exception as exc:
            messagebox.showerror("Editor de entrevistas", f"No se pudo abrir el texto:\n\n{exc}", parent=self)
            return False

        self.transcript_path = path.resolve()
        self._encoding = encoding
        self._newline = newline
        self._set_document_text(text)
        self._set_status(f"Documento abierto · {self.transcript_path.name}")
        self.after(30, self.text.focus_set)
        return True

    def _open_dialog(self) -> None:
        if not self._confirm_safe_to_replace_document():
            return
        selected = filedialog.askopenfilename(
            title="Abrir transcripción",
            filetypes=TXT_FILETYPES,
            parent=self,
        )
        if selected:
            self._load_path(Path(selected))

    def _save(self) -> bool:
        if self.transcript_path is None:
            return self._save_as()
        return self._save_to_path(self.transcript_path, preserve_current_format=True)

    def _save_as(self) -> bool:
        initial_name = self.transcript_path.name if self.transcript_path else "entrevista_corregida.txt"
        initial_dir = str(self.transcript_path.parent) if self.transcript_path else None
        selected = filedialog.asksaveasfilename(
            title="Guardar transcripción como",
            defaultextension=".txt",
            filetypes=TXT_FILETYPES,
            initialfile=initial_name,
            initialdir=initial_dir,
            parent=self,
        )
        if not selected:
            return False
        return self._save_to_path(Path(selected), preserve_current_format=False)

    def _save_to_path(self, path: Path, preserve_current_format: bool) -> bool:
        path = Path(path)
        encoding = self._encoding if preserve_current_format else "utf-8"
        newline = self._newline if preserve_current_format else "\n"

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                backup = path.with_name(path.name + ".bak")
                shutil.copy2(path, backup)
            _write_text_file(path, self._document_text(), encoding, newline)
        except Exception as exc:
            messagebox.showerror("Editor de entrevistas", f"No se pudo guardar el documento:\n\n{exc}", parent=self)
            return False

        self.transcript_path = path.resolve()
        self._encoding = encoding
        self._newline = newline
        self._dirty = False
        self.text.edit_modified(False)
        self._refresh_document_meta()
        self._set_status(f"Guardado · {self.transcript_path.name}")
        return True

    # --------------------------------------------------------------- playback
    def _pick_audio(self) -> None:
        initial_dir = str(self.transcript_path.parent) if self.transcript_path else None
        selected = filedialog.askopenfilename(
            title="Abrir audio de la entrevista",
            filetypes=AUDIO_FILETYPES,
            initialdir=initial_dir,
            parent=self,
        )
        if selected:
            self._load_audio_path(Path(selected), report_errors=True)

    def _load_audio_path(self, path: Path, report_errors: bool = True) -> bool:
        path = Path(path)
        if not path.exists() or not path.is_file():
            if report_errors:
                messagebox.showerror("Reproductor", f"No se encontró el archivo:\n\n{path}", parent=self)
            return False

        duration = _probe_media_duration(path)
        self._transport.load(path, duration=duration)
        self._dragging_seek = False
        self._seek_preview = 0.0
        self._refresh_audio_meta()
        self._sync_player_ui(force_position=0.0)

        if duration > 0:
            self._set_status(f"Audio abierto · {path.name}")
        else:
            self._set_status(f"Audio abierto · {path.name} · duración no disponible")
            if report_errors and not shutil.which("ffprobe"):
                messagebox.showwarning(
                    "Reproductor",
                    "El audio se cargó, pero no se encontró ffprobe. La reproducción puede funcionar, "
                    "aunque la barra de avance queda deshabilitada porque no se conoce la duración.",
                    parent=self,
                )
        return True

    def _refresh_audio_meta(self) -> None:
        path = self._transport.audio_path
        if path is None:
            self.lbl_audio.configure(text="Audio: ninguno", text_color=COLORS["muted"])
            self.seek_slider.configure(from_=0, to=1, state="disabled")
            self.seek_slider.set(0)
            self.lbl_time.configure(text="00:00:00 / --:--:--")
            return

        self.lbl_audio.configure(text=f"Audio: {path.name}", text_color=COLORS["text"])
        if self._transport.duration > 0:
            self.seek_slider.configure(from_=0, to=self._transport.duration, state="normal")
        else:
            self.seek_slider.configure(from_=0, to=1, state="disabled")
            self.seek_slider.set(0)

    def _toggle_playback(self) -> None:
        if self._transport.is_playing():
            self._transport.pause()
            self._sync_player_ui()
            return
        try:
            self._transport.play()
            self._sync_player_ui()
        except Exception as exc:
            messagebox.showwarning("Reproducción", str(exc), parent=self)
            self._sync_player_ui()

    def _skip_audio(self, delta: float) -> None:
        if self._transport.audio_path is None:
            messagebox.showinfo("Reproductor", "Primero abre el audio de la entrevista.", parent=self)
            return
        try:
            self._transport.skip(delta)
        except Exception as exc:
            messagebox.showwarning("Reproducción", str(exc), parent=self)
        self._sync_player_ui()

    def _on_seek_press(self, _event=None) -> None:
        if self._transport.duration <= 0:
            return
        self._dragging_seek = True
        self._drag_was_playing = self._transport.is_playing()
        if self._drag_was_playing:
            # Se detiene una sola vez al iniciar el drag. Durante el movimiento
            # no se crea ningún proceso nuevo.
            self._transport.pause()
        self._seek_preview = float(self.seek_slider.get())
        self._sync_play_button()

    def _on_seek_slider(self, value) -> None:
        if not self._dragging_seek:
            return
        self._seek_preview = _clamp_position(value, self._transport.duration)
        self._update_time_label(self._seek_preview)
        self._sync_text_to_position(self._seek_preview)

    def _on_seek_release(self, _event=None) -> None:
        if not self._dragging_seek:
            return
        target = _clamp_position(self.seek_slider.get(), self._transport.duration)
        resume = self._drag_was_playing
        self._dragging_seek = False
        self._drag_was_playing = False
        try:
            self._transport.seek(target, resume=resume)
        except Exception as exc:
            messagebox.showwarning("Reproducción", str(exc), parent=self)
        self._sync_player_ui()

    def _schedule_playback_tick(self) -> None:
        try:
            self._tick_job = self.after(self.PLAYBACK_TICK_MS, self._playback_tick)
        except tk.TclError:
            self._tick_job = None

    def _playback_tick(self) -> None:
        self._tick_job = None
        try:
            if not self._dragging_seek:
                position = self._transport.current_position()
                self._sync_player_ui(force_position=position)
            else:
                # El reloj muestra el destino que se está eligiendo, no el reloj
                # interno del reproductor detenido durante el arrastre.
                self._update_time_label(self._seek_preview)
                self._sync_play_button()
        except tk.TclError:
            return
        self._schedule_playback_tick()

    def _sync_player_ui(self, force_position: float | None = None) -> None:
        position = self._transport.current_position() if force_position is None else force_position
        position = _clamp_position(position, self._transport.duration)
        if self._transport.duration > 0 and not self._dragging_seek:
            try:
                self.seek_slider.set(position)
            except tk.TclError:
                pass
        self._update_time_label(position)
        self._sync_play_button()
        self._sync_text_to_position(position)

    def _update_time_label(self, position: float) -> None:
        current = _format_media_time(position)
        total = _format_media_time(self._transport.duration) if self._transport.duration > 0 else "--:--:--"
        self.lbl_time.configure(text=f"{current} / {total}")

    def _sync_play_button(self) -> None:
        self.btn_play.configure(text="⏸ Pausar" if self._transport.is_playing() else "▶ Reproducir")

    # ---------------------------------------------------------- text/audio sync
    def _schedule_sync_rebuild(self) -> None:
        if self._sync_rebuild_job is not None:
            try:
                self.after_cancel(self._sync_rebuild_job)
            except Exception:
                pass
        try:
            self._sync_rebuild_job = self.after(self.SYNC_REBUILD_DELAY_MS, self._rebuild_timed_turns)
        except tk.TclError:
            self._sync_rebuild_job = None

    def _rebuild_timed_turns(self) -> None:
        self._sync_rebuild_job = None
        self._timed_turns = _parse_timed_turns(self._document_text())
        self._active_turn_signature = None
        try:
            self.text.tag_remove(self.ACTIVE_TURN_TAG, "1.0", "end")
        except tk.TclError:
            return

        count = len(self._timed_turns)
        if count:
            self.lbl_sync.configure(
                text=f"Sincronización por marcas de tiempo · {count} turno{'s' if count != 1 else ''}",
                text_color=COLORS["water"],
            )
            self.switch_follow.configure(state="normal")
            self._sync_text_to_position(self._transport.current_position())
        else:
            self.lbl_sync.configure(
                text="Sin marcas temporales · seguimiento no disponible",
                text_color=COLORS["muted"],
            )
            self.switch_follow.configure(state="disabled")

    def _sync_text_to_position(self, position: float, force_scroll: bool = False) -> None:
        turn = _find_timed_turn(self._timed_turns, position)
        signature = None if turn is None else (turn["char_start"], turn["char_end"], turn["start"], turn["end"])
        if signature == self._active_turn_signature and not force_scroll:
            return

        try:
            self.text.tag_remove(self.ACTIVE_TURN_TAG, "1.0", "end")
            if turn is None:
                self._active_turn_signature = None
                return

            start_index = f"1.0 + {int(turn['char_start'])} chars"
            end_index = f"1.0 + {int(turn['char_end'])} chars"
            self.text.tag_add(self.ACTIVE_TURN_TAG, start_index, end_index)
            # La selección del usuario debe seguir viéndose por encima del
            # resaltado de reproducción. No se mueve el cursor de inserción.
            self.text.tag_raise("sel")
            self._active_turn_signature = signature
            if bool(self._follow_audio_var.get()):
                self.text.see(start_index)
        except tk.TclError:
            self._active_turn_signature = None

    def _on_follow_audio_toggle(self) -> None:
        if not bool(self._follow_audio_var.get()):
            return
        self._sync_text_to_position(self._transport.current_position(), force_scroll=True)

    # --------------------------------------------------------------- state/UI
    def _on_text_modified(self, _event=None) -> None:
        if self._loading_document:
            self.text.edit_modified(False)
            return
        if self.text.edit_modified():
            self._dirty = True
            self.text.edit_modified(False)
            self._refresh_document_meta()
            self._schedule_sync_rebuild()

    def _refresh_document_meta(self) -> None:
        if not hasattr(self, "lbl_file"):
            return
        name = self.transcript_path.name if self.transcript_path else "Documento sin guardar"
        self.lbl_file.configure(text=name)
        self.lbl_state.configure(
            text="Cambios sin guardar" if self._dirty else "Guardado" if self.transcript_path else "Sin cambios",
            text_color=COLORS["terracotta"] if self._dirty else COLORS["muted"],
        )
        marker = " *" if self._dirty else ""
        self.title(f"Jerónimo Abya Yala — Editor de entrevistas — {name}{marker}")

    def _set_status(self, text: str) -> None:
        self.lbl_status.configure(text=text)

    def _confirm_safe_to_replace_document(self) -> bool:
        if not self._dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Cambios sin guardar",
            "Hay cambios sin guardar. ¿Quieres guardarlos antes de continuar?",
            parent=self,
        )
        if answer is None:
            return False
        if answer is True:
            return self._save()
        return True

    # --------------------------------------------------------------- shortcuts
    def _shortcut_save(self, _event=None):
        self._save()
        return "break"

    def _shortcut_save_as(self, _event=None):
        self._save_as()
        return "break"

    def _shortcut_open(self, _event=None):
        self._open_dialog()
        return "break"

    def _shortcut_redo(self, _event=None):
        try:
            self.text.edit_redo()
        except tk.TclError:
            pass
        return "break"

    # ---------------------------------------------------------------- close
    def _on_close(self) -> None:
        if not self._confirm_safe_to_replace_document():
            return
        if self._tick_job is not None:
            try:
                self.after_cancel(self._tick_job)
            except Exception:
                pass
            self._tick_job = None
        if self._sync_rebuild_job is not None:
            try:
                self.after_cancel(self._sync_rebuild_job)
            except Exception:
                pass
            self._sync_rebuild_job = None
        self._transport.close()
        self.destroy()


def open_interview_text_editor(
    master=None,
    transcript_path: str | Path | None = None,
    audio_path: str | Path | None = None,
):
    return InterviewTextEditor(master=master, transcript_path=transcript_path, audio_path=audio_path)
