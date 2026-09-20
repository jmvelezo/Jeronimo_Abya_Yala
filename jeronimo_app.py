from __future__ import annotations

import faulthandler
import json
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from queue import Empty, Queue
from pathlib import Path
from subprocess_utils import hidden_process_kwargs


def _ensure_gui_standard_streams() -> None:
    """Evita fallos de librerías que escriben a stdout/stderr en un EXE sin consola.

    PyInstaller ``console=False`` puede dejar ``sys.stdout``/``sys.stderr`` en
    ``None``. Algunas dependencias de descarga (por ejemplo barras de progreso
    de Hugging Face/tqdm) asumen un stream con ``write()`` y fallan con
    ``'NoneType' object has no attribute 'write'``. Sólo sustituimos streams
    ausentes por ``os.devnull``; si existe una consola/captura real se conserva.
    """
    modes = (("stdin", "r"), ("stdout", "w"), ("stderr", "w"))
    for name, mode in modes:
        if getattr(sys, name, None) is not None:
            continue
        try:
            stream = open(os.devnull, mode, encoding="utf-8")
            setattr(sys, name, stream)
        except Exception:
            # La aplicación puede seguir; las descargas desactivan además sus
            # barras de progreso de consola explícitamente.
            pass


_ensure_gui_standard_streams()
# La interfaz ya presenta su propio progreso. Evitar barras de consola de HF
# también previene escrituras a stderr en ejecutables GUI de Windows.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import tkinter.filedialog as fd
import tkinter.messagebox as mb

from portable_runtime import bootstrap_portable_environment, portable_status, probe_diarization_worker
bootstrap_portable_environment()

import customtkinter as ctk
from PIL import Image

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    class _DnDRoot(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            ctk.CTk.__init__(self, *args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
    HAS_DND = True
except Exception:
    DND_FILES = None
    HAS_DND = False
    class _DnDRoot(ctk.CTk):
        pass

from core_transcriber import (
    APP_LOG_DIR,
    LOG_DIR_ENV,
    TranscriberConfig,
    check_environment_status,
    load_api_key_from_env,
    run_batch,
)
from credential_store import resolve_secret, set_secure_secret
from job_engine import (
    Job,
    JobTracker,
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_ERROR,
    format_duration,
    stage_label,
)
from model_manager import TEXT_MODELS, ollama_inventory, ollama_model_present
from model_manager_ui import open_model_manager, ask_recommended_model_action
from onboarding import load_state, startup_configuration_decision
from onboarding_ui import OnboardingWizard
from runtime_isolation import assess_runtime
from system_diagnostics import run_system_diagnostics, format_diagnostics_report
from system_monitor import SystemResourceMonitor, SystemSnapshot
from text_providers import DEFAULT_OPENAI_COMPATIBLE_URL, DEFAULT_OLLAMA_URL, TextProviderSettings, describe_endpoint, test_provider
from transcript_editor import open_transcript_editor
from interview_text_editor import open_interview_text_editor
from ui_help import HelpBubbleButton
from visual_theme import CARD_RADIUS, COLORS, SIDEBAR_WIDTH, asset_path
from ui_workflow import (
    DEFAULT_CLEAN_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
    GENERIC_WHISPER_PROMPT,
    PROFILE_DIARIZATION,
    PROFILE_LOCAL,
    PROFILE_OPENAI,
    TEXT_COMPATIBLE,
    TEXT_NONE,
    TEXT_OLLAMA,
    TEXT_OPENAI,
    WorkflowSelection,
    assess_privacy,
    build_config,
    find_primary_transcript,
    new_output_files,
    recommended_profile,
    recommended_text_model,
    snapshot_directory,
)


ctk.set_appearance_mode("system")
ctk.set_default_color_theme("green")

APP_TITLE = "Jerónimo Abya Yala — Desgrabador de entrevistas"


_FATAL_CRASH_STREAM = None


def _enable_fatal_crash_log() -> None:
    """Activa un rastro persistente para fallos nativos que no llegan a Python.

    El archivo se mantiene abierto durante toda la vida del proceso porque
    ``faulthandler`` escribe directamente sobre su descriptor. No contiene
    audio ni texto de entrevistas; sólo trazas técnicas del intérprete.
    """
    global _FATAL_CRASH_STREAM
    if _FATAL_CRASH_STREAM is not None:
        return
    try:
        raw_dir = os.getenv(LOG_DIR_ENV, "").strip()
        log_dir = Path(raw_dir).expanduser() if raw_dir else APP_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = open(log_dir / "fatal_crash.log", "a", encoding="utf-8", buffering=1)
        stream.write("\n=== Inicio de sesión Jerónimo Abya Yala ===\n")
        stream.flush()
        faulthandler.enable(file=stream, all_threads=True)
        _FATAL_CRASH_STREAM = stream
    except Exception:
        # El diagnóstico nunca debe impedir que la aplicación arranque.
        _FATAL_CRASH_STREAM = None


def _record_ui_dispatch_exception() -> None:
    """Conserva una excepción del despachador sin detener el bucle de Tk."""
    stream = _FATAL_CRASH_STREAM
    if stream is None:
        return
    try:
        stream.write("ERROR Python en despachador de UI:\n")
        traceback.print_exc(file=stream)
        stream.flush()
    except Exception:
        pass


_enable_fatal_crash_log()


# EDIT-4: resolución conservadora resultado -> audio. El manifiesto conserva sólo
# nombres de archivo por privacidad; la ruta se reconstruye únicamente contra
# las entradas que pertenecen al trabajo actual y siempre se verifica en disco.
def _is_interview_editor_result(path: Path) -> bool:
    return Path(path).name.lower().endswith(".raw.txt")


def _read_manifest_object(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _find_manifest_for_result(result_path: Path) -> tuple[Path | None, dict | None]:
    result_path = Path(result_path)
    parent = result_path.parent
    name = result_path.name

    # Camino directo para el RAW actual: <base>.raw.txt -> <base>.manifest.json.
    if name.lower().endswith(".raw.txt"):
        candidate = parent / (name[:-8] + ".manifest.json")
        data = _read_manifest_object(candidate) if candidate.is_file() else None
        if data is not None:
            generated = data.get("generated_files")
            if isinstance(generated, list) and name in {str(item) for item in generated}:
                return candidate, data

    # Fallback defensivo: buscar un manifiesto que declare exactamente el archivo.
    matches: list[tuple[Path, dict]] = []
    try:
        manifests = parent.glob("*.manifest.json")
    except Exception:
        manifests = ()
    for manifest in manifests:
        data = _read_manifest_object(manifest)
        if data is None:
            continue
        generated = data.get("generated_files")
        if isinstance(generated, list) and name in {str(item) for item in generated}:
            matches.append((manifest, data))
    return matches[0] if len(matches) == 1 else (None, None)


def _resolve_audio_for_result(
    result_path: Path,
    *,
    input_path: Path | None = None,
    selected_files: list[Path] | None = None,
) -> tuple[Path | None, Path | None]:
    """Devuelve (audio, manifiesto) sólo cuando la relación es verificable.

    Nunca busca por parecido ni por stem: exige el nombre exacto registrado por
    el manifiesto y una única entrada existente del trabajo actual.
    """
    manifest_path, data = _find_manifest_for_result(Path(result_path))
    if manifest_path is None or data is None:
        return None, None
    input_name = str(data.get("input_file_name") or "").strip()
    if (
        not input_name
        or "/" in input_name
        or "\\" in input_name
        or Path(input_name).name != input_name
        or Path(input_name).suffix.lower() not in {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".mkv", ".ogg", ".flac"}
    ):
        return None, manifest_path

    candidates: list[Path] = []
    for item in selected_files or []:
        candidate = Path(item)
        if candidate.name == input_name and candidate.is_file():
            candidates.append(candidate)

    if input_path is not None:
        root = Path(input_path)
        if root.is_file() and root.name == input_name:
            candidates.append(root)
        elif root.is_dir():
            candidate = root / input_name
            if candidate.is_file():
                candidates.append(candidate)

    # Deduplicar por ruta absoluta sin resolver symlinks inexistentes.
    unique: dict[str, Path] = {}
    for candidate in candidates:
        try:
            key = os.path.normcase(str(candidate.absolute()))
        except Exception:
            key = os.path.normcase(str(candidate))
        unique[key] = candidate
    if len(unique) != 1:
        return None, manifest_path
    return next(iter(unique.values())), manifest_path

PROFILE_LABELS = {
    "Calidad + hablantes": PROFILE_DIARIZATION,
    "Local sin hablantes": PROFILE_LOCAL,
    "OpenAI API": PROFILE_OPENAI,
}
PROFILE_LABELS_INV = {v: k for k, v in PROFILE_LABELS.items()}

TEXT_LABELS = {
    "Local · Ollama": TEXT_OLLAMA,
    "OpenAI": TEXT_OPENAI,
    "API compatible": TEXT_COMPATIBLE,
    "Sin IA de texto": TEXT_NONE,
}
TEXT_LABELS_INV = {v: k for k, v in TEXT_LABELS.items()}

STAGE_ORDER = [
    "preparing", "transcribing", "diarizing", "cleaning", "summarizing", "exporting", "finalizing"
]


def _open_path(path: Path) -> None:
    path = Path(path)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif os.sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], **hidden_process_kwargs())
        else:
            subprocess.Popen(["xdg-open", str(path)], **hidden_process_kwargs())
    except Exception as exc:
        mb.showerror("Abrir", f"No se pudo abrir:\n{path}\n\n{exc}")


def _short_path(path: str | Path, max_chars: int = 74) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text
    keep = max(12, (max_chars - 3) // 2)
    return text[:keep] + "..." + text[-keep:]


HelpButton = HelpBubbleButton


class Card(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=COLORS["card"], corner_radius=CARD_RADIUS, border_width=1, border_color=COLORS["border"], **kwargs)


class AdvancedDialog(ctk.CTkToplevel):
    """Configuración técnica fuera del flujo principal.

    No altera nada hasta pulsar Guardar. Las claves pueden mantenerse sólo en la sesión
    o guardarse en el almacén seguro del sistema.
    """

    def __init__(self, master: "JeronimoApp"):
        super().__init__(master)
        self.app = master
        self.title("Jerónimo Abya Yala — Opciones avanzadas")
        self.geometry("940x800")
        self.minsize(820, 680)
        self.configure(fg_color=COLORS["canvas"])
        try:
            self.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass
        self.transient(master)
        self.grab_set()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Cabecera visual Territorio Vivo. No interviene en la configuración.
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="CONFIGURACIÓN TÉCNICA", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["terracotta"], anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Opciones avanzadas", font=ctk.CTkFont(size=28, weight="bold"), text_color=COLORS["text"], anchor="w").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ctk.CTkLabel(header, text="Ajustes técnicos fuera del flujo principal. Nada se aplica hasta pulsar Guardar.", text_color=COLORS["muted"], anchor="w").grid(row=2, column=0, sticky="w", pady=(3, 0))
        sovereignty = ctk.CTkFrame(header, fg_color=COLORS["accent_soft"], corner_radius=10)
        sovereignty.grid(row=0, column=1, rowspan=3, sticky="e", padx=(18, 0))
        ctk.CTkLabel(sovereignty, text="●  CONTROL LOCAL", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["success"]).pack(padx=14, pady=9)

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(4, 8))
        scroll.grid_columnconfigure(0, weight=1)

        def make_section(row_index: int, number: str, title: str, subtitle: str, help_title: str | None = None, help_text: str | None = None):
            card = Card(scroll)
            card.grid(row=row_index, column=0, sticky="ew", pady=7)
            card.grid_columnconfigure(1, weight=1)
            badge = ctk.CTkLabel(card, text=number, width=36, height=26, corner_radius=13, fg_color=COLORS["accent_soft"], text_color=COLORS["accent"], font=ctk.CTkFont(size=11, weight="bold"))
            badge.grid(row=0, column=0, padx=(18, 10), pady=(17, 11), sticky="nw")
            title_box = ctk.CTkFrame(card, fg_color="transparent")
            title_box.grid(row=0, column=1, sticky="ew", pady=(14, 10))
            ctk.CTkLabel(title_box, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=COLORS["text"], anchor="w").pack(anchor="w")
            ctk.CTkLabel(title_box, text=subtitle, text_color=COLORS["muted"], anchor="w").pack(anchor="w", pady=(1, 0))
            if help_title and help_text:
                HelpButton(card, title=help_title, help_text=help_text).grid(row=0, column=2, padx=(8, 18), pady=(17, 10), sticky="ne")
            body = ctk.CTkFrame(card, fg_color="transparent")
            body.grid(row=1, column=0, columnspan=3, sticky="ew", padx=18, pady=(0, 18))
            body.grid_columnconfigure(1, weight=1)
            return body

        # 01 · Transcripción local
        section = make_section(0, "01", "Transcripción", "Motor, dispositivo y tratamiento previo del audio.", "Transcripción", "Estos ajustes conservan los mismos motores y defaults del programa. Sólo cambian cuando guardas esta ventana.")
        row = 0
        ctk.CTkLabel(section, text="Modelo faster-whisper:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.local_model = ctk.CTkComboBox(section, values=["large-v3", "large-v3-turbo", "medium", "small"], state="readonly", fg_color=COLORS["card_alt"], border_color=COLORS["border"], button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
        self.local_model.set(master.local_fast_model)
        self.local_model.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="Modelo local", help_text="Se usa en el perfil 'Local sin hablantes'. El baseline de diarización sigue usando WhisperX large-v2.").grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="Dispositivo local:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.local_device = ctk.CTkComboBox(section, values=["auto", "cuda", "cpu"], state="readonly", fg_color=COLORS["card_alt"], border_color=COLORS["border"], button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
        self.local_device.set(master.local_device)
        self.local_device.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="Dispositivo local", help_text="Auto intenta usar CUDA cuando está disponible y cae a CPU si corresponde. CUDA usa una GPU NVIDIA compatible y suele acelerar transcripción; CPU funciona sin GPU pero puede ser considerablemente más lento.").grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="Hablantes esperados:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.speakers = ctk.CTkComboBox(section, values=["Auto", "1", "2", "3", "4", "5", "6"], state="readonly", fg_color=COLORS["card_alt"], border_color=COLORS["border"], button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
        self.speakers.set("Auto" if master.target_speakers == 0 else str(master.target_speakers))
        self.speakers.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="Hablantes esperados", help_text="0/Auto no fuerza una cantidad. Jerónimo conserva el comportamiento actual del pipeline; este dato no se usa para inventar identidades.").grid(row=row, column=2)
        row += 1

        self.audio_enhance = ctk.BooleanVar(value=master.audio_enhancement)
        ctk.CTkCheckBox(section, text="Reducción de ruido + normalización", variable=self.audio_enhance, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], border_color=COLORS["border"]).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))
        HelpButton(section, title="Mejora de audio", help_text="Aplica el filtro histórico del programa antes de transcribir. Puede ayudar con ruido, pero también modificar señales acústicas débiles. Se conserva activado para no cambiar el baseline actual.").grid(row=row, column=2)

        # 02 · Texto local y APIs
        section = make_section(1, "02", "Modelos y APIs de texto", "Distingue recursos en tu equipo de servicios externos.", "Procesamiento de texto", "Ollama local mantiene el texto en este equipo. OpenAI y endpoints remotos implican salida de datos según la configuración elegida.")
        row = 0
        local_tag = ctk.CTkLabel(section, text="LOCAL", width=54, height=22, corner_radius=11, fg_color=COLORS["accent_soft"], text_color=COLORS["success"], font=ctk.CTkFont(size=9, weight="bold"))
        local_tag.grid(row=row, column=0, sticky="w", pady=(2, 8))
        ctk.CTkLabel(section, text="Ollama · procesamiento bajo tu control", text_color=COLORS["muted"], anchor="w").grid(row=row, column=1, columnspan=2, sticky="w", padx=10, pady=(2, 8))
        row += 1

        ctk.CTkLabel(section, text="URL Ollama:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.ollama_url = ctk.CTkEntry(section, fg_color=COLORS["card_alt"], border_color=COLORS["border"])
        self.ollama_url.insert(0, master.ollama_url)
        self.ollama_url.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="URL de Ollama", help_text=f"Por defecto {DEFAULT_OLLAMA_URL} es el motor privado administrado por Jerónimo. Si indicas otra IP, hostname o servidor remoto, el texto será enviado a ese equipo y Jerónimo lo tratará como procesamiento externo.").grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="Modelo Ollama:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.ollama_model = ctk.CTkComboBox(section, values=master.ollama_models or [master.ollama_model or "qwen3:4b"], state="normal", fg_color=COLORS["card_alt"], border_color=COLORS["border"], button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"])
        self.ollama_model.set(master.ollama_model)
        self.ollama_model.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        ctk.CTkButton(section, text="Probar", width=76, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=self._test_ollama).grid(row=row, column=2)
        row += 1

        external_separator = ctk.CTkFrame(section, height=1, fg_color=COLORS["border"])
        external_separator.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 12))
        row += 1
        external_tag = ctk.CTkLabel(section, text="EXTERNO", width=62, height=22, corner_radius=11, fg_color=COLORS["external_soft"], text_color=COLORS["terracotta"], font=ctk.CTkFont(size=9, weight="bold"))
        external_tag.grid(row=row, column=0, sticky="w", pady=(0, 8))
        ctk.CTkLabel(section, text="Se utiliza sólo cuando seleccionas un proveedor externo.", text_color=COLORS["muted"], anchor="w").grid(row=row, column=1, columnspan=2, sticky="w", padx=10, pady=(0, 8))
        row += 1

        ctk.CTkLabel(section, text="Modelo OpenAI texto:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.openai_chat_model = ctk.CTkEntry(section, fg_color=COLORS["card_alt"], border_color=COLORS["border"])
        self.openai_chat_model.insert(0, master.openai_chat_model)
        self.openai_chat_model.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="Modelo OpenAI de texto", help_text="Se usa sólo cuando eliges OpenAI para limpieza/resumen. El texto de la entrevista se envía al proveedor externo y la disponibilidad/costo dependen de tu cuenta y del modelo indicado.").grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="Modelo OpenAI STT:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.openai_stt_model = ctk.CTkEntry(section, fg_color=COLORS["card_alt"], border_color=COLORS["border"])
        self.openai_stt_model.insert(0, master.openai_stt_model)
        self.openai_stt_model.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="Modelo OpenAI STT", help_text="Se usa sólo en el perfil OpenAI API para transcribir. En ese modo el archivo de audio se envía al servicio externo; no afecta al baseline local WhisperX.").grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="OpenAI API key:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.openai_key = ctk.CTkEntry(section, show="•", placeholder_text="Vacía = usar credencial segura/entorno", fg_color=COLORS["card_alt"], border_color=COLORS["border"])
        self.openai_key.insert(0, master.session_openai_key)
        self.openai_key.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        ctk.CTkButton(section, text="Guardar seguro", width=112, fg_color=COLORS["water"], hover_color=COLORS["accent_hover"], command=self._save_openai).grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="API compatible URL:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.compat_url = ctk.CTkEntry(section, fg_color=COLORS["card_alt"], border_color=COLORS["border"])
        self.compat_url.insert(0, master.compatible_api_url)
        self.compat_url.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="URL API compatible", help_text="Debe apuntar a un endpoint compatible con OpenAI chat/completions. localhost puede ser local; una IP de red o dominio remoto significa que el texto se envía fuera de esta computadora.").grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="Modelo compatible:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.compat_model = ctk.CTkEntry(section, fg_color=COLORS["card_alt"], border_color=COLORS["border"])
        self.compat_model.insert(0, master.compatible_model)
        self.compat_model.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        HelpButton(section, title="Modelo compatible", help_text="Escribe el identificador exacto que espera el servidor compatible. Jerónimo no inventa ni sustituye este nombre; si el servidor no conoce el modelo, la prueba fallará antes de procesar entrevistas.").grid(row=row, column=2)
        row += 1

        ctk.CTkLabel(section, text="API key compatible:", anchor="w").grid(row=row, column=0, sticky="w", pady=6)
        self.compat_key = ctk.CTkEntry(section, show="•", placeholder_text="Opcional según servidor", fg_color=COLORS["card_alt"], border_color=COLORS["border"])
        self.compat_key.insert(0, master.session_compatible_key)
        self.compat_key.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        compat_actions = ctk.CTkFrame(section, fg_color="transparent")
        compat_actions.grid(row=row, column=2, sticky="e")
        ctk.CTkButton(compat_actions, text="Probar", width=62, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=self._test_compatible).pack(side="left", padx=2)
        ctk.CTkButton(compat_actions, text="Guardar", width=68, fg_color=COLORS["water"], hover_color=COLORS["accent_hover"], command=self._save_compatible).pack(side="left", padx=2)

        # 03 · Exportación
        section = make_section(2, "03", "Exportación", "El original y los derivados permanecen diferenciados.", "Formatos de exportación", "RAW conserva una representación directa de la transcripción. ATLAS prepara formatos pensados para análisis cualitativo y puede incluir timecodes, turnos, speaker_id y etiquetas en negrita. Elegir exportaciones no cambia la transcripción original.")
        self.exp_docx_raw = ctk.BooleanVar(value=master.export_docx_raw)
        self.exp_docx_atlas = ctk.BooleanVar(value=master.export_docx_atlas)
        self.exp_rtf_raw = ctk.BooleanVar(value=master.export_rtf_raw)
        self.exp_rtf_atlas = ctk.BooleanVar(value=master.export_rtf_atlas)
        self.atlas_timecodes = ctk.BooleanVar(value=master.atlas_timecodes)
        self.atlas_turns = ctk.BooleanVar(value=master.atlas_turn_numbers)
        self.atlas_id = ctk.BooleanVar(value=master.atlas_speaker_id)
        self.atlas_bold = ctk.BooleanVar(value=master.atlas_bold_speaker)
        export_items = [
            ("DOCX RAW", self.exp_docx_raw), ("DOCX ATLAS", self.exp_docx_atlas),
            ("RTF RAW", self.exp_rtf_raw), ("RTF ATLAS", self.exp_rtf_atlas),
            ("ATLAS: incluir timecodes", self.atlas_timecodes), ("ATLAS: numerar turnos", self.atlas_turns),
            ("ATLAS: speaker_id", self.atlas_id), ("ATLAS: etiqueta de hablante en negrita", self.atlas_bold),
        ]
        for idx, (item_text, var) in enumerate(export_items):
            col = idx % 2
            r = idx // 2
            ctk.CTkCheckBox(section, text=item_text, variable=var, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], border_color=COLORS["border"]).grid(row=r, column=col, sticky="w", padx=(0, 22), pady=5)
        section.grid_columnconfigure(0, weight=1)
        section.grid_columnconfigure(1, weight=1)

        # 04 · Prompts
        section = make_section(3, "04", "Prompts", "Contexto para ASR y derivados de texto; el raw permanece separado.", "Prompts", "El prompt de transcripción aporta vocabulario/contexto al ASR cuando el motor lo admite. Limpieza y resumen guían los derivados de texto. Cambiar prompts puede alterar resultados derivados; la transcripción raw se conserva separada.")
        section.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(section, text="Prompt de transcripción:", anchor="w").grid(row=0, column=0, sticky="nw", pady=6)
        self.whisper_prompt = ctk.CTkTextbox(section, height=92, fg_color=COLORS["card_alt"], border_width=1, border_color=COLORS["border"])
        self.whisper_prompt.insert("1.0", master.whisper_prompt)
        self.whisper_prompt.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)
        ctk.CTkLabel(section, text="Prompt de limpieza:", anchor="w").grid(row=1, column=0, sticky="nw", pady=6)
        self.clean_prompt = ctk.CTkTextbox(section, height=122, fg_color=COLORS["card_alt"], border_width=1, border_color=COLORS["border"])
        self.clean_prompt.insert("1.0", master.clean_prompt)
        self.clean_prompt.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)
        ctk.CTkLabel(section, text="Prompt de resumen:", anchor="w").grid(row=2, column=0, sticky="nw", pady=6)
        self.summary_prompt = ctk.CTkTextbox(section, height=122, fg_color=COLORS["card_alt"], border_width=1, border_color=COLORS["border"])
        self.summary_prompt.insert("1.0", master.summary_prompt)
        self.summary_prompt.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=6)

        footer = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=0, border_width=1, border_color=COLORS["border"])
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(footer, text="Los cambios se aplican sólo al guardar.", text_color=COLORS["muted"], anchor="w").grid(row=0, column=0, padx=26, pady=16, sticky="w")
        actions = ctk.CTkFrame(footer, fg_color="transparent")
        actions.grid(row=0, column=1, padx=26, pady=12, sticky="e")
        ctk.CTkButton(actions, text="Cancelar", fg_color=COLORS["card_alt"], hover_color=COLORS["border"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], command=self.destroy).pack(side="left", padx=5)
        ctk.CTkButton(actions, text="Guardar cambios", width=140, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=self._save).pack(side="left", padx=5)

    def _save_openai(self):
        value = self.openai_key.get().strip()
        if not value:
            mb.showwarning("Credencial", "Escribe una API key antes de guardarla.", parent=self)
            return
        try:
            set_secure_secret("openai", value)
            mb.showinfo("Credencial", "OpenAI guardada en el almacén seguro del sistema.", parent=self)
        except Exception as exc:
            mb.showerror("Credencial", str(exc), parent=self)

    def _save_compatible(self):
        value = self.compat_key.get().strip()
        if not value:
            mb.showwarning("Credencial", "Escribe una API key antes de guardarla. Algunos servidores locales no requieren clave y pueden dejarla vacía.", parent=self)
            return
        try:
            set_secure_secret("openai_compatible", value)
            mb.showinfo("Credencial", "Credencial compatible guardada en el almacén seguro del sistema.", parent=self)
        except Exception as exc:
            mb.showerror("Credencial", str(exc), parent=self)

    def _test_ollama(self):
        settings = TextProviderSettings(engine="ollama", ollama_model=self.ollama_model.get().strip(), ollama_url=self.ollama_url.get().strip())
        self._test_provider(settings)

    def _test_compatible(self):
        settings = TextProviderSettings(
            engine="openai_compatible",
            model=self.compat_model.get().strip(),
            base_url=self.compat_url.get().strip(),
            api_key=self.compat_key.get().strip() or resolve_secret("openai_compatible"),
        )
        self._test_provider(settings)

    def _test_provider(self, settings):
        def worker():
            try:
                result = test_provider(settings)
                self.app._post_ui(mb.showinfo, "Prueba de conexión", str(result), parent=self)
            except Exception as exc:
                self.app._post_ui(mb.showerror, "Prueba de conexión", str(exc), parent=self)
        threading.Thread(target=worker, daemon=True).start()

    def _save(self):
        app = self.app
        app.local_fast_model = self.local_model.get().strip() or "large-v3"
        app.local_device = self.local_device.get().strip() or "auto"
        spk = self.speakers.get().strip()
        app.target_speakers = 0 if spk.lower() == "auto" else int(spk)
        app.audio_enhancement = bool(self.audio_enhance.get())
        app.ollama_url = self.ollama_url.get().strip() or DEFAULT_OLLAMA_URL
        app.ollama_model = self.ollama_model.get().strip()
        app.openai_chat_model = self.openai_chat_model.get().strip() or "gpt-4.1-mini"
        app.openai_stt_model = self.openai_stt_model.get().strip() or "gpt-4o-mini-transcribe"
        app.session_openai_key = self.openai_key.get().strip()
        app.compatible_api_url = self.compat_url.get().strip() or DEFAULT_OPENAI_COMPATIBLE_URL
        app.compatible_model = self.compat_model.get().strip()
        app.session_compatible_key = self.compat_key.get().strip()
        app.export_docx_raw = bool(self.exp_docx_raw.get())
        app.export_docx_atlas = bool(self.exp_docx_atlas.get())
        app.export_rtf_raw = bool(self.exp_rtf_raw.get())
        app.export_rtf_atlas = bool(self.exp_rtf_atlas.get())
        app.atlas_timecodes = bool(self.atlas_timecodes.get())
        app.atlas_turn_numbers = bool(self.atlas_turns.get())
        app.atlas_speaker_id = bool(self.atlas_id.get())
        app.atlas_bold_speaker = bool(self.atlas_bold.get())
        app.whisper_prompt = self.whisper_prompt.get("1.0", "end").strip()
        app.clean_prompt = self.clean_prompt.get("1.0", "end").strip()
        app.summary_prompt = self.summary_prompt.get("1.0", "end").strip()
        app._refresh_privacy_card()
        self.destroy()


class PerformanceAdviceDialog(ctk.CTkToplevel):
    """Confirmación breve de rendimiento antes de lanzar un trabajo largo."""

    def __init__(self, master):
        super().__init__(master)
        self.result = False
        self.title("Jerónimo Abya Yala — Antes de comenzar")
        self.geometry("650x430")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["canvas"])
        try:
            self.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 10))
        ctk.CTkLabel(
            header, text="ANTES DE COMENZAR", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["terracotta"], anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text="Reserva recursos para la transcripción",
            font=ctk.CTkFont(size=25, weight="bold"), text_color=COLORS["text"], anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        card = Card(self)
        card.grid(row=1, column=0, sticky="nsew", padx=28, pady=(4, 14))
        ctk.CTkLabel(
            card,
            text=(
                "Para obtener el mejor rendimiento, cierra o pausa aplicaciones que compitan "
                "por CPU, RAM o GPU antes de iniciar."
            ),
            font=ctk.CTkFont(size=14, weight="bold"), text_color=COLORS["text"],
            anchor="w", justify="left", wraplength=550,
        ).pack(fill="x", padx=22, pady=(20, 12))
        ctk.CTkLabel(
            card,
            text=(
                "• juegos, edición de video y otras tareas pesadas\n"
                "• música, video y streaming\n"
                "• navegadores con muchas pestañas o sitios exigentes\n\n"
                "Puedes seguir usando el equipo para tareas livianas, pero compartir recursos "
                "puede aumentar el tiempo de procesamiento."
            ),
            text_color=COLORS["muted"], anchor="w", justify="left", wraplength=550,
        ).pack(fill="x", padx=22, pady=(0, 20))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="e", padx=28, pady=(0, 24))
        ctk.CTkButton(
            actions, text="Volver", width=110, fg_color=COLORS["card_alt"],
            hover_color=COLORS["border"], text_color=COLORS["text"], command=self._cancel,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            actions, text="Iniciar proceso", width=150, fg_color=COLORS["success"],
            hover_color=COLORS["success_hover"], command=self._accept,
        ).pack(side="left")
        self.after(60, self._present_front)

    def _present_front(self):
        try:
            self.lift()
            self.focus_force()
        except Exception:
            pass

    def _accept(self):
        self.result = True
        self.destroy()

    def _cancel(self):
        self.result = False
        self.destroy()


class JeronimoApp(_DnDRoot):
    def __init__(self):
        super().__init__()

        # En el primer inicio la aplicación principal NO debe verse detrás del
        # onboarding. Se mantiene retirada hasta completar la configuración.
        # En inicios posteriores, si el estado ya está completo, abre directo.
        try:
            startup_decision = startup_configuration_decision()
            self._startup_configuration_mode = startup_decision.mode
            self._startup_repair_reasons = tuple(startup_decision.reasons)
            self._startup_onboarding_required = startup_decision.required
        except Exception as exc:
            self._startup_configuration_mode = "full"
            self._startup_repair_reasons = (f"No se pudo validar la configuración persistida: {exc}",)
            self._startup_onboarding_required = True
        if self._startup_onboarding_required:
            try:
                self.withdraw()
            except Exception:
                pass

        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.minsize(1060, 700)
        self.configure(fg_color=COLORS["canvas"])
        # EDIT-4: Tk/CustomTkinter puede ignorar un ``zoomed`` solicitado antes
        # de que Windows haya mapeado realmente la raíz. Conservamos el intento
        # inmediato existente y añadimos un único reintento posterior al primer
        # <Map> real. El onboarding sigue siendo quien decide cuándo la raíz se
        # hace visible; no se modifica su flujo ni su instalador.
        self._main_window_mapped = False
        self._startup_maximize_after_map_done = False
        self.bind("<Map>", self._on_main_window_mapped, add="+")
        # Sólo maximizar inmediatamente cuando la configuración inicial ya fue
        # completada en una ejecución anterior. En primer inicio se maximiza
        # después de cerrar el onboarding.
        if not self._startup_onboarding_required:
            self.after_idle(self._maximize_main_window)
        try:
            self.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass

        # Estado de producto: local-first y fidelidad como defaults de la GUI nueva.
        self.input_path: Path | None = None
        self.output_dir: Path = Path.cwd() / "transcripciones"
        self.profile = PROFILE_DIARIZATION
        self.text_engine = TEXT_OLLAMA
        self.do_summary = True
        self.do_clean = False
        self.language = "es"
        self.target_speakers = 0
        self.audio_enhancement = True
        self.local_fast_model = "large-v3"
        self.local_device = "auto"
        self.openai_stt_model = "gpt-4o-mini-transcribe"
        self.openai_chat_model = "gpt-4.1-mini"
        self.ollama_url = DEFAULT_OLLAMA_URL
        self.ollama_model = ""
        self.ollama_models: list[str] = []
        self.compatible_api_url = DEFAULT_OPENAI_COMPATIBLE_URL
        self.compatible_model = ""
        self.session_openai_key = ""
        self.session_compatible_key = ""
        self.interviewer = ""
        self.interviewee = ""
        self.project = ""
        self.place = ""
        self.date_label = ""
        self.extra_context = ""
        self.whisper_prompt = GENERIC_WHISPER_PROMPT
        self.clean_prompt = DEFAULT_CLEAN_PROMPT
        self.summary_prompt = DEFAULT_SUMMARY_PROMPT
        self.export_docx_raw = False
        self.export_docx_atlas = True
        self.export_rtf_raw = False
        self.export_rtf_atlas = False
        self.atlas_timecodes = True
        self.atlas_turn_numbers = False
        self.atlas_speaker_id = False
        self.atlas_bold_speaker = True

        self._diag_report = None
        self._is_running = False
        self._cancel_event = threading.Event()
        self._job_tracker: JobTracker | None = None
        self._current_cfg: TranscriberConfig | None = None
        self._before_outputs: set[Path] = set()
        self._new_outputs: list[Path] = []
        self._last_audio: Path | None = None
        self._last_transcript: Path | None = None
        self._last_status = ""
        self._resource_monitor: SystemResourceMonitor | None = None
        self._resource_poll_job = None

        # HOTFIX 4: frontera thread -> Tk. Los workers sólo depositan trabajo
        # Python en esta cola; únicamente el hilo principal ejecuta callbacks
        # de interfaz al drenar la cola mediante un ``after`` creado aquí.
        self._ui_dispatch_queue: Queue = Queue()
        self._ui_dispatch_after_id = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_pages()
        self._show_page("home")
        self._configure_drag_drop()
        self._ui_dispatch_after_id = self.after(40, self._drain_ui_dispatch_queue)
        self.after(100, self._load_diagnostics_async)
        # En primer inicio abrir el asistente cuanto antes. La raíz permanece
        # retirada, por lo que en pantalla sólo aparece Configuración inicial.
        self.after(80 if self._startup_onboarding_required else 650, self._maybe_start_onboarding)

    def _maximize_main_window(self):
        try:
            self.state("zoomed")
            return
        except Exception:
            pass
        try:
            self.attributes("-zoomed", True)
        except Exception:
            # Mantener la geometría de respaldo si la plataforma no soporta zoomed.
            pass

    def _on_main_window_mapped(self, event=None):
        # <Map> también puede propagarse desde hijos; sólo interesa la raíz.
        if event is not None and getattr(event, "widget", self) is not self:
            return
        self._main_window_mapped = True
        if self._startup_onboarding_required or self._startup_maximize_after_map_done:
            return
        # El pequeño diferido permite que Windows termine de aplicar la geometría
        # inicial de CustomTkinter antes del único zoom definitivo.
        self.after(60, self._maximize_after_first_map)

    def _maximize_after_first_map(self):
        if self._startup_maximize_after_map_done:
            return
        if self._startup_onboarding_required or not self._main_window_mapped:
            return
        try:
            self.update_idletasks()
        except Exception:
            pass
        self._maximize_main_window()
        self._startup_maximize_after_map_done = True

    # ---------- layout ----------
    def _build_sidebar(self):
        bar = ctk.CTkFrame(self, width=SIDEBAR_WIDTH, corner_radius=0, fg_color=COLORS["sidebar"])
        bar.grid(row=0, column=0, sticky="nsew")
        bar.grid_propagate(False)
        bar.grid_rowconfigure(8, weight=1)

        # Identidad Territorio Vivo: marca abstracta territorio/onda y jerarquía sobria.
        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.grid(row=0, column=0, rowspan=2, padx=18, pady=(24, 22), sticky="ew")
        brand.grid_columnconfigure(1, weight=1)
        self._visual_images = {}
        try:
            mark_img = Image.open(asset_path("brand", "jeronimo_mark_light.png"))
            self._visual_images["brand"] = ctk.CTkImage(light_image=mark_img, dark_image=mark_img, size=(54, 54))
            ctk.CTkLabel(brand, text="", image=self._visual_images["brand"], width=56).grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 10))
        except Exception:
            pass
        ctk.CTkLabel(brand, text="JERÓNIMO", font=ctk.CTkFont(size=17, weight="bold"), text_color=COLORS["sidebar_text"], anchor="w").grid(row=0, column=1, sticky="sw")
        ctk.CTkLabel(brand, text="ABYA YALA", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["sidebar_muted"], anchor="w").grid(row=1, column=1, sticky="nw")

        self.nav_buttons = {}
        nav = [("home", "Inicio"), ("new", "Nueva desgrabación"), ("process", "Proceso"), ("results", "Revisar y exportar")]
        for i, (key, label) in enumerate(nav, start=2):
            image = None
            try:
                icon = Image.open(asset_path("icons", f"{key}.png"))
                image = ctk.CTkImage(light_image=icon, dark_image=icon, size=(20, 20))
                self._visual_images[key] = image
            except Exception:
                pass
            btn = ctk.CTkButton(
                bar, text=label, image=image, compound="left", anchor="w", width=192, height=40, corner_radius=10,
                fg_color="transparent", hover_color=COLORS["nav_hover"], text_color=COLORS["sidebar_text"],
                font=ctk.CTkFont(size=13, weight="normal"),
                command=lambda k=key: self._show_page(k),
            )
            btn.grid(row=i, column=0, padx=16, pady=3, sticky="ew")
            self.nav_buttons[key] = btn

        # Herramienta documental independiente. Se ubica inmediatamente debajo
        # de Revisar y exportar sin alterar las páginas ni el flujo de proceso.
        editor_image = None
        try:
            editor_icon = Image.open(asset_path("icons", "review.png"))
            editor_image = ctk.CTkImage(light_image=editor_icon, dark_image=editor_icon, size=(20, 20))
            self._visual_images["interview_text_editor"] = editor_image
        except Exception:
            pass
        ctk.CTkButton(
            bar, text="Editor de entrevistas", image=editor_image, compound="left", anchor="w",
            width=192, height=40, corner_radius=10,
            fg_color="transparent", hover_color=COLORS["nav_hover"], text_color=COLORS["sidebar_text"],
            font=ctk.CTkFont(size=13, weight="normal"), command=self._open_interview_text_editor,
        ).grid(row=6, column=0, padx=16, pady=3, sticky="ew")

        tools = [("settings", "Configuración inicial", self._open_onboarding), ("models", "Modelos locales", self._open_models), ("diagnostics", "Diagnóstico", self._show_diagnostics)]
        for row, (key, label, command) in enumerate(tools, start=9):
            image = None
            try:
                icon = Image.open(asset_path("icons", f"{key}.png"))
                image = ctk.CTkImage(light_image=icon, dark_image=icon, size=(19, 19))
                self._visual_images[key] = image
            except Exception:
                pass
            ctk.CTkButton(
                bar, text=label, image=image, compound="left", anchor="w", width=192, height=36, corner_radius=10,
                fg_color="transparent", hover_color=COLORS["nav_hover"], text_color=COLORS["sidebar_muted"],
                font=ctk.CTkFont(size=12), command=command,
            ).grid(row=row, column=0, padx=16, pady=2, sticky="ew")

        # Franja territorial mínima. Se divide en dos líneas para que el lema
        # nunca quede recortado por el ancho fijo del sidebar ni por el DPI de Windows.
        footer = ctk.CTkFrame(bar, fg_color="transparent", height=80)
        footer.grid(row=12, column=0, padx=18, pady=(16, 16), sticky="sew")
        ctk.CTkLabel(
            footer,
            text="TECNOLOGÍA LOCAL\nDATOS BAJO TU CONTROL",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=COLORS["sidebar_muted"],
            anchor="w",
            justify="left",
        ).pack(fill="x")
        line = ctk.CTkFrame(footer, height=2, fg_color=COLORS["terracotta"], corner_radius=1)
        line.pack(fill="x", pady=(8, 6))
        chaos = ctk.CTkLabel(
            footer, text="Chaos Reigns", font=ctk.CTkFont(size=9),
            text_color=COLORS["sidebar_muted"], anchor="w",
        )
        chaos.pack(fill="x", anchor="w")
        try:
            chaos.configure(cursor="hand2")
        except Exception:
            pass
        chaos.bind("<Button-1>", lambda _event: self._open_chaos_reigns())
        chaos.bind("<Enter>", lambda _event: chaos.configure(text_color=COLORS["maize"]))
        chaos.bind("<Leave>", lambda _event: chaos.configure(text_color=COLORS["sidebar_muted"]))

    def _build_pages(self):
        self.page_host = ctk.CTkFrame(self, fg_color=COLORS["canvas"], corner_radius=0)
        self.page_host.grid(row=0, column=1, sticky="nsew")
        self.page_host.grid_columnconfigure(0, weight=1)
        self.page_host.grid_rowconfigure(0, weight=1)
        self.pages = {}
        for name, builder in [("home", self._build_home), ("new", self._build_new), ("process", self._build_process), ("results", self._build_results)]:
            frame = ctk.CTkFrame(self.page_host, fg_color="transparent")
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = frame
            builder(frame)

    def _page_scroll(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=26, pady=20)
        scroll.grid_columnconfigure(0, weight=1)
        return scroll

    def _show_page(self, name: str):
        if name == "process" and not self._is_running and self._job_tracker is None:
            name = "new"
        if name == "results" and not self._new_outputs and not self._last_status:
            name = "home"
        self.pages[name].tkraise()
        for key, btn in self.nav_buttons.items():
            btn.configure(fg_color=COLORS["nav_active"] if key == name else "transparent", font=ctk.CTkFont(size=13, weight="bold" if key == name else "normal"))

    # ---------- home ----------
    def _build_home(self, parent):
        scroll = self._page_scroll(parent)

        # Encabezado editorial Territorio Vivo. Sólo presentación.
        eyebrow = ctk.CTkLabel(
            scroll, text="LOCAL-FIRST  ·  SOBERANÍA DIGITAL", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["terracotta"], anchor="w",
        )
        eyebrow.grid(row=0, column=0, sticky="ew", pady=(4, 4))
        home_title = ctk.CTkFrame(scroll, fg_color="transparent")
        home_title.grid(row=1, column=0, sticky="ew", pady=(0, 2))
        ctk.CTkLabel(
            home_title, text="Desgrabar sin perder de vista los datos",
            font=ctk.CTkFont(size=32, weight="bold"), anchor="w",
        ).pack(side="left")
        HelpButton(
            home_title, title="Inicio",
            help_text="Desde aquí puedes crear una nueva desgrabación, revisar una transcripción existente, administrar modelos locales y consultar el estado del equipo. Jerónimo Abya Yala prioriza procesamiento local y te avisa antes de cualquier uso de servicios externos.",
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            scroll,
            text="El conocimiento tiene territorio. Procesamiento local por defecto, trazabilidad de cada etapa y APIs sólo cuando las eliges.",
            font=ctk.CTkFont(size=15), text_color=COLORS["muted"], anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 18))

        hero = Card(scroll)
        hero.grid(row=3, column=0, sticky="ew", pady=8)
        hero.grid_columnconfigure(0, weight=5)
        hero.grid_columnconfigure(1, weight=4)

        hero_copy = ctk.CTkFrame(hero, fg_color="transparent")
        hero_copy.grid(row=0, column=0, padx=(26, 10), pady=24, sticky="nsew")
        ctk.CTkLabel(
            hero_copy, text="NUEVA ENTREVISTA", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["accent"], anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            hero_copy, text="Escuchar.\nTranscribir. Documentar.",
            font=ctk.CTkFont(size=25, weight="bold"), justify="left", anchor="w",
        ).pack(fill="x", pady=(5, 8))
        ctk.CTkLabel(
            hero_copy,
            text="Selecciona o arrastra una grabación. Jerónimo Abya Yala indica qué se procesa en este equipo y cuándo interviene un servicio externo.",
            text_color=COLORS["muted"], wraplength=480, justify="left", anchor="w",
        ).pack(fill="x")
        hero_actions = ctk.CTkFrame(hero_copy, fg_color="transparent")
        hero_actions.pack(fill="x", pady=(17, 0))
        ctk.CTkButton(
            hero_actions, text="Nueva desgrabación", width=184, height=42,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=lambda: self._show_page("new"),
        ).pack(side="left")
        ctk.CTkLabel(
            hero_actions, text="●  DATOS BAJO TU CONTROL", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["success"],
        ).pack(side="left", padx=(16, 0))

        try:
            terrain = Image.open(asset_path("textures", "territorio_hero.png"))
            self._visual_images["home_terrain"] = ctk.CTkImage(light_image=terrain, dark_image=terrain, size=(420, 121))
            ctk.CTkLabel(hero, text="", image=self._visual_images["home_terrain"], anchor="e").grid(
                row=0, column=1, padx=(0, 18), pady=18, sticky="e"
            )
        except Exception:
            ctk.CTkFrame(hero, fg_color=COLORS["accent_soft"], corner_radius=12, width=360, height=120).grid(
                row=0, column=1, padx=(0, 20), pady=24, sticky="e"
            )

        tools = ctk.CTkFrame(scroll, fg_color="transparent")
        tools.grid(row=4, column=0, sticky="ew", pady=8)
        tools.grid_columnconfigure((0, 1), weight=1)
        tool_specs = [
            (0, "review", "Revisar una transcripción", "Waveform, bloques, audio y retranscripción A/B.", self._open_existing_review, "Revisar"),
            (1, "model_local", "Modelos locales", "Descarga y prueba modelos de resumen y transcripción sin terminal.", self._open_models, "Gestionar"),
        ]
        for col, icon_name, title, desc, cmd, action in tool_specs:
            card = Card(tools)
            card.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else (6, 0))
            card.grid_columnconfigure(1, weight=1)
            image = None
            try:
                icon = Image.open(asset_path("icons", f"{icon_name}.png"))
                image = ctk.CTkImage(light_image=icon, dark_image=icon, size=(38, 38))
                self._visual_images[f"home_{icon_name}"] = image
            except Exception:
                pass
            ctk.CTkLabel(card, text="", image=image, width=46).grid(row=0, column=0, rowspan=2, padx=(20, 8), pady=(18, 6), sticky="nw")
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=17, weight="bold"), anchor="w").grid(row=0, column=1, padx=(0, 18), pady=(18, 2), sticky="ew")
            ctk.CTkLabel(card, text=desc, text_color=COLORS["muted"], wraplength=330, justify="left", anchor="w").grid(row=1, column=1, padx=(0, 18), sticky="ew")
            ctk.CTkButton(
                card, text=action, width=108, height=34, fg_color=COLORS["accent_soft"],
                hover_color=COLORS["nav_hover"], text_color=COLORS["text"], command=cmd,
            ).grid(row=2, column=1, padx=(0, 18), pady=(12, 18), sticky="w")

        self.home_system_card = Card(scroll)
        self.home_system_card.grid(row=5, column=0, sticky="ew", pady=8)
        self.home_system_card.grid_columnconfigure(1, weight=1)
        shield_image = None
        try:
            shield = Image.open(asset_path("icons", "local_shield.png"))
            shield_image = ctk.CTkImage(light_image=shield, dark_image=shield, size=(38, 38))
            self._visual_images["home_shield"] = shield_image
        except Exception:
            pass
        ctk.CTkLabel(self.home_system_card, text="", image=shield_image, width=48).grid(row=0, column=0, rowspan=3, padx=(20, 8), pady=18, sticky="nw")
        ctk.CTkLabel(
            self.home_system_card, text="ESTE EQUIPO", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["terracotta"], anchor="w",
        ).grid(row=0, column=1, padx=(0, 20), pady=(17, 0), sticky="ew")
        ctk.CTkLabel(
            self.home_system_card, text="Territorio de procesamiento", font=ctk.CTkFont(size=18, weight="bold"), anchor="w"
        ).grid(row=1, column=1, padx=(0, 20), pady=(1, 1), sticky="ew")
        self.lbl_home_system = ctk.CTkLabel(
            self.home_system_card, text="Comprobando hardware y motores…", text_color=COLORS["muted"],
            anchor="w", justify="left", wraplength=820,
        )
        self.lbl_home_system.grid(row=2, column=1, padx=(0, 20), pady=(1, 18), sticky="ew")

    # ---------- new workflow ----------
    def _build_new(self, parent):
        scroll = self._page_scroll(parent)

        ctk.CTkLabel(
            scroll, text="NUEVO TRABAJO  ·  FLUJO CONTROLADO", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["terracotta"], anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(4, 4))
        new_title = ctk.CTkFrame(scroll, fg_color="transparent")
        new_title.grid(row=1, column=0, sticky="ew", pady=(0, 2))
        ctk.CTkLabel(new_title, text="Nueva desgrabación", font=ctk.CTkFont(size=31, weight="bold"), anchor="w").pack(side="left")
        HelpButton(
            new_title, title="Nueva desgrabación",
            help_text="Este flujo reúne sólo las decisiones que afectan el tratamiento de la entrevista: archivo de entrada, carpeta de resultados, motor de transcripción, generación de derivados de texto y contexto opcional. Las opciones técnicas menos frecuentes están en 'Opciones avanzadas'.",
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            scroll, text="Cinco decisiones visibles. El audio original se conserva y cada derivado mantiene su trazabilidad.",
            text_color=COLORS["muted"], anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 18))

        source = Card(scroll)
        source.grid(row=3, column=0, sticky="ew", pady=7)
        source.grid_columnconfigure(0, weight=1)
        source.grid_columnconfigure(1, weight=0)
        source_head = ctk.CTkFrame(source, fg_color="transparent")
        source_head.grid(row=0, column=0, columnspan=2, padx=20, pady=(17, 7), sticky="ew")
        ctk.CTkLabel(source_head, text="01", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["terracotta"]).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(source_head, text="Grabación", font=ctk.CTkFont(size=19, weight="bold")).pack(side="left")
        HelpButton(
            source_head, title="Grabación",
            help_text="Selecciona un archivo de audio/video o una carpeta. El audio original no se modifica. Jerónimo crea temporales de trabajo y los elimina al finalizar o cancelar. Para revisión detallada conviene procesar una entrevista individual.",
        ).pack(side="left", padx=8)

        drop = ctk.CTkFrame(source, fg_color=COLORS["card_alt"], corner_radius=14, border_width=1, border_color=COLORS["border"])
        drop.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        drop.grid_columnconfigure(1, weight=1)
        upload_image = None
        try:
            upload = Image.open(asset_path("icons", "upload.png"))
            upload_image = ctk.CTkImage(light_image=upload, dark_image=upload, size=(48, 48))
            self._visual_images["new_upload"] = upload_image
        except Exception:
            pass
        ctk.CTkLabel(drop, text="", image=upload_image, width=64).grid(row=0, column=0, rowspan=2, padx=(20, 10), pady=18)
        self.lbl_drop = ctk.CTkLabel(
            drop, text="Arrastra un audio o video aquí\no selecciónalo desde el equipo", height=66,
            fg_color="transparent", justify="left", anchor="w", text_color=COLORS["muted"],
            font=ctk.CTkFont(size=14),
        )
        self.lbl_drop.grid(row=0, column=1, rowspan=2, padx=(0, 12), pady=10, sticky="ew")
        self.lbl_drop.bind("<Button-1>", lambda _e: self._select_file())
        ctk.CTkLabel(
            drop, text="WAV · MP3 · M4A · MP4 · MOV · MKV · OGG · FLAC",
            font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"],
        ).grid(row=0, column=2, padx=(8, 20), pady=(18, 0), sticky="e")
        ctk.CTkLabel(
            drop, text="ORIGINAL CONSERVADO", font=ctk.CTkFont(size=9, weight="bold"),
            text_color=COLORS["success"],
        ).grid(row=1, column=2, padx=(8, 20), pady=(0, 18), sticky="e")

        source_actions = ctk.CTkFrame(source, fg_color="transparent")
        source_actions.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 18), sticky="ew")
        ctk.CTkButton(
            source_actions, text="Seleccionar archivo", height=36, fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"], command=self._select_file,
        ).pack(side="left")
        ctk.CTkButton(
            source_actions, text="Procesar carpeta", height=36, fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"], text_color=COLORS["text"], command=self._select_folder,
        ).pack(side="left", padx=8)
        HelpButton(
            source_actions, title="Archivo o carpeta",
            help_text="'Seleccionar archivo' procesa una entrevista y habilita el flujo de revisión. 'Procesar carpeta' ejecuta un lote de archivos compatibles; los resultados se guardan individualmente en la carpeta elegida.",
        ).pack(side="right")

        out = Card(scroll)
        out.grid(row=4, column=0, sticky="ew", pady=7)
        out.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(out, text="02", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["terracotta"]).grid(row=0, column=0, padx=(20, 8), pady=(17, 6), sticky="w")
        ctk.CTkLabel(out, text="Resultados", font=ctk.CTkFont(size=19, weight="bold")).grid(row=0, column=1, pady=(17, 6), sticky="w")
        HelpButton(
            out, title="Carpeta de resultados",
            help_text="Aquí se guardan la transcripción original automática y, si los activas, sus derivados normalizados, resumen, documentos y manifiesto de trazabilidad. La carpeta puede cambiarse antes de iniciar el trabajo.",
        ).grid(row=0, column=2, padx=(4, 20), pady=(15, 5), sticky="e")
        ctk.CTkLabel(out, text="Carpeta de destino", text_color=COLORS["muted"], font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=(20, 8), pady=(0, 18), sticky="w")
        self.entry_output = ctk.CTkEntry(out, height=36, border_color=COLORS["border"])
        self.entry_output.insert(0, str(self.output_dir))
        self.entry_output.grid(row=1, column=1, sticky="ew", pady=(0, 18))
        ctk.CTkButton(
            out, text="Elegir", width=82, height=36, fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"], text_color=COLORS["text"], command=self._select_output,
        ).grid(row=1, column=2, padx=20, pady=(0, 18))

        profile = Card(scroll)
        profile.grid(row=5, column=0, sticky="ew", pady=7)
        profile.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(profile, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=20, pady=(17, 7))
        ctk.CTkLabel(top, text="03", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["terracotta"]).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(top, text="Cómo transcribir", font=ctk.CTkFont(size=19, weight="bold")).pack(side="left")
        HelpButton(
            top, title="Cómo transcribir",
            help_text="Calidad + hablantes: WhisperX large-v2 y diarización local, el baseline de mayor fidelidad validado hasta ahora. Local sin hablantes: faster-whisper local, más simple y normalmente más rápido. OpenAI API: envía el audio al servicio externo y requiere credencial. El perfil elegido cambia dónde se procesa el audio y si se separan voces.",
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            top, text="LOCAL / EXTERNO", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"]
        ).pack(side="right")
        self.seg_profile = ctk.CTkSegmentedButton(
            profile, values=list(PROFILE_LABELS.keys()), command=self._on_profile_changed,
            selected_color=COLORS["accent"], selected_hover_color=COLORS["accent_hover"],
            unselected_color=COLORS["card_alt"], unselected_hover_color=COLORS["accent_soft"],
            text_color=COLORS["text"], height=38, corner_radius=10,
        )
        self.seg_profile.set(PROFILE_LABELS_INV[self.profile])
        self.seg_profile.grid(row=1, column=0, padx=20, pady=6, sticky="ew")
        note_box = ctk.CTkFrame(profile, fg_color=COLORS["accent_soft"], corner_radius=10)
        note_box.grid(row=2, column=0, padx=20, pady=(4, 18), sticky="ew")
        self.lbl_profile_note = ctk.CTkLabel(
            note_box, text="WhisperX large-v2 + diarización local. Es el baseline actual de máxima fidelidad.",
            text_color=COLORS["text"], anchor="w", justify="left", wraplength=820,
        )
        self.lbl_profile_note.pack(fill="x", padx=14, pady=10)

        text = Card(scroll)
        text.grid(row=6, column=0, sticky="ew", pady=7)
        text.grid_columnconfigure(0, weight=1)
        title_row = ctk.CTkFrame(text, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=20, pady=(17, 7))
        ctk.CTkLabel(title_row, text="04", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["terracotta"]).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(title_row, text="Resumen y texto", font=ctk.CTkFont(size=19, weight="bold")).pack(side="left")
        HelpButton(
            title_row, title="Resumen y texto",
            help_text="Generar resumen crea un documento derivado sin alterar la transcripción original. Versión normalizada crea otra copia con correcciones de puntuación/forma. Local · Ollama procesa el texto en este equipo si el endpoint es local; OpenAI o una API remota envían texto fuera del equipo. 'Sin IA de texto' desactiva ambos derivados automáticos.",
        ).pack(side="left", padx=8)
        ctk.CTkLabel(title_row, text="DERIVADOS SEPARADOS", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"]).pack(side="right")
        controls = ctk.CTkFrame(text, fg_color=COLORS["card_alt"], corner_radius=10)
        controls.grid(row=1, column=0, padx=20, pady=4, sticky="ew")
        controls.grid_columnconfigure(2, weight=1)
        self.var_summary = ctk.BooleanVar(value=True)
        self.var_clean = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            controls, text="Generar resumen", variable=self.var_summary, command=self._refresh_privacy_card,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
        ).grid(row=0, column=0, sticky="w", padx=(14, 18), pady=12)
        ctk.CTkCheckBox(
            controls, text="Versión normalizada", variable=self.var_clean, command=self._refresh_privacy_card,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
        ).grid(row=0, column=1, sticky="w", padx=(0, 18), pady=12)
        self.combo_text_engine = ctk.CTkComboBox(
            controls, values=list(TEXT_LABELS.keys()), state="readonly", command=self._on_text_engine_changed,
            height=34, border_color=COLORS["border"], button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"],
        )
        self.combo_text_engine.set(TEXT_LABELS_INV[self.text_engine])
        self.combo_text_engine.grid(row=0, column=2, padx=(0, 14), pady=12, sticky="ew")
        self.lbl_text_model = ctk.CTkLabel(text, text="Modelo local: detectando recomendación…", text_color=COLORS["muted"], anchor="w")
        self.lbl_text_model.grid(row=2, column=0, padx=20, pady=(3, 18), sticky="ew")

        context = Card(scroll)
        context.grid(row=7, column=0, sticky="ew", pady=7)
        context.grid_columnconfigure((1, 3), weight=1)
        context_title = ctk.CTkFrame(context, fg_color="transparent")
        context_title.grid(row=0, column=0, columnspan=4, padx=20, pady=(16, 6), sticky="ew")
        ctk.CTkLabel(context_title, text="05", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["terracotta"]).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(context_title, text="Contexto opcional", font=ctk.CTkFont(size=19, weight="bold")).pack(side="left")
        HelpButton(
            context_title, title="Contexto opcional",
            help_text="Estos campos ayudan a identificar el trabajo y pueden aportar contexto a los derivados de texto. No son obligatorios. Para entrevistas sensibles puedes usar códigos o seudónimos en lugar de nombres. Proyecto y lugar/fecha son metadatos de organización; no cambian el audio original.",
        ).pack(side="left", padx=8)
        ctk.CTkLabel(context_title, text="PUEDES USAR SEUDÓNIMOS", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"]).pack(side="right")
        ctk.CTkLabel(context, text="Entrevistador/a", text_color=COLORS["muted"], font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=(20, 6), pady=4, sticky="w")
        self.entry_interviewer = ctk.CTkEntry(context, placeholder_text="Nombre o código", height=34, border_color=COLORS["border"])
        self.entry_interviewer.grid(row=1, column=1, padx=(0, 14), pady=4, sticky="ew")
        ctk.CTkLabel(context, text="Entrevistado/a", text_color=COLORS["muted"], font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=2, padx=(0, 6), pady=4, sticky="w")
        self.entry_interviewee = ctk.CTkEntry(context, placeholder_text="Nombre o código", height=34, border_color=COLORS["border"])
        self.entry_interviewee.grid(row=1, column=3, padx=(0, 20), pady=4, sticky="ew")
        ctk.CTkLabel(context, text="Proyecto", text_color=COLORS["muted"], font=ctk.CTkFont(size=11, weight="bold")).grid(row=2, column=0, padx=(20, 6), pady=(4, 18), sticky="w")
        self.entry_project = ctk.CTkEntry(context, placeholder_text="Opcional", height=34, border_color=COLORS["border"])
        self.entry_project.grid(row=2, column=1, padx=(0, 14), pady=(4, 18), sticky="ew")
        ctk.CTkLabel(context, text="Lugar / fecha", text_color=COLORS["muted"], font=ctk.CTkFont(size=11, weight="bold")).grid(row=2, column=2, padx=(0, 6), pady=(4, 18), sticky="w")
        self.entry_place_date = ctk.CTkEntry(context, placeholder_text="Opcional", height=34, border_color=COLORS["border"])
        self.entry_place_date.grid(row=2, column=3, padx=(0, 20), pady=(4, 18), sticky="ew")

        self.privacy_card = Card(scroll)
        self.privacy_card.grid(row=8, column=0, sticky="ew", pady=7)
        self.privacy_card.grid_columnconfigure(1, weight=1)
        privacy_mark = ctk.CTkFrame(self.privacy_card, width=5, corner_radius=2, fg_color=COLORS["success"])
        privacy_mark.grid(row=0, column=0, rowspan=2, padx=(14, 10), pady=14, sticky="ns")
        self.lbl_privacy_title = ctk.CTkLabel(
            self.privacy_card, text="Procesamiento local-first", font=ctk.CTkFont(size=17, weight="bold"), anchor="w"
        )
        self.lbl_privacy_title.grid(row=0, column=1, padx=(0, 4), pady=(15, 2), sticky="ew")
        HelpButton(
            self.privacy_card, title="Privacidad de este trabajo",
            help_text="Esta tarjeta resume, con la configuración actual, qué se procesa dentro de esta computadora y qué datos podrían salir mediante una API o un servidor remoto. Revisa este bloque antes de iniciar si trabajas con información sensible.",
        ).grid(row=0, column=2, padx=(4, 20), pady=(12, 0), sticky="e")
        self.lbl_privacy = ctk.CTkLabel(
            self.privacy_card, text="", text_color=COLORS["muted"], justify="left", anchor="w", wraplength=820
        )
        self.lbl_privacy.grid(row=1, column=1, columnspan=2, padx=(0, 20), pady=(0, 16), sticky="ew")

        actions = ctk.CTkFrame(scroll, fg_color="transparent")
        actions.grid(row=9, column=0, sticky="ew", pady=(10, 26))
        actions.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(
            actions, text="Opciones avanzadas", height=40, fg_color=COLORS["accent_soft"],
            hover_color=COLORS["nav_hover"], text_color=COLORS["text"], command=self._open_advanced,
        ).grid(row=0, column=0, sticky="w")
        HelpButton(
            actions, title="Opciones avanzadas",
            help_text="Permite cambiar modelo local, dispositivo, cantidad esperada de hablantes, mejora de audio, endpoints, modelos de texto, prompts y formatos de exportación. Si no necesitas una decisión técnica concreta, deja los valores recomendados.",
        ).grid(row=0, column=1, sticky="w", padx=8)
        self.btn_start = ctk.CTkButton(
            actions, text="Iniciar desgrabación", width=198, height=44,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"], command=self._start_job,
        )
        self.btn_start.grid(row=0, column=2, sticky="e")
        HelpButton(
            actions, title="Iniciar desgrabación",
            help_text="Antes de comenzar Jerónimo verifica los componentes necesarios, el modelo local seleccionado y cualquier credencial requerida. Si una configuración envía datos fuera del equipo, solicita confirmación explícita antes de iniciar.",
        ).grid(row=0, column=3, sticky="e", padx=(8, 0))
        self._refresh_privacy_card()

    # ---------- process ----------
    def _build_process(self, parent):
        scroll = self._page_scroll(parent)

        # Cabecera editorial Territorio Vivo.
        process_title = ctk.CTkFrame(scroll, fg_color="transparent")
        process_title.grid(row=0, column=0, sticky="ew", pady=(4, 2))
        process_title.grid_columnconfigure(0, weight=1)
        title_box = ctk.CTkFrame(process_title, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(title_box, text="RECORRIDO DE TRANSCRIPCIÓN", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["terracotta"], anchor="w").pack(anchor="w")
        title_line = ctk.CTkFrame(title_box, fg_color="transparent")
        title_line.pack(anchor="w")
        ctk.CTkLabel(title_line, text="Procesando entrevista", font=ctk.CTkFont(size=29, weight="bold"), anchor="w").pack(side="left")
        HelpButton(title_line, title="Proceso", help_text="La barra y las etapas muestran en qué punto está el trabajo. La ETA se calcula con rendimiento observado cuando hay datos suficientes; durante operaciones monolíticas puede ser aproximada. Cancelar solicita cierre seguro y limpieza de temporales.").pack(side="left", padx=8)
        local_badge = ctk.CTkLabel(process_title, text="●  PROCESO CONTROLADO", height=28, corner_radius=14, fg_color=COLORS["accent_soft"], text_color=COLORS["success"], font=ctk.CTkFont(size=9, weight="bold"))
        local_badge.grid(row=0, column=1, sticky="e", padx=(18, 0))
        self.lbl_process_file = ctk.CTkLabel(scroll, text="", text_color=COLORS["muted"], anchor="w")
        self.lbl_process_file.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        # Recomendación visible durante todo el procesamiento. No bloquea el uso
        # del equipo ni altera el motor; explica únicamente la competencia por recursos.
        performance_notice = ctk.CTkFrame(
            scroll, fg_color=COLORS["external_soft"], corner_radius=12,
            border_width=1, border_color=COLORS["terracotta"],
        )
        performance_notice.grid(row=2, column=0, sticky="ew", pady=(0, 7))
        performance_notice.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            performance_notice, text="!", width=34, height=34, corner_radius=17,
            fg_color=COLORS["terracotta"], text_color=COLORS["on_accent"],
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=(16, 10), pady=13, sticky="w")
        notice_text = ctk.CTkFrame(performance_notice, fg_color="transparent")
        notice_text.grid(row=0, column=1, padx=(0, 16), pady=10, sticky="ew")
        ctk.CTkLabel(
            notice_text, text="RECOMENDACIÓN DE RENDIMIENTO",
            font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["terracotta"], anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            notice_text,
            text=("Para el máximo rendimiento se recomienda no usar el equipo durante el procesamiento. "
                  "Si necesitas usarlo, evita video, música/streaming, juegos y otras tareas pesadas."),
            text_color=COLORS["text"], anchor="w", justify="left", wraplength=850,
        ).pack(fill="x", anchor="w", pady=(2, 0))

        # Etapa actual: mismos labels y progress bar usados por el motor de UI.
        main = Card(scroll)
        main.grid(row=3, column=0, sticky="ew", pady=7)
        main.grid_columnconfigure(0, weight=1)
        stage_header = ctk.CTkFrame(main, fg_color="transparent")
        stage_header.grid(row=0, column=0, padx=22, pady=(18, 0), sticky="ew")
        stage_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(stage_header, text="ETAPA ACTUAL", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["terracotta"], anchor="w").grid(row=0, column=0, sticky="w")
        self.lbl_percent = ctk.CTkLabel(stage_header, text="0 %", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLORS["accent"])
        self.lbl_percent.grid(row=0, column=1, sticky="e")
        self.lbl_stage = ctk.CTkLabel(main, text="Preparando", font=ctk.CTkFont(size=22, weight="bold"), anchor="w")
        self.lbl_stage.grid(row=1, column=0, padx=22, pady=(4, 2), sticky="ew")
        self.lbl_stage_message = ctk.CTkLabel(main, text="", text_color=COLORS["muted"], anchor="w", wraplength=850)
        self.lbl_stage_message.grid(row=2, column=0, padx=22, sticky="ew")
        self.progress = ctk.CTkProgressBar(main, height=12, corner_radius=6, fg_color=COLORS["card_alt"], progress_color=COLORS["terracotta"])
        self.progress.set(0)
        self.progress.grid(row=3, column=0, padx=22, pady=(18, 8), sticky="ew")
        meta = ctk.CTkFrame(main, fg_color="transparent")
        meta.grid(row=4, column=0, padx=22, pady=(0, 18), sticky="ew")
        meta.grid_columnconfigure(0, weight=1)
        self.lbl_eta = ctk.CTkLabel(meta, text="Calculando velocidad…", text_color=COLORS["muted"], anchor="w")
        self.lbl_eta.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(meta, text="El original no se sobrescribe", text_color=COLORS["muted"], font=ctk.CTkFont(size=10), anchor="e").grid(row=0, column=1, sticky="e")

        # Telemetría local, informativa y desacoplada del pipeline. No condiciona
        # el trabajo: si un contador no está disponible, se muestra simplemente —.
        resources = ctk.CTkFrame(main, fg_color=COLORS["card_alt"], corner_radius=10)
        resources.grid(row=5, column=0, padx=22, pady=(0, 18), sticky="ew")
        for col in range(5):
            resources.grid_columnconfigure(col, weight=1)
        ctk.CTkLabel(
            resources, text="CARGA DEL EQUIPO · MEDICIÓN LOCAL",
            font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["water"], anchor="w",
        ).grid(row=0, column=0, columnspan=5, padx=14, pady=(10, 5), sticky="ew")

        def resource_cell(column, title):
            box = ctk.CTkFrame(resources, fg_color="transparent")
            box.grid(row=1, column=column, padx=10, pady=(0, 11), sticky="ew")
            ctk.CTkLabel(box, text=title, font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"], anchor="w").pack(fill="x")
            value = ctk.CTkLabel(box, text="—", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text"], anchor="w")
            value.pack(fill="x", pady=(1, 3))
            bar = ctk.CTkProgressBar(box, height=5, corner_radius=3, fg_color=COLORS["border"], progress_color=COLORS["water"])
            bar.set(0)
            bar.pack(fill="x")
            return value, bar

        self.lbl_resource_cpu, self.bar_resource_cpu = resource_cell(0, "CPU")
        self.lbl_resource_ram, self.bar_resource_ram = resource_cell(1, "RAM")
        self.lbl_resource_gpu, self.bar_resource_gpu = resource_cell(2, "GPU")
        self.lbl_resource_vram, self.bar_resource_vram = resource_cell(3, "VRAM")
        temp_box = ctk.CTkFrame(resources, fg_color="transparent")
        temp_box.grid(row=1, column=4, padx=10, pady=(0, 11), sticky="ew")
        ctk.CTkLabel(temp_box, text="TEMP. GPU", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"], anchor="w").pack(fill="x")
        self.lbl_resource_gpu_temp = ctk.CTkLabel(temp_box, text="—", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text"], anchor="w")
        self.lbl_resource_gpu_temp.pack(fill="x", pady=(1, 8))
        ctk.CTkLabel(temp_box, text="CPU temp. no estimada", font=ctk.CTkFont(size=8), text_color=COLORS["muted"], anchor="w").pack(fill="x")

        # Recorrido por etapas: la actualización dinámica sigue usando self.stage_labels.
        stages = Card(scroll)
        stages.grid(row=4, column=0, sticky="ew", pady=7)
        stages.grid_columnconfigure(0, weight=1)
        stages_title = ctk.CTkFrame(stages, fg_color="transparent")
        stages_title.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 9))
        ctk.CTkLabel(stages_title, text="Recorrido", font=ctk.CTkFont(size=18, weight="bold"), anchor="w").pack(side="left")
        ctk.CTkLabel(stages_title, text="FUENTE → TRANSCRIPCIÓN → DERIVADOS → ARCHIVO", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"]).pack(side="right")
        self.stage_labels = {}
        labels = [
            ("preparing", "Preparar audio"), ("transcribing", "Transcribir"), ("diarizing", "Separar hablantes"),
            ("cleaning", "Versión normalizada"), ("summarizing", "Resumen"), ("exporting", "Guardar / exportar"),
            ("finalizing", "Finalizar"),
        ]
        for i, (key, text) in enumerate(labels, start=1):
            row_box = ctk.CTkFrame(stages, fg_color="transparent")
            row_box.grid(row=i, column=0, padx=20, pady=0, sticky="ew")
            row_box.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row_box, text=f"{i:02d}", width=34, text_color=COLORS["muted"], font=ctk.CTkFont(size=9, weight="bold")).grid(row=0, column=0, sticky="w")
            label = ctk.CTkLabel(row_box, text=f"○  {text}", anchor="w", text_color=COLORS["muted"], height=30)
            label.grid(row=0, column=1, sticky="ew")
            self.stage_labels[key] = label
            if i < len(labels):
                ctk.CTkFrame(row_box, width=1, height=7, fg_color=COLORS["border"]).grid(row=1, column=0, padx=(16, 0), sticky="n")
        ctk.CTkLabel(stages, text="", height=8).grid(row=len(labels) + 1, column=0)

        details = Card(scroll)
        details.grid(row=5, column=0, sticky="ew", pady=7)
        details.grid_columnconfigure(0, weight=1)
        row = ctk.CTkFrame(details, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 6))
        ctk.CTkLabel(row, text="Detalles técnicos", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        HelpButton(row, title="Detalles técnicos", help_text="Este registro muestra mensajes de ejecución útiles para diagnóstico. Los logs persistentes evitan guardar el contenido completo de las entrevistas y pseudonimizan rutas/nombres cuando corresponde.").pack(side="left", padx=8)
        ctk.CTkLabel(row, text="REGISTRO", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["water"]).pack(side="right")
        self.txt_log = ctk.CTkTextbox(details, height=145, fg_color=COLORS["card_alt"], border_width=1, border_color=COLORS["border"])
        self.txt_log.grid(row=1, column=0, padx=20, pady=(0, 18), sticky="ew")
        self.txt_log.configure(state="disabled")

        self.btn_cancel = ctk.CTkButton(scroll, text="Cancelar de forma segura", width=190, fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"], state="disabled", command=self._cancel_job)
        self.btn_cancel.grid(row=6, column=0, sticky="e", pady=(10, 24))


    # ---------- results ----------
    def _build_results(self, parent):
        scroll = self._page_scroll(parent)
        result_title_row = ctk.CTkFrame(scroll, fg_color="transparent")
        result_title_row.grid(row=0, column=0, sticky="ew", pady=(4, 2))
        result_title_row.grid_columnconfigure(0, weight=1)
        title_box = ctk.CTkFrame(result_title_row, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(title_box, text="ARCHIVO DE RESULTADOS", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["terracotta"], anchor="w").pack(anchor="w")
        title_line = ctk.CTkFrame(title_box, fg_color="transparent")
        title_line.pack(anchor="w")
        self.lbl_result_title = ctk.CTkLabel(title_line, text="Resultados", font=ctk.CTkFont(size=29, weight="bold"), anchor="w")
        self.lbl_result_title.pack(side="left")
        HelpButton(title_line, title="Resultados", help_text="Aquí puedes revisar la transcripción junto al audio, abrir la carpeta generada y consultar los archivos derivados. La transcripción original automática se conserva separada de normalizaciones y resúmenes.").pack(side="left", padx=8)
        trace_badge = ctk.CTkLabel(result_title_row, text="TRAZABILIDAD ACTIVA", height=28, corner_radius=14, fg_color=COLORS["accent_soft"], text_color=COLORS["success"], font=ctk.CTkFont(size=9, weight="bold"))
        trace_badge.grid(row=0, column=1, sticky="e", padx=(18, 0))
        self.lbl_result_subtitle = ctk.CTkLabel(scroll, text="", text_color=COLORS["muted"], anchor="w")
        self.lbl_result_subtitle.grid(row=1, column=0, sticky="ew", pady=(0, 18))

        actions = Card(scroll)
        actions.grid(row=2, column=0, sticky="ew", pady=7)
        for col in range(3):
            actions.grid_columnconfigure(col, weight=1)
        review_box = ctk.CTkFrame(actions, fg_color="transparent")
        review_box.grid(row=0, column=0, sticky="nsew", padx=(18, 8), pady=16)
        ctk.CTkLabel(review_box, text="01 · REVISAR", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["terracotta"], anchor="w").pack(fill="x")
        self.btn_review = ctk.CTkButton(review_box, text="Revisar transcripción", height=38, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=self._open_last_review)
        self.btn_review.pack(fill="x", pady=(6, 0))

        folder_box = ctk.CTkFrame(actions, fg_color="transparent")
        folder_box.grid(row=0, column=1, sticky="nsew", padx=8, pady=16)
        ctk.CTkLabel(folder_box, text="02 · EXPLORAR", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["water"], anchor="w").pack(fill="x")
        ctk.CTkButton(folder_box, text="Abrir carpeta", height=38, fg_color=COLORS["card_alt"], hover_color=COLORS["border"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], command=lambda: _open_path(self.output_dir)).pack(fill="x", pady=(6, 0))

        new_box = ctk.CTkFrame(actions, fg_color="transparent")
        new_box.grid(row=0, column=2, sticky="nsew", padx=(8, 18), pady=16)
        ctk.CTkLabel(new_box, text="03 · CONTINUAR", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["maize"], anchor="w").pack(fill="x")
        ctk.CTkButton(new_box, text="Nueva desgrabación", height=38, fg_color=COLORS["card_alt"], hover_color=COLORS["border"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], command=self._reset_for_new).pack(fill="x", pady=(6, 0))

        files = Card(scroll)
        files.grid(row=3, column=0, sticky="ew", pady=7)
        files.grid_columnconfigure(0, weight=1)
        files_header = ctk.CTkFrame(files, fg_color="transparent")
        files_header.grid(row=0, column=0, padx=20, pady=(17, 7), sticky="ew")
        files_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(files_header, text="Archivos generados", font=ctk.CTkFont(size=18, weight="bold"), anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(files_header, text="ORIGINALES + DERIVADOS", font=ctk.CTkFont(size=9, weight="bold"), text_color=COLORS["muted"]).grid(row=0, column=1, sticky="e")
        self.results_files_frame = ctk.CTkFrame(files, fg_color="transparent")
        self.results_files_frame.grid(row=1, column=0, padx=20, pady=(0, 18), sticky="ew")
        self.results_files_frame.grid_columnconfigure(0, weight=1)

        note = Card(scroll)
        note.grid(row=4, column=0, sticky="ew", pady=7)
        note.grid_columnconfigure(1, weight=1)
        accent = ctk.CTkFrame(note, width=4, fg_color=COLORS["terracotta"], corner_radius=2)
        accent.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 0), pady=0)
        trace_title = ctk.CTkFrame(note, fg_color="transparent")
        trace_title.grid(row=0, column=1, sticky="ew", padx=20, pady=(14, 2))
        ctk.CTkLabel(trace_title, text="Trazabilidad", font=ctk.CTkFont(size=17, weight="bold"), anchor="w").pack(side="left")
        HelpButton(trace_title, title="Trazabilidad", help_text="El manifiesto registra motor, modelo, modo local/remoto, transformaciones y estado del trabajo sin almacenar API keys. La transcripción raw se mantiene como referencia separada de los derivados.").pack(side="left", padx=8)
        ctk.CTkLabel(note, text="La transcripción original se conserva como .raw.txt. Las versiones normalizadas y los resúmenes se guardan como derivados separados; el manifiesto registra el flujo sin guardar claves.", text_color=COLORS["muted"], justify="left", wraplength=850, anchor="w").grid(row=1, column=1, sticky="ew", padx=20, pady=(0, 16))


    # ---------- interactions ----------
    def _configure_drag_drop(self):
        if not HAS_DND:
            return
        try:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        try:
            paths = list(self.tk.splitlist(event.data))
        except Exception:
            paths = [str(event.data).strip("{}")]
        if not paths:
            return
        path = Path(paths[0])
        if path.exists():
            self._set_input(path)
            self._show_page("new")

    def _select_file(self):
        value = fd.askopenfilename(title="Seleccionar entrevista", filetypes=[("Audio y video", "*.wav *.mp3 *.m4a *.mp4 *.mov *.mkv *.ogg *.flac"), ("Todos", "*.*")])
        if value:
            self._set_input(Path(value))

    def _select_folder(self):
        value = fd.askdirectory(title="Seleccionar carpeta de entrevistas")
        if value:
            self._set_input(Path(value))

    def _set_input(self, path: Path):
        self.input_path = Path(path)
        self.lbl_drop.configure(text=f"Seleccionado\n{_short_path(path)}", text_color=COLORS["text"])
        if path.is_file() and self.entry_output.get().strip() == str(Path.cwd() / "transcripciones"):
            suggested = path.parent / "transcripciones"
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, str(suggested))
        self._refresh_privacy_card()

    def _select_output(self):
        value = fd.askdirectory(title="Carpeta para resultados")
        if value:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, value)

    def _on_profile_changed(self, label: str):
        self.profile = PROFILE_LABELS.get(label, PROFILE_DIARIZATION)
        notes = {
            PROFILE_DIARIZATION: "WhisperX large-v2 + diarización local. Es el baseline actual de máxima fidelidad.",
            PROFILE_LOCAL: "faster-whisper local sin separación de hablantes. Menos dependencias y normalmente más rápido.",
            PROFILE_OPENAI: "Transcripción mediante API. El archivo de audio saldrá del equipo.",
        }
        self.lbl_profile_note.configure(text=notes[self.profile])
        self._refresh_privacy_card()

    def _on_text_engine_changed(self, label: str):
        self.text_engine = TEXT_LABELS.get(label, TEXT_OLLAMA)
        if self.text_engine == TEXT_NONE:
            self.var_summary.set(False)
            self.var_clean.set(False)
        self._refresh_text_model_label()
        self._refresh_privacy_card()

    def _current_selection(self) -> WorkflowSelection:
        self.profile = PROFILE_LABELS.get(self.seg_profile.get(), self.profile)
        self.text_engine = TEXT_LABELS.get(self.combo_text_engine.get(), self.text_engine)
        self.do_summary = bool(self.var_summary.get())
        self.do_clean = bool(self.var_clean.get())
        if self.text_engine == TEXT_NONE:
            self.do_summary = False
            self.do_clean = False
        self.output_dir = Path(self.entry_output.get().strip() or (Path.cwd() / "transcripciones"))
        self.interviewer = self.entry_interviewer.get().strip()
        self.interviewee = self.entry_interviewee.get().strip()
        self.project = self.entry_project.get().strip()
        place_date = self.entry_place_date.get().strip()
        self.place = place_date
        return WorkflowSelection(
            input_path=self.input_path or Path("."), output_dir=self.output_dir, profile=self.profile,
            language=self.language, target_speakers=self.target_speakers, audio_enhancement=self.audio_enhancement,
            do_clean=self.do_clean, do_summary=self.do_summary, text_engine=self.text_engine,
            ollama_url=self.ollama_url, ollama_model=self.ollama_model,
            openai_api_key=self.session_openai_key or resolve_secret("openai") or (load_api_key_from_env() or ""),
            openai_stt_model=self.openai_stt_model, openai_chat_model=self.openai_chat_model,
            compatible_api_url=self.compatible_api_url, compatible_api_key=self.session_compatible_key or resolve_secret("openai_compatible"),
            compatible_model=self.compatible_model, local_fast_model=self.local_fast_model, local_device=self.local_device,
            interviewer=self.interviewer, interviewee=self.interviewee, project=self.project, place=self.place,
            date_label=self.date_label, extra_context=self.extra_context,
            whisper_prompt=self.whisper_prompt, clean_prompt=self.clean_prompt, summary_prompt=self.summary_prompt,
            export_docx_raw=self.export_docx_raw, export_docx_atlas=self.export_docx_atlas,
            export_rtf_raw=self.export_rtf_raw, export_rtf_atlas=self.export_rtf_atlas,
            atlas_turn_numbers=self.atlas_turn_numbers, atlas_timecodes=self.atlas_timecodes,
            atlas_speaker_id=self.atlas_speaker_id, atlas_bold_speaker=self.atlas_bold_speaker,
            diarization_runtime_python=os.environ.get("JERONIMO_DIARIZATION_PYTHON", "").strip(),
            diarization_worker_executable=os.environ.get("JERONIMO_DIARIZATION_WORKER_EXE", "").strip(),
        )

    def _refresh_privacy_card(self):
        if not hasattr(self, "privacy_card"):
            return
        try:
            selection = self._current_selection()
        except Exception:
            return
        assessment = assess_privacy(selection)
        lines = list(assessment.details)
        if selection.text_engine == TEXT_OLLAMA:
            endpoint = describe_endpoint(TextProviderSettings(engine="ollama", ollama_model=selection.ollama_model, ollama_url=selection.ollama_url))
            if endpoint.sends_text_off_device:
                lines.append(f"Atención: Ollama apunta a {endpoint.endpoint}; el texto no permanecerá en esta PC.")
                title = "Procesamiento híbrido / externo"
            else:
                title = assessment.title
        else:
            title = assessment.title
        self.lbl_privacy_title.configure(text=title, text_color=COLORS["warning"] if "externo" in title.lower() else COLORS["success"])
        self.lbl_privacy.configure(text="\n".join("• " + x for x in lines))

    def _refresh_text_model_label(self):
        if not hasattr(self, "lbl_text_model"):
            return
        if self.text_engine == TEXT_OLLAMA:
            model = self.ollama_model or (recommended_text_model(self._diag_report) if self._diag_report else "por determinar")
            self.lbl_text_model.configure(text=f"Modelo local: {model}. Puedes descargarlo desde 'Modelos locales'.")
        elif self.text_engine == TEXT_OPENAI:
            self.lbl_text_model.configure(text=f"Modelo API: {self.openai_chat_model}. El texto será enviado a OpenAI.")
        elif self.text_engine == TEXT_COMPATIBLE:
            self.lbl_text_model.configure(text=f"Modelo API compatible: {self.compatible_model or 'especificar en Opciones avanzadas'}.")
        else:
            self.lbl_text_model.configure(text="No se ejecutará limpieza ni resumen mediante IA.")

    def _open_advanced(self):
        AdvancedDialog(self)

    def _start_job(self):
        if self._is_running:
            mb.showwarning("En ejecución", "Ya hay un trabajo en curso.", parent=self)
            return
        if self.input_path is None or not self.input_path.exists():
            mb.showerror("Grabación", "Selecciona un archivo o carpeta antes de iniciar.", parent=self)
            return
        try:
            selection = self._current_selection()
            cfg = build_config(selection)
        except Exception as exc:
            mb.showerror("Configuración", str(exc), parent=self)
            return

        # Validaciones que explican el problema antes de lanzar un trabajo largo.
        if cfg.use_diarization:
            env = check_environment_status(cfg.api_key)
            if not env.get("ffmpeg"):
                mb.showerror("Diarización", "El perfil de calidad con hablantes requiere FFmpeg. Abre Diagnóstico para ver cómo está el equipo.", parent=self)
                return
            if not env.get("whisperx") and not cfg.diarization_runtime_python and not cfg.diarization_worker_executable:
                if not mb.askyesno("WhisperX no detectado", "Jerónimo Abya Yala no detectó WhisperX en este entorno. Si está en un runtime aislado todavía no configurado, el trabajo fallará.\n\n¿Intentar igualmente?", parent=self):
                    return
            if not env.get("pyannote_local_model") and not env.get("huggingface_token"):
                mb.showerror(
                    "Modelo de hablantes",
                    "No se encontró Community-1 instalado localmente ni una credencial de Hugging Face de contingencia. "
                    "Abrí Configuración inicial para preparar el modelo de hablantes.",
                    parent=self,
                )
                return
        if cfg.stt_engine == "openai" and not cfg.api_key:
            mb.showerror("OpenAI", "El perfil OpenAI API necesita una credencial configurada.", parent=self)
            return
        if (cfg.do_clean or cfg.do_summary) and cfg.text_engine == TEXT_OLLAMA and not cfg.ollama_model:
            if self._diag_report:
                self.ollama_model = recommended_text_model(self._diag_report)
                cfg.ollama_model = self.ollama_model
            else:
                mb.showerror("Modelo local", "Selecciona un modelo Ollama en Opciones avanzadas o abre Modelos locales.", parent=self)
                return
        if (cfg.do_clean or cfg.do_summary) and cfg.text_engine == TEXT_OLLAMA:
            installed = ollama_inventory(cfg.ollama_url)
            model_present = ollama_model_present(installed, cfg.ollama_model)
            if not model_present:
                rec_id = recommended_text_model(self._diag_report) if self._diag_report else ""
                rec_spec = next((spec for spec in TEXT_MODELS if spec.model_id == rec_id), None)
                if rec_spec is not None:
                    rec_present = ollama_model_present(installed, rec_spec.model_id)
                    action = ask_recommended_model_action(
                        self, cfg.ollama_model, rec_spec, recommended_installed=rec_present,
                    )
                    if action == "recommended" and rec_present:
                        self.ollama_model = rec_spec.model_id
                        cfg.ollama_model = rec_spec.model_id
                        if rec_spec.model_id not in self.ollama_models:
                            self.ollama_models.append(rec_spec.model_id)
                        self.ollama_models = sorted(set(self.ollama_models))
                        self._refresh_text_model_label()
                    elif action == "recommended":
                        open_model_manager(
                            self, self.ollama_url, str(self.output_dir), auto_recommended=True,
                        )
                        return
                    elif action == "other":
                        self._open_models()
                        return
                    else:
                        return
                else:
                    mb.showwarning(
                        "Modelo local no disponible",
                        f"No se encontró '{cfg.ollama_model}'. Abre Modelos locales para elegir o descargar un modelo compatible.",
                        parent=self,
                    )
                    self._open_models()
                    return
        if (cfg.do_clean or cfg.do_summary) and cfg.text_engine == TEXT_OPENAI and not cfg.api_key:
            mb.showerror("OpenAI", "El resumen/limpieza con OpenAI necesita una API key.", parent=self)
            return
        if (cfg.do_clean or cfg.do_summary) and cfg.text_engine == TEXT_COMPATIBLE and not cfg.compatible_model:
            mb.showerror("API compatible", "Especifica el modelo compatible en Opciones avanzadas.", parent=self)
            return

        privacy = assess_privacy(selection)
        endpoint_warning = None
        if cfg.text_engine == TEXT_OLLAMA:
            info = describe_endpoint(TextProviderSettings(engine="ollama", ollama_model=cfg.ollama_model, ollama_url=cfg.ollama_url))
            if info.sends_text_off_device:
                endpoint_warning = f"El texto será enviado a {info.endpoint}."
        if privacy.audio_leaves_device or privacy.text_leaves_device or endpoint_warning:
            message = "Esta configuración enviará datos fuera de este equipo.\n\n"
            if privacy.audio_leaves_device:
                message += "• Se enviará la grabación para transcripción.\n"
            if privacy.text_leaves_device:
                message += "• Se enviará la transcripción para limpieza/resumen.\n"
            if endpoint_warning:
                message += f"• {endpoint_warning}\n"
            message += "\n¿Continuar?"
            if not mb.askyesno("Procesamiento externo", message, parent=self):
                return

        # Última confirmación antes de reservar recursos y crear el JobTracker.
        # Se presenta después de todas las validaciones para no molestar si el
        # trabajo todavía no está en condiciones de iniciar.
        advice = PerformanceAdviceDialog(self)
        self.wait_window(advice)
        if not advice.result:
            return

        self._current_cfg = cfg
        self._cancel_event.clear()
        cfg.cancel_check = self._cancel_event.is_set
        self._before_outputs = snapshot_directory(cfg.output_dir)
        self._new_outputs = []
        self._last_audio = self.input_path if self.input_path.is_file() else None
        self._last_transcript = None
        self._last_status = ""

        self._job_tracker = JobTracker(Job(kind="transcription"), on_event=lambda event: self._post_ui(self._render_job_event, event))
        self._job_tracker.job.configuration = {
            "profile": selection.profile,
            "use_diarization": cfg.use_diarization,
            "stt_model": cfg.whisperx_model if cfg.use_diarization else (cfg.local_model_name if cfg.stt_engine != "openai" else cfg.stt_model),
            "text_engine": cfg.text_engine,
            "do_clean": cfg.do_clean,
            "do_summary": cfg.do_summary,
        }
        start_event = self._job_tracker.start(message="Preparando el trabajo.")
        cfg.job_id = self._job_tracker.job.job_id
        cfg.job_event_callback = self._handle_core_event

        self._is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self._clear_log()
        self.lbl_process_file.configure(text=_short_path(self.input_path))
        self._render_job_event(start_event)
        self._show_page("process")
        self._start_resource_monitor()

        thread = threading.Thread(target=self._job_worker, args=(cfg,), daemon=True)
        thread.start()

    def _job_worker(self, cfg: TranscriberConfig):
        # Este método corre fuera del hilo de Tk. No debe ejecutar ``after``,
        # messageboxes, widgets ni ninguna otra operación Tcl/Tk.
        def progress(message: str):
            self._post_ui(self._append_log, message)

        def set_progress(value: float):
            tracker = self._job_tracker
            if tracker is not None:
                tracker.update(progress=value)

        def on_file_done(audio_path: Path):
            self._post_ui(self._remember_last_audio, Path(audio_path))

        try:
            run_batch(cfg, progress, set_progress, on_file_done=on_file_done)
        except Exception as exc:
            tracker = self._job_tracker
            if tracker is not None and tracker.job.status not in {JOB_STATUS_COMPLETED, JOB_STATUS_CANCELLED, JOB_STATUS_ERROR}:
                tracker.fail(str(exc))
            self._post_ui(self._append_log, f"ERROR general: {exc}")
            self._post_ui(mb.showerror, "Jerónimo Abya Yala", str(exc), parent=self)
        finally:
            self._post_ui(self._finish_job_ui)

    def _remember_last_audio(self, audio_path: Path):
        if self._last_audio is None:
            self._last_audio = Path(audio_path)

    def _handle_core_event(self, payload: dict):
        tracker = self._job_tracker
        if tracker is None:
            return
        tracker.update(stage=payload.get("stage"), progress=payload.get("progress"), message=payload.get("message"), current_item=payload.get("current_item"), status=payload.get("status"))

    def _render_job_event(self, event):
        label = stage_label(event.stage)
        pct = max(0, min(100, int(round(event.progress * 100))))
        self.lbl_stage.configure(text=label)
        self.lbl_stage_message.configure(text=event.message or event.current_item or "")
        self.progress.set(event.progress)
        self.lbl_percent.configure(text=f"{pct} %")
        elapsed = format_duration(event.elapsed_seconds)
        if event.status == JOB_STATUS_COMPLETED:
            eta = f"Finalizado · {elapsed}"
        elif event.status == JOB_STATUS_CANCELLED:
            eta = f"Cancelado · {elapsed}"
        elif event.status == JOB_STATUS_ERROR:
            eta = f"Finalizado con errores · {elapsed}"
        elif event.eta_seconds is None:
            eta = f"Transcurrido {elapsed} · calculando velocidad…"
        else:
            eta = f"Transcurrido {elapsed} · restante aprox. {format_duration(event.eta_seconds)}"
        self.lbl_eta.configure(text=eta)
        self._render_stage_list(event.stage, event.status)

    def _render_stage_list(self, stage: str, status: str):
        current_index = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1
        cfg = self._current_cfg
        for i, key in enumerate(STAGE_ORDER):
            label = self.stage_labels.get(key)
            if label is None:
                continue
            # Saltar visualmente etapas que no aplican.
            if key == "diarizing" and cfg is not None and not cfg.use_diarization:
                symbol, color = "–", COLORS["muted"]
            elif key == "cleaning" and cfg is not None and not cfg.do_clean:
                symbol, color = "–", COLORS["muted"]
            elif key == "summarizing" and cfg is not None and not cfg.do_summary:
                symbol, color = "–", COLORS["muted"]
            elif i < current_index or status == JOB_STATUS_COMPLETED:
                symbol, color = "✓", COLORS["success"]
            elif i == current_index:
                symbol = "×" if status == JOB_STATUS_ERROR else ("■" if status == JOB_STATUS_CANCELLED else "●")
                color = COLORS["danger"] if status == JOB_STATUS_ERROR else COLORS["accent"]
            else:
                symbol, color = "○", COLORS["muted"]
            original = label.cget("text")
            text = original[3:] if len(original) > 3 else original
            label.configure(text=f"{symbol}  {text}", text_color=color)

    def _start_resource_monitor(self):
        self._stop_resource_monitor()
        self._render_resource_snapshot(None)
        monitor = SystemResourceMonitor(interval=1.0, gpu_interval=2.0).start()
        self._resource_monitor = monitor
        self._resource_poll_job = self.after(300, self._poll_resource_monitor)

    def _poll_resource_monitor(self):
        self._resource_poll_job = None
        monitor = self._resource_monitor
        if monitor is None or not self._is_running:
            return
        snapshot = monitor.latest()
        if snapshot is not None:
            self._render_resource_snapshot(snapshot)
        self._resource_poll_job = self.after(1000, self._poll_resource_monitor)

    def _stop_resource_monitor(self):
        job = self._resource_poll_job
        self._resource_poll_job = None
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        monitor = self._resource_monitor
        self._resource_monitor = None
        if monitor is not None:
            monitor.stop(wait=False)

    @staticmethod
    def _resource_fraction(value):
        if value is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(value) / 100.0))
        except Exception:
            return 0.0

    def _render_resource_snapshot(self, snapshot: SystemSnapshot | None):
        if snapshot is None:
            self.lbl_resource_cpu.configure(text="Midiendo…")
            self.lbl_resource_ram.configure(text="Midiendo…")
            self.lbl_resource_gpu.configure(text="Midiendo…")
            self.lbl_resource_vram.configure(text="Midiendo…")
            self.lbl_resource_gpu_temp.configure(text="—")
            for bar in (self.bar_resource_cpu, self.bar_resource_ram, self.bar_resource_gpu, self.bar_resource_vram):
                bar.set(0)
            return

        cpu = snapshot.cpu_percent
        if cpu is None:
            self.lbl_resource_cpu.configure(text="—")
            self.bar_resource_cpu.set(0)
        else:
            self.lbl_resource_cpu.configure(text=f"{cpu:.0f} %")
            self.bar_resource_cpu.set(self._resource_fraction(cpu))

        ram_pct = snapshot.ram_percent
        if ram_pct is None or not snapshot.ram_total_bytes:
            self.lbl_resource_ram.configure(text="—")
            self.bar_resource_ram.set(0)
        else:
            used_gb = (snapshot.ram_used_bytes or 0) / (1024 ** 3)
            total_gb = snapshot.ram_total_bytes / (1024 ** 3)
            self.lbl_resource_ram.configure(text=f"{ram_pct:.0f} % · {used_gb:.1f}/{total_gb:.1f} GB")
            self.bar_resource_ram.set(self._resource_fraction(ram_pct))

        if not snapshot.gpu_available:
            self.lbl_resource_gpu.configure(text="No disponible")
            self.lbl_resource_vram.configure(text="No disponible")
            self.lbl_resource_gpu_temp.configure(text="—")
            self.bar_resource_gpu.set(0)
            self.bar_resource_vram.set(0)
            return

        gpu_pct = snapshot.gpu_percent
        if gpu_pct is None:
            self.lbl_resource_gpu.configure(text="—")
            self.bar_resource_gpu.set(0)
        else:
            self.lbl_resource_gpu.configure(text=f"{gpu_pct:.0f} %")
            self.bar_resource_gpu.set(self._resource_fraction(gpu_pct))

        used_mb = snapshot.vram_used_mb
        total_mb = snapshot.vram_total_mb
        if used_mb is None or total_mb is None or total_mb <= 0:
            self.lbl_resource_vram.configure(text="—")
            self.bar_resource_vram.set(0)
        else:
            vram_pct = max(0.0, min(100.0, used_mb / total_mb * 100.0))
            self.lbl_resource_vram.configure(text=f"{vram_pct:.0f} % · {used_mb / 1024:.1f}/{total_mb / 1024:.1f} GB")
            self.bar_resource_vram.set(self._resource_fraction(vram_pct))

        temp = snapshot.gpu_temperature_c
        self.lbl_resource_gpu_temp.configure(text="—" if temp is None else f"{temp:.0f} °C")

    def _cancel_job(self):
        if not self._is_running:
            return
        if mb.askyesno("Cancelar", "¿Cancelar el proceso? Jerónimo Abya Yala intentará cerrar de forma segura y eliminar sus temporales.", parent=self):
            self._cancel_event.set()
            self.btn_cancel.configure(state="disabled", text="Cancelando…")
            self._append_log("Cancelación solicitada por el usuario.")

    def _finish_job_ui(self):
        self._stop_resource_monitor()
        self._is_running = False
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled", text="Cancelar de forma segura")
        cfg = self._current_cfg
        if cfg is None:
            return
        self._new_outputs = new_output_files(cfg.output_dir, self._before_outputs)
        self._last_transcript = find_primary_transcript(self._new_outputs)
        status = self._job_tracker.job.status if self._job_tracker else JOB_STATUS_ERROR
        self._last_status = status
        self._populate_results(status)
        self._show_page("results")

    def _populate_results(self, status: str):
        if status == JOB_STATUS_COMPLETED:
            self.lbl_result_title.configure(text="Desgrabación finalizada", text_color=COLORS["success"])
            self.lbl_result_subtitle.configure(text=f"Se generaron {len(self._new_outputs)} archivos en {_short_path(self.output_dir)}")
        elif status == JOB_STATUS_CANCELLED:
            self.lbl_result_title.configure(text="Proceso cancelado", text_color=COLORS["warning"])
            self.lbl_result_subtitle.configure(text="Los resultados completados antes de cancelar se conservaron; los temporales propios se limpiaron.")
        else:
            self.lbl_result_title.configure(text="Proceso finalizado con errores", text_color=COLORS["danger"])
            self.lbl_result_subtitle.configure(text="Revisa los archivos generados y el manifiesto. Los resultados válidos no se descartan.")
        self.btn_review.configure(state="normal" if self._last_transcript and self._last_audio else "disabled")
        for child in self.results_files_frame.winfo_children():
            child.destroy()
        if not self._new_outputs:
            empty = ctk.CTkFrame(self.results_files_frame, fg_color=COLORS["card_alt"], corner_radius=10)
            empty.grid(row=0, column=0, sticky="ew")
            ctk.CTkLabel(empty, text="No se detectaron archivos nuevos en la carpeta de salida.", text_color=COLORS["muted"], anchor="w").pack(fill="x", padx=14, pady=12)
            return
        for i, path in enumerate(self._new_outputs):
            row = ctk.CTkFrame(self.results_files_frame, fg_color=COLORS["card_alt"], corner_radius=10, border_width=1, border_color=COLORS["border"])
            row.grid(row=i, column=0, sticky="ew", pady=4)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=f"{i + 1:02d}", width=38, text_color=COLORS["terracotta"], font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, padx=(10, 2), pady=9)
            ctk.CTkLabel(row, text=path.name, anchor="w").grid(row=0, column=1, padx=8, pady=9, sticky="ew")
            ctk.CTkButton(row, text="Abrir ↗", width=78, height=28, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=lambda p=path: _open_path(p)).grid(row=0, column=2, padx=(8, 2) if _is_interview_editor_result(path) else 8, pady=6)
            if _is_interview_editor_result(path):
                ctk.CTkButton(
                    row, text="Abrir con editor", width=126, height=28,
                    fg_color=COLORS["card"], hover_color=COLORS["border"],
                    text_color=COLORS["text"], border_width=1, border_color=COLORS["border"],
                    command=lambda p=path: self._open_result_in_interview_editor(p),
                ).grid(row=0, column=3, padx=8, pady=6)


    # ---------- utilities ----------
    def _post_ui(self, callback, *args, **kwargs):
        """Encola una operación de UI sin tocar Tcl/Tk desde el thread llamador."""
        self._ui_dispatch_queue.put((callback, args, kwargs))

    def _drain_ui_dispatch_queue(self):
        """Ejecuta en el hilo principal las operaciones encoladas por workers."""
        processed = 0
        while processed < 200:
            try:
                callback, args, kwargs = self._ui_dispatch_queue.get_nowait()
            except Empty:
                break
            try:
                callback(*args, **kwargs)
            except Exception:
                _record_ui_dispatch_exception()
            finally:
                processed += 1
        try:
            self._ui_dispatch_after_id = self.after(40, self._drain_ui_dispatch_queue)
        except Exception:
            # La raíz puede estar destruyéndose; no recrear el temporizador.
            self._ui_dispatch_after_id = None

    def _clear_log(self):
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")

    def _append_log(self, message: str):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", message.rstrip() + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _load_diagnostics_async(self):
        def worker():
            try:
                report = run_system_diagnostics(work_dir=self.output_dir, ollama_url=self.ollama_url)
                inventory = ollama_inventory(self.ollama_url)
                self._post_ui(self._apply_diagnostics, report, inventory)
            except Exception as exc:
                self._post_ui(self._set_diagnostics_error, str(exc))
        threading.Thread(target=worker, daemon=True).start()

    def _set_diagnostics_error(self, message: str):
        self.lbl_home_system.configure(text=f"No se pudo completar el diagnóstico automático: {message}")

    def _apply_diagnostics(self, report, inventory):
        self._diag_report = report
        self.ollama_models = sorted(inventory.keys())
        model = recommended_text_model(report)
        if not self.ollama_model:
            self.ollama_model = model
        profile, note = recommended_profile(report)
        gpu = report.gpus[0]["name"] if report.gpus else "sin GPU NVIDIA detectada"
        ram = report.ram_total_bytes / (1024 ** 3) if report.ram_total_bytes else 0
        self.lbl_home_system.configure(text=f"{gpu} · {ram:.0f} GB RAM\nRecomendación: {note}\nResumen local sugerido: {model}")
        self._refresh_text_model_label()
        # La recomendación se muestra; sólo la aplicamos si el usuario todavía no tocó el default.
        if self.profile == PROFILE_DIARIZATION and profile != PROFILE_DIARIZATION:
            self.lbl_profile_note.configure(text=f"Recomendación del equipo: {note} Puedes cambiar de perfil si priorizas velocidad/compatibilidad.")

    def _show_diagnostics(self):
        if self._diag_report is None:
            mb.showinfo("Diagnóstico", "El diagnóstico todavía se está ejecutando. Intenta nuevamente en unos segundos.", parent=self)
            return
        win = ctk.CTkToplevel(self)
        win.title("Jerónimo Abya Yala — Diagnóstico del equipo")
        win.geometry("860x680")
        win.configure(fg_color=COLORS["canvas"])
        try:
            win.iconbitmap(str(asset_path("brand", "app.ico")))
        except Exception:
            pass
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(
            win, corner_radius=CARD_RADIUS, fg_color=COLORS["card"],
            border_width=1, border_color=COLORS["border"],
        )
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header, text="DIAGNÓSTICO LOCAL", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS["terracotta"], anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(
            header, text="Estado de este equipo", font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["text"], anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 2))
        ctk.CTkLabel(
            header, text="La comprobación se ejecuta localmente y no envía información del equipo.",
            text_color=COLORS["muted"], anchor="w",
        ).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 12))
        ctk.CTkLabel(
            header, text="LOCAL", font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COLORS["accent_soft"], text_color=COLORS["text"], corner_radius=9,
        ).grid(row=0, column=1, rowspan=3, padx=16, pady=14, ipadx=10, ipady=4)

        panel = ctk.CTkFrame(
            win, corner_radius=CARD_RADIUS, fg_color=COLORS["card"],
            border_width=1, border_color=COLORS["border"],
        )
        panel.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)
        box = ctk.CTkTextbox(
            panel, fg_color=COLORS["card_alt"], border_width=0, corner_radius=10,
            text_color=COLORS["text"], font=ctk.CTkFont(family="Consolas", size=12),
        )
        box.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        box.insert("1.0", format_diagnostics_report(self._diag_report))
        box.configure(state="disabled")

    def _maybe_start_onboarding(self):
        try:
            decision = startup_configuration_decision()
            required = bool(self._startup_onboarding_required or decision.required)
            if required:
                mode = getattr(self, "_startup_configuration_mode", "") or decision.mode
                reasons = getattr(self, "_startup_repair_reasons", ()) or decision.reasons
                self._open_onboarding(required=True, startup_mode=mode, repair_reasons=reasons)
        except Exception as exc:
            # Si el primer inicio falla, hacer visible la raíz únicamente para
            # poder mostrar el error y permitir un reintento; nunca continuar
            # silenciosamente sin onboarding.
            try:
                self.deiconify()
                self._maximize_main_window()
            except Exception:
                pass
            mb.showerror("Configuración inicial", f"No se pudo iniciar la configuración obligatoria:\n\n{exc}", parent=self)
            self.after(800, lambda: self._open_onboarding(required=True, startup_mode="full"))

    def _open_onboarding(self, required=None, startup_mode=None, repair_reasons=()):
        try:
            if required is None or startup_mode is None:
                decision = startup_configuration_decision()
                if required is None:
                    required = decision.required
                if startup_mode is None:
                    startup_mode = decision.mode if decision.required else "manual"
                if not repair_reasons:
                    repair_reasons = decision.reasons
            OnboardingWizard(
                self,
                on_complete=self._on_onboarding_complete,
                required=bool(required),
                startup_mode=str(startup_mode or "manual"),
                repair_reasons=tuple(repair_reasons or ()),
            )
        except Exception as exc:
            mb.showerror("Configuración inicial", str(exc), parent=self)

    def _on_onboarding_complete(self):
        # Primer inicio: el onboarding ya se destruyó. Recién ahora aparece la
        # aplicación principal y lo hace maximizada. En aperturas manuales del
        # asistente, la ventana principal ya estaba visible y simplemente se
        # refresca.
        was_startup = bool(getattr(self, "_startup_onboarding_required", False))
        self._startup_onboarding_required = False
        self._startup_configuration_mode = "none"
        self._startup_repair_reasons = ()
        if was_startup:
            try:
                self._startup_maximize_after_map_done = False
                self.deiconify()
                self.update_idletasks()
                self._maximize_main_window()
                # El <Map> posterior al deiconify realizará el único reintento
                # diferido si Windows todavía no había aplicado el estado zoomed.
                self.lift()
                self.focus_force()
            except Exception:
                pass
        self._load_diagnostics_async()
        self._refresh_privacy_card()
        self._refresh_text_model_label()

    def _open_chaos_reigns(self):
        try:
            webbrowser.open_new_tab("https://github.com/jmvelezo")
        except Exception as exc:
            mb.showerror("Chaos Reigns", f"No se pudo abrir el enlace.\n\n{exc}", parent=self)

    def _open_models(self):
        try:
            open_model_manager(self, self.ollama_url, str(self.output_dir))
        except Exception as exc:
            mb.showerror("Modelos locales", str(exc), parent=self)

    def _open_interview_text_editor(self):
        try:
            open_interview_text_editor(self)
        except Exception as exc:
            mb.showerror("Editor de entrevistas", str(exc), parent=self)

    def _open_result_in_interview_editor(self, transcript_path: Path):
        transcript_path = Path(transcript_path)
        selected_files = None
        cfg = self._current_cfg
        if cfg is not None:
            selected_files = list(getattr(cfg, "selected_files", None) or [])
        audio_path, manifest_path = _resolve_audio_for_result(
            transcript_path,
            input_path=self.input_path,
            selected_files=selected_files,
        )
        try:
            editor = open_interview_text_editor(
                self,
                transcript_path=transcript_path,
                audio_path=audio_path,
            )
        except Exception as exc:
            mb.showerror("Editor de entrevistas", str(exc), parent=self)
            return
        if audio_path is None:
            detail = (
                "No se pudo verificar automáticamente el audio original de este resultado. "
                "La transcripción se abrió sin audio; puedes seleccionarlo desde el editor."
            )
            if manifest_path is None:
                detail += " No se encontró un manifiesto inequívoco para este archivo."
            mb.showwarning("Audio no asociado", detail, parent=editor)
            try:
                editor.after(60, editor._bring_editor_to_front)
            except Exception:
                pass

    def _open_existing_review(self):
        audio = fd.askopenfilename(title="Seleccionar audio", filetypes=[("Audio/video", "*.wav *.mp3 *.m4a *.mp4 *.mov *.mkv *.ogg *.flac"), ("Todos", "*.*")])
        if not audio:
            return
        txt = fd.askopenfilename(title="Seleccionar transcripción", filetypes=[("Texto", "*.txt"), ("Todos", "*.*")])
        if not txt:
            return
        try:
            open_transcript_editor(self, audio_path=audio, transcript_path=txt, cfg=None)
        except Exception as exc:
            mb.showerror("Editor", str(exc), parent=self)

    def _open_last_review(self):
        if not self._last_audio or not self._last_transcript:
            return
        try:
            open_transcript_editor(self, audio_path=str(self._last_audio), transcript_path=str(self._last_transcript), cfg=self._current_cfg)
        except Exception as exc:
            mb.showerror("Editor", str(exc), parent=self)

    def _reset_for_new(self):
        self.input_path = None
        self.lbl_drop.configure(text="Arrastra un audio o video aquí\no selecciónalo desde el equipo", text_color=COLORS["muted"])
        self._new_outputs = []
        self._last_status = ""
        self._show_page("new")

def _portable_self_test() -> int:
    import json
    from core_transcriber import APP_VERSION, check_environment_status
    manager_ok = False
    manager_error = ""
    try:
        from model_manager_ui import ModelManagerWindow as _ModelManagerWindow
        manager_ok = _ModelManagerWindow is not None
    except Exception as exc:
        manager_error = f"{type(exc).__name__}: {exc}"
    payload = {
        "app_version": APP_VERSION,
        "portable": portable_status(),
        "environment": check_environment_status(""),
        "diarization_worker": probe_diarization_worker(timeout=45),
        "model_manager_ui": {"ok": manager_ok, "error": manager_error},
    }
    # Nunca serializar secretos; check_environment_status sólo expone presencia/origen.
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    ffmpeg_ok = bool(payload["environment"].get("ffmpeg"))
    worker_cfg = bool(payload["portable"].get("worker_present"))
    worker_ok = bool(payload["diarization_worker"].get("ok")) if worker_cfg else True
    return 0 if ffmpeg_ok and worker_ok and manager_ok else 2


if __name__ == "__main__":
    if "--self-test-json" in sys.argv:
        raise SystemExit(_portable_self_test())
    app = JeronimoApp()
    app.mainloop()
