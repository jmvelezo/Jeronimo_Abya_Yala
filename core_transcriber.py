from __future__ import annotations
import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Callable, Dict, List, Optional
from dotenv import load_dotenv
from credential_store import resolve_secret, credential_source
from subprocess_utils import hidden_process_kwargs
from job_engine import (
    JOB_STATUS_CANCELLED, JOB_STATUS_COMPLETED, JOB_STATUS_ERROR, JOB_STATUS_RUNNING,
    STAGE_CANCELLED, STAGE_CLEANING, STAGE_COMPLETED, STAGE_DIARIZING, STAGE_ERROR,
    STAGE_EXPORTING, STAGE_FINALIZING, STAGE_PREPARING, STAGE_SUMMARIZING, STAGE_TRANSCRIBING,
)
from runtime_isolation import DiarizationWorkerClient
from text_providers import (
    TEXT_ENGINE_NONE as TP_TEXT_ENGINE_NONE,
    TEXT_ENGINE_OPENAI as TP_TEXT_ENGINE_OPENAI,
    TEXT_ENGINE_OLLAMA as TP_TEXT_ENGINE_OLLAMA,
    TEXT_ENGINE_OPENAI_COMPATIBLE as TP_TEXT_ENGINE_OPENAI_COMPATIBLE,
    DEFAULT_OPENAI_COMPATIBLE_URL,
    DEFAULT_OLLAMA_URL as PROVIDER_DEFAULT_OLLAMA_URL,
    TextProviderSettings,
    create_provider,
    describe_endpoint,
    endpoint_scope,
    get_ollama_models as _provider_get_ollama_models,
    normalize_engine as _provider_normalize_engine,
    process_text as _provider_process_text,
    split_text_for_llm as _provider_split_text_for_llm,
)
try:
    from openai import OpenAI  # type: ignore
    HAS_OPENAI = True
except Exception:  # permite usar el modo local sin tener openai instalado
    OpenAI = None  # type: ignore
    HAS_OPENAI = False
import contextlib
import wave
import json
import re
import urllib.request
import urllib.error
import tempfile
import hashlib
import time

# ---------------------------------------------------------------------------
# La compatibilidad PyTorch 2.6+ para checkpoints de WhisperX/PyAnnote
# se aplica de forma temporal dentro de diarize_and_transcribe_local().
# No se parchea torch.load globalmente al importar este módulo.
# ---------------------------------------------------------------------------

# diarización local
# WhisperX 3.8.x + pyannote 4.x pueden emitir un warning de TorchCodec al
# importarse en Windows si FFmpeg no incluye DLLs compartidas. Jerónimo
# carga el audio con whisperx.load_audio y entrega la forma de onda en memoria
# al pipeline de diarización, por lo que ese decoder de archivo no se usa en
# nuestro flujo. Se silencia únicamente ese warning específico.
import warnings
try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torchcodec is not installed correctly.*",
            category=UserWarning,
        )
        import whisperx
    HAS_WHISPERX = True
except Exception:
    whisperx = None  # type: ignore
    HAS_WHISPERX = False

try:
    import inspect
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torchcodec is not installed correctly.*",
            category=UserWarning,
        )
        from whisperx.diarize import DiarizationPipeline as _WhisperXDiarizationPipeline

    def DiarizationPipeline(*args, **kwargs):
        """Adaptador compatible con WhisperX viejo/3.8.x y Community-1 local."""
        local_model = str(os.environ.get("JERONIMO_PYANNOTE_MODEL_DIR", "") or "").strip()
        if local_model and Path(local_model, "config.yaml").is_file():
            # Cuando Jerónimo instaló el release verificado local, pyannote acepta
            # directamente la carpeta que contiene config.yaml. No se necesita
            # token ni red durante el uso.
            kwargs["model_name"] = local_model
            kwargs.pop("use_auth_token", None)
            kwargs.pop("token", None)
        elif "use_auth_token" in kwargs:
            try:
                params = inspect.signature(_WhisperXDiarizationPipeline.__init__).parameters
            except Exception:
                params = {}
            if "token" in params and "use_auth_token" not in params:
                kwargs["token"] = kwargs.pop("use_auth_token")
        return _WhisperXDiarizationPipeline(*args, **kwargs)
except ImportError:
    _WhisperXDiarizationPipeline = None  # type: ignore
    DiarizationPipeline = None  # mas abajo pa


ProgressFn = Callable[[str], None]
ProgressBarFn = Callable[[float], None]
MAX_SEGMENT_SECONDS = 600  # 20min tiempo recorte?

STT_ENGINE_OPENAI = "openai"
STT_ENGINE_LOCAL_FAST_WHISPER = "local_faster_whisper"
LOCAL_FAST_WHISPER_DEFAULT_MODEL = "large-v3"
LOCAL_FAST_WHISPER_MODELS = ("large-v3", "large-v3-turbo", "medium", "small")
LOCAL_FAST_WHISPER_DEVICES = ("auto", "cuda", "cpu")

TEXT_ENGINE_NONE = "none"
TEXT_ENGINE_OPENAI = "openai"
TEXT_ENGINE_OLLAMA = "ollama"
TEXT_ENGINE_OPENAI_COMPATIBLE = "openai_compatible"
TEXT_ENGINE_DEFAULT = TEXT_ENGINE_OPENAI
DEFAULT_OLLAMA_URL = PROVIDER_DEFAULT_OLLAMA_URL
DEFAULT_OLLAMA_MODEL = ""
OLLAMA_CHUNK_CHARS = 12000

APP_NAME = "Jeronimo Abya Yala Transcriptor"
APP_VERSION = "1.0.0-prelocal-0.9.3"



@contextlib.contextmanager
def _temporary_torch_full_load_for_diarization(torch_module, progress: Optional[ProgressFn] = None):
    """
    Compatibilidad acotada para WhisperX/PyAnnote con PyTorch 2.6+.

    PyTorch 2.6 cambió torch.load para priorizar weights_only=True. Algunos
    checkpoints de pyannote/whisperx pueden requerir carga completa. En vez de
    parchear torch.load globalmente al importar la app, este helper modifica
    torch.load solo mientras se ejecuta la diarización local.

    Usar únicamente con modelos descargados desde fuentes confiables
    (Hugging Face/pyannote/whisperx aceptados explícitamente por el usuario).
    """
    if torch_module is None:
        yield
        return

    orig_load = getattr(torch_module, "load", None)
    serialization = getattr(torch_module, "serialization", None)
    orig_serialization_load = getattr(serialization, "load", None) if serialization is not None else None

    if orig_load is None:
        yield
        return

    def _patched_load(*args, **kwargs):
        # Solo completamos el argumento cuando la librería no lo envía.
        kwargs.setdefault("weights_only", False)
        try:
            return orig_load(*args, **kwargs)
        except TypeError as exc:
            # Versiones antiguas de torch no aceptan weights_only.
            if "weights_only" in str(exc):
                kwargs.pop("weights_only", None)
                return orig_load(*args, **kwargs)
            raise

    try:
        torch_module.load = _patched_load  # type: ignore[attr-defined]
        if serialization is not None and orig_serialization_load is not None:
            serialization.load = _patched_load  # type: ignore[attr-defined]
        logger.info("Compatibilidad torch.load aplicada temporalmente solo para diarización local.")
        if progress:
            progress("Compatibilidad PyTorch aplicada solo durante la diarización local.")
        yield
    finally:
        try:
            torch_module.load = orig_load  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if serialization is not None and orig_serialization_load is not None:
                serialization.load = orig_serialization_load  # type: ignore[attr-defined]
        except Exception:
            pass

LOG_DIR_ENV = "JERONIMO_TRANSCRIPTOR_LOG_DIR"
APP_LOG_DIR = Path.home() / ".jeronimo_abya_yala_transcriptor" / "logs"
LOG_FILE = "transcribir_stt.log"


def _resolve_log_path() -> Path:
    """Devuelve una ruta local para logs fuera del paquete distribuible."""
    raw_dir = os.getenv(LOG_DIR_ENV, "").strip()
    log_dir = Path(raw_dir).expanduser() if raw_dir else APP_LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / LOG_FILE
    except Exception:
        # Fallback defensivo: no bloquear la app por problemas de permisos.
        return Path.cwd() / LOG_FILE


def _configure_logger() -> logging.Logger:
    logger_obj = logging.getLogger("stt_transcriber")
    logger_obj.setLevel(logging.INFO)

    if logger_obj.handlers:
        return logger_obj

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    try:
        file_handler = RotatingFileHandler(
            _resolve_log_path(),
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        logger_obj.addHandler(file_handler)
    except Exception:
        # Si no se puede escribir log a archivo, se mantiene solo consola.
        pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger_obj.addHandler(stream_handler)
    logger_obj.propagate = False
    return logger_obj


logger = _configure_logger()


@dataclass
class TranscriberConfig:
    api_key: str
    input_dir: Path          # carpeta base (por si se usa modo carpeta)
    output_dir: Path
    work_dir: Path

    #lista opcional de archivos concretos a procesar
    selected_files: list[Path] | None = None

    stt_engine: str = STT_ENGINE_OPENAI
    stt_model: str = "gpt-4o-mini-transcribe"
    local_model_name: str = LOCAL_FAST_WHISPER_DEFAULT_MODEL
    local_device: str = "auto"
    chat_model: str = "gpt-4.1-mini"
    text_engine: str = TEXT_ENGINE_DEFAULT
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    compatible_api_url: str = DEFAULT_OPENAI_COMPATIBLE_URL
    compatible_api_key: str = ""
    text_processing_error: str = ""
    cancel_check: Optional[Callable[[], bool]] = None
    job_id: str = ""
    job_event_callback: Optional[Callable[[dict], None]] = None
    diarization_runtime_python: str = ""
    diarization_worker_executable: str = ""

    language: str = "es"
    use_ffmpeg: bool = True
    audio_enhancement: bool = True

    whisper_prompt: str = ""
    clean_prompt: str = ""
    summary_prompt: str = ""

    do_clean: bool = True
    do_summary: bool = True
    #diarización local opcional con WhisperX
    use_diarization: bool = False
    whisperx_model: str = "large-v2"      # modelo por defecto de WhisperX
    diarization_device: str = "cuda"      # "cuda" o "cpu"

    # Cantidad esperada de hablantes para diarización (0 = Auto)
    target_speakers: int = 0

    #nombres para etiquetar hablantes cuando haya diarización
    interviewer_name: str = ""
    interviewee_name: str = ""


    # Metadatos para cabecera (ATLAS/Word)
    project: str = ""
    place: str = ""
    date_label: str = ""
    duration_label: str = ""
    extra_context: str = ""

    # Exportaciones adicionales
    export_docx_raw: bool = True
    export_docx_atlas: bool = True
    export_rtf_raw: bool = True
    export_rtf_atlas: bool = True

    # --- Fase 3: opciones de formato ATLAS.ti ---
    atlas_include_turn_numbers: bool = False
    atlas_include_timecodes: bool = False
    atlas_include_speaker_id: bool = False
    atlas_blank_line_between_turns: bool = True
    atlas_bold_speaker_prefix: bool = False
class ProcessingCancelled(RuntimeError):
    """Cancelación cooperativa solicitada por la interfaz."""


def _check_cancelled(cfg: TranscriberConfig) -> None:
    checker = getattr(cfg, "cancel_check", None)
    if checker is not None:
        try:
            if checker():
                raise ProcessingCancelled("Proceso cancelado por el usuario.")
        except ProcessingCancelled:
            raise
        except Exception:
            # Un callback de cancelación defectuoso no debe romper una transcripción.
            return


def _emit_job_event(
    cfg: TranscriberConfig,
    *,
    stage: str,
    status: str = JOB_STATUS_RUNNING,
    progress: Optional[float] = None,
    message: str = "",
    current_item: str = "",
) -> None:
    """Emite estado efímero hacia la GUI sin persistir contenido ni rutas completas."""
    callback = getattr(cfg, "job_event_callback", None)
    if callback is None:
        return
    payload = {
        "job_id": str(getattr(cfg, "job_id", "") or ""),
        "stage": stage,
        "status": status,
        "progress": None if progress is None else max(0.0, min(1.0, float(progress))),
        "message": str(message or ""),
        "current_item": str(current_item or ""),
    }
    try:
        callback(payload)
    except Exception:
        logger.debug("Callback de Job falló; se continúa sin afectar el procesamiento.", exc_info=True)


def _diarize_via_optional_worker(
    audio_path: Path,
    cfg: TranscriberConfig,
    progress: ProgressFn,
) -> str:
    python_executable = str(getattr(cfg, "diarization_runtime_python", "") or "").strip()
    worker_executable = str(getattr(cfg, "diarization_worker_executable", "") or "").strip()
    if not python_executable and not worker_executable:
        return diarize_and_transcribe_local(audio_path, cfg, progress)

    worker_script = Path(
        os.environ.get("JERONIMO_DIARIZATION_WORKER_SCRIPT", "")
        or (Path(__file__).resolve().parent / "workers" / "diarization_worker.py")
    )
    progress("Diarización aislada activada: WhisperX se ejecutará en un runtime separado.")
    if worker_executable:
        client = DiarizationWorkerClient(worker_executable=worker_executable)
    else:
        client = DiarizationWorkerClient(python_executable, worker_script)
    payload = {
        "op": "diarize",
        "audio_path": str(audio_path),
        "config": {
            "output_dir": str(cfg.output_dir),
            "work_dir": str(cfg.work_dir),
            "language": cfg.language,
            "whisperx_model": cfg.whisperx_model,
            "diarization_device": cfg.diarization_device,
            "target_speakers": cfg.target_speakers,
            "interviewer_name": cfg.interviewer_name,
            "interviewee_name": cfg.interviewee_name,
            "whisper_prompt": cfg.whisper_prompt,
            "atlas_include_turn_numbers": cfg.atlas_include_turn_numbers,
            "atlas_include_timecodes": cfg.atlas_include_timecodes,
            "atlas_include_speaker_id": cfg.atlas_include_speaker_id,
            "atlas_blank_line_between_turns": cfg.atlas_blank_line_between_turns,
        },
    }

    def _worker_event(event: dict) -> None:
        msg = str(event.get("message") or "").strip()
        if msg:
            progress(msg)

    try:
        result = client.run(
            payload,
            on_event=_worker_event,
            cancel_check=getattr(cfg, "cancel_check", None),
        )
    except InterruptedError as exc:
        raise ProcessingCancelled(str(exc)) from exc
    return str(result.get("text") or "")


def _privacy_file_ref(path: Path) -> str:
    """Referencia pseudónima para logs persistentes: no guarda el nombre real."""
    raw = str(Path(path).name).encode("utf-8", errors="replace")
    digest = hashlib.sha256(raw).hexdigest()[:10]
    return f"{digest}{Path(path).suffix.lower()}"


def _sanitize_error_message(message: object, cfg: Optional[TranscriberConfig] = None) -> str:
    """Evita que una excepción termine persistiendo claves conocidas en logs/manifiestos."""
    text = str(message or "")
    secrets: list[str] = []
    if cfg is not None:
        api_key = str(getattr(cfg, "api_key", "") or "").strip()
        if api_key:
            secrets.append(api_key)
        compatible_key = str(getattr(cfg, "compatible_api_key", "") or "").strip()
        if compatible_key:
            secrets.append(compatible_key)
    for provider in ("openai", "huggingface", "openai_compatible"):
        try:
            value = resolve_secret(provider)
        except Exception:
            value = ""
        if value:
            secrets.append(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:1000]


def load_api_key_from_env() -> Optional[str]:
    # Compatibilidad: el nombre se conserva porque la GUI existente lo usa.
    # Desde FASE 2 también puede resolver el almacén seguro del sistema.
    load_dotenv()
    return resolve_secret("openai") or None


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            **hidden_process_kwargs(),
        )
        return True
    except Exception:
        return False


def check_environment_status(api_key: str = "") -> Dict[str, object]:
    """Chequeo liviano del entorno. No bloquea la app; solo informa estado."""
    load_dotenv()
    resolved_openai = resolve_secret("openai", api_key)
    resolved_hf = resolve_secret("huggingface")
    status: Dict[str, object] = {
        "ffmpeg": check_ffmpeg(),
        "api_key": bool(resolved_openai),
        "api_key_source": credential_source("openai", api_key),
        "openai_pkg": HAS_OPENAI,
        "whisperx": HAS_WHISPERX,
        "diarization_pipeline": DiarizationPipeline is not None,
        "huggingface_token": bool(resolved_hf),
        "huggingface_token_source": credential_source("huggingface"),
        "pyannote_local_model": bool(
            str(os.environ.get("JERONIMO_PYANNOTE_MODEL_DIR", "") or "").strip()
            and Path(str(os.environ.get("JERONIMO_PYANNOTE_MODEL_DIR", "")), "config.yaml").is_file()
        ),
        "faster_whisper": False,
        "ollama": False,
        "torch": False,
        "cuda": False,
        "log_file": str(_resolve_log_path()),
    }

    try:
        import faster_whisper  # type: ignore  # noqa: F401
        status["faster_whisper"] = True
    except Exception:
        status["faster_whisper"] = False


    try:
        url = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL).strip() or DEFAULT_OLLAMA_URL
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1.5):
            status["ollama"] = True
    except Exception:
        status["ollama"] = False

    try:
        import torch  # type: ignore
        status["torch"] = True
        status["cuda"] = bool(torch.cuda.is_available())
        status["torch_version"] = getattr(torch, "__version__", "desconocida")
    except Exception:
        status["torch_version"] = "no disponible"

    status["ffmpeg_path"] = shutil.which("ffmpeg") or ""
    return status


def format_environment_status(status: Dict[str, object]) -> str:
    """Texto breve para mostrar en la GUI/log sin filtrar secretos."""
    yes = "OK"
    no = "NO"
    lines = [
        "Chequeo de entorno:",
        f"- ffmpeg: {yes if status.get('ffmpeg') else no}",
        f"- API key disponible: {yes if status.get('api_key') else no} ({status.get('api_key_source', 'missing')})",
        f"- paquete openai instalado: {yes if status.get('openai_pkg') else no}",
        f"- faster-whisper instalado: {yes if status.get('faster_whisper') else no}",
        f"- Ollama local responde: {yes if status.get('ollama') else no}",
        f"- whisperx instalado: {yes if status.get('whisperx') else no}",
        f"- pyannote/diarización disponible: {yes if status.get('diarization_pipeline') else no}",
        f"- Community-1 local: {yes if status.get('pyannote_local_model') else no}",
        f"- token Hugging Face de contingencia: {yes if status.get('huggingface_token') else no} ({status.get('huggingface_token_source', 'missing')})",
        f"- torch: {yes if status.get('torch') else no} ({status.get('torch_version', 'desconocida')})",
        f"- CUDA: {yes if status.get('cuda') else no}",
        f"- log local: {status.get('log_file', '')}",
    ]
    return "\n".join(lines)




def _normalize_text_engine(value: str | None) -> str:
    return _provider_normalize_engine(value)


def _text_processing_requested(cfg: TranscriberConfig) -> bool:
    return bool(
        getattr(cfg, "do_clean", False) and str(getattr(cfg, "clean_prompt", "") or "").strip()
        or getattr(cfg, "do_summary", False) and str(getattr(cfg, "summary_prompt", "") or "").strip()
    )


def _text_processing_uses_openai(cfg: TranscriberConfig) -> bool:
    return _text_processing_requested(cfg) and _normalize_text_engine(getattr(cfg, "text_engine", TEXT_ENGINE_DEFAULT)) == TEXT_ENGINE_OPENAI


def _text_processing_uses_ollama(cfg: TranscriberConfig) -> bool:
    return _text_processing_requested(cfg) and _normalize_text_engine(getattr(cfg, "text_engine", TEXT_ENGINE_DEFAULT)) == TEXT_ENGINE_OLLAMA


def _text_processing_uses_compatible_api(cfg: TranscriberConfig) -> bool:
    return _text_processing_requested(cfg) and _normalize_text_engine(getattr(cfg, "text_engine", TEXT_ENGINE_DEFAULT)) == TEXT_ENGINE_OPENAI_COMPATIBLE


def _text_provider_settings(cfg: TranscriberConfig) -> TextProviderSettings:
    engine = _normalize_text_engine(getattr(cfg, "text_engine", TEXT_ENGINE_DEFAULT))
    api_key = ""
    base_url = ""
    if engine == TEXT_ENGINE_OPENAI:
        api_key = str(getattr(cfg, "api_key", "") or "").strip()
    elif engine == TEXT_ENGINE_OPENAI_COMPATIBLE:
        api_key = resolve_secret("openai_compatible", str(getattr(cfg, "compatible_api_key", "") or ""))
        base_url = str(getattr(cfg, "compatible_api_url", "") or DEFAULT_OPENAI_COMPATIBLE_URL).strip()
    return TextProviderSettings(
        engine=engine,
        model=str(getattr(cfg, "chat_model", "") or "").strip(),
        api_key=api_key,
        base_url=base_url,
        ollama_url=str(getattr(cfg, "ollama_url", "") or DEFAULT_OLLAMA_URL).strip(),
        ollama_model=str(getattr(cfg, "ollama_model", "") or "").strip(),
        chunk_chars=OLLAMA_CHUNK_CHARS,
    )


def _manifest_privacy_mode(cfg: TranscriberConfig) -> str:
    """Clasifica el procesamiento sin exponer contenido ni secretos."""
    stt_engine = (getattr(cfg, "stt_engine", STT_ENGINE_OPENAI) or STT_ENGINE_OPENAI).strip()
    uses_api_stt = (not getattr(cfg, "use_diarization", False)) and stt_engine == STT_ENGINE_OPENAI
    if uses_api_stt:
        return "api"

    if not _text_processing_requested(cfg):
        return "local"

    info = describe_endpoint(_text_provider_settings(cfg))
    return "hybrid" if info.sends_text_off_device else "local"


def write_manifest(
    cfg: TranscriberConfig,
    *,
    input_file: Path,
    output_base_name: str,
    generated_files: list[Path | str],
    use_ffmpeg: bool,
    status: str = "success",
    error_message: str = "",
) -> Path:
    """Genera un manifiesto técnico/ético sin claves ni contenido de entrevistas."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.output_dir / f"{output_base_name}.manifest.json"

    clean_generated: list[str] = []
    for item in generated_files or []:
        try:
            clean_generated.append(Path(item).name)
        except Exception:
            clean_generated.append(str(item))

    stt_engine = (getattr(cfg, "stt_engine", STT_ENGINE_OPENAI) or STT_ENGINE_OPENAI).strip()
    uses_openai_stt = (not getattr(cfg, "use_diarization", False)) and stt_engine == STT_ENGINE_OPENAI
    text_engine = _normalize_text_engine(getattr(cfg, "text_engine", TEXT_ENGINE_DEFAULT))
    uses_openai_chat = _text_processing_uses_openai(cfg)
    uses_ollama_chat = _text_processing_uses_ollama(cfg)

    if bool(getattr(cfg, "use_diarization", False)):
        manifest_stt_model = getattr(cfg, "whisperx_model", "")
    elif uses_openai_stt:
        manifest_stt_model = getattr(cfg, "stt_model", "")
    elif stt_engine == STT_ENGINE_LOCAL_FAST_WHISPER:
        manifest_stt_model = getattr(cfg, "local_model_name", "")
    else:
        manifest_stt_model = getattr(cfg, "stt_model", "")

    data: dict[str, object] = {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "input_file_name": input_file.name,
        "input_file_suffix": input_file.suffix.lower(),
        "output_base_name": output_base_name,
        "language": getattr(cfg, "language", ""),
        "stt_engine": stt_engine,
        "stt_model": manifest_stt_model,
        "local_model_name": getattr(cfg, "local_model_name", "") if stt_engine == STT_ENGINE_LOCAL_FAST_WHISPER else "",
        "openai_model": getattr(cfg, "stt_model", "") if uses_openai_stt else "",
        "whisperx_model": getattr(cfg, "whisperx_model", "") if bool(getattr(cfg, "use_diarization", False)) else "",
        "chat_model": getattr(cfg, "chat_model", "") if uses_openai_chat else "",
        "text_model": (getattr(cfg, "ollama_model", "") if uses_ollama_chat else getattr(cfg, "chat_model", "")) if _text_processing_requested(cfg) else "",
        "text_engine": text_engine,
        "ollama_url": getattr(cfg, "ollama_url", DEFAULT_OLLAMA_URL) if uses_ollama_chat else "",
        "ollama_model": getattr(cfg, "ollama_model", "") if uses_ollama_chat else "",
        "compatible_api_url": getattr(cfg, "compatible_api_url", "") if _text_processing_uses_compatible_api(cfg) else "",
        "text_endpoint_scope": describe_endpoint(_text_provider_settings(cfg)).scope if _text_processing_requested(cfg) else "disabled",
        "text_sends_data_off_device": describe_endpoint(_text_provider_settings(cfg)).sends_text_off_device if _text_processing_requested(cfg) else False,
        "text_processing_error": str(getattr(cfg, "text_processing_error", "") or "")[:500],
        "use_ffmpeg": bool(use_ffmpeg),
        "ffmpeg_requested": bool(getattr(cfg, "use_ffmpeg", False)),
        "audio_enhancement_requested": bool(getattr(cfg, "audio_enhancement", True)),
        "temporary_files_policy": "automatic_cleanup",
        "do_clean": bool(getattr(cfg, "do_clean", False)),
        "do_summary": bool(getattr(cfg, "do_summary", False)),
        "use_diarization": bool(getattr(cfg, "use_diarization", False)),
        "target_speakers": int(getattr(cfg, "target_speakers", 0) or 0),
        "export_docx_raw": bool(getattr(cfg, "export_docx_raw", False)),
        "export_docx_atlas": bool(getattr(cfg, "export_docx_atlas", False)),
        "export_rtf_raw": bool(getattr(cfg, "export_rtf_raw", False)),
        "export_rtf_atlas": bool(getattr(cfg, "export_rtf_atlas", False)),
        "generated_files": clean_generated,
        "privacy_mode": _manifest_privacy_mode(cfg),
        "ethical_note": "Este manifiesto registra el procesamiento técnico. No reemplaza consentimiento informado ni revisión humana.",
    }

    if error_message:
        data["error_message"] = _sanitize_error_message(error_message, cfg)[:500]

    manifest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path

def _check_directory_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fd = None
    probe = None
    try:
        fd, name = tempfile.mkstemp(prefix=".jeronimo_write_test_", dir=str(path))
        probe = Path(name)
    finally:
        if fd is not None:
            os.close(fd)
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass


def _storage_preflight(cfg: TranscriberConfig, progress: ProgressFn) -> None:
    """Valida escritura y espacio mínimo sin inspeccionar contenido de entrevistas."""
    for label, path in (("salida", cfg.output_dir), ("temporales", cfg.work_dir)):
        try:
            _check_directory_writable(path)
        except Exception as exc:
            raise RuntimeError(f"La carpeta de {label} no es escribible: {path}. Detalle: {exc}") from exc

        try:
            free = shutil.disk_usage(path).free
        except Exception:
            continue
        free_mb = free / (1024 * 1024)
        if free_mb < 64:
            raise RuntimeError(
                f"Hay menos de 64 MB libres en la carpeta de {label}. "
                "Jerónimo detiene el proceso para evitar archivos incompletos."
            )
        if free_mb < 512:
            progress(f"AVISO: quedan sólo {free_mb:.0f} MB libres en la carpeta de {label}.")


def _cleanup_stale_workdirs(work_dir: Path, *, older_than_hours: float = 24.0) -> int:
    """Elimina sólo temporales propios antiguos, nunca archivos arbitrarios del usuario."""
    if not work_dir.exists():
        return 0
    threshold = time.time() - max(1.0, float(older_than_hours)) * 3600.0
    removed = 0
    for child in work_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(".jeronimo_tmp_"):
            continue
        try:
            if child.stat().st_mtime >= threshold:
                continue
            shutil.rmtree(child)
            removed += 1
        except Exception:
            continue
    return removed


def get_wav_duration_seconds(path: Path) -> float:
    """Devuelve duración del wav en segundos (usa wave, sin dependencias raras)."""
    with contextlib.closing(wave.open(str(path), "rb")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        if rate == 0:
            return 0.0
        return frames / float(rate)


def segment_wav_if_needed(
    wav_path: Path,
    base_name: str,
    work_dir: Path,
    progress: Callable[[str], None],
) -> list[Path]:
    """
    Si el wav dura más que MAX_SEGMENT_SECONDS, lo corta en segmentos.
    Devuelve una lista de paths de segmentos (o el original si es corto).
    """
    if wav_path.suffix.lower() != ".wav":
        progress(
            f"El archivo activo ({wav_path.suffix.lower() or 'sin extensión'}) no es WAV; "
            "se omite la segmentación WAV y se transcribe el archivo original."
        )
        return [wav_path]

    try:
        duration = get_wav_duration_seconds(wav_path)
    except (wave.Error, OSError, EOFError) as exc:
        progress(f"AVISO: no se pudo leer el WAV para segmentar ({exc}); se usará el archivo completo.")
        return [wav_path]
    if duration <= 0:
        progress(f"AVISO: no se pudo leer duración de {wav_path.name}, se usa tal cual.")
        return [wav_path]

    if duration <= MAX_SEGMENT_SECONDS:
        progress(f"Duración {duration:.1f} s (< {MAX_SEGMENT_SECONDS}s), se transcribe en un solo bloque.")
        return [wav_path]

    progress(
        f"Duración {duration/60:.1f} min, se dividirá en segmentos de "
        f"{MAX_SEGMENT_SECONDS/60:.1f} min para la transcripción."
    )

    segment_pattern = work_dir / f"{base_name}_seg_%03d.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-f", "segment",
        "-segment_time", str(MAX_SEGMENT_SECONDS),
        "-c", "copy",
        str(segment_pattern),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **hidden_process_kwargs())
    except subprocess.CalledProcessError as e:
        progress(f"ERROR al segmentar audio, se usará el archivo completo. Detalle: {e}")
        return [wav_path]

    segments = sorted(work_dir.glob(f"{base_name}_seg_*.wav"))
    if not segments:
        progress("No se generaron segmentos, se usará el archivo completo.")
        return [wav_path]

    progress(f"Se generaron {len(segments)} segmentos para {wav_path.name}.")
    return segments



def preprocess_audio(
    input_path: Path,
    output_path: Path,
    use_ffmpeg: bool,
    progress: Callable[[str], None],
    enhance_audio: bool = True,
) -> Path:
    """Prepara audio sin falsificar extensiones ni conservar copias innecesarias.

    Si ffmpeg no se solicita/no está disponible, devuelve el archivo original.
    Si ffmpeg falla, elimina cualquier salida parcial y vuelve al original.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not use_ffmpeg:
        progress("Preprocesamiento ffmpeg omitido: se usará el archivo original sin crear una copia temporal falsa.")
        return input_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
    ]
    if enhance_audio:
        cmd.extend(["-af", "loudnorm,afftdn=nf=-20"])
    cmd.append(str(output_path))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden_process_kwargs())
        progress("Audio preparado (mono, 16k, reducción de ruido + normalización)." if enhance_audio else "Audio convertido técnicamente (mono, 16k), sin reducción de ruido.")
        return output_path
    except (subprocess.CalledProcessError, OSError) as exc:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        progress(
            "AVISO: ffmpeg no pudo preprocesar este archivo. "
            "Se usará el archivo original sin renombrarlo ni copiarlo como WAV. "
            f"Detalle: {exc}"
        )
        logger.error("ffmpeg falló; se continúa con el original. tipo=%s", type(exc).__name__)
        return input_path


def transcribe_audio(
    client: OpenAI,
    cfg: TranscriberConfig,
    audio_path: Path,
    progress: Callable[[str], None]
) -> str:
    progress("Enviando a la API de transcripción...")
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model=cfg.stt_model,
            file=f,
            language=cfg.language,
            prompt=cfg.whisper_prompt or None,
            temperature=0,
            response_format="json"
        )
    text = resp.text
    progress("Transcripción recibida.")
    return text


def _fmt_seconds_brief(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _resolve_local_fast_whisper_device(cfg: TranscriberConfig, progress: Callable[[str], None]) -> tuple[str, str]:
    requested = (getattr(cfg, "local_device", "auto") or "auto").lower().strip()
    if requested not in LOCAL_FAST_WHISPER_DEVICES:
        requested = "auto"

    cuda_ok = False
    try:
        import torch  # type: ignore
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if requested == "cuda" and not cuda_ok:
        progress("AVISO: se pidió CUDA para faster-whisper, pero no está disponible. Se usará CPU (int8).")
        device = "cpu"
    elif requested == "cpu":
        device = "cpu"
    else:
        device = "cuda" if cuda_ok else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def transcribe_audio_local_faster_whisper(
    cfg: TranscriberConfig,
    audio_path: Path,
    progress: Callable[[str], None]
) -> str:
    """Transcripción local simple con faster-whisper.

    Mantiene el enfoque de entrevistas: no envía audio a API externa y no registra
    fragmentos completos de la entrevista en el log.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "El motor local requiere 'faster-whisper'. Instálalo con:\n"
            "    pip install -r requirements-local.txt\n"
            "o:\n"
            "    pip install faster-whisper"
        ) from e

    model_name = (getattr(cfg, "local_model_name", "") or LOCAL_FAST_WHISPER_DEFAULT_MODEL).strip()
    if not model_name:
        model_name = LOCAL_FAST_WHISPER_DEFAULT_MODEL

    device, compute_type = _resolve_local_fast_whisper_device(cfg, progress)
    progress(
        f"Cargando faster-whisper local: modelo={model_name}, "
        f"device={device}, compute_type={compute_type}..."
    )

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            progress(
                "AVISO: faster-whisper no pudo cargar en CUDA/float16. "
                "Reintentando en CPU (int8)."
            )
            logger.warning("faster-whisper CUDA falló; fallback a CPU: %s", e)
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        else:
            raise

    progress("Transcribiendo localmente con faster-whisper (el audio no sale de la máquina)...")

    segments, info = model.transcribe(
        str(audio_path),
        language=cfg.language or "es",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
            threshold=0.45,
        ),
        temperature=0.0,
        initial_prompt=(cfg.whisper_prompt or None),
        condition_on_previous_text=False,
        word_timestamps=False,
    )

    detected_lang = getattr(info, "language", None)
    if detected_lang:
        progress(f"Idioma detectado por faster-whisper: {detected_lang}")

    parts: list[str] = []
    seg_count = 0
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        seg_count += 1
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", start) or start)
        if seg_count == 1 or seg_count % 10 == 0:
            progress(
                f"Segmentos locales procesados: {seg_count} "
                f"(último: {_fmt_seconds_brief(start)}-{_fmt_seconds_brief(end)})"
            )
        parts.append(text)

    progress(f"Transcripción local finalizada: {seg_count} segmentos útiles.")
    return "\n".join(parts).strip()



def get_ollama_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    """Compatibilidad pública: delega el inventario de Ollama al módulo de proveedores."""
    return _provider_get_ollama_models(base_url)


def _split_text_for_llm(text: str, max_chars: int = OLLAMA_CHUNK_CHARS) -> list[str]:
    """Compatibilidad pública para herramientas/tests existentes."""
    return _provider_split_text_for_llm(text, max_chars)


def chat_process(
    client: Optional[OpenAI],
    cfg: TranscriberConfig,
    transcript: str,
    system_prompt: str,
    progress: Callable[[str], None],
    etapa: str
) -> str:
    # `client` se conserva en la firma para no romper callers anteriores. La capa
    # de texto ya crea/gestiona su propio proveedor; STT y metadatos siguen usando
    # el cliente OpenAI histórico de core_transcriber.
    del client
    settings = _text_provider_settings(cfg)
    if settings.engine == TEXT_ENGINE_NONE:
        progress(f"Etapa de {etapa} omitida: procesamiento de texto desactivado.")
        return transcript
    provider = create_provider(settings)
    if provider is None:
        return transcript
    return _provider_process_text(
        provider,
        transcript,
        system_prompt,
        progress,
        etapa,
        chunk_chars=settings.chunk_chars,
    )


def diarize_and_transcribe_local(
    audio_path: Path,
    cfg: TranscriberConfig,
    progress: ProgressFn,
) -> str:
    """
    Usa WhisperX para hacer transcripción + diarización local.

    - Intenta primero en CUDA (si está disponible).
    - Si falla por "no CUDA-capable device" o no hay CUDA, cae a CPU.
    - Ajusta compute_type según el dispositivo (float16 en GPU, int8 en CPU).
    - Informa en el log y en la barra de estado qué dispositivo usó.
    """
    if not HAS_WHISPERX:
        raise RuntimeError(
            "La diarización local está activada, pero el paquete 'whisperx' no está instalado.\n"
            "Instálalo en tu entorno local con:\n"
            "    pip install whisperx\n"
            "Si usas un entorno virtual, activalo antes de instalar. El programa conservará los modos sin diarización."
        )

    # Importamos torch solo dentro del flujo de diarización para chequear CUDA
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore

    # --- FIX PyTorch 2.6+ (weights_only=True) para modelos de pyannote/whisperx ---
    # --- FIX PyTorch 2.6+ (weights_only=True) para modelos de pyannote/whisperx ---
    def _allowlist_pyannote_omegaconf():
        """
        PyTorch 2.6+ carga checkpoints con weights_only=True por defecto.
        Los modelos de pyannote/whisperx suelen incluir objetos OmegaConf y typing,
        y torch los bloquea salvo allowlist explícita.
        """
        if torch is None:
            return
        try:
            import typing
            from omegaconf.listconfig import ListConfig
            from omegaconf.dictconfig import DictConfig

            safe = [ListConfig, DictConfig, typing.Any, list, dict, tuple, set]

            # Algunos checkpoints meten metadata de OmegaConf
            try:
                from omegaconf.base import ContainerMetadata
                safe.append(ContainerMetadata)
            except Exception:
                pass

            if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals(safe)

        except Exception:
            return


    language = cfg.language or "es"
    model_name = getattr(cfg, "whisperx_model", None) or "large-v2"

    # --- Selección inicial de dispositivo ---
    # Si hay CUDA disponible, intentamos 'cuda'; si no, vamos directo a 'cpu'.
    preferred = (getattr(cfg, "diarization_device", "") or "").lower().strip()

    cuda_ok = (
        torch is not None
        and getattr(torch, "cuda", None) is not None
        and torch.cuda.is_available()
    )

    if preferred == "cpu":
        device = "cpu"
    elif preferred == "cuda":
        if cuda_ok:
            device = "cuda"
        else:
            progress("AVISO: se seleccionó CUDA para diarización, pero no está disponible. Se usará CPU (más lento).")
            device = "cpu"
    else:
        # auto
        device = "cuda" if cuda_ok else "cpu"
        if device == "cpu":
            progress("Diarización: CUDA no disponible; usando CPU (int8).")

    # compute_type recomendado:
    # - GPU: float16 (rápido)
    # - CPU: int8 (más liviano, evita el error de float16 no soportado)
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "WhisperX: intentando cargar modelo '%s' en dispositivo '%s' (compute_type=%s)",
        model_name,
        device,
        compute_type,
    )
    progress(f"Intentando cargar WhisperX ({model_name}) en dispositivo '{device}'...")

    # --- 1) Cargar modelo ASR con helper y fallback ---
    # --- 1) Cargar modelo ASR con helper y fallback ---
    def _load_model_with(device_to_use: str, compute_type_to_use: str):
        # Arreglo mínimo: allowlist para que pyannote/whisperx no rompa con torch>=2.6
        _allowlist_pyannote_omegaconf()

        with _temporary_torch_full_load_for_diarization(torch, progress):
            return whisperx.load_model(
                model_name,
                device=device_to_use,
                language=language,
                compute_type=compute_type_to_use,
            )

    try:
        model = _load_model_with(device, compute_type)
    except (RuntimeError, ValueError) as e:
        msg = str(e)
        # Errores típicos: no CUDA o float16 no soportado
        if (
            "no CUDA-capable device is detected" in msg
            or "CUDA" in msg
            or "float16" in msg
        ):
            logger.warning(
                "WhisperX: error en dispositivo '%s' (error: %s). Probando en CPU con int8...",
                device,
                msg,
            )
            progress(
                "CUDA/float16 no disponibles; usando CPU (int8) para diarización. "
                "Esto puede ser más lento."
            )
            device = "cpu"
            compute_type = "int8"
            model = _load_model_with(device, compute_type)
        else:
            # otro error distinto -> lo propagamos
            raise

    logger.info(
        "WhisperX: modelo '%s' cargado en dispositivo '%s' (compute_type=%s)",
        model_name,
        device,
        compute_type,
    )
    progress(f"WhisperX cargado en dispositivo '{device}' (compute_type={compute_type}).")

    # --- 2) Transcripción base ---
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=language)
    segments = result.get("segments", [])
    if not segments:
        progress("AVISO: WhisperX no devolvió segmentos, se devuelve texto vacío.")
        return ""

    # --- 3) Alineación (opcional) ---
    try:
        with _temporary_torch_full_load_for_diarization(torch, progress):
            model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result_aligned = whisperx.align(segments, model_a, metadata, audio, device)
    except Exception as e:
        logger.warning("WhisperX: no se pudo cargar/alinear modelo de alineación: %s", e)
        progress("AVISO: no se pudo hacer alineación fina, se continúa igual.")
        result_aligned = {"segments": segments}

    # --- 4) Diarización (Community-1 local verificado; HF sólo como compatibilidad) ---
    load_dotenv()
    local_pyannote = str(os.environ.get("JERONIMO_PYANNOTE_MODEL_DIR", "") or "").strip()
    local_pyannote_ready = bool(local_pyannote and Path(local_pyannote, "config.yaml").is_file())
    hf_token = resolve_secret("huggingface")
    if not local_pyannote_ready and not hf_token:
        raise RuntimeError(
            "No está disponible el modelo local Community-1 ni una credencial de Hugging Face de contingencia.\n"
            "Abrí Configuración inicial para preparar el modelo de hablantes.\n"
            "Cuando Community-1 está instalado localmente, la diarización funciona sin token y sin conexión."
        )

    if DiarizationPipeline is None:
        raise RuntimeError(
            "La versión instalada de whisperx no expone DiarizationPipeline en whisperx.diarize.\n"
            "Verifica que tienes una versión reciente de whisperx y pyannote instaladas en el mismo entorno.\n"
            "Sugerencia: pip install -U whisperx"
        )

    progress("Cargando pipeline de diarización (pyannote.audio)...")
    try:
        with _temporary_torch_full_load_for_diarization(torch, progress):
            diarize_kwargs = {"device": device}
            if local_pyannote_ready:
                diarize_kwargs["model_name"] = local_pyannote
            else:
                diarize_kwargs["use_auth_token"] = hf_token
            diarize_model = DiarizationPipeline(**diarize_kwargs)
            diarize_segments = diarize_model(audio)
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()
        if any(x in low for x in ("401", "403", "gated", "unauthorized", "forbidden", "access", "token", "accept")):
            raise RuntimeError(
                "No se pudo cargar el modelo de diarización de pyannote. Revisa tres cosas:\n"
                "1) que HUGGINGFACE_TOKEN esté definido,\n"
                "2) que el token tenga permisos correctos,\n"
                "3) que hayas aceptado las condiciones de uso de los modelos pyannote en Hugging Face.\n"
                "El audio no fue enviado a OpenAI por este paso; la diarización es local, pero descarga/usa modelos de Hugging Face."
            ) from exc
        if "cuda" in low:
            raise RuntimeError(
                "La diarización falló al usar CUDA. Prueba seleccionar CPU en el selector de diarización.\n"
                f"Detalle técnico resumido: {msg[:300]}"
            ) from exc
        raise RuntimeError(
            "La diarización local falló durante la carga o ejecución de pyannote/WhisperX.\n"
            f"Detalle técnico resumido: {msg[:300]}"
        ) from exc

    result_diarized = whisperx.assign_word_speakers(
        diarize_segments,
        result_aligned,
    )

    diar_segments = result_diarized.get("segments", [])
    if not diar_segments:
        progress("AVISO: no se generaron segmentos diarizados, se usa transcripción sin speakers.")
        return result.get("text", "")

    # --- 5) Calcular cuánto habla cada SPEAKER_XX y etiquetar sin borrar extras ---
    speaker_durations: dict[str, float] = {}
    for seg in diar_segments:
        spk = str(seg.get("speaker", "SPEAKER_?") or "SPEAKER_?").strip()
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start))
        except Exception:
            start, end = 0.0, 0.0
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + max(0.0, end - start)

    sorted_speakers = sorted(speaker_durations.items(), key=lambda kv: kv[1], reverse=True)
    target = int(getattr(cfg, "target_speakers", 0) or 0)

    if target > 0 and len(sorted_speakers) > target:
        msg = (
            f"AVISO: la diarización detectó {len(sorted_speakers)} hablante(s), "
            f"más que los {target} esperado(s). Se conservarán como HABLANTE 3, HABLANTE 4, etc.; revisar manualmente."
        )
        progress(msg)
        logger.warning(msg)

    interviewer_name = (getattr(cfg, "interviewer_name", "") or "").strip()
    interviewee_name = (getattr(cfg, "interviewee_name", "") or "").strip()
    has_interview_names = bool(interviewer_name or interviewee_name)

    # Heurística revisable, no definitiva:
    # - Si se esperan 2 hablantes y hay nombres/contexto, el que más habla se etiqueta como entrevistado/a.
    # - El segundo se etiqueta como entrevistador/a.
    # - Si no hay nombres, se usan etiquetas neutras HABLANTE 1, HABLANTE 2...
    # - Los hablantes extra nunca se eliminan ni se colapsan automáticamente.
    use_interview_roles = has_interview_names and (target == 2 or len(sorted_speakers) == 2)

    speaker_labels: dict[str, str] = {}
    for rank, (spk, _dur) in enumerate(sorted_speakers, start=1):
        if use_interview_roles and rank == 1:
            speaker_labels[spk] = f"ENTREVISTADO/A ({interviewee_name})" if interviewee_name else "ENTREVISTADO/A"
        elif use_interview_roles and rank == 2:
            speaker_labels[spk] = f"ENTREVISTADOR/A ({interviewer_name})" if interviewer_name else "ENTREVISTADOR/A"
        else:
            speaker_labels[spk] = f"HABLANTE {rank}"

    # Fallback defensivo para segmentos que no aparecieron en el conteo.
    for seg in diar_segments:
        spk = str(seg.get("speaker", "SPEAKER_?") or "SPEAKER_?").strip()
        speaker_labels.setdefault(spk, f"HABLANTE {len(speaker_labels) + 1}")

    # --- 6) Construir texto tipo guion con formato ATLAS (fase 3) ---
    include_turn_numbers = bool(getattr(cfg, 'atlas_include_turn_numbers', False))
    include_timecodes = bool(getattr(cfg, 'atlas_include_timecodes', False))
    include_speaker_id = bool(getattr(cfg, 'atlas_include_speaker_id', False))
    blank_line_between_turns = bool(getattr(cfg, 'atlas_blank_line_between_turns', True))

    def _fmt_ts(seconds: float) -> str:
        try:
            seconds = float(seconds)
        except Exception:
            seconds = 0.0
        if seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    pause_split_sec = 3.5
    max_turn_sec = 120.0
    max_turn_chars = 1800



    def _join_words_readable(words_list) -> str:
        toks = []
        for w in (words_list or []):
            if not isinstance(w, dict):
                continue
            tok = w.get('word') or w.get('text') or ''
            tok = str(tok).strip()
            if not tok:
                continue
            toks.append(tok)
        if not toks:
            return ''
        s = ' '.join(toks)
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        s = re.sub(r"([¿¡])\s+", r"\1", s)
        s = re.sub(r"\s+([\)\]\}])", r"\1", s)
        s = re.sub(r"([\(\[\{¿¡])\s+", r"\1", s)
        s = re.sub(r"\s{2,}", ' ', s)
        return s.strip()

    turns = []  # list[dict]
    current_turn = None
    for seg in diar_segments:
        raw_speaker = seg.get('speaker') or 'SPEAKER_?'
        speaker_label = speaker_labels.get(raw_speaker, str(raw_speaker))
        text = _join_words_readable(seg.get('words')) or (seg.get('text') or '').strip()
        if not text:
            continue
        try:
            seg_start = float(seg.get('start', 0.0) or 0.0)
        except Exception:
            seg_start = 0.0
        try:
            seg_end = float(seg.get('end', seg_start) or seg_start)
        except Exception:
            seg_end = seg_start

        need_new_turn = False
        if current_turn is None:
            need_new_turn = True
        elif current_turn['raw_speaker'] != raw_speaker:
            need_new_turn = True
        else:
            # mismo speaker: cortamos por pausa/duración/tamaño
            prev_end = float(current_turn.get('end', seg_start))
            gap = seg_start - prev_end
            if gap > pause_split_sec:
                need_new_turn = True

            turn_dur = seg_end - float(current_turn.get('start', seg_start))
            if turn_dur > max_turn_sec:
                need_new_turn = True

            current_chars = sum(len(t) for t in current_turn.get('texts', []))
            if current_chars > max_turn_chars:
                need_new_turn = True

        if need_new_turn:
            current_turn = {
                'raw_speaker': raw_speaker,
                'speaker_label': speaker_label,
                'start': seg_start,
                'end': seg_end,
                'texts': [text],
            }
            turns.append(current_turn)
        else:
            current_turn['end'] = seg_end
            current_turn['texts'].append(text)

    # Render a texto (compatible con el parser de ATLAS.docx)
    lines = []
    turn_idx = 1
    for t in turns:
        speaker_part = t['speaker_label']
        if include_speaker_id:
            speaker_part = f"{speaker_part} ({t['raw_speaker']})"

        meta = []
        if include_turn_numbers:
            meta.append(f"T{turn_idx:03d}")
        if include_timecodes:
            meta.append(f"{_fmt_ts(t['start'])}-{_fmt_ts(t['end'])}")
        meta_txt = f"[{' '.join(meta)}] " if meta else ""

        # Primera línea con etiqueta + metadata, y líneas siguientes como continuación
        first = t['texts'][0] if t['texts'] else ''
        lines.append(f"{speaker_part}: {meta_txt}{first}")
        for extra in t['texts'][1:]:
            lines.append(extra)
        if blank_line_between_turns:
            lines.append("")
        turn_idx += 1

    if lines and lines[-1] == "":
        lines.pop()
    progress("Diarización y transcripción local completadas.")
    return "\n".join(lines)



# ==========================
# Export DOCX (fase 1 y 2)
# ==========================
def _docx_available() -> bool:
    try:
        import docx  # noqa: F401
        return True
    except Exception:
        return False

def _build_header_lines(cfg: TranscriberConfig, *, always: bool = False) -> List[str]:
    items = [
        ("Proyecto / investigación", getattr(cfg, "project", "") or ""),
        ("Entrevistador/a / Docente", getattr(cfg, "interviewer_name", "") or ""),
        ("Entrevistado/a", getattr(cfg, "interviewee_name", "") or ""),
        ("Lugar / institución", getattr(cfg, "place", "") or ""),
        ("Fecha / cohorte", getattr(cfg, "date_label", "") or ""),
        ("Notas contextuales adicionales", getattr(cfg, "extra_context", "") or ""),
    ]

    if always:
        # Siempre aparecen, aunque queden en blanco
        return [f"{label}: {value}".rstrip() for (label, value) in items]

    # Modo “clásico”: solo si hay valor
    out: List[str] = []
    for label, value in items:
        if str(value).strip():
            out.append(f"{label}: {value}")
    return out


def _atlas_paragraphs_from_text(text: str) -> List[str]:
    import re
    # Un "turno" por párrafo, combinando líneas de continuidad
    speaker_re = re.compile(r"^[^:\n]{1,40}:\s+")
    out: List[str] = []
    current: Optional[str] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                out.append(current.strip())
                current = None
            # Mantener un separador visual (párrafo vacío)
            out.append("")
            continue

        if speaker_re.match(line):
            if current:
                out.append(current.strip())
            current = line
        else:
            if current is None:
                current = line
            else:
                current += " " + line

    if current:
        out.append(current.strip())

    # Normalizar: colapsar múltiple vacíos seguidos a un solo vacío
    cleaned: List[str] = []
    prev_empty = False
    for p in out:
        if p == "":
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(p)
            prev_empty = False
    return cleaned

def _build_atlas_body_paragraphs(transcript_atlas_paras: List[str], summary_text: str) -> List[str]:
    out: List[str] = []

    s = (summary_text or "").strip()
    if s:
        out.append("RESUMEN")
        out.append("")
        out.extend(s.splitlines())
        out.append("")
        out.append("TRANSCRIPCIÓN (ATLAS)")
        out.append("")

    out.extend(transcript_atlas_paras)
    return out


def _write_docx(
    path: str,
    title: str,
    header_lines: List[str],
    paragraphs: List[str],
    *,
    bold_before_colon: bool = False,
    skip_empty_paragraphs: bool = False,
    space_after_pt: Optional[float] = None,
) -> None:
    """Escribe un .docx simple (compatible con ATLAS.ti).

    - bold_before_colon: pone en negrita todo lo anterior a ':' (útil para speaker labels).
    - skip_empty_paragraphs: evita párrafos vacíos (si preferís usar spacing en lugar de líneas en blanco).
    - space_after_pt: si se setea (p.ej. 6), aplica espacio después de cada párrafo.
    """
    try:
        from docx import Document
        from docx.shared import Pt
    except Exception as e:
        raise RuntimeError("Falta python-docx. Instalá 'python-docx' y reintentá.") from e

    doc = Document()
    # Título
    doc.add_heading(title, level=1)

    # Encabezado (metadata)
    for h in header_lines:
        if skip_empty_paragraphs and not str(h).strip():
            continue
        doc.add_paragraph(str(h))

    if header_lines and not skip_empty_paragraphs:
        doc.add_paragraph("")

    for t in paragraphs:
        t = "" if t is None else str(t)
        if skip_empty_paragraphs and not t.strip():
            continue
        p = doc.add_paragraph()
        if bold_before_colon and ":" in t:
            left, right = t.split(":", 1)
            r1 = p.add_run(left + ":")
            r1.bold = True
            p.add_run(right)  # incluye el espacio inicial si existía
        else:
            p.add_run(t)
        if space_after_pt is not None:
            p.paragraph_format.space_after = Pt(space_after_pt)

    doc.save(path)

def _rtf_escape(text: str) -> str:
    """Escape text for minimal RTF, preserving Unicode characters."""
    if text is None:
        return ""
    out = []
    for ch in str(text):
        o = ord(ch)
        if ch in ("\\", "{", "}"):
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\par\n")
        elif o < 128:
            out.append(ch)
        else:
            # RTF unicode escape: \uN? where ? is a fallback char
            out.append(f"\\u{o}?")
    return "".join(out)


def _write_rtf(
    path: Path,
    title: str,
    header_lines: list[str],
    paragraphs: list[str],
    *,
    bold_before_colon: bool = False,
) -> None:
    """Write a simple RTF transcript compatible with ATLAS.ti (RTF import)."""
    # Use a basic font (Calibri) and 12pt default (\fs24).
    parts: list[str] = []
    parts.append("{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Calibri;}}\n")
    parts.append("\\fs24\n")

    # Title
    if title:
        parts.append("\\b \\fs32 " + _rtf_escape(title) + " \\b0 \\fs24\\par\n")
        parts.append("\\par\n")

    # Header / metadata
    for line in header_lines or []:
        if line:
            parts.append("\\b " + _rtf_escape(line) + " \\b0\\par\n")
    if header_lines:
        parts.append("\\par\n")

    # Body
    for p in paragraphs or []:
        p = "" if p is None else str(p).strip()
        if p == "":
            parts.append("\\par\n")
            continue

        if bold_before_colon and ":" in p:
            left, right = p.split(":", 1)
            parts.append("\\b " + _rtf_escape(left.strip()) + ": \\b0 " + _rtf_escape(right.lstrip()) + "\\par\n")
        else:
            parts.append(_rtf_escape(p) + "\\par\n")

    parts.append("}")
    path.write_text("".join(parts), encoding="utf-8")


def _parse_first_json_object(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _autofill_metadata_if_missing(
    client: OpenAI,
    cfg: TranscriberConfig,
    *,
    transcript_text: str,
    summary_text: str,
    progress: ProgressFn,
) -> None:
    # Solo si falta algo importante
    keys = ["project", "interviewer_name", "interviewee_name", "place", "date_label", "extra_context"]
    missing = [k for k in keys if not str(getattr(cfg, k, "") or "").strip()]
    if not missing:
        return

    progress("Intentando inferir metadatos (encabezado) desde el análisis...")

    system = (
        "Eres un asistente que extrae metadatos SOLO si están explícitos o son muy inferibles. "
        "Si no estás seguro, devuelve cadena vacía. No inventes.\n\n"
        "Devuelve ÚNICAMENTE un JSON con estas claves:\n"
        "project, interviewer_name, interviewee_name, place, date_label, extra_context\n"
        "Todas deben existir y ser strings."
    )

    user = (
        "TEXTO (transcripción limpia o base):\n"
        f"{transcript_text}\n\n"
        "RESUMEN (si existe):\n"
        f"{summary_text}\n\n"
        "Extrae metadatos. Si no hay información confiable, usa \"\"."
    )

    try:
        resp = client.chat.completions.create(
            model=cfg.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        data = _parse_first_json_object((resp.choices[0].message.content or "").strip())
        if not isinstance(data, dict):
            return

        # Solo rellenar los campos que estaban vacíos
        for k in keys:
            if not str(getattr(cfg, k, "") or "").strip():
                v = str(data.get(k, "") or "").strip()
                # Si el modelo no está seguro, debería traer ""
                setattr(cfg, k, v)

    except Exception as e:
        progress(f"AVISO: no se pudieron inferir metadatos automáticamente: {e}")

def run_batch(
    cfg: TranscriberConfig,
    progress: ProgressFn,
    set_progress: Optional[ProgressBarFn] = None,
    on_file_done: Optional[Callable[[Path], None]] = None
):
    if set_progress is None:
        set_progress = lambda x: None  # no-op

    _emit_job_event(cfg, stage=STAGE_PREPARING, progress=0.0, message="Preparando trabajo de transcripción.")
    logger.info("=== Inicio de proceso de transcripción por lote ===")
    logger.info("Rutas de entrada/salida configuradas para el lote (no se persisten rutas por privacidad).")

    stt_engine = (getattr(cfg, "stt_engine", STT_ENGINE_OPENAI) or STT_ENGINE_OPENAI).strip()
    needs_openai_stt = (not getattr(cfg, "use_diarization", False)) and stt_engine == STT_ENGINE_OPENAI
    text_engine = _normalize_text_engine(getattr(cfg, "text_engine", TEXT_ENGINE_DEFAULT))
    needs_openai_chat = _text_processing_uses_openai(cfg)
    needs_compatible_chat = _text_processing_uses_compatible_api(cfg)
    needs_openai = needs_openai_stt or needs_openai_chat

    if needs_compatible_chat and not str(getattr(cfg, "compatible_api_url", "") or "").strip():
        raise RuntimeError("La API compatible requiere una URL base para limpieza/resumen.")

    if needs_openai and not cfg.api_key:
        raise RuntimeError(
            "No se proporcionó API key de OpenAI. "
            "No hace falta para transcripción local simple ni para Ollama local, pero sí para STT OpenAI o limpieza/resumen con OpenAI."
        )
    if needs_openai and not HAS_OPENAI:
        raise RuntimeError("Falta el paquete 'openai'. Instalá requirements.txt para usar STT OpenAI o limpieza/resumen con OpenAI.")

    client = OpenAI(api_key=cfg.api_key) if needs_openai else None

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {cfg.input_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    _storage_preflight(cfg, progress)
    stale_removed = _cleanup_stale_workdirs(cfg.work_dir)
    if stale_removed:
        progress(f"Se eliminaron {stale_removed} carpeta(s) temporal(es) antiguas de Jerónimo.")

    ffmpeg_available = check_ffmpeg()
    use_ffmpeg = cfg.use_ffmpeg and ffmpeg_available
    if cfg.use_diarization and not ffmpeg_available:
        raise RuntimeError(
            "La diarización local con WhisperX requiere ffmpeg disponible en el sistema. "
            "Jerónimo no intentará renombrar/copiar el archivo como WAV para ocultar este problema."
        )
    if cfg.use_ffmpeg and not use_ffmpeg:
        progress("AVISO: se solicitó ffmpeg pero no se encontró. Se usará el archivo original sin crear un WAV falso.")
    elif use_ffmpeg:
        progress("ffmpeg detectado: se preparará el audio con reducción de ruido y normalización." if cfg.audio_enhancement else "ffmpeg detectado: se hará sólo conversión técnica a mono/16k.")

    if cfg.selected_files:
        audio_files = list(cfg.selected_files)
    else:
        audio_files = [
            p for p in cfg.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".mp4"}
        ]

    audio_files.sort()


    if not audio_files:
        progress("No se encontraron archivos de audio en la carpeta de entrada.")
        logger.info("No hay archivos de audio para procesar.")
        set_progress(0.0)
        _emit_job_event(cfg, stage=STAGE_COMPLETED, status=JOB_STATUS_COMPLETED, progress=1.0, message="No había archivos para procesar.")
        return

    total_files = len(audio_files)
    progress(f"Se encontraron {total_files} archivos para procesar.")
    set_progress(0.0)

    # --- Evitar contaminación de metadatos entre archivos en modo lote ---
    _base_meta = {
        "interviewer_name": cfg.interviewer_name,
        "interviewee_name": cfg.interviewee_name,
        "project": cfg.project,
        "place": cfg.place,
        "date_label": cfg.date_label,
        "duration_label": getattr(cfg, "duration_label", "") or "",
        "extra_context": cfg.extra_context,
    }

    cancelled_run = False
    had_errors = False
    for idx, audio in enumerate(audio_files, start=1):
        progress(f"\n[{idx}/{total_files}] Procesando: {audio.name}")
        logger.info("Procesando archivo ref=%s", _privacy_file_ref(audio))

        base_name = audio.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_name = f"{base_name}.{timestamp}"
        generated_files: list[Path] = []

        # Reset de metadatos por archivo (evita heredar inferencias del anterior)
        cfg.interviewer_name = _base_meta["interviewer_name"]
        cfg.interviewee_name = _base_meta["interviewee_name"]
        cfg.project = _base_meta["project"]
        cfg.place = _base_meta["place"]
        cfg.date_label = _base_meta["date_label"]
        cfg.duration_label = _base_meta["duration_label"]
        cfg.extra_context = _base_meta["extra_context"]

        # Reset por archivo: evita arrastrar resumen/errores entre iteraciones del lote
        resumen_text = ""
        cfg.text_processing_error = ""

        # rango de progreso para este archivo
        file_base = (idx - 1) / total_files
        file_span = 1.0 / total_files

        current_job_stage = STAGE_PREPARING

        def stage(pct_in_file: float, stage_name: Optional[str] = None, message: str = ""):
            """Actualiza barra y estado estructurado dentro del rango de este archivo."""
            nonlocal current_job_stage
            if stage_name:
                current_job_stage = stage_name
            overall = file_base + pct_in_file * file_span
            set_progress(overall)
            _emit_job_event(
                cfg, stage=current_job_stage, progress=overall, message=message, current_item=audio.name
            )

        job_work_dir = Path(tempfile.mkdtemp(prefix=".jeronimo_tmp_", dir=str(cfg.work_dir)))
        ffmpeg_applied = False

        try:
            _check_cancelled(cfg)
            processed_path = job_work_dir / "audio_proc.wav"
            processed_path = preprocess_audio(audio, processed_path, use_ffmpeg, progress, cfg.audio_enhancement)
            ffmpeg_applied = processed_path != audio and processed_path.exists()
            # Duración automática (desde WAV procesado)
            try:
                dur_sec = get_wav_duration_seconds(processed_path)
                if dur_sec and dur_sec > 0:
                    h = int(dur_sec // 3600)
                    m = int((dur_sec % 3600) // 60)
                    s = int(dur_sec % 60)
                    cfg.duration_label = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    cfg.duration_label = ""
            except Exception:
                cfg.duration_label = ""

            stage(0.1, STAGE_PREPARING, "Audio preparado.")

            _check_cancelled(cfg)
            if cfg.use_diarization:
                # 👉 Modo diarización local con WhisperX
                progress("Diarización local activada: usando WhisperX en vez de la API de audio de OpenAI.")
                stage(0.12, STAGE_DIARIZING, "Iniciando WhisperX y separación de hablantes.")
                raw_text = _diarize_via_optional_worker(processed_path, cfg, progress)
                stage(0.8, STAGE_DIARIZING, "Transcripción y diarización finalizadas.")
            else:
                # 👉 Modo sin diarización: segmentación + motor STT elegido
                segments = segment_wav_if_needed(processed_path, base_name, job_work_dir, progress)

                if stt_engine == STT_ENGINE_LOCAL_FAST_WHISPER:
                    progress("Motor STT: Local faster-whisper (sin enviar audio a API externa).")
                else:
                    progress("Motor STT: OpenAI API.")
                stage(0.2, STAGE_TRANSCRIBING, "Iniciando transcripción.")

                raw_parts: list[str] = []

                if segments:
                    per_seg_span = 0.6 / len(segments)
                else:
                    per_seg_span = 0.6

                for i, seg in enumerate(segments):
                    _check_cancelled(cfg)
                    progress(f"Transcribiendo segmento {i+1}/{len(segments)}: {seg.name}")
                    stage(0.2 + per_seg_span * i, STAGE_TRANSCRIBING, f"Transcribiendo segmento {i+1}/{len(segments)}.")

                    if stt_engine == STT_ENGINE_LOCAL_FAST_WHISPER:
                        segment_text = transcribe_audio_local_faster_whisper(cfg, seg, progress)
                    else:
                        if client is None:
                            raise RuntimeError("Cliente OpenAI no inicializado para STT OpenAI.")
                        segment_text = transcribe_audio(client, cfg, seg, progress)

                    raw_parts.append(
                        f"\n\n[=== SEGMENTO {i+1}/{len(segments)} ===]\n\n{segment_text.strip()}"
                    )

                raw_text = "\n".join(raw_parts).strip()
                stage(0.8, STAGE_TRANSCRIBING, "Transcripción finalizada.")



            _check_cancelled(cfg)
            if raw_text is None or (isinstance(raw_text, str) and not raw_text.strip()):
                logger.error("Transcripción devolvió None o vacío. Se marca este archivo como error.")
                raise RuntimeError("La transcripción devolvió texto vacío.")

            raw_out = cfg.output_dir / f"{output_base_name}.raw.txt"
            raw_out.write_text(raw_text, encoding="utf-8")
            generated_files.append(raw_out)

            # Export RTF (RAW y ATLAS) - no depende de python-docx
            try:
                header_lines = _build_header_lines(cfg)
                title = base_name

                if getattr(cfg, "export_rtf_raw", False):
                    rtf_path = cfg.output_dir / f"{output_base_name}.RAW.rtf"
                    _write_rtf(rtf_path, title, header_lines, raw_text.splitlines(), bold_before_colon=False)
                    generated_files.append(rtf_path)
                    logger.info("RTF RAW generado.")


            except Exception as e:
                logger.warning("No se pudo generar RTF RAW. tipo=%s", type(e).__name__)


            # Export DOCX (fase 1: RAW) y (fase 2: ATLAS) para ATLAS.ti


            if (_docx_available() and getattr(cfg, "export_docx_raw", False)):
       
                try:
                    header_lines = _build_header_lines(cfg)
                    title = base_name
                    if getattr(cfg, "export_docx_raw", False):
                        docx_path = cfg.output_dir / f"{output_base_name}.RAW.docx"
                        _write_docx(docx_path, title, header_lines, raw_text.splitlines())
                        generated_files.append(docx_path)
                        logger.info("DOCX RAW generado.")

                except Exception as e:

                    logger.warning("No se pudo generar DOCX RAW. tipo=%s", type(e).__name__)

            progress(f"Transcripción cruda guardada en: {raw_out.name}")
            stage(0.82, STAGE_EXPORTING, "Guardando transcripción original.")

            _check_cancelled(cfg)
            clean_text = raw_text
            did_clean = False
            if cfg.do_clean and cfg.clean_prompt.strip():
                try:
                    stage(0.83, STAGE_CLEANING, "Generando versión limpia.")
                    clean_text = chat_process(
                        client, cfg, raw_text, cfg.clean_prompt, progress, "limpieza"
                    )
                    did_clean = True
                    stage(0.9, STAGE_CLEANING, "Versión limpia generada.")
                except Exception as e:
                    safe_error = _sanitize_error_message(e, cfg)
                    progress(f"ERROR en limpieza, se usará texto crudo. Detalle: {safe_error}")
                    cfg.text_processing_error = (cfg.text_processing_error + " | " if cfg.text_processing_error else "") + f"limpieza: {safe_error}"
                    logger.error("Error en limpieza. tipo=%s", type(e).__name__)
                    clean_text = raw_text

            if cfg.do_clean:
                clean_out = cfg.output_dir / f"{output_base_name}.clean.txt"
                clean_out.write_text(clean_text, encoding="utf-8")
                generated_files.append(clean_out)
                progress(f"Versión limpia guardada en: {clean_out.name}")
                if did_clean:
                    stage(0.92)

            _check_cancelled(cfg)
            if cfg.do_summary and cfg.summary_prompt.strip():
                try:
                    stage(0.93, STAGE_SUMMARIZING, "Generando resumen.")
                    resumen_text = chat_process(
                        client, cfg, clean_text, cfg.summary_prompt, progress, "resumen"
                    )
                    resumen_out = cfg.output_dir / f"{output_base_name}.resumen.txt"
                    resumen_out.write_text(resumen_text, encoding="utf-8")
                    generated_files.append(resumen_out)
                    progress(f"Resumen guardado en: {resumen_out.name}")
                    stage(0.98, STAGE_SUMMARIZING, "Resumen generado.")
                except Exception as e:
                    safe_error = _sanitize_error_message(e, cfg)
                    progress(f"ERROR en resumen. Detalle: {safe_error}")
                    cfg.text_processing_error = (cfg.text_processing_error + " | " if cfg.text_processing_error else "") + f"resumen: {safe_error}"
                    logger.error("Error en resumen. tipo=%s", type(e).__name__)

            _check_cancelled(cfg)
            # --- Autocompletar metadatos si faltan (solo si están vacíos) ---
            if client is not None and needs_openai_chat:
                _autofill_metadata_if_missing(
                    client,
                    cfg,
                    transcript_text=clean_text if (clean_text or "").strip() else raw_text,
                    summary_text=resumen_text if (cfg.do_summary and (resumen_text or "").strip()) else "",
                    progress=progress,
                )
            else:
                progress("Metadatos automáticos omitidos: no se usó OpenAI en este flujo local.")

            # --- Export ATLAS con encabezado + resumen + transcripción ---
            atlas_header = _build_header_lines(cfg, always=True)

            # IMPORTANTE:
            # - Si hay diarización (WhisperX), los timecodes/turnos viven en raw_text.
            # - Si usamos clean_text, GPT puede borrar [Txxx 00:..-00:..].
            if cfg.use_diarization:
                atlas_source = raw_text
            else:
                atlas_source = clean_text if cfg.do_clean else raw_text

            atlas_paras = _atlas_paragraphs_from_text(atlas_source)

            atlas_body = _build_atlas_body_paragraphs(
                atlas_paras,
                resumen_text if (cfg.do_summary and (resumen_text or "").strip()) else "",
            )

            # RTF ATLAS
            if getattr(cfg, "export_rtf_atlas", False):
                try:
                    rtf_path = cfg.output_dir / f"{output_base_name}.ATLAS.rtf"
                    _write_rtf(
                        rtf_path,
                        base_name,
                        atlas_header,
                        atlas_body,
                        bold_before_colon=bool(getattr(cfg, "atlas_bold_speaker_prefix", False)),
                    )
                    generated_files.append(rtf_path)
                    logger.info("RTF ATLAS generado.")
                except Exception as e:
                    logger.warning("No se pudo generar RTF ATLAS. tipo=%s", type(e).__name__)

            # DOCX ATLAS
            if getattr(cfg, "export_docx_atlas", False) and _docx_available():
                try:
                    docx_path = cfg.output_dir / f"{output_base_name}.ATLAS.docx"
                    _write_docx(
                        docx_path,
                        base_name,
                        atlas_header,
                        atlas_body,
                        bold_before_colon=bool(getattr(cfg, "atlas_bold_speaker_prefix", False)),
                    )
                    generated_files.append(docx_path)
                    logger.info("DOCX ATLAS generado.")
                except Exception as e:
                    logger.warning("No se pudo generar DOCX ATLAS. tipo=%s", type(e).__name__)


            stage(0.99, STAGE_EXPORTING, "Exportando y registrando resultados.")
            manifest_path = write_manifest(
                cfg,
                input_file=audio,
                output_base_name=output_base_name,
                generated_files=generated_files + [cfg.output_dir / f"{output_base_name}.manifest.json"],
                use_ffmpeg=ffmpeg_applied,
                status="success",
            )
            generated_files.append(manifest_path)
            progress(f"Manifiesto generado: {manifest_path.name}")

            progress(f"Archivo finalizado: {audio.name}")
            stage(1.0, STAGE_FINALIZING, "Archivo finalizado.")

            if on_file_done:
                on_file_done(audio)

        except ProcessingCancelled as e:
            cancelled_run = True
            safe_error = _sanitize_error_message(e, cfg)
            progress("Cancelación solicitada. Jerónimo cerrará este trabajo y limpiará los temporales.")
            logger.info("Proceso cancelado en archivo ref=%s", _privacy_file_ref(audio))
            try:
                manifest_path = write_manifest(
                    cfg,
                    input_file=audio,
                    output_base_name=output_base_name,
                    generated_files=generated_files + [cfg.output_dir / f"{output_base_name}.manifest.json"],
                    use_ffmpeg=ffmpeg_applied,
                    status="cancelled",
                    error_message=safe_error,
                )
                progress(f"Manifiesto de cancelación generado: {manifest_path.name}")
            except Exception as manifest_error:
                logger.warning("No se pudo generar manifiesto de cancelación: %s", type(manifest_error).__name__)
            stage(1.0, STAGE_CANCELLED, "Trabajo cancelado de forma segura.")

        except Exception as e:
            had_errors = True
            safe_error = _sanitize_error_message(e, cfg)
            msg = f"ERROR procesando {audio.name}: {safe_error}"
            progress(msg)
            logger.error(
                "Error procesando archivo ref=%s tipo=%s detalle=%s",
                _privacy_file_ref(audio),
                type(e).__name__,
                safe_error,
            )
            try:
                manifest_path = write_manifest(
                    cfg,
                    input_file=audio,
                    output_base_name=output_base_name,
                    generated_files=generated_files + [cfg.output_dir / f"{output_base_name}.manifest.json"],
                    use_ffmpeg=ffmpeg_applied,
                    status="error",
                    error_message=safe_error,
                )
                progress(f"Manifiesto generado: {manifest_path.name}")
            except Exception as manifest_error:
                logger.warning(
                    "No se pudo generar manifiesto de error ref=%s tipo=%s",
                    _privacy_file_ref(audio),
                    type(manifest_error).__name__,
                )
            stage(1.0, STAGE_ERROR, "El archivo terminó con error.")

        finally:
            shutil.rmtree(job_work_dir, ignore_errors=True)
            if job_work_dir.exists():
                progress("AVISO: no se pudieron eliminar todos los temporales de este archivo.")
                logger.warning("Limpieza temporal incompleta en un directorio de trabajo de Jerónimo.")
            else:
                progress("Temporales de este archivo eliminados.")

        if cancelled_run:
            break

    if cancelled_run:
        progress("\nProceso cancelado de forma segura.")
        _emit_job_event(cfg, stage=STAGE_CANCELLED, status=JOB_STATUS_CANCELLED, progress=1.0, message="Proceso cancelado.")
    elif had_errors:
        progress("\nProceso finalizado con uno o más errores. Revisa el log y los manifiestos.")
        _emit_job_event(cfg, stage=STAGE_ERROR, status=JOB_STATUS_ERROR, progress=1.0, message="Proceso finalizado con errores.")
    else:
        progress("\nProceso completado.")
        _emit_job_event(cfg, stage=STAGE_COMPLETED, status=JOB_STATUS_COMPLETED, progress=1.0, message="Proceso completado.")
    set_progress(1.0)
    logger.info("=== Fin del proceso de transcripción por lote ===")



def run_analysis_only(
    cfg: TranscriberConfig,
    progress: ProgressFn,
    set_progress: Optional[ProgressBarFn] = None,
    on_file_done: Optional[Callable[[Path], None]] = None
):
    if set_progress is None:
        set_progress = lambda x: None

    _emit_job_event(cfg, stage=STAGE_PREPARING, progress=0.0, message="Preparando análisis de textos.")
    logger.info("=== Inicio de análisis de textos por lote ===")
    logger.info("Rutas de análisis configuradas (no se persisten rutas por privacidad).")

    text_engine = _normalize_text_engine(getattr(cfg, "text_engine", TEXT_ENGINE_DEFAULT))
    needs_openai_chat = _text_processing_uses_openai(cfg)
    needs_compatible_chat = _text_processing_uses_compatible_api(cfg)

    if needs_compatible_chat and not str(getattr(cfg, "compatible_api_url", "") or "").strip():
        raise RuntimeError("La API compatible requiere una URL base para limpieza/resumen.")

    if needs_openai_chat and not cfg.api_key:
        raise RuntimeError("No se proporcionó API key de OpenAI para limpieza/resumen con OpenAI.")
    if needs_openai_chat and not HAS_OPENAI:
        raise RuntimeError("Falta el paquete 'openai'. Instalá requirements.txt para limpieza/resumen con OpenAI.")

    client = OpenAI(api_key=cfg.api_key) if needs_openai_chat else None

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {cfg.input_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _check_directory_writable(cfg.output_dir)
    except Exception as exc:
        raise RuntimeError(f"La carpeta de salida no es escribible: {cfg.output_dir}. Detalle: {exc}") from exc

    if cfg.selected_files:
        text_files = list(cfg.selected_files)
    else:
        text_files = [
            p for p in cfg.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".txt"}
        ]

    text_files.sort()


    if not text_files:
        progress("No se encontraron archivos .txt en la carpeta de entrada.")
        logger.info("No hay archivos de texto para analizar.")
        set_progress(0.0)
        _emit_job_event(cfg, stage=STAGE_COMPLETED, status=JOB_STATUS_COMPLETED, progress=1.0, message="No había textos para analizar.")
        return

    total_files = len(text_files)
    progress(f"Se encontraron {total_files} archivos de texto para analizar.")
    set_progress(0.0)

    cancelled_run = False
    had_errors = False
    for idx, txt in enumerate(text_files, start=1):
        progress(f"\n[{idx}/{total_files}] Analizando: {txt.name}")
        logger.info("Analizando texto ref=%s", _privacy_file_ref(txt))

        base_name = txt.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_base = (idx - 1) / total_files
        file_span = 1.0 / total_files

        current_job_stage = STAGE_PREPARING

        def stage(pct_in_file: float, stage_name: Optional[str] = None, message: str = ""):
            nonlocal current_job_stage
            if stage_name:
                current_job_stage = stage_name
            overall = file_base + pct_in_file * file_span
            set_progress(overall)
            _emit_job_event(cfg, stage=current_job_stage, progress=overall, message=message, current_item=txt.name)

        try:
            _check_cancelled(cfg)
            raw_text = txt.read_text(encoding="utf-8")
            stage(0.1, STAGE_PREPARING, "Texto cargado.")

            _check_cancelled(cfg)
            clean_text = raw_text
            if cfg.do_clean and cfg.clean_prompt.strip():
                try:
                    stage(0.15, STAGE_CLEANING, "Generando versión limpia.")
                    clean_text = chat_process(
                        client, cfg, raw_text, cfg.clean_prompt, progress, "limpieza"
                    )
                    stage(0.6, STAGE_CLEANING, "Versión limpia generada.")
                except Exception as e:
                    safe_error = _sanitize_error_message(e, cfg)
                    progress(f"ERROR en limpieza. Detalle: {safe_error}")
                    logger.error("Error en limpieza de texto. tipo=%s", type(e).__name__)
                    clean_text = raw_text

            if cfg.do_clean:
                clean_out = cfg.output_dir / f"{base_name}.{timestamp}.clean.txt"
                clean_out.write_text(clean_text, encoding="utf-8")
                progress(f"Versión limpia guardada en: {clean_out.name}")
                stage(0.7, STAGE_EXPORTING, "Versión limpia guardada.")

            _check_cancelled(cfg)
            if cfg.do_summary and cfg.summary_prompt.strip():
                try:
                    stage(0.72, STAGE_SUMMARIZING, "Generando resumen.")
                    resumen_text = chat_process(
                        client, cfg, clean_text, cfg.summary_prompt, progress, "resumen"
                    )
                    resumen_out = cfg.output_dir / f"{base_name}.{timestamp}.resumen.txt"
                    resumen_out.write_text(resumen_text, encoding="utf-8")
                    progress(f"Resumen guardado en: {resumen_out.name}")
                    stage(0.95, STAGE_SUMMARIZING, "Resumen generado.")
                except Exception as e:
                    safe_error = _sanitize_error_message(e, cfg)
                    progress(f"ERROR en resumen. Detalle: {safe_error}")
                    logger.error("Error en resumen de texto. tipo=%s", type(e).__name__)

            progress(f"Archivo de texto finalizado: {txt.name}")
            stage(1.0, STAGE_FINALIZING, "Archivo de texto finalizado.")

            if on_file_done:
                on_file_done(txt)

        except ProcessingCancelled:
            cancelled_run = True
            progress("Cancelación solicitada. No se iniciarán más análisis de texto.")
            logger.info("Análisis cancelado en texto ref=%s", _privacy_file_ref(txt))
            stage(1.0, STAGE_CANCELLED, "Análisis cancelado.")

        except Exception as e:
            had_errors = True
            safe_error = _sanitize_error_message(e, cfg)
            msg = f"ERROR procesando {txt.name}: {safe_error}"
            progress(msg)
            logger.error(
                "Error procesando texto ref=%s tipo=%s detalle=%s",
                _privacy_file_ref(txt),
                type(e).__name__,
                safe_error,
            )
            stage(1.0, STAGE_ERROR, "El análisis terminó con error.")

        if cancelled_run:
            break

    if cancelled_run:
        progress("\nAnálisis de textos cancelado de forma segura.")
        _emit_job_event(cfg, stage=STAGE_CANCELLED, status=JOB_STATUS_CANCELLED, progress=1.0, message="Análisis cancelado.")
    elif had_errors:
        progress("\nAnálisis finalizado con uno o más errores.")
        _emit_job_event(cfg, stage=STAGE_ERROR, status=JOB_STATUS_ERROR, progress=1.0, message="Análisis finalizado con errores.")
    else:
        progress("\nAnálisis de textos completado.")
        _emit_job_event(cfg, stage=STAGE_COMPLETED, status=JOB_STATUS_COMPLETED, progress=1.0, message="Análisis completado.")
    set_progress(1.0)
    logger.info("=== Fin del análisis de textos por lote ===")

