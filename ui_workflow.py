from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Iterable

from core_transcriber import (
    TranscriberConfig,
    STT_ENGINE_LOCAL_FAST_WHISPER,
    STT_ENGINE_OPENAI,
)
from text_providers import DEFAULT_OPENAI_COMPATIBLE_URL, DEFAULT_OLLAMA_URL, endpoint_scope


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".mkv", ".ogg", ".flac"}
TEXT_EXTENSIONS = {".txt"}

PROFILE_DIARIZATION = "quality_diarization"
PROFILE_LOCAL = "local_fast"
PROFILE_OPENAI = "openai_api"

TEXT_NONE = "none"
TEXT_OLLAMA = "ollama"
TEXT_OPENAI = "openai"
TEXT_COMPATIBLE = "openai_compatible"

GENERIC_WHISPER_PROMPT = (
    "Entrevista o conversación. Mantén nombres propios, vocabulario especializado, "
    "muletillas relevantes y expresiones tal como se escuchan. No completes contenido dudoso. "
    "Si una palabra no puede reconocerse con seguridad, evita inventarla."
)

DEFAULT_CLEAN_PROMPT = (
    "Eres un asistente de apoyo a investigación cualitativa. Limpia la transcripción sin alterar el testimonio.\n\n"
    "Reglas:\n"
    "1. Corrige puntuación, cortes de frase y errores evidentes de reconocimiento.\n"
    "2. Conserva el sentido, el orden y la voz de cada persona.\n"
    "3. No resumas, interpretes ni agregues información.\n"
    "4. No conviertas el testimonio a lenguaje académico.\n"
    "5. Conserva contradicciones y vacilaciones significativas.\n"
    "6. Mantén etiquetas de hablantes y timecodes si existen.\n"
    "7. Marca [inaudible] o [dudoso] cuando corresponda; no inventes.\n\n"
    "Devuelve solamente la versión limpia."
)

DEFAULT_SUMMARY_PROMPT = (
    "A partir de la transcripción, elabora un resumen analítico fiel.\n\n"
    "Incluye:\n"
    "1. Un resumen denso de las ideas principales.\n"
    "2. Temas centrales.\n"
    "3. Conceptos, nombres o instituciones mencionadas, sólo si aparecen.\n"
    "4. Posibles citas textuales relevantes, sin inventarlas.\n"
    "5. Dudas o zonas que requieran revisión humana.\n\n"
    "No agregues contexto externo ni sustituyas la revisión humana."
)


@dataclass
class WorkflowSelection:
    input_path: Path
    output_dir: Path
    profile: str = PROFILE_DIARIZATION
    language: str = "es"
    target_speakers: int = 0
    audio_enhancement: bool = True
    do_clean: bool = False
    do_summary: bool = True
    text_engine: str = TEXT_OLLAMA
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = ""
    openai_api_key: str = ""
    openai_stt_model: str = "gpt-4o-mini-transcribe"
    openai_chat_model: str = "gpt-4.1-mini"
    compatible_api_url: str = DEFAULT_OPENAI_COMPATIBLE_URL
    compatible_api_key: str = ""
    compatible_model: str = ""
    local_fast_model: str = "large-v3"
    local_device: str = "auto"
    interviewer: str = ""
    interviewee: str = ""
    project: str = ""
    place: str = ""
    date_label: str = ""
    extra_context: str = ""
    whisper_prompt: str = GENERIC_WHISPER_PROMPT
    clean_prompt: str = DEFAULT_CLEAN_PROMPT
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    export_docx_raw: bool = False
    export_docx_atlas: bool = True
    export_rtf_raw: bool = False
    export_rtf_atlas: bool = False
    atlas_turn_numbers: bool = False
    atlas_timecodes: bool = True
    atlas_speaker_id: bool = False
    atlas_bold_speaker: bool = True
    diarization_runtime_python: str = ""
    diarization_worker_executable: str = ""


@dataclass(frozen=True)
class PrivacyAssessment:
    audio_leaves_device: bool
    text_leaves_device: bool
    title: str
    details: tuple[str, ...]


def default_work_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    if root:
        return Path(root) / "JeronimoAbyaYala" / "work"
    return Path(tempfile.gettempdir()) / "JeronimoAbyaYala" / "work"


def resolve_input(path: Path, *, text_only: bool = False) -> tuple[Path, list[Path] | None]:
    path = Path(path)
    if path.is_file():
        allowed = TEXT_EXTENSIONS if text_only else AUDIO_EXTENSIONS
        if path.suffix.lower() not in allowed:
            expected = "texto .txt" if text_only else "audio o video compatible"
            raise ValueError(f"El archivo seleccionado no es un {expected}: {path.name}")
        return path.parent, [path]
    if path.is_dir():
        return path, None
    raise FileNotFoundError(f"La entrada no existe: {path}")


def build_context_prompt(base_prompt: str, selection: WorkflowSelection) -> str:
    parts: list[str] = []
    if selection.project.strip():
        parts.append(f"Proyecto: {selection.project.strip()}.")
    if selection.interviewer.strip():
        parts.append(f"Entrevistador/a: {selection.interviewer.strip()}.")
    if selection.interviewee.strip():
        parts.append(f"Persona entrevistada o código: {selection.interviewee.strip()}.")
    if selection.place.strip():
        parts.append(f"Lugar o institución: {selection.place.strip()}.")
    if selection.date_label.strip():
        parts.append(f"Fecha o cohorte: {selection.date_label.strip()}.")
    if selection.extra_context.strip():
        parts.append(f"Contexto adicional: {selection.extra_context.strip()}")
    context = " ".join(parts)
    base = (base_prompt or "").strip()
    if base and context:
        return base + "\n\nContexto específico de esta grabación:\n" + context
    return base or context


def _profile_settings(profile: str) -> tuple[str, bool, str]:
    if profile == PROFILE_DIARIZATION:
        return STT_ENGINE_LOCAL_FAST_WHISPER, True, "large-v2"
    if profile == PROFILE_LOCAL:
        return STT_ENGINE_LOCAL_FAST_WHISPER, False, "large-v3"
    if profile == PROFILE_OPENAI:
        return STT_ENGINE_OPENAI, False, ""
    raise ValueError(f"Perfil de transcripción desconocido: {profile}")


def build_config(selection: WorkflowSelection) -> TranscriberConfig:
    input_dir, selected_files = resolve_input(selection.input_path)
    stt_engine, use_diarization, whisperx_model = _profile_settings(selection.profile)
    requested_clean = bool(selection.do_clean)
    requested_summary = bool(selection.do_summary)
    text_engine = selection.text_engine if (requested_clean or requested_summary) else TEXT_NONE
    if text_engine == TEXT_NONE:
        requested_clean = False
        requested_summary = False

    cfg = TranscriberConfig(
        api_key=selection.openai_api_key.strip(),
        input_dir=input_dir,
        output_dir=Path(selection.output_dir),
        work_dir=default_work_dir(),
        selected_files=selected_files,
        stt_engine=stt_engine,
        stt_model=selection.openai_stt_model.strip() or "gpt-4o-mini-transcribe",
        local_model_name=(selection.local_fast_model.strip() or "large-v3"),
        local_device=(selection.local_device.strip() or "auto"),
        chat_model=(selection.compatible_model.strip() if text_engine == TEXT_COMPATIBLE else selection.openai_chat_model.strip()) or "gpt-4.1-mini",
        text_engine=text_engine,
        ollama_url=selection.ollama_url.strip() or DEFAULT_OLLAMA_URL,
        ollama_model=selection.ollama_model.strip(),
        compatible_api_url=selection.compatible_api_url.strip() or DEFAULT_OPENAI_COMPATIBLE_URL,
        compatible_api_key=selection.compatible_api_key.strip(),
        diarization_runtime_python=selection.diarization_runtime_python.strip(),
        diarization_worker_executable=selection.diarization_worker_executable.strip(),
        language=selection.language or "es",
        use_ffmpeg=True,
        audio_enhancement=bool(selection.audio_enhancement),
        whisper_prompt=build_context_prompt(selection.whisper_prompt, selection),
        clean_prompt=selection.clean_prompt.strip(),
        summary_prompt=selection.summary_prompt.strip(),
        do_clean=requested_clean,
        do_summary=requested_summary,
        use_diarization=use_diarization,
        whisperx_model=whisperx_model or "large-v2",
        diarization_device="cuda",
        target_speakers=max(0, int(selection.target_speakers or 0)),
        interviewer_name=selection.interviewer.strip(),
        interviewee_name=selection.interviewee.strip(),
        project=selection.project.strip(),
        place=selection.place.strip(),
        date_label=selection.date_label.strip(),
        extra_context=selection.extra_context.strip(),
        export_docx_raw=bool(selection.export_docx_raw),
        export_docx_atlas=bool(selection.export_docx_atlas),
        export_rtf_raw=bool(selection.export_rtf_raw),
        export_rtf_atlas=bool(selection.export_rtf_atlas),
        atlas_include_turn_numbers=bool(selection.atlas_turn_numbers),
        atlas_include_timecodes=bool(selection.atlas_timecodes),
        atlas_include_speaker_id=bool(selection.atlas_speaker_id),
        atlas_bold_speaker_prefix=bool(selection.atlas_bold_speaker),
    )
    return cfg


def assess_privacy(selection: WorkflowSelection) -> PrivacyAssessment:
    audio_external = selection.profile == PROFILE_OPENAI
    text_requested = bool(selection.do_clean or selection.do_summary)
    text_external = text_requested and selection.text_engine in {TEXT_OPENAI, TEXT_COMPATIBLE}
    if text_requested and selection.text_engine == TEXT_OLLAMA:
        text_external = endpoint_scope(selection.ollama_url) != "local"
    details: list[str] = []
    if audio_external:
        details.append("La grabación será enviada al proveedor de transcripción configurado.")
    else:
        details.append("La transcripción se ejecutará en este equipo.")
    if selection.do_clean or selection.do_summary:
        if text_external:
            details.append("El texto de la entrevista será enviado a un proveedor externo para limpieza/resumen.")
        elif selection.text_engine == TEXT_OLLAMA:
            details.append("La limpieza/resumen usará Ollama en este equipo.")
        else:
            details.append("No se enviará texto a una API para limpieza/resumen.")
    else:
        details.append("No se ejecutará limpieza ni resumen automático.")
    if audio_external or text_external:
        title = "Procesamiento híbrido / externo"
    else:
        title = "Procesamiento local-first"
    return PrivacyAssessment(audio_external, text_external, title, tuple(details))


def recommended_profile(report) -> tuple[str, str]:
    """Recomendación explicativa. Nunca modifica una configuración por sí sola."""
    ffmpeg_ok = bool(getattr(report, "ffmpeg", {}).get("available"))
    worker = getattr(report, "diarization_worker", {}) or {}
    whisperx = bool(getattr(report, "whisperx_version", "")) or bool(worker.get("available"))
    faster = bool(getattr(report, "faster_whisper_version", ""))
    gpus = getattr(report, "gpus", []) or []
    cuda = bool(getattr(report, "torch_cuda_available", False)) or bool(getattr(report, "ctranslate2_cuda_devices", 0))
    if ffmpeg_ok and whisperx and gpus and cuda:
        return PROFILE_DIARIZATION, "WhisperX + hablantes: el equipo parece apto para el baseline de máxima fidelidad."
    if faster:
        return PROFILE_LOCAL, "faster-whisper local: compatible sin depender de una API externa."
    return PROFILE_DIARIZATION, "Se mantiene el perfil local de calidad, pero conviene revisar Diagnóstico antes de iniciar."


def recommended_text_model(report) -> str:
    recommendations = getattr(report, "recommendations", {}) or {}
    return str(recommendations.get("text_primary") or "qwen3:4b")


def snapshot_directory(directory: Path) -> set[Path]:
    directory = Path(directory)
    if not directory.exists():
        return set()
    try:
        return {p.resolve() for p in directory.iterdir() if p.is_file()}
    except Exception:
        return set()


def new_output_files(directory: Path, before: Iterable[Path]) -> list[Path]:
    before_set = {Path(p).resolve() for p in before}
    directory = Path(directory)
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and p.resolve() not in before_set]
    return sorted(files, key=lambda p: (p.suffix.lower(), p.name.lower()))


def find_primary_transcript(files: Iterable[Path]) -> Path | None:
    candidates = [Path(p) for p in files]
    raw = [p for p in candidates if p.name.lower().endswith(".raw.txt")]
    if raw:
        return sorted(raw)[-1]
    txt = [p for p in candidates if p.suffix.lower() == ".txt" and ".resumen." not in p.name.lower() and ".clean." not in p.name.lower()]
    return sorted(txt)[-1] if txt else None
