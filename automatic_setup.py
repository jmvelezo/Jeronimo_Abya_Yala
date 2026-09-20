"""Configuración automática local-first de Jerónimo Abya Yala.

El modo automático elige una configuración compatible según el diagnóstico,
prepara componentes locales, descarga los modelos faltantes con fuentes verificadas
y evita configurar APIs externas.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from threading import Event
from typing import Any, Callable

from credential_store import get_secure_secret, set_secure_secret
from model_manager import (
    ASR_MODELS,
    TEXT_MODELS,
    download_faster_whisper_model,
    resolve_cached_faster_whisper_model,
    is_asr_model_cached,
    ollama_inventory,
    pull_ollama_model,
    benchmark_ollama_model,
    ollama_model_present,
    delete_ollama_model,
)
from team_credentials import get_embedded_team_hf_token
from local_text_runtime import (
    PRIVATE_OLLAMA_URL,
    ensure_private_ollama,
    is_private_ollama_url,
)
from diarization_runtime_manager import (
    RUNTIME_APPROX_GB_CUDA,
    RUNTIME_APPROX_GB_CPU,
    ensure_diarization_runtime,
    validate_diarization_stack,
    runtime_status as diarization_runtime_status,
)
from pyannote_model_manager import (
    ensure_pyannote_model,
    install_from_huggingface as install_pyannote_from_huggingface,
    load_mirror_release,
    local_model_status as pyannote_local_model_status,
)

ProgressFn = Callable[[str, float | None], None]
PYANNOTE_REPO_ID = "pyannote/speaker-diarization-community-1"
BASELINE_ASR_ID = "large-v2"
DEFAULT_OLLAMA_URL = PRIVATE_OLLAMA_URL


@dataclass(frozen=True)
class AutomaticPlan:
    text_model: str
    text_model_size_gb: float
    asr_model: str
    asr_model_size_gb: float
    cuda: bool
    needs_ollama: bool
    needs_text_model: bool
    needs_diarization_runtime: bool
    diarization_runtime_size_gb: float
    needs_asr_model: bool
    needs_pyannote_model: bool
    free_disk_gb: float
    warnings: tuple[str, ...]


def _find_text_spec(model_id: str):
    return next((item for item in TEXT_MODELS if item.model_id == model_id), None)


def _baseline_asr_spec():
    return next(item for item in ASR_MODELS if item.model_id == BASELINE_ASR_ID)



def _text_fallback_chain(primary: str) -> list[str]:
    """Orden de degradación por memoria; nunca baja sólo porque un modelo sea lento."""
    ordered = ["qwen3:30b", "qwen3:14b", "qwen3:8b", "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b"]
    if primary not in ordered:
        return [primary]
    return ordered[ordered.index(primary):]


def _is_model_resource_error(exc: Exception) -> bool:
    text = str(exc).lower()
    signals = (
        "out of memory", "not enough memory", "insufficient memory",
        "requires more system memory", "cannot allocate", "failed to allocate",
        "cuda out of memory", "cublas_status_alloc_failed",
    )
    return any(signal in text for signal in signals)


def _pyannote_cached(report: Any) -> bool:
    # El release 1.0-prelocal estandariza Community-1 en runtime/models para
    # poder cargarlo sin red ni token durante el uso. La caché histórica de HF
    # puede seguir existiendo, pero ya no se toma como instalación canónica.
    try:
        return bool(pyannote_local_model_status(verify_hashes=False).get("ready"))
    except Exception:
        return False


def build_automatic_plan(report: Any, ollama_models: list[str] | None = None) -> AutomaticPlan:
    recommendations = getattr(report, "recommendations", {}) or {}
    text_model = str(recommendations.get("text_primary") or "qwen3:4b")
    text_spec = _find_text_spec(text_model) or _find_text_spec("qwen3:4b")
    asr_spec = _baseline_asr_spec()

    models = set(ollama_models or [])
    has_text = ollama_model_present(models, text_model)
    ollama_state = (getattr(report, "ollama", {}) or {})
    ollama_ok = bool(ollama_state.get("available"))
    runtime_installed = bool(ollama_state.get("runtime_installed"))
    cuda = bool(getattr(report, "torch_cuda_available", False)) or bool(getattr(report, "ctranslate2_cuda_devices", 0))
    free_bytes = int((getattr(report, "disk", {}) or {}).get("free_bytes", 0) or 0)
    free_gb = free_bytes / (1024 ** 3) if free_bytes else 0.0

    runtime = diarization_runtime_status()
    runtime_size_gb = RUNTIME_APPROX_GB_CUDA if bool(getattr(report, "gpus", None)) else RUNTIME_APPROX_GB_CPU

    warnings: list[str] = []
    try:
        mirror_configured = load_mirror_release() is not None
    except Exception:
        mirror_configured = False
    if not mirror_configured and not _pyannote_cached(report):
        warnings.append("El mirror verificado de Community-1 todavía no está configurado; se usará Hugging Face oficial como contingencia.")
    if not cuda:
        warnings.append("No se confirmó CUDA; la configuración seguirá siendo local pero puede ser lenta.")
    required_gb = (text_spec.approx_size_gb if text_spec else 0) + asr_spec.approx_size_gb + (runtime_size_gb if not runtime.ready else 0) + 4
    if free_gb and free_gb < required_gb:
        warnings.append("El espacio libre puede ser insuficiente para el runtime y los modelos recomendados.")
    if not bool((getattr(report, "disk", {}) or {}).get("writable", False)):
        warnings.append("La carpeta de trabajo no es escribible.")

    return AutomaticPlan(
        text_model=text_model,
        text_model_size_gb=float(text_spec.approx_size_gb if text_spec else 0),
        asr_model=BASELINE_ASR_ID,
        asr_model_size_gb=float(asr_spec.approx_size_gb),
        cuda=cuda,
        needs_ollama=not (ollama_ok or runtime_installed),
        needs_text_model=not has_text,
        needs_diarization_runtime=bool(runtime.required and not runtime.ready),
        diarization_runtime_size_gb=float(runtime_size_gb),
        needs_asr_model=not is_asr_model_cached(asr_spec),
        needs_pyannote_model=not _pyannote_cached(report),
        free_disk_gb=free_gb,
        warnings=tuple(warnings),
    )


def bootstrap_team_hf_credential() -> tuple[str, str]:
    """Asegura una credencial HF utilizable sin pedir nada al usuario del equipo.

    Devuelve (token, source). Si ya existe una credencial segura, se respeta.
    Si no existe, migra la credencial interna al almacén seguro del sistema.
    """
    existing = get_secure_secret("huggingface")
    if existing:
        return existing, "keyring"
    embedded = get_embedded_team_hf_token()
    if not embedded:
        raise RuntimeError("La credencial interna de Hugging Face del equipo no está disponible.")
    try:
        set_secure_secret("huggingface", embedded)
        return embedded, "team→keyring"
    except Exception:
        # El setup puede continuar con la credencial en memoria. No se escribe en disco.
        return embedded, "team-memory"


def ensure_ollama(
    base_url: str = DEFAULT_OLLAMA_URL,
    progress: ProgressFn | None = None,
    *,
    report: Any | None = None,
) -> None:
    """Asegura el motor de texto sin instalar Ollama globalmente.

    El endpoint privado de Jerónimo descarga/usa el runtime standalone propio.
    Un endpoint avanzado distinto se considera responsabilidad del usuario y sólo
    se acepta si ya responde; nunca se modifica su instalación.
    """
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    if is_private_ollama_url(base):
        ensure_private_ollama(report=report, progress=progress)
        return
    try:
        import urllib.request
        with urllib.request.urlopen(base + "/api/version", timeout=3) as response:
            if response.status == 200:
                return
    except Exception as exc:
        raise RuntimeError(
            "El endpoint Ollama configurado en modo avanzado no responde. "
            "Jerónimo sólo instala automáticamente su runtime privado local."
        ) from exc


def _download_pyannote(token: str, progress: ProgressFn | None = None) -> Path:
    """Compatibilidad: descarga oficial directa, no usada como ruta primaria."""
    return install_pyannote_from_huggingface(token, progress=progress)


def run_automatic_setup(
    report: Any,
    *,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    progress: ProgressFn | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Instala/configura la selección recomendada por el diagnóstico.

    No configura APIs externas. Audio y texto quedan en el equipo; sólo se usan
    conexiones de red para descargar componentes/modelos.
    """
    inv = ollama_inventory(ollama_url)
    plan = build_automatic_plan(report, list(inv.keys()))

    def emit(message: str, fraction: float | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Configuración automática cancelada.")
        if progress:
            progress(message, fraction)

    pyannote_result: dict[str, Any] = {"source": "local"}
    token_source = "not-needed"

    # Community-1 es pequeño comparado con el runtime CUDA. Se valida primero:
    # así una caída del mirror/HF no hace descargar varios GB antes de descubrirla.
    if plan.needs_pyannote_model:
        emit("Preparando modelo local de separación de hablantes…", 0.03)
        pyannote_result = ensure_pyannote_model(
            progress=lambda msg, frac: emit(
                msg,
                None if frac is None else 0.03 + 0.09 * float(frac),
            ),
            cancel_event=cancel_event,
        )
        token_source = str(pyannote_result.get("source") or "not-needed")
        emit("Community-1 quedó validado para uso local/offline.", 0.12)
    else:
        emit("Community-1 local ya está disponible y validado.", 0.12)

    if plan.needs_diarization_runtime:
        emit("Preparando runtime local de transcripción con hablantes…", 0.14)
        ensure_diarization_runtime(
            report=report,
            progress=lambda msg, frac: emit(
                msg,
                None if frac is None else 0.14 + 0.20 * float(frac),
            ),
        )
    else:
        emit("Runtime local de transcripción con hablantes ya disponible.", 0.34)

    asr_spec = _baseline_asr_spec()
    if plan.needs_asr_model:
        emit("Descargando Whisper large-v2 (baseline de transcripción)…", 0.36)
        asr_path = download_faster_whisper_model(
            asr_spec,
            progress=lambda e: emit(
                str(e.get("status", "Descargando Whisper…")),
                0.50 if e.get("fraction") is None else 0.36 + 0.14 * float(e.get("fraction")),
            ),
        )
    else:
        emit("Whisper large-v2 ya está disponible.", 0.50)
        asr_path = None

    # En la BASE compilada el runtime pesado es externo al EXE. Tras tener el
    # snapshot local, hacemos una carga real de WhisperX/CTranslate2 sin audio:
    # así CUDA/cuBLAS/cuDNN/VAD se comprueban antes del primer trabajo del usuario.
    runtime_now = diarization_runtime_status()
    stack_probe: dict[str, Any] = {"skipped": True, "reason": "integrated-development"}
    if runtime_now.required:
        if asr_path is None:
            asr_path = resolve_cached_faster_whisper_model(asr_spec)
        emit("Validando WhisperX/CTranslate2 en el runtime aislado…", 0.52)
        stack_probe = validate_diarization_stack(
            asr_path,
            device=("cuda" if plan.cuda else "cpu"),
        )
        emit("Runtime de transcripción con hablantes validado con carga real del modelo.", 0.56)

    emit("Comprobando motor de resumen local…", 0.58)
    ensure_ollama(ollama_url, progress=lambda msg, _: emit(msg, 0.66), report=report)

    inv = ollama_inventory(ollama_url)
    selected_model = plan.text_model
    benchmark: dict[str, Any] = {}
    last_resource_error: Exception | None = None

    for candidate_index, candidate in enumerate(_text_fallback_chain(plan.text_model)):
        candidate_present = ollama_model_present(inv, candidate)
        downloaded_here = False
        if not candidate_present:
            label = "modelo recomendado" if candidate_index == 0 else "modelo alternativo compatible"
            emit(f"Descargando {candidate} ({label})…", 0.70 if candidate_index == 0 else None)

            def ollama_progress(event: dict[str, Any]) -> None:
                frac = event.get("fraction")
                mapped = None if frac is None or candidate_index else 0.70 + 0.24 * float(frac)
                emit(str(event.get("status", "Descargando modelo de análisis…")), mapped)

            pull_ollama_model(candidate, ollama_url, progress=ollama_progress, cancel_event=cancel_event)
            downloaded_here = True
        else:
            emit(f"{candidate} ya está disponible.", 0.94 if candidate_index == 0 else None)

        emit(f"Probando {candidate} en este equipo…", 0.96 if candidate_index == 0 else None)
        try:
            benchmark = benchmark_ollama_model(candidate, ollama_url)
            selected_model = candidate
            last_resource_error = None
            break
        except Exception as exc:
            if not _is_model_resource_error(exc) or candidate_index >= len(_text_fallback_chain(plan.text_model)) - 1:
                raise
            last_resource_error = exc
            emit(
                f"{candidate} supera la memoria disponible ahora. Jerónimo probará un modelo menor sin sacrificar el análisis jerárquico…",
                None,
            )
            # Si esta ejecución acaba de descargar un modelo que no puede cargarse,
            # lo retira para no dejar varios GB inútiles. Nunca borra un modelo que
            # ya existía antes de la configuración.
            if downloaded_here:
                try:
                    delete_ollama_model(candidate, ollama_url)
                except Exception:
                    pass

    if last_resource_error is not None and not benchmark:
        raise last_resource_error

    tps = float(benchmark.get("tokens_per_second", 0.0) or 0.0)
    gpu_fraction = benchmark.get("gpu_fraction")
    if gpu_fraction is None:
        placement = "carga de CPU/GPU no informada"
    elif float(gpu_fraction) >= 0.95:
        placement = "modelo prácticamente completo en GPU"
    elif float(gpu_fraction) > 0.05:
        placement = f"carga mixta CPU/GPU ({float(gpu_fraction) * 100:.0f}% en GPU)"
    else:
        placement = "modelo principalmente en CPU/RAM"
    emit(
        f"Motor local verificado: {selected_model} · {tps:.1f} tok/s · {placement}.",
        0.99,
    )

    emit("Configuración automática completada.", 1.0)
    return {
        "ok": True,
        "text_model": selected_model,
        "asr_model": plan.asr_model,
        "cuda": plan.cuda,
        "hf_credential_source": token_source,
        "pyannote_source": str(pyannote_result.get("source") or "local"),
        "pyannote_model": pyannote_local_model_status(verify_hashes=False),
        "diarization_runtime": diarization_runtime_status().to_dict(),
        "local_only": True,
        "ollama_url": ollama_url,
        "text_benchmark": benchmark,
        "diarization_stack_probe": stack_probe,
    }
