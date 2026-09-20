"""Lógica de onboarding de Jerónimo Abya Yala.

Esta capa NO almacena secretos. El estado persistente sólo registra si el asistente
se completó y preferencias no sensibles. Las credenciales siguen en keyring,
variables de entorno o memoria de la sesión.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credential_store import credential_source, resolve_secret
from model_manager import ASR_MODELS, is_asr_model_cached, ollama_model_present
from diarization_runtime_manager import (
    runtime_stack_validation_status,
    runtime_status as diarization_runtime_status,
)
from portable_runtime import is_frozen
from pyannote_model_manager import local_model_status as pyannote_local_model_status
from local_text_runtime import ollama_executable

STATE_SCHEMA = 5
HF_SIGNUP_URL = "https://huggingface.co/join"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
HF_PYANNOTE_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
HF_PYANNOTE_CONFIG_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml"
OPENAI_API_KEYS_URL = "https://platform.openai.com/api-keys"


@dataclass
class OnboardingState:
    schema: int = STATE_SCHEMA
    completed: bool = False
    completed_at: str = ""
    privacy_acknowledged: bool = False
    preferred_text_mode: str = "local"
    last_step: str = "intro"
    setup_mode: str = ""


@dataclass(frozen=True)
class StartupConfigurationDecision:
    mode: str  # none | full | repair
    reasons: tuple[str, ...] = field(default_factory=tuple)
    stored_schema: int = 0

    @property
    def required(self) -> bool:
        return self.mode != "none"

    @property
    def repair(self) -> bool:
        return self.mode == "repair"


@dataclass(frozen=True)
class ReadinessItem:
    key: str
    label: str
    status: str  # ok | warning | missing | optional
    detail: str
    required_for: str = ""


@dataclass(frozen=True)
class ReadinessReport:
    items: tuple[ReadinessItem, ...] = field(default_factory=tuple)

    @property
    def blocking(self) -> tuple[ReadinessItem, ...]:
        return tuple(i for i in self.items if i.status == "missing" and i.required_for)

    @property
    def warnings(self) -> tuple[ReadinessItem, ...]:
        return tuple(i for i in self.items if i.status == "warning")


def settings_dir() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "JeronimoAbyaYala"
    root = os.environ.get("XDG_CONFIG_HOME")
    if root:
        return Path(root) / "JeronimoAbyaYala"
    return Path.home() / ".config" / "JeronimoAbyaYala"


def state_path() -> Path:
    override = str(os.environ.get("JERONIMO_STATE_PATH", "") or "").strip()
    return Path(override) if override else settings_dir() / "settings.json"


def _read_state_payload(path: Path | None = None) -> dict[str, Any]:
    target = Path(path or state_path())
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def startup_technical_issues() -> tuple[str, ...]:
    """Comprueba sólo requisitos locales que pueden invalidar una instalación.

    No usa red ni descarga nada. En desarrollo fuente no fuerza el onboarding:
    esta política protege el release compilado, cuyo runtime pesado vive fuera
    del EXE y puede faltar aunque ``settings.json`` diga ``completed=true``.
    """
    if not is_frozen():
        return ()

    issues: list[str] = []
    try:
        runtime = diarization_runtime_status()
        if runtime.required and not runtime.ready:
            issues.append("Falta o es incompatible el runtime aislado de WhisperX/PyTorch.")
        elif runtime.required:
            stack = runtime_stack_validation_status()
            if not bool(stack.get("ok")):
                issues.append("El runtime aislado aún no tiene validación profunda WhisperX/CTranslate2 para esta versión.")
    except Exception as exc:
        issues.append(f"No se pudo validar el runtime aislado: {exc}")

    try:
        baseline = next((spec for spec in ASR_MODELS if spec.model_id == "large-v2"), None)
        if baseline is None or not is_asr_model_cached(baseline):
            issues.append("Falta el modelo baseline Whisper large-v2.")
    except Exception as exc:
        issues.append(f"No se pudo comprobar Whisper large-v2: {exc}")

    try:
        if not bool(pyannote_local_model_status(verify_hashes=False).get("ready")):
            issues.append("Falta Community-1 local validado para separación de hablantes.")
    except Exception as exc:
        issues.append(f"No se pudo comprobar Community-1 local: {exc}")

    try:
        if not ollama_executable().is_file():
            issues.append("Falta el runtime privado de Ollama para análisis y resumen local.")
    except Exception as exc:
        issues.append(f"No se pudo comprobar el runtime privado de Ollama: {exc}")

    return tuple(issues)


def startup_configuration_decision(path: Path | None = None) -> StartupConfigurationDecision:
    """Decide si abrir normal, primer onboarding o reparación técnica.

    Un estado viejo completado NO se convierte en un primer inicio: conserva el
    hecho de que la persona ya pasó por privacidad/bienvenida, pero exige una
    reparación automática antes de entrar si cambió el contrato de instalación.
    Un estado actual también se repara si faltan componentes locales esenciales.
    """
    raw = _read_state_payload(path)
    if not raw:
        return StartupConfigurationDecision("full", ("No existe una configuración inicial válida.",), 0)

    try:
        stored_schema = int(raw.get("schema", 0) or 0)
    except Exception:
        stored_schema = 0
    was_completed = bool(raw.get("completed", False))
    privacy_ack = bool(raw.get("privacy_acknowledged", False))
    raw_setup_mode = str(raw.get("setup_mode", "") or "").strip().lower()
    current = load_state(path)

    if current.completed:
        # Sólo el modo Automático promete un baseline local completo. El modo
        # Avanzado puede elegir deliberadamente APIs externas o no instalar la
        # pila pesada, por lo que no se le fuerza una reparación local.
        if str(current.setup_mode or "").strip().lower() == "automatic":
            issues = startup_technical_issues()
            if issues:
                return StartupConfigurationDecision("repair", issues, stored_schema)
        return StartupConfigurationDecision("none", (), stored_schema)

    # Una instalación anterior completa ya pasó por la presentación y privacidad.
    if was_completed and privacy_ack:
        if raw_setup_mode == "advanced":
            # Migración metadata-only: no instala nada ni cambia las elecciones
            # avanzadas. Evita obligar a una persona que eligió APIs/manual a
            # descargar el baseline local por un simple cambio de schema.
            migrated = OnboardingState(
                schema=STATE_SCHEMA,
                completed=True,
                completed_at=str(raw.get("completed_at", "") or ""),
                privacy_acknowledged=True,
                preferred_text_mode=str(raw.get("preferred_text_mode", "local") or "local"),
                last_step="done",
                setup_mode="advanced",
            )
            save_state(migrated, path)
            return StartupConfigurationDecision("none", (), STATE_SCHEMA)

        # Automático (o un estado legacy sin modo explícito) sí prometía la pila
        # local recomendada: al cambiar el contrato técnico debe repararse antes
        # de entrar, pero sin repetir bienvenida ni privacidad.
        reasons = [
            f"La configuración pertenece al esquema {stored_schema}; esta versión requiere el esquema {STATE_SCHEMA}."
        ]
        reasons.extend(startup_technical_issues())
        return StartupConfigurationDecision("repair", tuple(reasons), stored_schema)

    return StartupConfigurationDecision("full", ("La configuración inicial no está completa para esta versión.",), stored_schema)


def load_state(path: Path | None = None) -> OnboardingState:
    target = Path(path or state_path())
    if not target.exists():
        return OnboardingState()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return OnboardingState()
        stored_schema = int(data.get("schema", 0) or 0)
        # Cambios del flujo de configuración pueden requerir una nueva aceptación.
        # Un estado de una versión anterior no permite saltarse el onboarding actual.
        completed = bool(data.get("completed", False)) if stored_schema == STATE_SCHEMA else False
        return OnboardingState(
            schema=STATE_SCHEMA,
            completed=completed,
            completed_at=str(data.get("completed_at", "") or "") if completed else "",
            privacy_acknowledged=bool(data.get("privacy_acknowledged", False)) if completed else False,
            preferred_text_mode=str(data.get("preferred_text_mode", "local") or "local"),
            last_step=str(data.get("last_step", "intro") or "intro") if completed else "intro",
            setup_mode=str(data.get("setup_mode", "") or "") if completed else "",
        )
    except Exception:
        # Un archivo de preferencias dañado nunca debe impedir abrir Jerónimo.
        return OnboardingState()


def save_state(state: OnboardingState, path: Path | None = None) -> Path:
    target = Path(path or state_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    # Defensa adicional: estos nombres jamás deben aparecer en el archivo de estado.
    forbidden = {"api_key", "token", "secret", "password", "openai_key", "hf_token"}
    if forbidden.intersection(payload):
        raise RuntimeError("El estado de onboarding intentó incluir un campo sensible.")
    fd, tmp_name = tempfile.mkstemp(prefix=".jeronimo_settings_", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass
    return target


def mark_step(step: str, *, path: Path | None = None) -> OnboardingState:
    state = load_state(path)
    state.last_step = str(step or "intro")
    save_state(state, path)
    return state


def complete_onboarding(*, privacy_acknowledged: bool = True, preferred_text_mode: str = "local", setup_mode: str = "", path: Path | None = None) -> OnboardingState:
    state = load_state(path)
    state.completed = True
    state.completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.privacy_acknowledged = bool(privacy_acknowledged)
    state.preferred_text_mode = str(preferred_text_mode or "local")
    state.last_step = "done"
    state.setup_mode = str(setup_mode or state.setup_mode or "")
    save_state(state, path)
    return state


def reset_onboarding(*, path: Path | None = None) -> OnboardingState:
    state = OnboardingState()
    save_state(state, path)
    return state


def _hf_request(url: str, token: str, *, timeout: float = 7.0) -> tuple[bool, str]:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "JeronimoAbyaYala/local-pre-release"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read(128)
        return True, "acceso confirmado"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, "token rechazado o condiciones del modelo aún no aceptadas"
        return False, f"Hugging Face respondió HTTP {exc.code}"
    except Exception as exc:
        return False, f"no se pudo comprobar la conexión: {exc}"


def validate_huggingface(*, explicit_token: str = "", test_model_access: bool = True) -> dict[str, Any]:
    token = resolve_secret("huggingface", explicit_token)
    source = credential_source("huggingface", explicit_token)
    if not token:
        return {
            "ok": False,
            "identity_ok": False,
            "model_access_ok": False,
            "source": source,
            "message": "No hay token de Hugging Face configurado.",
        }

    whoami = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "JeronimoAbyaYala/local-pre-release"},
    )
    try:
        with urllib.request.urlopen(whoami, timeout=7) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        name = str(data.get("name", "") or data.get("fullname", "") or "").strip()
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "identity_ok": False,
            "model_access_ok": False,
            "source": source,
            "message": "El token fue rechazado por Hugging Face." if exc.code in (401, 403) else f"Hugging Face respondió HTTP {exc.code}.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "identity_ok": False,
            "model_access_ok": False,
            "source": source,
            "message": f"No se pudo validar el token por red: {exc}",
        }

    model_ok = True
    model_message = "no comprobado"
    if test_model_access:
        model_ok, model_message = _hf_request(HF_PYANNOTE_CONFIG_URL, token)
    ok = bool(model_ok)
    if ok:
        message = f"Token válido{f' para {name}' if name else ''}; acceso a Community-1 confirmado."
    else:
        message = (
            f"Token válido{f' para {name}' if name else ''}, pero {model_message}. "
            "Abre la página del modelo, inicia sesión y acepta sus condiciones; luego vuelve a probar."
        )
    return {
        "ok": ok,
        "identity_ok": True,
        "model_access_ok": bool(model_ok),
        "source": source,
        "user": name,
        "message": message,
    }


def build_readiness(report: Any, *, ollama_models: list[str] | None = None, recommended_model: str = "") -> ReadinessReport:
    items: list[ReadinessItem] = []
    ffmpeg_ok = bool(getattr(report, "ffmpeg", {}).get("available"))
    items.append(ReadinessItem(
        "ffmpeg", "FFmpeg", "ok" if ffmpeg_ok else "missing",
        "Disponible para preparar audio." if ffmpeg_ok else "No está disponible; la diarización no debe iniciarse así.",
        "transcripción con diarización",
    ))

    worker_info = getattr(report, "diarization_worker", {}) or {}
    worker_ok = bool(worker_info.get("available"))
    managed_runtime = diarization_runtime_status()
    whisperx_ok = bool(getattr(report, "whisperx_version", "")) or worker_ok or managed_runtime.ready
    if whisperx_ok:
        wx_status = "ok"
        wx_detail = (
            "Runtime local de diarización disponible."
            if (worker_ok or managed_runtime.ready)
            else f"Versión {getattr(report, 'whisperx_version', '')}."
        )
    elif managed_runtime.required:
        wx_status = "warning"
        wx_detail = "Runtime pesado aún no descargado; el modo Automático lo instala dentro de Jerónimo sólo si se necesita."
    else:
        wx_status = "missing"
        wx_detail = "No se detectó WhisperX ni runtime local de diarización."
    items.append(ReadinessItem(
        "whisperx", "WhisperX / hablantes", wx_status, wx_detail,
        "perfil Calidad + hablantes" if wx_status == "missing" else "",
    ))

    writable = bool(getattr(report, "disk", {}).get("writable"))
    items.append(ReadinessItem(
        "disk", "Carpeta de trabajo", "ok" if writable else "missing",
        "Escribible." if writable else "Jerónimo no puede escribir en la carpeta de trabajo.",
        "cualquier procesamiento",
    ))

    ollama = getattr(report, "ollama", {}) or {}
    ollama_ok = bool(ollama.get("available"))
    managed = bool(ollama.get("managed_by_jeronimo"))
    runtime_installed = bool(ollama.get("runtime_installed"))
    models = set(ollama_models or [str(x.get("name", "")) for x in ollama.get("models", [])])
    model_installed = bool(recommended_model and ollama_model_present(models, recommended_model))
    if not ollama_ok and managed:
        detail = (
            "El runtime privado ya está preparado, pero ahora no responde. Jerónimo intentará reiniciarlo automáticamente."
            if runtime_installed else
            "Todavía no está preparado. El modo Automático descargará y configurará el runtime standalone privado sin instalar Ollama globalmente."
        )
        items.append(ReadinessItem("ollama", "Motor de análisis local", "warning", detail))
    elif not ollama_ok:
        items.append(ReadinessItem(
            "ollama", "Motor de análisis configurado", "warning",
            "El endpoint Ollama indicado en modo avanzado no responde. Jerónimo no modificará instalaciones externas."
        ))
    elif recommended_model and not model_installed:
        items.append(ReadinessItem(
            "ollama_model", "Modelo de análisis local", "warning",
            f"El motor responde, pero {recommended_model} todavía no está descargado."
        ))
    else:
        items.append(ReadinessItem(
            "ollama", "Análisis local", "ok",
            f"Motor local disponible{f' con {recommended_model}' if recommended_model else ''}."
        ))

    hf_source = str(getattr(report, "hf_token_source", "missing") or "missing")
    cached = [str(x).lower() for x in (getattr(report, "hf_cached_models", []) or [])]
    pyannote_cached = any("pyannote" in x and ("diarization" in x or "segmentation" in x) for x in cached)
    if pyannote_cached:
        hf_detail = "Se detectó al menos un modelo pyannote en caché local; el token sigue siendo útil para validar/actualizar acceso."
    elif hf_source != "missing":
        hf_detail = "Token configurado; conviene probar acceso al modelo antes de la primera diarización."
    else:
        hf_detail = "Sin token. Sólo es necesario para acceder/descargar los modelos gated de diarización."
    items.append(ReadinessItem(
        "huggingface", "Hugging Face", "ok" if (hf_source != "missing" or pyannote_cached) else "warning",
        hf_detail,
    ))

    cuda = bool(getattr(report, "torch_cuda_available", False)) or bool(getattr(report, "ctranslate2_cuda_devices", 0))
    items.append(ReadinessItem(
        "acceleration", "Aceleración", "ok" if cuda else "warning",
        "CUDA disponible." if cuda else "No se confirmó CUDA; el procesamiento local puede ser considerablemente más lento.",
    ))
    return ReadinessReport(tuple(items))
