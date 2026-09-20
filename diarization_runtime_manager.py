from __future__ import annotations

"""Gestor del runtime pesado de diarización descargable de Jerónimo.

El ejecutable principal permanece liviano. En un release frozen, WhisperX,
PyTorch y pyannote se instalan bajo ``runtime/managed_diarization`` sólo cuando
el usuario elige una configuración que requiere hablantes.

No instala Python ni paquetes en el sistema: usa uv standalone + Python
administrado por uv dentro de la carpeta de Jerónimo.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from typing import Any, Callable

from portable_runtime import app_root, is_frozen
from subprocess_utils import hidden_process_kwargs

ProgressFn = Callable[[str, float | None], None]

RUNTIME_SCHEMA = 1
RUNTIME_PYTHON = "3.10"
RUNTIME_APPROX_GB_CUDA = 5.0
RUNTIME_APPROX_GB_CPU = 2.2

# uv se usa sólo como bootstrap local del runtime. Está fijado y verificado por
# SHA-256 para que una build de Jerónimo no dependa de "latest".
UV_VERSION = "0.12.17"
UV_WINDOWS_X64_URL = (
    "https://releases.astral.sh/github/uv/releases/download/0.12.17/"
    "uv-x86_64-pc-windows-msvc.zip"
)
UV_WINDOWS_X64_SHA256 = "a252121d5b59398fcb137c6ea448176459a44010f33f67e0072305a637119ca7"

TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCHAUDIO_VERSION = "2.8.0"
WHISPERX_VERSION = "3.8.6"
PYANNOTE_VERSION = "4.0.4"
TORCHCODEC_VERSION = "0.7.0"


@dataclass(frozen=True)
class DiarizationRuntimeStatus:
    required: bool
    ready: bool
    mode: str
    root: str
    python: str
    worker_script: str
    variant: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def managed_runtime_root() -> Path:
    return app_root() / "runtime" / "managed_diarization"


def managed_python() -> Path:
    if os.name == "nt":
        return managed_runtime_root() / "venv" / "Scripts" / "python.exe"
    return managed_runtime_root() / "venv" / "bin" / "python"


def runtime_source_root() -> Path:
    return app_root() / "runtime_source"


def worker_script_path() -> Path:
    return runtime_source_root() / "workers" / "diarization_worker.py"


def _marker_path() -> Path:
    return managed_runtime_root() / "runtime.json"


def _emit(progress: ProgressFn | None, message: str, fraction: float | None = None) -> None:
    if progress:
        progress(message, fraction)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().lower()


def _configure_environment(python: Path | None = None) -> None:
    py = python or managed_python()
    script = worker_script_path()
    if py.is_file() and script.is_file():
        os.environ["JERONIMO_DIARIZATION_PYTHON"] = str(py)
        os.environ["JERONIMO_DIARIZATION_WORKER_SCRIPT"] = str(script)


def runtime_status() -> DiarizationRuntimeStatus:
    # En desarrollo local se conserva exactamente el runtime integrado probado.
    if not is_frozen():
        return DiarizationRuntimeStatus(
            required=False,
            ready=True,
            mode="integrated-development",
            root=str(Path(sys.executable).resolve().parent),
            python=sys.executable,
            worker_script=str(Path(__file__).resolve().parent / "workers" / "diarization_worker.py"),
            message="Entorno fuente: se conserva el runtime local probado.",
        )

    root = managed_runtime_root()
    py = managed_python()
    script = worker_script_path()
    marker: dict[str, Any] = {}
    try:
        if _marker_path().is_file():
            marker = json.loads(_marker_path().read_text(encoding="utf-8"))
    except Exception:
        marker = {}
    ready = bool(
        py.is_file()
        and script.is_file()
        and int(marker.get("schema", 0) or 0) == RUNTIME_SCHEMA
        and marker.get("torch") == TORCH_VERSION
        and marker.get("whisperx") == WHISPERX_VERSION
        and marker.get("pyannote") == PYANNOTE_VERSION
    )
    if ready:
        _configure_environment(py)
    return DiarizationRuntimeStatus(
        required=True,
        ready=ready,
        mode="managed-download" if ready else "not-installed",
        root=str(root),
        python=str(py),
        worker_script=str(script),
        variant=str(marker.get("variant") or ""),
        message=("Runtime local descargable listo." if ready else "Runtime de diarización todavía no descargado."),
    )


def _download(url: str, dest: Path, expected_sha256: str, progress: ProgressFn | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Jeronimo-Abya-Yala/0.9.3"})
    with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            frac = (done / total) if total else None
            _emit(progress, "Descargando bootstrap del runtime local…", frac)
    actual = _sha256(dest)
    if actual != expected_sha256.lower():
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            "La descarga del bootstrap local no coincide con el SHA-256 fijado. "
            "Se canceló la instalación para no ejecutar un binario no verificado."
        )


def _ensure_uv(progress: ProgressFn | None = None) -> Path:
    if os.name != "nt":
        raise RuntimeError("El instalador administrado de diarización está preparado actualmente para Windows x64.")
    tool_dir = managed_runtime_root() / "tools" / "uv"
    uv_exe = tool_dir / "uv.exe"
    if uv_exe.is_file():
        return uv_exe
    tool_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jeronimo_uv_") as td:
        archive = Path(td) / "uv.zip"
        _download(UV_WINDOWS_X64_URL, archive, UV_WINDOWS_X64_SHA256, progress)
        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            candidate = next((name for name in names if Path(name).name.lower() == "uv.exe"), "")
            if not candidate:
                raise RuntimeError("El paquete verificado de uv no contiene uv.exe.")
            with zf.open(candidate) as src, uv_exe.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return uv_exe


def _run(command: list[str], *, env: dict[str, str], progress: ProgressFn | None, label: str) -> None:
    _emit(progress, label, None)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        **hidden_process_kwargs(),
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.strip()
        if line:
            tail.append(line)
            if len(tail) > 25:
                tail.pop(0)
    code = proc.wait()
    if code != 0:
        detail = "\n".join(tail[-8:])
        raise RuntimeError(f"{label} falló con código {code}.\n{detail}".strip())


def _runtime_env() -> dict[str, str]:
    root = managed_runtime_root()
    env = dict(os.environ)
    env["UV_PYTHON_INSTALL_DIR"] = str(root / "python")
    env["UV_PYTHON_BIN_DIR"] = str(root / "python-bin")
    env["UV_CACHE_DIR"] = str(root / "cache")
    env["UV_PYTHON_NO_REGISTRY"] = "1"
    env["UV_NO_PROGRESS"] = "1"
    return env


def _probe_runtime(
    python: Path,
    *,
    model_path: str | Path | None = None,
    device: str = "cuda",
    timeout: int = 90,
) -> dict[str, Any]:
    script = worker_script_path()
    if not script.is_file():
        raise RuntimeError("El paquete base no contiene el worker fuente de diarización.")
    env = dict(os.environ)
    env["JERONIMO_PORTABLE_ROOT"] = str(app_root())
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if model_path is None:
        command = [str(python), str(script), "--probe"]
        operation_label = "autodiagnóstico básico"
    else:
        command = [str(python), str(script), "--probe-stack", str(model_path), str(device)]
        operation_label = "validación profunda WhisperX/CTranslate2"
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
        **hidden_process_kwargs(),
    )
    payload: dict[str, Any] = {}
    for raw in reversed((proc.stdout or "").splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if proc.returncode != 0 or payload.get("type") != "result":
        protocol_message = str(payload.get("message") or "").strip()
        aux = (proc.stderr or "").strip()
        if not aux:
            aux = "\n".join(
                line for line in (proc.stdout or "").splitlines()
                if line.strip() and not line.lstrip().startswith("{")
            ).strip()
        detail_parts = [part for part in (protocol_message, aux[-1800:] if aux else "") if part]
        detail = "\n".join(detail_parts) or "sin detalle adicional"
        raise RuntimeError(
            f"El runtime de diarización no superó {operation_label}.\n{detail}"
        )
    return payload


def runtime_stack_validation_status() -> dict[str, Any]:
    """Estado persistido de la validación profunda del runtime aislado.

    ``runtime_status().ready`` confirma que las versiones fijadas y el Python
    administrado existen. Esta segunda marca confirma que esa misma instalación
    llegó a cargar realmente WhisperX/Faster-Whisper/CTranslate2 con el modelo
    baseline. Se mantiene separada porque la validación profunda sólo puede
    ejecutarse después de descargar ``large-v2``.
    """
    status = runtime_status()
    if not status.required:
        return {"required": False, "ok": True, "reason": "development"}
    if not status.ready:
        return {"required": True, "ok": False, "reason": "runtime-not-ready"}
    try:
        marker = json.loads(_marker_path().read_text(encoding="utf-8")) if _marker_path().is_file() else {}
    except Exception:
        marker = {}
    validation = marker.get("stack_validation") if isinstance(marker, dict) else {}
    if not isinstance(validation, dict):
        validation = {}
    ok = bool(validation.get("ok") and validation.get("whisperx_model_load"))
    return {
        "required": True,
        "ok": ok,
        "reason": "validated" if ok else "deep-validation-missing",
        "device": str(validation.get("device") or ""),
        "compute_type": str(validation.get("compute_type") or ""),
        "cuda_available": bool(validation.get("cuda_available")),
        "ctranslate2_cuda_devices": validation.get("ctranslate2_cuda_devices"),
        "whisperx_model_load": bool(validation.get("whisperx_model_load")),
    }


def probe_diarization_runtime() -> dict[str, Any]:
    """Probe liviano del runtime administrado (Torch + CTranslate2, sin modelo)."""
    status = runtime_status()
    if not status.required:
        return {"type": "result", "operation": "probe", "skipped": True, "reason": "development"}
    if not status.ready:
        return {"type": "result", "operation": "probe", "skipped": True, "reason": "not-installed"}
    return _probe_runtime(managed_python())


def validate_diarization_stack(
    model_path: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Valida la misma inicialización de WhisperX usada al comenzar un trabajo.

    Sólo se ejecuta en el release frozen y exige un snapshot ASR ya descargado;
    no instala paquetes ni procesa audio. WhisperX puede inicializar su caché VAD
    exactamente igual que en el primer uso real.
    """
    status = runtime_status()
    if not status.required:
        return {"type": "result", "operation": "probe-stack", "skipped": True, "reason": "development"}
    if not status.ready:
        raise RuntimeError("El runtime aislado de diarización no está listo para su validación profunda.")
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError("No se encontró el snapshot local de Whisper large-v2 para validar el runtime.")
    result = _probe_runtime(managed_python(), model_path=path, device=device, timeout=240)
    try:
        marker = json.loads(_marker_path().read_text(encoding="utf-8")) if _marker_path().is_file() else {}
        marker["stack_validation"] = {
            "ok": True,
            "operation": str(result.get("operation") or "probe-stack"),
            "device": str(result.get("device") or device),
            "compute_type": str(result.get("compute_type") or ""),
            "cuda_available": bool(result.get("cuda_available")),
            "ctranslate2_cuda_devices": result.get("ctranslate2_cuda_devices"),
            "whisperx_model_load": bool(result.get("whisperx_model_load")),
        }
        _marker_path().write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # El marker de validación es diagnóstico adicional; jamás debe convertir
        # una carga correcta del modelo en un fallo de instalación.
        pass
    return result


def ensure_diarization_runtime(
    *,
    report: Any | None = None,
    progress: ProgressFn | None = None,
) -> DiarizationRuntimeStatus:
    """Asegura el runtime pesado sólo en releases frozen.

    En desarrollo no modifica el entorno probado. En portable descarga uv
    verificado, un Python administrado y dependencias fijadas dentro de Jerónimo.
    """
    status = runtime_status()
    if status.ready:
        return status
    if not is_frozen():
        return status

    if os.name != "nt" or (os.environ.get("PROCESSOR_ARCHITECTURE", "").lower() not in {"amd64", "x86_64", ""}):
        raise RuntimeError("La descarga automática del runtime está preparada para Windows x64.")

    root = managed_runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    uv = _ensure_uv(progress)
    env = _runtime_env()
    venv = root / "venv"

    _run(
        [str(uv), "venv", "--no-config", "--managed-python", "--python", RUNTIME_PYTHON, str(venv)],
        env=env,
        progress=progress,
        label="Preparando Python local aislado para diarización…",
    )
    py = managed_python()
    if not py.is_file():
        raise RuntimeError("uv terminó sin crear el Python aislado esperado.")

    has_nvidia = bool(getattr(report, "gpus", None)) if report is not None else False
    variant = "cuda128" if has_nvidia else "cpu"
    torch_index = (
        "https://download.pytorch.org/whl/cu128"
        if has_nvidia
        else "https://download.pytorch.org/whl/cpu"
    )
    _run(
        [
            str(uv), "pip", "install", "--no-config", "--link-mode", "copy", "--python", str(py),
            f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}", f"torchaudio=={TORCHAUDIO_VERSION}",
            "--index-url", torch_index,
        ],
        env=env,
        progress=progress,
        label=("Descargando PyTorch CUDA 12.8…" if has_nvidia else "Descargando PyTorch CPU…"),
    )
    _run(
        [
            str(uv), "pip", "install", "--no-config", "--link-mode", "copy", "--python", str(py),
            f"whisperx=={WHISPERX_VERSION}", f"pyannote-audio=={PYANNOTE_VERSION}", f"torchcodec=={TORCHCODEC_VERSION}",
            "python-dotenv==1.2.3", "keyring==25.7.0",
        ],
        env=env,
        progress=progress,
        label="Descargando WhisperX y pyannote para hablantes…",
    )

    payload = _probe_runtime(py)
    marker = {
        "schema": RUNTIME_SCHEMA,
        "variant": variant,
        "python": RUNTIME_PYTHON,
        "torch": TORCH_VERSION,
        "whisperx": WHISPERX_VERSION,
        "pyannote": PYANNOTE_VERSION,
        "torchcodec": TORCHCODEC_VERSION,
        "cuda_available": bool(payload.get("cuda_available")),
        "ctranslate2_cuda_devices": payload.get("ctranslate2_cuda_devices"),
    }
    _marker_path().write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")

    # uv instaló por copia; el cache ya no es necesario y puede duplicar varios GB.
    shutil.rmtree(root / "cache", ignore_errors=True)
    _configure_environment(py)
    final = runtime_status()
    if not final.ready:
        raise RuntimeError("El runtime terminó de instalarse pero no quedó marcado como listo.")
    _emit(progress, "Runtime local de transcripción con hablantes listo.", 1.0)
    return final
