from __future__ import annotations

"""Rutas y bootstrap del release portable de Jerónimo.

No almacena datos de entrevistas. Sólo resuelve componentes distribuidos junto al
programa (FFmpeg y workers auxiliares) y los expone al resto de la aplicación.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from subprocess_utils import hidden_process_kwargs


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Raíz visible del portable: carpeta que contiene Jeronimo.exe."""
    override = str(os.environ.get("JERONIMO_PORTABLE_ROOT", "") or "").strip()
    if override:
        return Path(override).resolve()
    if is_frozen():
        here = Path(sys.executable).resolve().parent
        # Workers auxiliares viven en <root>/runtime/{diarization,classic}.
        if here.parent.name.lower() == "runtime" and here.name.lower() in {"diarization", "classic"}:
            return here.parent.parent
        return here
    return Path(__file__).resolve().parent


def runtime_root() -> Path:
    return app_root() / "runtime"


def ffmpeg_bin_dir() -> Path:
    return runtime_root() / "ffmpeg" / "bin"


def bundled_worker_executable() -> Path:
    name = "JeronimoDiarizationWorker.exe" if os.name == "nt" else "JeronimoDiarizationWorker"
    return runtime_root() / "diarization" / name


def managed_diarization_python() -> Path:
    root = runtime_root() / "managed_diarization" / "venv"
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def managed_diarization_worker_script() -> Path:
    return app_root() / "runtime_source" / "workers" / "diarization_worker.py"


def bundled_classic_executable() -> Path:
    name = "JeronimoClassic.exe" if os.name == "nt" else "JeronimoClassic"
    return runtime_root() / "classic" / name


def _prepend_path(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    normalized = {os.path.normcase(os.path.abspath(p)) for p in parts}
    target = os.path.normcase(os.path.abspath(str(directory)))
    if target not in normalized:
        os.environ["PATH"] = str(directory) + (os.pathsep + current if current else "")
    return True


def bootstrap_portable_environment() -> dict[str, Any]:
    """Activa componentes incluidos sin sobrescribir configuración explícita."""
    root = app_root()
    if is_frozen() and not os.environ.get("JERONIMO_PORTABLE_ROOT"):
        os.environ["JERONIMO_PORTABLE_ROOT"] = str(root)
    ffmpeg_ready = _prepend_path(ffmpeg_bin_dir())

    worker = bundled_worker_executable()
    if worker.is_file() and not os.environ.get("JERONIMO_DIARIZATION_WORKER_EXE"):
        os.environ["JERONIMO_DIARIZATION_WORKER_EXE"] = str(worker)

    managed_python = managed_diarization_python()
    managed_script = managed_diarization_worker_script()
    if (
        not worker.is_file()
        and managed_python.is_file()
        and managed_script.is_file()
        and not os.environ.get("JERONIMO_DIARIZATION_PYTHON")
    ):
        os.environ["JERONIMO_DIARIZATION_PYTHON"] = str(managed_python)
        os.environ["JERONIMO_DIARIZATION_WORKER_SCRIPT"] = str(managed_script)

    classic = bundled_classic_executable()
    if classic.is_file() and not os.environ.get("JERONIMO_CLASSIC_EXE"):
        os.environ["JERONIMO_CLASSIC_EXE"] = str(classic)

    pyannote_model = ""
    try:
        from pyannote_model_manager import activate_local_model
        pyannote_model = activate_local_model()
    except Exception:
        pyannote_model = ""

    return {
        "frozen": is_frozen(),
        "app_root": str(app_root()),
        "ffmpeg_bin": str(ffmpeg_bin_dir()),
        "ffmpeg_bin_present": ffmpeg_ready,
        "worker_executable": str(worker),
        "worker_present": worker.is_file(),
        "managed_diarization_python": str(managed_python),
        "managed_diarization_python_present": managed_python.is_file(),
        "managed_diarization_worker_script": str(managed_script),
        "managed_diarization_worker_script_present": managed_script.is_file(),
        "classic_executable": str(classic),
        "classic_present": classic.is_file(),
        "pyannote_local_model": pyannote_model,
        "pyannote_local_model_present": bool(pyannote_model),
    }


def portable_status() -> dict[str, Any]:
    status = bootstrap_portable_environment()
    import shutil
    status.update({
        "ffmpeg": shutil.which("ffmpeg") or "",
        "ffprobe": shutil.which("ffprobe") or "",
        "ffplay": shutil.which("ffplay") or "",
    })
    return status


def probe_diarization_worker(timeout: float = 30.0) -> dict[str, Any]:
    worker = str(os.environ.get("JERONIMO_DIARIZATION_WORKER_EXE", "") or "").strip()
    if not worker:
        worker_path = bundled_worker_executable()
        worker = str(worker_path) if worker_path.is_file() else ""

    managed_python = str(os.environ.get("JERONIMO_DIARIZATION_PYTHON", "") or "").strip()
    managed_script = str(os.environ.get("JERONIMO_DIARIZATION_WORKER_SCRIPT", "") or "").strip()
    if not managed_python:
        py_path = managed_diarization_python()
        script_path = managed_diarization_worker_script()
        if py_path.is_file() and script_path.is_file():
            managed_python, managed_script = str(py_path), str(script_path)

    if worker and Path(worker).is_file():
        command = [worker, "--probe"]
        label = worker
    elif managed_python and managed_script and Path(managed_python).is_file() and Path(managed_script).is_file():
        command = [managed_python, managed_script, "--probe"]
        label = managed_python
    else:
        return {"ok": False, "message": "runtime de diarización no disponible", "worker": worker, "python": managed_python}
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            **hidden_process_kwargs(),
        )
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {}
        return {
            "ok": proc.returncode == 0 and payload.get("type") == "result",
            "returncode": proc.returncode,
            "worker": worker,
            "python": managed_python,
            "command_target": label,
            "payload": payload,
            "stderr": (proc.stderr or "")[:500],
        }
    except Exception as exc:
        return {"ok": False, "worker": worker, "python": managed_python, "message": f"{type(exc).__name__}: {exc}"}

