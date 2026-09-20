from __future__ import annotations

"""Runtime local de IA de texto administrado por Jerónimo Abya Yala.

Jerónimo no requiere una instalación global de Ollama. En Windows descarga una
versión standalone validada, la guarda dentro de los datos propios de la app,
levanta un servidor sólo en loopback y almacena sus modelos en una carpeta
privada. El runtime sólo se descarga cuando una función local de texto lo necesita.
"""

import atexit
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from subprocess_utils import hidden_process_kwargs

# Versión fijada para reproducibilidad. Actualizar sólo después de validarla con
# Jerónimo. La URL apunta a los artefactos oficiales de Ollama/GitHub.
OLLAMA_RUNTIME_VERSION = "0.34.2"
OLLAMA_RELEASE_BASE = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_RUNTIME_VERSION}"
OLLAMA_SHA256_URL = f"{OLLAMA_RELEASE_BASE}/sha256sum.txt"
# Digests oficiales del release fijado, tomados de los metadatos de assets de
# GitHub para v0.34.2. Se mantienen junto a la versión para que la instalación
# siga siendo reproducible incluso si sha256sum.txt no puede leerse o cambia su
# formato. Al actualizar OLLAMA_RUNTIME_VERSION hay que actualizar esta tabla.
PINNED_OLLAMA_SHA256 = {
    "ollama-windows-amd64.zip": "8f3fd071a2a2f9497b562f43502c77c2b701a99d1ee5dfda28da8c786373063b",
    "ollama-windows-arm64.zip": "fc27d07955c44678f2e492af1fef16033e554fa6a92edfdfa7aad3d69677cdb2",
    "ollama-windows-amd64-rocm.zip": "a019f59490a28ab04f65e291716ee2147d92df373aa3978aa9073e1f6cb1ce82",
}
PRIVATE_OLLAMA_HOST = "127.0.0.1"
PRIVATE_OLLAMA_PORT = 11435
PRIVATE_OLLAMA_URL = f"http://{PRIVATE_OLLAMA_HOST}:{PRIVATE_OLLAMA_PORT}"

ProgressFn = Callable[[str, float | None], None]
_PROCESS: subprocess.Popen | None = None


def _data_root() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "JeronimoAbyaYala"
    root = os.environ.get("XDG_DATA_HOME")
    if root:
        return Path(root) / "JeronimoAbyaYala"
    return Path.home() / ".local" / "share" / "JeronimoAbyaYala"


def ollama_runtime_dir() -> Path:
    return _data_root() / "runtime" / "ollama" / OLLAMA_RUNTIME_VERSION


def ollama_models_dir() -> Path:
    return _data_root() / "models" / "ollama"


def ollama_executable() -> Path:
    name = "ollama.exe" if os.name == "nt" else "ollama"
    return ollama_runtime_dir() / name


def is_private_ollama_url(url: str | None) -> bool:
    value = str(url or "").strip().rstrip("/").lower()
    return value in {
        PRIVATE_OLLAMA_URL.lower(),
        f"http://localhost:{PRIVATE_OLLAMA_PORT}",
    }


def _asset_name() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "ollama-windows-arm64.zip"
    return "ollama-windows-amd64.zip"


def _amd_rocm_asset() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return ""
    return "ollama-windows-amd64-rocm.zip"


def _report_has_amd_gpu(report: Any | None) -> bool:
    if report is None:
        return False
    names = [str(item.get("name", "")).lower() for item in (getattr(report, "gpus", []) or [])]
    # system_diagnostics hoy enumera NVIDIA. Este chequeo queda preparado para
    # reportes futuros que incluyan AMD/Radeon sin forzar una descarga extra.
    return any("amd" in name or "radeon" in name for name in names)


def _request_bytes(url: str, timeout: float = 45.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Jeronimo-Abya-Yala/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _release_hashes() -> dict[str, str]:
    # La tabla fijada es la fuente primaria: evita que una caída, redirección o
    # cambio de formato de sha256sum.txt bloquee la instalación. El archivo
    # oficial se usa sólo para completar assets no fijados explícitamente.
    result: dict[str, str] = dict(PINNED_OLLAMA_SHA256)
    try:
        raw = _request_bytes(OLLAMA_SHA256_URL, timeout=30).decode("utf-8", errors="replace")
    except Exception:
        return result
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            name = parts[-1].lstrip("*./")
            result.setdefault(name, parts[0].lower())
    return result


def _download_file(url: str, target: Path, *, expected_sha256: str, progress: ProgressFn | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Jeronimo-Abya-Yala/1.0"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        while True:
            block = response.read(4 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            digest.update(block)
            done += len(block)
            if progress:
                progress(
                    "Descargando motor local de análisis…",
                    (done / total) if total else None,
                )
    actual = digest.hexdigest().lower()
    if expected_sha256 and actual != expected_sha256.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError("La verificación SHA-256 del runtime local de análisis no coincide. Se canceló la instalación.")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    dest_resolved = destination.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(dest_resolved)
            except ValueError as exc:
                raise RuntimeError("El paquete del runtime contiene una ruta insegura.") from exc
        zf.extractall(destination)


def install_private_ollama(*, report: Any | None = None, progress: ProgressFn | None = None) -> Path:
    """Descarga y verifica el Ollama standalone privado de Jerónimo.

    No usa WinGet, no registra Ollama en Windows y no modifica PATH global.
    """
    exe = ollama_executable()
    if exe.is_file():
        return exe
    if os.name != "nt":
        raise RuntimeError("La instalación automática del runtime local está preparada para Windows.")

    if progress:
        progress("Preparando motor privado de análisis local…", None)
    hashes = _release_hashes()
    assets = [_asset_name()]
    if _report_has_amd_gpu(report):
        rocm = _amd_rocm_asset()
        if rocm:
            assets.append(rocm)

    runtime = ollama_runtime_dir()
    runtime.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="jaya_ollama_", dir=str(runtime.parent)))
    try:
        for index, asset in enumerate(assets):
            expected = hashes.get(asset, "")
            if not expected:
                raise RuntimeError(f"No se encontró SHA-256 oficial para {asset}.")
            archive = staging / asset
            base_fraction = index / max(1, len(assets))
            span = 1.0 / max(1, len(assets))

            def mapped(message: str, fraction: float | None) -> None:
                if progress:
                    progress(message, None if fraction is None else base_fraction + span * fraction)

            _download_file(f"{OLLAMA_RELEASE_BASE}/{asset}", archive, expected_sha256=expected, progress=mapped)
            _safe_extract(archive, staging / "payload")

        payload = staging / "payload"
        candidate = payload / ("ollama.exe" if os.name == "nt" else "ollama")
        if not candidate.is_file():
            matches = list(payload.rglob("ollama.exe" if os.name == "nt" else "ollama"))
            if not matches:
                raise RuntimeError("El paquete oficial se descargó, pero no contiene el ejecutable de Ollama esperado.")
        if runtime.exists():
            shutil.rmtree(runtime, ignore_errors=True)
        runtime.mkdir(parents=True, exist_ok=True)
        for item in payload.iterdir():
            shutil.move(str(item), str(runtime / item.name))
        exe = ollama_executable()
        if not exe.is_file():
            # Algunos zips incluyen una carpeta superior; normalizamos sin alterar
            # las librerías relativas que acompañan al ejecutable.
            matches = list(runtime.rglob("ollama.exe"))
            if matches:
                root = matches[0].parent
                normalized = runtime.parent / (runtime.name + "_normalized")
                normalized.mkdir(parents=True, exist_ok=True)
                for item in root.iterdir():
                    shutil.move(str(item), str(normalized / item.name))
                shutil.rmtree(runtime, ignore_errors=True)
                normalized.replace(runtime)
        if not ollama_executable().is_file():
            raise RuntimeError("No se pudo preparar el ejecutable del motor local.")
        return ollama_executable()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    models = ollama_models_dir()
    models.mkdir(parents=True, exist_ok=True)
    env.update({
        "OLLAMA_HOST": f"{PRIVATE_OLLAMA_HOST}:{PRIVATE_OLLAMA_PORT}",
        "OLLAMA_MODELS": str(models),
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })
    return env


def _api_version(timeout: float = 1.5) -> str:
    try:
        with urllib.request.urlopen(PRIVATE_OLLAMA_URL + "/api/version", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        return str(data.get("version", "") or "")
    except Exception:
        return ""


def private_ollama_running() -> bool:
    return bool(_api_version())


def _stop_owned_process() -> None:
    global _PROCESS
    proc = _PROCESS
    _PROCESS = None
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                **hidden_process_kwargs(),
                check=False,
            )
        else:
            proc.terminate()
    except Exception:
        pass


atexit.register(_stop_owned_process)


def start_private_ollama_if_installed(*, wait: bool = True, progress: ProgressFn | None = None) -> bool:
    global _PROCESS
    if private_ollama_running():
        return True
    exe = ollama_executable()
    if not exe.is_file():
        return False
    if progress:
        progress("Iniciando motor privado de análisis local…", None)
    try:
        _PROCESS = subprocess.Popen(
            [str(exe), "serve"],
            cwd=str(exe.parent),
            env=_server_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_process_kwargs(),
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudo iniciar el motor privado de análisis: {exc}") from exc
    if not wait:
        return True
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if private_ollama_running():
            return True
        if _PROCESS is not None and _PROCESS.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError("El motor privado de análisis se inició, pero su API local no respondió a tiempo.")


def repair_private_ollama(*, report: Any | None = None, progress: ProgressFn | None = None) -> dict[str, Any]:
    """Reinstala sólo el runtime privado sin borrar los modelos locales.

    Los modelos viven fuera de ``ollama_runtime_dir()``, por lo que una reparación
    del ejecutable/librerías no obliga a volver a descargar varios GB de modelos.
    """
    if progress:
        progress("Reparando motor privado de análisis local…", None)
    _stop_owned_process()
    runtime = ollama_runtime_dir()
    if runtime.exists():
        try:
            shutil.rmtree(runtime)
        except Exception as exc:
            raise RuntimeError(
                "El runtime privado de análisis está dañado o bloqueado y no pudo reemplazarse. "
                "Cierra otros procesos de Jerónimo/Ollama y vuelve a intentar."
            ) from exc
    install_private_ollama(report=report, progress=progress)
    start_private_ollama_if_installed(wait=True, progress=progress)
    return private_ollama_status(start_if_installed=False)


def ensure_private_ollama(*, report: Any | None = None, progress: ProgressFn | None = None) -> dict[str, Any]:
    """Asegura runtime instalado + servidor local activo y repara una vez si falla."""
    exe = ollama_executable()
    if not exe.is_file():
        install_private_ollama(report=report, progress=progress)
    try:
        start_private_ollama_if_installed(wait=True, progress=progress)
        return private_ollama_status(start_if_installed=False)
    except Exception as first_exc:
        # Una instalación parcial/corrupta no debe obligar a la persona a abrir
        # Ollama manualmente. Se hace un único intento controlado de reparación.
        if progress:
            progress("El motor privado no respondió; intentando reparación automática…", None)
        try:
            return repair_private_ollama(report=report, progress=progress)
        except Exception as repair_exc:
            raise RuntimeError(
                "Jerónimo no pudo iniciar ni reparar automáticamente su motor privado de análisis. "
                f"Detalle de reparación: {repair_exc}"
            ) from first_exc


def private_ollama_status(*, start_if_installed: bool = False) -> dict[str, Any]:
    exe = ollama_executable()
    if start_if_installed and exe.is_file() and not private_ollama_running():
        try:
            start_private_ollama_if_installed(wait=True)
        except Exception:
            pass
    version = _api_version()
    return {
        "managed_by_jeronimo": True,
        "runtime_version": OLLAMA_RUNTIME_VERSION,
        "runtime_installed": exe.is_file(),
        "executable": str(exe) if exe.is_file() else "",
        "models_path": str(ollama_models_dir()),
        "url": PRIVATE_OLLAMA_URL,
        "available": bool(version),
        "version": version,
        "cloud_disabled": True,
    }
