from __future__ import annotations

import ctypes
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from credential_store import credential_source, resolve_secret
from subprocess_utils import hidden_process_kwargs
from local_text_runtime import (
    PRIVATE_OLLAMA_URL,
    is_private_ollama_url,
    private_ollama_status,
    start_private_ollama_if_installed,
)

DEFAULT_OLLAMA_URL = PRIVATE_OLLAMA_URL


def _run(cmd: list[str], timeout: float = 4.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, **hidden_process_kwargs())
        out = (p.stdout or p.stderr or "").strip()
        return p.returncode, out
    except Exception as exc:
        return 1, str(exc)


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return ""


def _human_bytes(value: int | float | None) -> str:
    if value is None:
        return "desconocido"
    n = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(n) < 1024 or unit == units[-1]:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _memory_status() -> tuple[int, int]:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page * pages), int(page * avail_pages)
    except Exception:
        return 0, 0


def _cpu_name() -> str:
    name = (platform.processor() or "").strip()
    if name:
        return name
    if os.name == "nt":
        rc, out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"], timeout=5)
        if rc == 0 and out:
            return out.splitlines()[0].strip()
    return platform.machine() or "CPU no identificada"


def _nvidia_gpus() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    rc, out = _run([
        exe,
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ], timeout=6)
    if rc != 0:
        return []
    result: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            total_mb = int(float(parts[1]))
            free_mb = int(float(parts[2]))
        except Exception:
            total_mb = free_mb = 0
        result.append({
            "name": parts[0],
            "memory_total_mb": total_mb,
            "memory_free_mb": free_mb,
            "driver_version": parts[3],
        })
    return result


def _tool_version(name: str) -> tuple[bool, str, str]:
    path = shutil.which(name) or ""
    if not path:
        return False, "", ""
    rc, out = _run([path, "-version"], timeout=4)
    first = out.splitlines()[0].strip() if out else ""
    return rc == 0, path, first


def _ollama_status(base_url: str) -> dict[str, Any]:
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    if is_private_ollama_url(base):
        # En arranques posteriores, si el runtime privado ya fue preparado por
        # Jerónimo, se inicia sin instalar nada ni tocar el sistema global.
        try:
            start_private_ollama_if_installed(wait=True)
        except Exception:
            pass
        result = private_ollama_status(start_if_installed=False)
        result.setdefault("models", [])
        try:
            with urllib.request.urlopen(base + "/api/tags", timeout=3.5) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
            result["models"] = [
                {
                    "name": str(item.get("name", "") or item.get("model", "")).strip(),
                    "size": int(item.get("size", 0) or 0),
                }
                for item in data.get("models", [])
                if str(item.get("name", "") or item.get("model", "")).strip()
            ]
            result["available"] = True
        except Exception as exc:
            result.setdefault("error", str(exc))
        return result

    # Endpoint Ollama avanzado/externo: sólo diagnosticar, nunca instalar ni
    # iniciar procesos del usuario.
    result: dict[str, Any] = {
        "available": False, "version": "", "models": [], "url": base,
        "executable": "", "executable_version": "", "managed_by_jeronimo": False,
    }
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=2.5) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
            result["version"] = str(data.get("version", ""))
        with urllib.request.urlopen(base + "/api/tags", timeout=3.5) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
            result["models"] = [
                {"name": str(item.get("name", "") or item.get("model", "")).strip(), "size": int(item.get("size", 0) or 0)}
                for item in data.get("models", [])
                if str(item.get("name", "") or item.get("model", "")).strip()
            ]
        result["available"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _hf_cache_models() -> list[str]:
    root = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    if not root.exists():
        return []
    found: list[str] = []
    try:
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("models--"):
                found.append(p.name[len("models--"):].replace("--", "/"))
    except Exception:
        return []
    return sorted(found)


def _validate_hf_token(token: str) -> tuple[bool | None, str]:
    if not token:
        return False, "no configurado"
    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "Jeronimo/phase3"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
            user = str(data.get("name", "") or data.get("fullname", "")).strip()
            return True, f"válido{f' ({user})' if user else ''}"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, "rechazado por Hugging Face"
        return None, f"no se pudo validar (HTTP {exc.code})"
    except Exception:
        return None, "no se pudo validar sin conexión"


@dataclass
class SystemReport:
    platform: str
    python_version: str
    cpu_name: str
    cpu_logical_cores: int
    ram_total_bytes: int
    ram_available_bytes: int
    gpus: list[dict[str, Any]] = field(default_factory=list)
    torch_version: str = ""
    torch_cuda_available: bool = False
    torch_cuda_version: str = ""
    ctranslate2_version: str = ""
    ctranslate2_cuda_devices: int | None = None
    faster_whisper_version: str = ""
    whisperx_version: str = ""
    ffmpeg: dict[str, Any] = field(default_factory=dict)
    ffprobe: dict[str, Any] = field(default_factory=dict)
    ffplay: dict[str, Any] = field(default_factory=dict)
    disk: dict[str, Any] = field(default_factory=dict)
    ollama: dict[str, Any] = field(default_factory=dict)
    hf_token_source: str = "missing"
    hf_token_validation: str = "no configurado"
    hf_cached_models: list[str] = field(default_factory=list)
    diarization_worker: dict[str, Any] = field(default_factory=dict)
    recommendations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _recommendations(report: SystemReport) -> dict[str, Any]:
    ram_gb = report.ram_total_bytes / (1024 ** 3) if report.ram_total_bytes else 0
    ram_available_gb = report.ram_available_bytes / (1024 ** 3) if report.ram_available_bytes else 0
    max_vram = max((int(g.get("memory_total_mb", 0) or 0) for g in report.gpus), default=0) / 1024
    free_vram = max((int(g.get("memory_free_mb", 0) or 0) for g in report.gpus), default=0) / 1024
    cuda_ok = bool(report.gpus) and (report.torch_cuda_available or (report.ctranslate2_cuda_devices or 0) > 0)

    # Priorizamos calidad del análisis, no velocidad. Un 30B sólo se selecciona
    # cuando hay margen real; en 12–16 GB VRAM preferimos 14B para evitar un
    # offload excesivo y mantener contexto útil.
    if max_vram >= 20 and ram_gb >= 32:
        text_primary = "qwen3:30b"
        text_alt = "qwen3:14b"
        text_note = "perfil máximo: prioriza profundidad analítica aunque tarde más"
    elif max_vram >= 12 or ram_gb >= 28:
        text_primary = "qwen3:14b"
        text_alt = "gemma3:12b"
        text_note = "perfil alto: prioriza calidad del análisis y del resumen"
    elif max_vram >= 7 or ram_gb >= 16:
        text_primary = "qwen3:8b"
        text_alt = "qwen3:4b"
        text_note = "perfil equilibrado con análisis jerárquico"
    elif max_vram >= 4 or ram_gb >= 8:
        text_primary = "qwen3:4b"
        text_alt = "qwen3:1.7b"
        text_note = "perfil ligero: mantiene análisis local con menor profundidad"
    else:
        text_primary = "qwen3:1.7b"
        text_alt = "qwen3:0.6b"
        text_note = "perfil muy limitado: prioriza poder ejecutar localmente"

    if cuda_ok and max_vram >= 5:
        asr = "Mantener el pipeline WhisperX actual para máxima calidad; el equipo parece apto para GPU."
    elif cuda_ok:
        asr = "GPU CUDA detectada, pero la VRAM es limitada; mantener baseline y esperar rendimiento menor/fallback si hace falta."
    else:
        asr = "No se confirma aceleración CUDA. Jerónimo Abya Yala puede usar CPU/fallback, pero diarización y modelos grandes pueden ser lentos."

    warnings: list[str] = []
    if ram_available_gb and ram_available_gb < 4:
        warnings.append("Hay menos de 4 GB de RAM disponible ahora; cerrar aplicaciones pesadas puede evitar fallos.")
    if report.disk.get("free_bytes", 0) and report.disk.get("free_bytes", 0) < 10 * 1024 ** 3:
        warnings.append("Hay menos de 10 GB libres en la unidad de trabajo; modelos locales pueden no caber.")
    if not report.disk.get("writable", False):
        warnings.append("La carpeta de trabajo no permite escritura; selecciona otra antes de procesar entrevistas.")
    if not report.ffmpeg.get("available"):
        warnings.append("FFmpeg no está disponible: la diarización actual no debe iniciarse hasta resolverlo.")
    if not report.ollama.get("available"):
        if report.ollama.get("runtime_installed"):
            warnings.append("El motor privado de análisis está instalado pero no respondió; Jerónimo intentará reiniciarlo cuando se use.")
        else:
            warnings.append("El motor privado de análisis aún no está preparado; el modo Automático lo descargará sin instalar Ollama globalmente.")

    return {
        "text_primary": text_primary,
        "text_alternative": text_alt,
        "text_note": text_note,
        "asr": asr,
        "warnings": warnings,
        "basis": {
            "ram_gb": round(ram_gb, 1),
            "ram_available_gb": round(ram_available_gb, 1),
            "vram_gb": round(max_vram, 1),
            "vram_free_gb": round(free_vram, 1),
            "cuda_ok": cuda_ok,
        },
    }


def run_system_diagnostics(work_dir: str | Path | None = None, ollama_url: str = DEFAULT_OLLAMA_URL, validate_hf: bool = False) -> SystemReport:
    total_ram, avail_ram = _memory_status()
    gpus = _nvidia_gpus()

    torch_version = _pkg_version("torch")
    torch_cuda_available = False
    torch_cuda_version = ""
    try:
        import torch  # type: ignore
        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_version = str(getattr(torch.version, "cuda", "") or "")
    except Exception:
        pass

    ct2_version = _pkg_version("ctranslate2")
    ct2_devices: int | None = None
    try:
        import ctranslate2  # type: ignore
        if hasattr(ctranslate2, "get_cuda_device_count"):
            ct2_devices = int(ctranslate2.get_cuda_device_count())
    except Exception:
        pass

    ffmpeg_ok, ffmpeg_path, ffmpeg_ver = _tool_version("ffmpeg")
    ffprobe_ok, ffprobe_path, ffprobe_ver = _tool_version("ffprobe")
    ffplay_ok, ffplay_path, ffplay_ver = _tool_version("ffplay")

    disk_root = Path(work_dir or Path.cwd())
    try:
        disk_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        disk_root = Path.cwd()
    writable = False
    write_error = ""
    try:
        probe = tempfile.NamedTemporaryFile(prefix=".jeronimo_diag_", dir=disk_root, delete=False)
        probe_path = Path(probe.name)
        probe.write(b"ok")
        probe.close()
        probe_path.unlink(missing_ok=True)
        writable = True
    except Exception as exc:
        write_error = str(exc)
    try:
        usage = shutil.disk_usage(disk_root)
        disk = {
            "path": str(disk_root), "total_bytes": usage.total, "free_bytes": usage.free,
            "writable": writable, "write_error": write_error,
        }
    except Exception as exc:
        disk = {
            "path": str(disk_root), "total_bytes": 0, "free_bytes": 0,
            "writable": writable, "write_error": write_error, "error": str(exc),
        }

    hf_source = credential_source("huggingface")
    token = resolve_secret("huggingface")
    hf_validation = "configurado (validación en red no solicitada)" if token else "no configurado"
    if validate_hf and token:
        _, hf_validation = _validate_hf_token(token)

    worker_probe: dict[str, Any] = {}
    try:
        from diarization_runtime_manager import probe_diarization_runtime
        candidate = probe_diarization_runtime()
        if not candidate.get("skipped"):
            worker_probe = candidate
    except Exception as exc:
        worker_probe = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    report = SystemReport(
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=sys.version.split()[0],
        cpu_name=_cpu_name(),
        cpu_logical_cores=int(os.cpu_count() or 0),
        ram_total_bytes=total_ram,
        ram_available_bytes=avail_ram,
        gpus=gpus,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_version=torch_cuda_version,
        ctranslate2_version=ct2_version,
        ctranslate2_cuda_devices=ct2_devices,
        faster_whisper_version=_pkg_version("faster-whisper"),
        whisperx_version=_pkg_version("whisperx"),
        ffmpeg={"available": ffmpeg_ok, "path": ffmpeg_path, "version": ffmpeg_ver},
        ffprobe={"available": ffprobe_ok, "path": ffprobe_path, "version": ffprobe_ver},
        ffplay={"available": ffplay_ok, "path": ffplay_path, "version": ffplay_ver},
        disk=disk,
        ollama=_ollama_status(ollama_url),
        hf_token_source=hf_source,
        hf_token_validation=hf_validation,
        hf_cached_models=_hf_cache_models(),
        diarization_worker={
            "path": str(os.environ.get("JERONIMO_DIARIZATION_WORKER_EXE", "") or ""),
            "available": bool(
                (os.environ.get("JERONIMO_DIARIZATION_WORKER_EXE") and Path(os.environ.get("JERONIMO_DIARIZATION_WORKER_EXE", "")).is_file())
                or (os.environ.get("JERONIMO_DIARIZATION_PYTHON") and Path(os.environ.get("JERONIMO_DIARIZATION_PYTHON", "")).is_file())
            ),
            "python": str(os.environ.get("JERONIMO_DIARIZATION_PYTHON", "") or ""),
            "worker_script": str(os.environ.get("JERONIMO_DIARIZATION_WORKER_SCRIPT", "") or ""),
            "probe": worker_probe,
        },
    )
    report.recommendations = _recommendations(report)
    return report


def format_diagnostics_report(report: SystemReport) -> str:
    lines: list[str] = []
    lines.append("DIAGNÓSTICO DEL EQUIPO — JERÓNIMO")
    lines.append("")
    lines.append(f"Sistema: {report.platform}")
    lines.append(f"Python: {report.python_version}")
    lines.append(f"CPU: {report.cpu_name} · {report.cpu_logical_cores} hilos lógicos")
    lines.append(f"RAM: {_human_bytes(report.ram_total_bytes)} total · {_human_bytes(report.ram_available_bytes)} disponible")
    if report.gpus:
        for idx, gpu in enumerate(report.gpus, 1):
            lines.append(
                f"GPU {idx}: {gpu.get('name', '')} · VRAM {gpu.get('memory_total_mb', 0)/1024:.1f} GB total / "
                f"{gpu.get('memory_free_mb', 0)/1024:.1f} GB libre · driver {gpu.get('driver_version', '')}"
            )
    else:
        lines.append("GPU NVIDIA: no detectada por nvidia-smi")
    lines.append("")
    lines.append(f"PyTorch: {report.torch_version or 'no instalado'} · CUDA disponible: {'sí' if report.torch_cuda_available else 'no'} · CUDA runtime: {report.torch_cuda_version or '-'}")
    ct2_cuda = "desconocido" if report.ctranslate2_cuda_devices is None else str(report.ctranslate2_cuda_devices)
    lines.append(f"CTranslate2: {report.ctranslate2_version or 'no instalado'} · dispositivos CUDA visibles: {ct2_cuda}")
    lines.append(f"faster-whisper: {report.faster_whisper_version or 'no instalado'}")
    lines.append(f"WhisperX: {report.whisperx_version or 'no instalado'}")
    worker_probe = report.diarization_worker.get("probe") or {}
    if worker_probe:
        if worker_probe.get("type") == "result":
            packages = worker_probe.get("packages") or {}
            lines.append(
                "Runtime aislado: "
                f"Torch {packages.get('torch') or '-'} · "
                f"CTranslate2 {packages.get('ctranslate2') or '-'} · "
                f"CUDA Torch {'sí' if worker_probe.get('cuda_available') else 'no'} · "
                f"CUDA CTranslate2 {worker_probe.get('ctranslate2_cuda_devices', 'desconocido')}"
            )
        elif worker_probe.get("type") == "error":
            lines.append(f"Runtime aislado: ERROR · {worker_probe.get('message', 'sin detalle')}")
    lines.append("")
    for label, item in (("FFmpeg", report.ffmpeg), ("ffprobe", report.ffprobe), ("ffplay", report.ffplay)):
        available = bool(item.get("available"))
        version = item.get("version") if available else None
        suffix = f" · {version}" if version else ""
        lines.append(f"{label}: {'OK' if available else 'no disponible'}{suffix}")
    write_state = "escribible" if report.disk.get("writable") else "SIN permiso de escritura"
    lines.append(f"Disco ({report.disk.get('path', '')}): {_human_bytes(report.disk.get('free_bytes', 0))} libres · {write_state}")
    lines.append("")
    ollama = report.ollama
    managed = bool(ollama.get("managed_by_jeronimo"))
    installed_text = "runtime privado instalado" if ollama.get("runtime_installed") else ("ejecutable detectado" if ollama.get("executable") else "runtime no preparado")
    runtime_text = "servidor activo" if ollama.get("available") else "servidor no responde"
    version_text = ollama.get("version") or ollama.get("runtime_version") or ollama.get("executable_version") or ""
    engine_label = "Motor local de Jerónimo (Ollama standalone)" if managed else "Ollama"
    lines.append(f"{engine_label}: {installed_text} · {runtime_text}{f' · {version_text}' if version_text else ''}")
    if ollama.get("models"):
        lines.append("Modelos Ollama instalados: " + ", ".join(m.get("name", "") for m in ollama["models"]))
    else:
        lines.append("Modelos Ollama instalados: ninguno detectado")
    lines.append(f"Hugging Face: fuente de token = {report.hf_token_source} · {report.hf_token_validation}")
    relevant_cache = [m for m in report.hf_cached_models if any(k in m.lower() for k in ("whisper", "pyannote"))]
    lines.append("Modelos HF relevantes en caché: " + (", ".join(relevant_cache) if relevant_cache else "ninguno detectado"))
    lines.append("")
    rec = report.recommendations
    lines.append("RECOMENDACIONES")
    lines.append(f"Resumen local recomendado: {rec.get('text_primary')} ({rec.get('text_note')})")
    lines.append(f"Alternativa: {rec.get('text_alternative')}")
    lines.append(f"Transcripción/diarización: {rec.get('asr')}")
    for warning in rec.get("warnings", []):
        lines.append(f"AVISO: {warning}")
    lines.append("")
    lines.append("Nota: la recomendación es preventiva. Jerónimo Abya Yala debe confirmar rendimiento real con una prueba breve del modelo instalado.")
    return "\n".join(lines)
