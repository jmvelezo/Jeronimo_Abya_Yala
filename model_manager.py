from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Any

from system_diagnostics import SystemReport, run_system_diagnostics
from local_text_runtime import (
    PRIVATE_OLLAMA_URL,
    ensure_private_ollama,
    is_private_ollama_url,
    start_private_ollama_if_installed,
)

DEFAULT_OLLAMA_URL = PRIVATE_OLLAMA_URL
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    backend: str
    model_id: str
    display_name: str
    approx_size_gb: float
    min_ram_gb: float
    recommended_vram_gb: float
    notes: str
    repo_id: str = ""


TEXT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("text", "ollama", "qwen3:0.6b", "Qwen3 0.6B", 0.5, 4, 0, "Modelo mínimo para equipos muy limitados; calidad inferior para resúmenes complejos."),
    ModelSpec("text", "ollama", "qwen3:1.7b", "Qwen3 1.7B", 1.4, 6, 0, "Muy ligero; útil si el equipo es limitado."),
    ModelSpec("text", "ollama", "qwen3:4b", "Qwen3 4B", 2.5, 8, 4, "Ligero y razonable para resúmenes locales."),
    ModelSpec("text", "ollama", "qwen3:8b", "Qwen3 8B", 5.2, 12, 7, "Equilibrio recomendado en equipos medios."),
    ModelSpec("text", "ollama", "qwen3:14b", "Qwen3 14B", 9.3, 20, 12, "Alta calidad para análisis de entrevistas; opción preferida en equipos potentes."),
    ModelSpec("text", "ollama", "qwen3:30b", "Qwen3 30B", 18.6, 32, 20, "Calidad máxima local; pensado para equipos con mucha RAM/VRAM aunque el análisis sea más lento."),
    ModelSpec("text", "ollama", "gemma3:4b", "Gemma 3 4B", 3.3, 8, 4, "Alternativa multilingüe ligera."),
    ModelSpec("text", "ollama", "gemma3:12b", "Gemma 3 12B", 8.1, 20, 12, "Alternativa multilingüe de mayor calidad."),
)

ASR_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("asr", "faster-whisper", "small", "Whisper Small", 0.5, 4, 0, "Modo ligero; no es baseline de máxima calidad.", "Systran/faster-whisper-small"),
    ModelSpec("asr", "faster-whisper", "medium", "Whisper Medium", 1.5, 8, 0, "Intermedio; no reemplaza el baseline sin pruebas.", "Systran/faster-whisper-medium"),
    ModelSpec("asr", "faster-whisper", "large-v2", "Whisper Large v2 (baseline diarización actual)", 3.1, 12, 5, "Baseline real: modelo que el código actual usa por defecto con WhisperX.", "Systran/faster-whisper-large-v2"),
    ModelSpec("asr", "faster-whisper", "large-v3", "Whisper Large v3", 3.1, 12, 5, "Candidato de máxima calidad; no cambia el baseline automáticamente.", "Systran/faster-whisper-large-v3"),
    ModelSpec("asr", "faster-whisper", "large-v3-turbo", "Whisper Large v3 Turbo", 1.6, 8, 3, "Candidato rápido; requiere benchmark antes de ser default.", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
)


def all_models() -> tuple[ModelSpec, ...]:
    return TEXT_MODELS + ASR_MODELS


def _json_request(url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 5.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _ensure_private_runtime_for_request(base_url: str, *, install: bool = False) -> None:
    if not is_private_ollama_url(base_url):
        return
    # El runtime standalone administrado por Jerónimo es una integración Windows.
    # En otras plataformas se deja que el endpoint configurado responda por sí mismo;
    # esto también mantiene las funciones HTTP testeables sin efectos de instalación.
    if os.name != "nt":
        return
    if install:
        ensure_private_ollama()
    else:
        try:
            start_private_ollama_if_installed(wait=True)
        except Exception:
            pass


def ollama_inventory(base_url: str = DEFAULT_OLLAMA_URL) -> dict[str, dict[str, Any]]:
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    _ensure_private_runtime_for_request(base, install=False)
    try:
        with _json_request(base + "/api/tags", timeout=3.5) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in data.get("models", []):
        name = str(item.get("name", "") or item.get("model", "")).strip()
        if name:
            result[name] = {"size": int(item.get("size", 0) or 0), "modified_at": item.get("modified_at", "")}
    return result


def ollama_model_present(models: Any, model_id: str) -> bool:
    """Comprueba un modelo sin confundir tamaños distintos de la misma familia."""
    names = set(models.keys()) if isinstance(models, dict) else {str(x) for x in (models or [])}
    target = str(model_id or "").strip()
    if not target:
        return False
    if target in names:
        return True
    aliases = {
        # En la biblioteca oficial actual estos aliases apuntan a esas variantes.
        "qwen3:8b": {"qwen3:latest", "qwen3"},
        "gemma3:4b": {"gemma3:latest", "gemma3"},
    }
    if names.intersection(aliases.get(target, set())):
        return True
    if ":" not in target:
        return f"{target}:latest" in names
    return False


def _ollama_model_details(inventory: dict[str, dict[str, Any]], model_id: str) -> dict[str, Any]:
    if model_id in inventory:
        return inventory[model_id]
    aliases = {
        "qwen3:8b": ("qwen3:latest", "qwen3"),
        "gemma3:4b": ("gemma3:latest", "gemma3"),
    }
    for alias in aliases.get(model_id, ()):
        if alias in inventory:
            return inventory[alias]
    if ":" not in model_id and f"{model_id}:latest" in inventory:
        return inventory[f"{model_id}:latest"]
    return {}


def pull_ollama_model(model: str, base_url: str = DEFAULT_OLLAMA_URL, progress: ProgressCallback | None = None, cancel_event: Event | None = None) -> None:
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    if is_private_ollama_url(base) and os.name == "nt":
        def runtime_progress(message: str, fraction: float | None) -> None:
            if progress:
                progress({"status": message, "fraction": fraction, "model": model, "completed": 0, "total": 0})
        ensure_private_ollama(progress=runtime_progress)
    req = urllib.request.Request(
        base + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as response:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Descarga cancelada por el usuario. Ollama podrá reanudar al solicitar el modelo nuevamente.")
            line = response.readline()
            if not line:
                break
            try:
                event = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                continue
            completed = int(event.get("completed", 0) or 0)
            total = int(event.get("total", 0) or 0)
            if progress:
                progress({
                    "status": str(event.get("status", "descargando")),
                    "completed": completed,
                    "total": total,
                    "fraction": (completed / total) if total else None,
                    "model": model,
                })
            if event.get("error"):
                raise RuntimeError(str(event["error"]))


def delete_ollama_model(model: str, base_url: str = DEFAULT_OLLAMA_URL) -> None:
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    _ensure_private_runtime_for_request(base, install=False)
    with _json_request(base + "/api/delete", method="DELETE", payload={"model": model}, timeout=30) as response:
        response.read()


def benchmark_ollama_model(model: str, base_url: str = DEFAULT_OLLAMA_URL) -> dict[str, Any]:
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
    _ensure_private_runtime_for_request(base, install=is_private_ollama_url(base))
    prompt = (
        "Analiza con fidelidad y resume en tres oraciones, sin inventar información: "
        "La persona entrevistada explicó que utiliza herramientas digitales para organizar sus clases, "
        "pero que revisa manualmente los resultados antes de utilizarlos con estudiantes."
    )
    try:
        started = time.perf_counter()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "2m",
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        with _json_request(base + "/api/generate", method="POST", payload=payload, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        elapsed = time.perf_counter() - started
        eval_count = int(data.get("eval_count", 0) or 0)
        eval_duration = int(data.get("eval_duration", 0) or 0)
        tps = (eval_count / (eval_duration / 1_000_000_000)) if eval_count and eval_duration else 0.0
        size = 0
        size_vram = 0
        try:
            with _json_request(base + "/api/ps", timeout=8) as response:
                running = json.loads(response.read().decode("utf-8", errors="replace"))
            for item in running.get("models", []):
                name = str(item.get("name", "") or item.get("model", ""))
                if name == model or name.split(":", 1)[0] == model.split(":", 1)[0]:
                    size = int(item.get("size", 0) or 0)
                    size_vram = int(item.get("size_vram", 0) or 0)
                    break
        except Exception:
            pass
        gpu_fraction = (size_vram / size) if size else None
        return {
            "elapsed_seconds": elapsed,
            "tokens_per_second": tps,
            "eval_count": eval_count,
            "size_bytes": size,
            "size_vram_bytes": size_vram,
            "gpu_fraction": gpu_fraction,
        }
    finally:
        # El benchmark sólo diagnostica. Libera el modelo inmediatamente para no
        # robar VRAM al pipeline WhisperX que se ejecute después.
        try:
            with _json_request(
                base + "/api/generate", method="POST",
                payload={"model": model, "keep_alive": 0, "stream": False}, timeout=30,
            ) as response:
                response.read()
        except Exception:
            pass


def hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")


def hf_repo_cache_path(repo_id: str) -> Path:
    return hf_cache_root() / ("models--" + repo_id.replace("/", "--"))


def is_asr_model_cached(spec: ModelSpec) -> bool:
    if not spec.repo_id:
        return False
    root = hf_repo_cache_path(spec.repo_id)
    if not root.exists():
        return False
    snapshots = root / "snapshots"
    return snapshots.exists() and any(p.is_dir() for p in snapshots.iterdir())


def resolve_cached_faster_whisper_model(spec: ModelSpec) -> Path:
    """Resuelve el snapshot ya presente sin acceder a red.

    Se usa para validar el runtime después del setup sin provocar una segunda
    descarga ni depender de que "main" apunte a una revisión concreta.
    """
    if spec.backend != "faster-whisper" or not spec.repo_id:
        raise ValueError("El modelo no es un modelo faster-whisper gestionable.")
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        path = snapshot_download(repo_id=spec.repo_id, local_files_only=True)
        return Path(path)
    except Exception as exc:
        raise RuntimeError(
            f"El modelo {spec.model_id} figura en caché pero no pudo resolverse localmente."
        ) from exc


def download_faster_whisper_model(spec: ModelSpec, progress: ProgressCallback | None = None) -> Path:
    if spec.backend != "faster-whisper" or not spec.repo_id:
        raise ValueError("El modelo no es un modelo faster-whisper gestionable.")
    if progress:
        progress({"status": "preparando descarga desde Hugging Face", "fraction": None, "model": spec.model_id})
    # La app compilada es GUI (sin consola). Hugging Face/tqdm no debe intentar
    # escribir barras de progreso en sys.stderr; Jerónimo ya muestra progreso
    # dentro de su propia interfaz.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        try:
            from huggingface_hub.utils import disable_progress_bars  # type: ignore
            disable_progress_bars()
        except Exception:
            pass
    except Exception as exc:
        raise RuntimeError("Falta huggingface_hub. Se instala junto con faster-whisper; ejecuta el instalador de Jerónimo Abya Yala.") from exc
    # Hugging Face gestiona caché, reintentos y reanudación. La API no expone un total agregado estable
    # para todos los archivos, por eso esta fase muestra progreso indeterminado para ASR.
    # En huggingface_hub actual la caché/reanudación se gestiona automáticamente.
    # No pasar `resume_download`: fue retirado de la API moderna.
    path = snapshot_download(repo_id=spec.repo_id)
    if progress:
        progress({"status": "descarga completada", "fraction": 1.0, "model": spec.model_id})
    return Path(path)


def delete_faster_whisper_model(spec: ModelSpec) -> bool:
    if not spec.repo_id:
        return False
    target = hf_repo_cache_path(spec.repo_id)
    root = hf_cache_root().resolve()
    try:
        resolved = target.resolve()
    except Exception:
        resolved = target
    if not str(resolved).startswith(str(root)) or not target.name.startswith("models--"):
        raise RuntimeError("Se rechazó el borrado porque la ruta no pertenece a la caché de Hugging Face.")
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def compatibility(spec: ModelSpec, report: SystemReport) -> tuple[str, str]:
    ram_gb = report.ram_total_bytes / (1024 ** 3) if report.ram_total_bytes else 0
    vram_gb = max((int(g.get("memory_total_mb", 0) or 0) for g in report.gpus), default=0) / 1024
    if ram_gb and ram_gb < spec.min_ram_gb:
        return "no_recomendado", f"RAM total {ram_gb:.1f} GB < recomendación {spec.min_ram_gb:.0f} GB"
    if spec.recommended_vram_gb and vram_gb >= spec.recommended_vram_gb:
        return "recomendado", f"VRAM {vram_gb:.1f} GB adecuada"
    if spec.recommended_vram_gb and vram_gb == 0:
        return "posible_lento", "sin GPU NVIDIA detectada; puede funcionar por CPU/RAM pero será más lento"
    if spec.recommended_vram_gb and vram_gb < spec.recommended_vram_gb:
        return "posible_lento", f"VRAM {vram_gb:.1f} GB por debajo de {spec.recommended_vram_gb:.0f} GB recomendados"
    return "compatible", "compatible por memoria detectada"


def recommended_text_model(report: SystemReport) -> str:
    return str(report.recommendations.get("text_primary", "qwen3:4b"))


class ModelManager:
    def __init__(self, ollama_url: str = DEFAULT_OLLAMA_URL):
        self.ollama_url = ollama_url or DEFAULT_OLLAMA_URL
        self._diagnostics: SystemReport | None = None

    def diagnostics(self, work_dir: str | Path | None = None, refresh: bool = False) -> SystemReport:
        if self._diagnostics is None or refresh:
            self._diagnostics = run_system_diagnostics(work_dir=work_dir, ollama_url=self.ollama_url)
        return self._diagnostics

    def inventory(self) -> dict[str, dict[str, Any]]:
        ollama = ollama_inventory(self.ollama_url)
        result: dict[str, dict[str, Any]] = {}
        for spec in all_models():
            if spec.backend == "ollama":
                installed = ollama_model_present(ollama, spec.model_id)
                details = _ollama_model_details(ollama, spec.model_id)
            else:
                installed = is_asr_model_cached(spec)
                details = {}
            result[spec.model_id] = {"installed": installed, "details": details, "spec": spec}
        return result
