from __future__ import annotations

"""Telemetría ligera y local del equipo durante una transcripción.

No forma parte del pipeline de audio ni condiciona su ejecución.  El monitor corre
como hilo daemon, conserva sólo la última muestra en memoria y falla de forma
silenciosa cuando un contador del sistema no está disponible.
"""

import ctypes
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from subprocess_utils import hidden_process_kwargs


@dataclass(frozen=True)
class SystemSnapshot:
    timestamp: float
    cpu_percent: float | None
    ram_used_bytes: int | None
    ram_total_bytes: int | None
    ram_percent: float | None
    gpu_available: bool
    gpu_name: str = ""
    gpu_percent: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    gpu_temperature_c: float | None = None


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]


def _filetime_value(value: _FILETIME) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


class CpuSampler:
    """Calcula utilización global por delta de contadores, sin psutil."""

    def __init__(self) -> None:
        self._previous: tuple[int, int, int] | None = None

    def _read_windows(self) -> tuple[int, int, int] | None:
        if os.name != "nt":
            return None
        idle = _FILETIME()
        kernel = _FILETIME()
        user = _FILETIME()
        try:
            ok = ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            )
        except Exception:
            return None
        if not ok:
            return None
        return _filetime_value(idle), _filetime_value(kernel), _filetime_value(user)

    def _read_proc(self) -> tuple[int, int, int] | None:
        # Fallback útil para desarrollo/pruebas fuera de Windows. Se normaliza al
        # mismo contrato: idle, kernel(total sistema), user.
        if os.name == "nt":
            return None
        try:
            first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
            if not first or first[0] != "cpu":
                return None
            nums = [int(v) for v in first[1:]]
            while len(nums) < 8:
                nums.append(0)
            user = nums[0] + nums[1]
            system = nums[2] + nums[5] + nums[6]
            idle = nums[3] + nums[4]
            total_kernel = system + idle
            return idle, total_kernel, user
        except Exception:
            return None

    def sample(self) -> float | None:
        current = self._read_windows() or self._read_proc()
        if current is None:
            return None
        previous = self._previous
        self._previous = current
        if previous is None:
            return None
        idle_delta = current[0] - previous[0]
        kernel_delta = current[1] - previous[1]
        user_delta = current[2] - previous[2]
        total_delta = kernel_delta + user_delta
        if total_delta <= 0:
            return None
        busy = max(0, total_delta - max(0, idle_delta))
        return max(0.0, min(100.0, (busy / total_delta) * 100.0))


def memory_usage() -> tuple[int | None, int | None, float | None]:
    """Devuelve (usada, total, porcentaje) de memoria física."""
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
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                total = int(status.ullTotalPhys)
                avail = int(status.ullAvailPhys)
                used = max(0, total - avail)
                pct = (used / total * 100.0) if total else None
                return used, total, pct
        except Exception:
            pass
    try:
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                info[key] = int(raw.strip().split()[0]) * 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        if total:
            used = max(0, total - avail)
            return used, total, used / total * 100.0
    except Exception:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = int(page * os.sysconf("SC_PHYS_PAGES"))
        avail = int(page * os.sysconf("SC_AVPHYS_PAGES"))
        used = max(0, total - avail)
        return used, total, (used / total * 100.0) if total else None
    except Exception:
        return None, None, None


def _number(value: str) -> float | None:
    try:
        cleaned = value.strip()
        if not cleaned or cleaned.upper() in {"N/A", "NA", "[N/A]", "NOT SUPPORTED"}:
            return None
        return float(cleaned)
    except Exception:
        return None


def nvidia_usage(timeout: float = 4.0) -> dict[str, object] | None:
    """Muestra la GPU NVIDIA más activa. Nunca genera una excepción al caller."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    cmd = [
        exe,
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **hidden_process_kwargs(),
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    candidates: list[dict[str, object]] = []
    for line in (proc.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        util = _number(parts[1])
        used = _number(parts[2])
        total = _number(parts[3])
        temp = _number(parts[4])
        candidates.append({
            "name": parts[0],
            "gpu_percent": util,
            "vram_used_mb": used,
            "vram_total_mb": total,
            "gpu_temperature_c": temp,
        })
    if not candidates:
        return None

    def score(item: dict[str, object]) -> tuple[float, float]:
        util = item.get("gpu_percent")
        used = item.get("vram_used_mb")
        return (
            float(util) if isinstance(util, (int, float)) else -1.0,
            float(used) if isinstance(used, (int, float)) else -1.0,
        )

    return max(candidates, key=score)


class SystemResourceMonitor:
    """Recolector en background con snapshot consultable por la UI."""

    def __init__(self, interval: float = 1.0, gpu_interval: float = 2.0) -> None:
        self.interval = max(0.5, float(interval))
        self.gpu_interval = max(self.interval, float(gpu_interval))
        self._cpu = CpuSampler()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest: SystemSnapshot | None = None
        self._gpu_cache: dict[str, object] | None = None
        self._next_gpu_at = 0.0

    def start(self) -> "SystemResourceMonitor":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jeronimo-system-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self, wait: bool = False) -> None:
        self._stop.set()
        thread = self._thread
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def latest(self) -> SystemSnapshot | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            cpu = self._cpu.sample()
            ram_used, ram_total, ram_pct = memory_usage()
            if now >= self._next_gpu_at:
                self._gpu_cache = nvidia_usage()
                self._next_gpu_at = now + self.gpu_interval
            gpu = self._gpu_cache
            snapshot = SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu,
                ram_used_bytes=ram_used,
                ram_total_bytes=ram_total,
                ram_percent=ram_pct,
                gpu_available=gpu is not None,
                gpu_name=str(gpu.get("name", "")) if gpu else "",
                gpu_percent=gpu.get("gpu_percent") if gpu else None,  # type: ignore[arg-type]
                vram_used_mb=gpu.get("vram_used_mb") if gpu else None,  # type: ignore[arg-type]
                vram_total_mb=gpu.get("vram_total_mb") if gpu else None,  # type: ignore[arg-type]
                gpu_temperature_c=gpu.get("gpu_temperature_c") if gpu else None,  # type: ignore[arg-type]
            )
            with self._lock:
                self._latest = snapshot
            self._stop.wait(self.interval)
