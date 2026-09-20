from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Optional
import math
import time
import uuid


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_CANCELLED = "cancelled"
JOB_STATUS_ERROR = "error"

STAGE_QUEUED = "queued"
STAGE_PREPARING = "preparing"
STAGE_TRANSCRIBING = "transcribing"
STAGE_DIARIZING = "diarizing"
STAGE_CLEANING = "cleaning"
STAGE_SUMMARIZING = "summarizing"
STAGE_EXPORTING = "exporting"
STAGE_FINALIZING = "finalizing"
STAGE_COMPLETED = "completed"
STAGE_CANCELLED = "cancelled"
STAGE_ERROR = "error"

STAGE_LABELS = {
    STAGE_QUEUED: "En espera",
    STAGE_PREPARING: "Preparando audio",
    STAGE_TRANSCRIBING: "Transcribiendo",
    STAGE_DIARIZING: "Transcribiendo y separando hablantes",
    STAGE_CLEANING: "Preparando versión limpia",
    STAGE_SUMMARIZING: "Generando resumen",
    STAGE_EXPORTING: "Exportando resultados",
    STAGE_FINALIZING: "Finalizando",
    STAGE_COMPLETED: "Completado",
    STAGE_CANCELLED: "Cancelado",
    STAGE_ERROR: "Error",
}


@dataclass
class JobEvent:
    job_id: str
    stage: str
    status: str = JOB_STATUS_RUNNING
    progress: float = 0.0
    message: str = ""
    current_item: str = ""
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Job:
    """Estado serializable de un trabajo sin persistir contenido de entrevista."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    kind: str = "transcription"
    stage: str = STAGE_QUEUED
    status: str = JOB_STATUS_QUEUED
    progress: float = 0.0
    message: str = ""
    current_item: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    configuration: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobTracker:
    """
    Mantiene estado y ETA a partir de progreso observado.

    La ETA no usa tablas teóricas de hardware. Aprende la velocidad de la
    ejecución actual y sólo se muestra cuando hay suficiente señal. Para evitar
    saltos, la tasa se suaviza con una media exponencial.
    """

    def __init__(
        self,
        job: Optional[Job] = None,
        *,
        on_event: Optional[Callable[[JobEvent], None]] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.job = job or Job()
        self.on_event = on_event
        self._clock = monotonic
        self._start_monotonic: Optional[float] = None
        self._last_monotonic: Optional[float] = None
        self._last_progress = 0.0
        self._rate_ema: Optional[float] = None
        self._lock = Lock()

    def start(self, *, stage: str = STAGE_PREPARING, message: str = "") -> JobEvent:
        with self._lock:
            now = self._clock()
            self._start_monotonic = now
            self._last_monotonic = now
            self._last_progress = 0.0
            self._rate_ema = None
            self.job.status = JOB_STATUS_RUNNING
            self.job.stage = stage
            self.job.started_at = datetime.now(timezone.utc).isoformat()
            self.job.message = message
            return self._event_locked()

    def update(
        self,
        *,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        current_item: Optional[str] = None,
        status: Optional[str] = None,
    ) -> JobEvent:
        with self._lock:
            if self._start_monotonic is None:
                now = self._clock()
                self._start_monotonic = now
                self._last_monotonic = now
                self.job.started_at = datetime.now(timezone.utc).isoformat()
                self.job.status = JOB_STATUS_RUNNING

            now = self._clock()
            old_progress = self.job.progress
            if progress is not None:
                p = max(0.0, min(1.0, float(progress)))
                # El progreso global no debe retroceder por un evento de etapa.
                p = max(old_progress, p)
                self.job.progress = p
                self._update_rate(now, p)
            else:
                self._update_elapsed(now)

            if stage is not None:
                self.job.stage = stage
            if message is not None:
                self.job.message = message
            if current_item is not None:
                self.job.current_item = current_item
            if status is not None:
                self.job.status = status

            self._refresh_eta()
            event = self._event_locked()
        self._notify(event)
        return event

    def complete(self, message: str = "Proceso completado.") -> JobEvent:
        return self._finish(JOB_STATUS_COMPLETED, STAGE_COMPLETED, message)

    def cancel(self, message: str = "Proceso cancelado.") -> JobEvent:
        return self._finish(JOB_STATUS_CANCELLED, STAGE_CANCELLED, message)

    def fail(self, message: str) -> JobEvent:
        self.job.error = message
        return self._finish(JOB_STATUS_ERROR, STAGE_ERROR, message, force_progress=False)

    def _finish(self, status: str, stage: str, message: str, *, force_progress: bool = True) -> JobEvent:
        with self._lock:
            now = self._clock()
            if self._start_monotonic is None:
                self._start_monotonic = now
                self.job.started_at = datetime.now(timezone.utc).isoformat()
            self._update_elapsed(now)
            self.job.status = status
            self.job.stage = stage
            self.job.message = message
            self.job.eta_seconds = 0.0 if status == JOB_STATUS_COMPLETED else None
            if force_progress and status == JOB_STATUS_COMPLETED:
                self.job.progress = 1.0
            self.job.finished_at = datetime.now(timezone.utc).isoformat()
            event = self._event_locked()
        self._notify(event)
        return event

    def _update_rate(self, now: float, progress: float) -> None:
        self._update_elapsed(now)
        if self._last_monotonic is None:
            self._last_monotonic = now
            self._last_progress = progress
            return
        dt = now - self._last_monotonic
        dp = progress - self._last_progress
        if dt >= 0.25 and dp > 0:
            instantaneous = dp / dt
            if math.isfinite(instantaneous) and instantaneous > 0:
                alpha = 0.28
                self._rate_ema = instantaneous if self._rate_ema is None else (
                    alpha * instantaneous + (1.0 - alpha) * self._rate_ema
                )
            self._last_monotonic = now
            self._last_progress = progress

    def _update_elapsed(self, now: float) -> None:
        if self._start_monotonic is not None:
            self.job.elapsed_seconds = max(0.0, now - self._start_monotonic)

    def _refresh_eta(self) -> None:
        p = self.job.progress
        elapsed = self.job.elapsed_seconds
        if self.job.status != JOB_STATUS_RUNNING or p >= 1.0:
            self.job.eta_seconds = 0.0 if self.job.status == JOB_STATUS_COMPLETED else None
            return
        # No fingir precisión al inicio: esperamos al menos 2 % y 3 s.
        if p < 0.02 or elapsed < 3.0:
            self.job.eta_seconds = None
            return
        rate = self._rate_ema
        if rate is None or rate <= 0:
            rate = p / elapsed if elapsed > 0 else 0.0
        if rate <= 0:
            self.job.eta_seconds = None
            return
        eta = (1.0 - p) / rate
        # Evita números absurdos por un único salto inicial.
        self.job.eta_seconds = min(max(0.0, eta), 7 * 24 * 3600.0)

    def _event_locked(self) -> JobEvent:
        return JobEvent(
            job_id=self.job.job_id,
            stage=self.job.stage,
            status=self.job.status,
            progress=self.job.progress,
            message=self.job.message,
            current_item=self.job.current_item,
            elapsed_seconds=self.job.elapsed_seconds,
            eta_seconds=self.job.eta_seconds,
        )

    def _notify(self, event: JobEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)


class CancellationToken:
    """Token simple compartible entre GUI, motor y workers."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "calculando…"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", " ").strip().capitalize())
