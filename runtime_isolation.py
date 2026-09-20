from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import importlib.metadata
import json
import os
import queue
import subprocess
import sys
import threading
import time

from subprocess_utils import hidden_process_kwargs


@dataclass
class RuntimeAssessment:
    current_python: str
    whisperx_installed: bool
    torch_installed: bool
    pyannote_installed: bool
    whisperx_version: str = ""
    torch_version: str = ""
    pyannote_version: str = ""
    cuda_available: Optional[bool] = None
    configured_worker_python: str = ""
    configured_worker_available: bool = False
    configured_worker_executable: str = ""
    configured_worker_executable_available: bool = False
    isolation_required: bool = False
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return ""


def assess_runtime(worker_python: str = "", worker_executable: str = "") -> RuntimeAssessment:
    whisperx_version = _version("whisperx")
    torch_version = _version("torch")
    pyannote_version = _version("pyannote.audio")
    cuda_available: Optional[bool] = None
    if torch_version:
        try:
            import torch  # type: ignore
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = None

    worker = str(worker_python or os.environ.get("JERONIMO_DIARIZATION_PYTHON", "")).strip()
    worker_available = bool(worker and Path(worker).is_file())
    worker_exe = str(worker_executable or os.environ.get("JERONIMO_DIARIZATION_WORKER_EXE", "")).strip()
    worker_exe_available = bool(worker_exe and Path(worker_exe).is_file())

    # No declaramos incompatibilidad sólo porque falte una dependencia en el
    # entorno de desarrollo. Aislar cambia el modo de ejecución y sólo debe
    # activarse ante conflicto reproducible o cuando el usuario lo configura.
    if worker_exe_available:
        recommendation = (
            "Hay un worker portable de diarización configurado. WhisperX/PyTorch se ejecutarán "
            "aislados del proceso principal."
        )
    elif worker_available:
        recommendation = (
            "Hay un runtime externo configurado para diarización. Puede usarse para "
            "congelar WhisperX/PyTorch sin modificar el entorno principal."
        )
    elif whisperx_version and torch_version:
        recommendation = (
            "No se detectó una necesidad demostrada de aislar el runtime. Mantener el "
            "pipeline integrado que ya funciona y capturar sus versiones antes del release."
        )
    else:
        recommendation = (
            "No puede evaluarse compatibilidad completa en este entorno porque faltan componentes "
            "de diarización. No se activa aislamiento automáticamente."
        )

    return RuntimeAssessment(
        current_python=sys.executable,
        whisperx_installed=bool(whisperx_version),
        torch_installed=bool(torch_version),
        pyannote_installed=bool(pyannote_version),
        whisperx_version=whisperx_version,
        torch_version=torch_version,
        pyannote_version=pyannote_version,
        cuda_available=cuda_available,
        configured_worker_python=worker,
        configured_worker_available=worker_available,
        configured_worker_executable=worker_exe,
        configured_worker_executable_available=worker_exe_available,
        isolation_required=False,
        recommendation=recommendation,
    )


def capture_runtime_profile(path: str | Path, *, worker_python: str = "") -> Path:
    path = Path(path)
    assessment = assess_runtime(worker_python)
    payload = {
        "schema": 1,
        "purpose": "diarization_runtime_baseline",
        "assessment": assessment.to_dict(),
        "packages": {
            name: version
            for name in ("whisperx", "torch", "torchaudio", "pyannote.audio", "ctranslate2", "faster-whisper")
            if (version := _version(name))
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class WorkerProtocolError(RuntimeError):
    pass


def _diagnostic_tail(lines: list[str], *, limit_lines: int = 12, limit_chars: int = 1800) -> str:
    """Devuelve un tail acotado para errores del worker sin volcar payloads completos."""
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    if not cleaned:
        return ""
    text = "\n".join(cleaned[-limit_lines:])
    if len(text) > limit_chars:
        text = text[-limit_chars:]
    return text


class DiarizationWorkerClient:
    """
    Cliente JSON-lines para el worker opcional de diarización.

    ``stdout`` es un canal de control: sólo los mensajes JSON de Jerónimo deben
    circular por él. Algunas dependencias de ML imprimen avisos directamente;
    por robustez el cliente los conserva como diagnóstico en vez de abortar en
    la primera línea ajena al protocolo. ``stderr`` se drena en paralelo para
    evitar bloqueos y se incorpora sólo cuando hay un error real.
    """

    def __init__(self, python_executable: str = "", worker_script: str | Path = "", *, worker_executable: str = "") -> None:
        self.python_executable = str(python_executable or "")
        self.worker_script = str(worker_script or "")
        self.worker_executable = str(worker_executable or "")
        if self.worker_executable:
            if not Path(self.worker_executable).is_file():
                raise FileNotFoundError(f"No existe el worker portable de diarización: {self.worker_executable}")
        else:
            if not Path(self.python_executable).is_file():
                raise FileNotFoundError(f"No existe el Python del runtime aislado: {self.python_executable}")
            if not Path(self.worker_script).is_file():
                raise FileNotFoundError(f"No existe el worker de diarización: {self.worker_script}")

    def run(
        self,
        payload: dict[str, Any],
        *,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        poll_interval: float = 0.1,
    ) -> dict[str, Any]:
        command = [self.worker_executable] if self.worker_executable else [self.python_executable, "-u", self.worker_script]
        env = dict(os.environ)
        # Windows puede elegir una página de códigos heredada para procesos con
        # pipes. El protocolo de Jerónimo es UTF-8 de extremo a extremo.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            **hidden_process_kwargs(),
        )
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        proc.stdin.close()

        output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

        def _reader(stream, name: str) -> None:
            try:
                for line in stream:
                    output_queue.put((name, line))
            finally:
                output_queue.put((name, None))

        stdout_thread = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        final: Optional[dict[str, Any]] = None
        auxiliary_stdout: list[str] = []
        stderr_lines: list[str] = []
        finished_streams: set[str] = set()
        try:
            while len(finished_streams) < 2:
                if cancel_check is not None and cancel_check():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise InterruptedError("Diarización aislada cancelada por el usuario.")
                try:
                    stream_name, raw = output_queue.get(timeout=poll_interval)
                except queue.Empty:
                    if proc.poll() is not None and not stdout_thread.is_alive() and not stderr_thread.is_alive():
                        break
                    continue
                if raw is None:
                    finished_streams.add(stream_name)
                    continue
                line = raw.strip()
                if not line:
                    continue
                if stream_name == "stderr":
                    stderr_lines.append(line)
                    if len(stderr_lines) > 60:
                        del stderr_lines[:-60]
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # No se descarta: puede ser el diagnóstico exacto de una DLL,
                    # CTranslate2, Pyannote o Lightning. Tampoco se considera por
                    # sí solo un fallo si el worker termina entregando un resultado.
                    auxiliary_stdout.append(line)
                    if len(auxiliary_stdout) > 30:
                        del auxiliary_stdout[:-30]
                    continue
                kind = msg.get("type")
                if kind == "event":
                    if on_event is not None:
                        on_event(msg)
                elif kind in {"result", "error"}:
                    final = msg

            returncode = proc.wait(timeout=5)
            diag_parts: list[str] = []
            aux_tail = _diagnostic_tail(auxiliary_stdout)
            err_tail = _diagnostic_tail(stderr_lines)
            if aux_tail:
                diag_parts.append("stdout auxiliar:\n" + aux_tail)
            if err_tail:
                diag_parts.append("stderr:\n" + err_tail)
            diagnostics = "\n".join(diag_parts)

            if final and final.get("type") == "error":
                message = str(final.get("message") or "Error en worker de diarización.")
                if diagnostics:
                    message += "\n\nDiagnóstico del worker:\n" + diagnostics
                raise RuntimeError(message)
            if returncode != 0 and not final:
                detail = diagnostics or "sin salida diagnóstica adicional"
                raise WorkerProtocolError(f"Worker finalizó con código {returncode}.\n{detail}")
            if not final or final.get("type") != "result":
                detail = diagnostics or "sin salida diagnóstica adicional"
                raise WorkerProtocolError(f"Worker finalizó sin resultado JSON válido.\n{detail}")
            return final
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
