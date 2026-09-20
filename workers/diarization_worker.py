from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

# Bootstrap portable antes de importar el motor para que FFmpeg incluido quede en PATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from portable_runtime import bootstrap_portable_environment
    bootstrap_portable_environment()
except Exception:
    pass

# El worker se ejecuta desde la raíz del proyecto en el release portable.
# IMPORTANTE: core_transcriber/WhisperX/pyannote se cargan de forma perezosa
# únicamente al ejecutar una diarización real. El probe de empaquetado debe poder
# validar el ejecutable y Torch sin disparar TorchCodec/pyannote durante el arranque.


def _load_transcriber():
    import warnings

    with warnings.catch_warnings():
        # pyannote 4 puede advertir sobre TorchCodec en Windows aunque Jerónimo
        # entregue el audio ya cargado en memoria. Ese decoder no interviene en
        # nuestro pipeline, por eso se silencia sólo esta advertencia al importar.
        warnings.filterwarnings(
            "ignore",
            message=r"(?s).*torchcodec is not installed correctly.*",
            category=UserWarning,
        )
        from core_transcriber import TranscriberConfig, diarize_and_transcribe_local
    return TranscriberConfig, diarize_and_transcribe_local


_PROTOCOL_STDOUT = sys.stdout


def emit(payload: dict) -> None:
    # stdout queda reservado al protocolo JSON aunque una dependencia cambie o
    # redirija temporalmente sys.stdout.
    _PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _PROTOCOL_STDOUT.flush()


def _external_output_to_stderr():
    """Aísla prints de terceros del canal JSON de control."""
    return contextlib.redirect_stdout(sys.stderr)


def _read_request() -> dict:
    # El autodiagnóstico del portable usa un argumento dedicado en lugar de
    # depender de stdin. En algunos hosts Windows/PowerShell el stdin redirigido
    # del ejecutable PyInstaller puede llegar vacío o alterado antes del primer
    # readline. El protocolo JSON por stdin se conserva intacto para la
    # diarización real usada por DiarizationWorkerClient.
    args = [str(arg).strip().lower() for arg in sys.argv[1:]]
    if args and args[0] in {"--probe", "probe"}:
        return {"op": "probe"}
    if args and args[0] in {"--probe-stack", "probe-stack"}:
        model_path = str(sys.argv[2]) if len(sys.argv) > 2 else ""
        device = str(sys.argv[3]) if len(sys.argv) > 3 else "cuda"
        return {"op": "probe-stack", "model_path": model_path, "device": device}

    line = sys.stdin.readline()
    if not line:
        raise EOFError("No se recibió solicitud.")
    return json.loads(line)


def main() -> int:
    try:
        req = _read_request()
        if req.get("op") in {"probe", "probe-stack"}:
            import importlib.metadata

            packages = {}
            for name in ("whisperx", "torch", "torchaudio", "pyannote.audio", "ctranslate2", "faster-whisper"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except Exception:
                    packages[name] = ""

            try:
                with _external_output_to_stderr():
                    import torch  # type: ignore
                    import ctranslate2  # type: ignore
                torch_ok = True
                cuda_available = bool(torch.cuda.is_available())
                cuda_runtime = str(getattr(torch.version, "cuda", "") or "")
                ct2_devices = (
                    int(ctranslate2.get_cuda_device_count())
                    if hasattr(ctranslate2, "get_cuda_device_count") else None
                )
            except Exception as exc:
                emit({
                    "type": "error",
                    "operation": req.get("op"),
                    "message": f"Runtime CUDA/CTranslate2 no pudo inicializarse: {type(exc).__name__}: {exc}",
                    "packages": packages,
                })
                return 1

            if req.get("op") == "probe-stack":
                model_path = Path(str(req.get("model_path") or ""))
                device = str(req.get("device") or "cuda").strip().lower() or "cuda"
                if not model_path.is_dir():
                    raise FileNotFoundError("No existe el snapshot local de Whisper large-v2 para validar el runtime.")
                if device == "cuda" and not cuda_available:
                    raise RuntimeError("PyTorch no informa CUDA disponible en el runtime aislado.")
                if device == "cuda" and not (ct2_devices or 0):
                    raise RuntimeError("CTranslate2 no informa dispositivos CUDA disponibles en el runtime aislado.")

                compute_type = "float16" if device == "cuda" else "int8"
                # Esta es deliberadamente la misma inicialización que fallaría al
                # comenzar una diarización real, pero sin procesar audio. El ASR
                # large-v2 ya fue descargado; WhisperX puede inicializar/cachear el
                # VAD que también necesitaría en el primer uso real.
                with _external_output_to_stderr():
                    import whisperx  # type: ignore
                    model = whisperx.load_model(
                        str(model_path),
                        device,
                        compute_type=compute_type,
                        language="es",
                    )
                    del model
                    if device == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                emit({
                    "type": "result",
                    "operation": "probe-stack",
                    "packages": packages,
                    "torch_import_ok": torch_ok,
                    "cuda_available": cuda_available,
                    "torch_cuda_runtime": cuda_runtime,
                    "ctranslate2_cuda_devices": ct2_devices,
                    "whisperx_model_load": True,
                    "device": device,
                    "compute_type": compute_type,
                })
                return 0

            emit({
                "type": "result",
                "operation": "probe",
                "packages": packages,
                "torch_import_ok": torch_ok,
                "cuda_available": cuda_available,
                "torch_cuda_runtime": cuda_runtime,
                "ctranslate2_cuda_devices": ct2_devices,
            })
            return 0
        if req.get("op") != "diarize":
            raise ValueError("Operación desconocida.")

        with _external_output_to_stderr():
            TranscriberConfig, diarize_and_transcribe_local = _load_transcriber()
        audio_path = Path(req["audio_path"])
        cfg_data = dict(req.get("config") or {})
        # El worker no recibe ni persiste API keys. La diarización resuelve HF
        # mediante el almacén seguro/entorno del propio runtime.
        cfg = TranscriberConfig(
            api_key="",
            input_dir=audio_path.parent,
            output_dir=Path(cfg_data.get("output_dir") or audio_path.parent),
            work_dir=Path(cfg_data.get("work_dir") or audio_path.parent),
            language=str(cfg_data.get("language") or "es"),
            use_diarization=True,
            whisperx_model=str(cfg_data.get("whisperx_model") or "large-v2"),
            diarization_device=str(cfg_data.get("diarization_device") or "cuda"),
            target_speakers=int(cfg_data.get("target_speakers") or 0),
            interviewer_name=str(cfg_data.get("interviewer_name") or ""),
            interviewee_name=str(cfg_data.get("interviewee_name") or ""),
            whisper_prompt=str(cfg_data.get("whisper_prompt") or ""),
            atlas_include_turn_numbers=bool(cfg_data.get("atlas_include_turn_numbers", False)),
            atlas_include_timecodes=bool(cfg_data.get("atlas_include_timecodes", False)),
            atlas_include_speaker_id=bool(cfg_data.get("atlas_include_speaker_id", False)),
            atlas_blank_line_between_turns=bool(cfg_data.get("atlas_blank_line_between_turns", True)),
        )

        def progress(message: str) -> None:
            emit({"type": "event", "stage": "diarizing", "message": str(message)})

        with _external_output_to_stderr():
            text = diarize_and_transcribe_local(audio_path, cfg, progress)
        emit({"type": "result", "operation": "diarize", "text": text})
        return 0
    except Exception as exc:
        # No imprimimos traceback ni configuración. Si una librería incluye la
        # ruta del audio en su error, se redacta antes de devolverlo.
        message = str(exc)
        try:
            audio_value = str(locals().get("audio_path", "") or "")
            if audio_value:
                message = message.replace(audio_value, "[audio]")
                message = message.replace(str(Path(audio_value).parent), "[carpeta]")
        except Exception:
            pass
        emit({"type": "error", "message": f"{type(exc).__name__}: {message}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
