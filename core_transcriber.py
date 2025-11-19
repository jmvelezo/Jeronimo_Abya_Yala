from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging
from typing import Callable, Optional
from dotenv import load_dotenv
from openai import OpenAI
import contextlib
import wave
#diarización local
try:
    import whisperx  
    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False
try:
    # WhisperX 3.3.4+ puto w no jala versiones desactualizadas, ojo, buscar mecanismo de actualizacion desde el git
    from whisperx.diarize import DiarizationPipeline
except ImportError:
    DiarizationPipeline = None  # mas abajo pa


ProgressFn = Callable[[str], None]
ProgressBarFn = Callable[[float], None]
MAX_SEGMENT_SECONDS = 1200  # 20min tiempo recorte?

LOG_FILE = "transcribir_stt.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("stt_transcriber")


@dataclass
class TranscriberConfig:
    api_key: str
    input_dir: Path          # carpeta base (por si se usa modo carpeta)
    output_dir: Path
    work_dir: Path

    #lista opcional de archivos concretos a procesar
    selected_files: list[Path] | None = None

    stt_model: str = "gpt-4o-mini-transcribe"
    chat_model: str = "gpt-4.1-mini"

    language: str = "es"
    use_ffmpeg: bool = True

    whisper_prompt: str = ""
    clean_prompt: str = ""
    summary_prompt: str = ""

    do_clean: bool = True
    do_summary: bool = True
    #diarización local opcional con WhisperX
    use_diarization: bool = False
    whisperx_model: str = "large-v2"      # modelo por defecto de WhisperX
    diarization_device: str = "cuda"      # "cuda" o "cpu"

    #nombres para etiquetar hablantes cuando haya diarización
    interviewer_name: str = ""
    interviewee_name: str = ""

def load_api_key_from_env() -> Optional[str]:
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except Exception:
        return False


def get_wav_duration_seconds(path: Path) -> float:
    """Devuelve duración del wav en segundos (usa wave, sin dependencias raras)."""
    with contextlib.closing(wave.open(str(path), "rb")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        if rate == 0:
            return 0.0
        return frames / float(rate)


def segment_wav_if_needed(
    wav_path: Path,
    base_name: str,
    work_dir: Path,
    progress: Callable[[str], None],
) -> list[Path]:
    """
    Si el wav dura más que MAX_SEGMENT_SECONDS, lo corta en segmentos.
    Devuelve una lista de paths de segmentos (o el original si es corto).
    """
    duration = get_wav_duration_seconds(wav_path)
    if duration <= 0:
        progress(f"AVISO: no se pudo leer duración de {wav_path.name}, se usa tal cual.")
        return [wav_path]

    if duration <= MAX_SEGMENT_SECONDS:
        progress(f"Duración {duration:.1f} s (< {MAX_SEGMENT_SECONDS}s), se transcribe en un solo bloque.")
        return [wav_path]

    progress(
        f"Duración {duration/60:.1f} min, se dividirá en segmentos de "
        f"{MAX_SEGMENT_SECONDS/60:.1f} min para la transcripción."
    )

    segment_pattern = work_dir / f"{base_name}_seg_%03d.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-f", "segment",
        "-segment_time", str(MAX_SEGMENT_SECONDS),
        "-c", "copy",
        str(segment_pattern),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        progress(f"ERROR al segmentar audio, se usará el archivo completo. Detalle: {e}")
        return [wav_path]

    segments = sorted(work_dir.glob(f"{base_name}_seg_*.wav"))
    if not segments:
        progress("No se generaron segmentos, se usará el archivo completo.")
        return [wav_path]

    progress(f"Se generaron {len(segments)} segmentos para {wav_path.name}.")
    return segments



def preprocess_audio(
    input_path: Path,
    output_path: Path,
    use_ffmpeg: bool,
    progress: Callable[[str], None]
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not use_ffmpeg:
        if input_path != output_path:
            output_path.write_bytes(input_path.read_bytes())
        progress("ffmpeg no disponible: se copia el audio sin procesar.")
        return output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-af", "loudnorm,afftdn=nf=-20",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        progress("Audio preprocesado (mono, 16k, reducción de ruido).")
    except subprocess.CalledProcessError as e:
        progress(f"ERROR en ffmpeg, se usará archivo original. Detalle: {e}")
        logger.exception("ffmpeg error")
        if input_path != output_path:
            output_path.write_bytes(input_path.read_bytes())

    return output_path


def transcribe_audio(
    client: OpenAI,
    cfg: TranscriberConfig,
    audio_path: Path,
    progress: Callable[[str], None]
) -> str:
    progress("Enviando a la API de transcripción...")
    with audio_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model=cfg.stt_model,
            file=f,
            language=cfg.language,
            prompt=cfg.whisper_prompt or None,
            temperature=0,
            response_format="json"
        )
    text = resp.text
    progress("Transcripción recibida.")
    return text


def chat_process(
    client: OpenAI,
    cfg: TranscriberConfig,
    transcript: str,
    system_prompt: str,
    progress: Callable[[str], None],
    etapa: str
) -> str:
    progress(f"Iniciando etapa de {etapa} con modelo de chat...")
    resp = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ],
        temperature=0
    )
    out = resp.choices[0].message.content.strip()
    progress(f"Etapa de {etapa} finalizada.")
    return out


def transcribe_audio_whisperx_local(
    audio_path: Path,
    cfg: TranscriberConfig,
    progress: ProgressFn,
) -> str:
    """
    Transcribe audio de forma local usando WhisperX (sin diarización).

    Usa cfg.whisperx_model como backend real (por ejemplo:
    'openai/whisper-large-v3-turbo', 'small', etc.).
    """
    if not HAS_WHISPERX:
        raise RuntimeError(
            "Se ha seleccionado un modelo STT local basado en WhisperX, "
            "pero el paquete 'whisperx' no está instalado.\n"
            "Instálalo con:\n"
            "    pip install whisperx"
        )

    # Importamos torch para decidir si hay CUDA disponible
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore

    language = cfg.language or "es"
    model_name = getattr(cfg, "whisperx_model", None) or "large-v2"

    # Elegir dispositivo: GPU si hay, si no CPU
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # En GPU usamos float16, en CPU int8 (más liviano)
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "WhisperX(local): cargando modelo '%s' en dispositivo '%s' (compute_type=%s)",
        model_name,
        device,
        compute_type,
    )
    progress(f"Cargando modelo local WhisperX ({model_name}) en '{device}'...")

    def _load_model_with(device_to_use: str, compute_type_to_use: str):
        return whisperx.load_model(
            model_name,
            device=device_to_use,
            language=language,
            compute_type=compute_type_to_use,
        )

    try:
        model = _load_model_with(device, compute_type)
    except (RuntimeError, ValueError) as e:
        msg = str(e)
        if (
            "no CUDA-capable device is detected" in msg
            or "CUDA" in msg
            or "float16" in msg
        ):
            logger.warning(
                "WhisperX(local): error en dispositivo '%s' (error: %s). Probando en CPU con int8...",
                device,
                msg,
            )
            progress(
                "CUDA/float16 no disponibles; usando CPU (int8) para transcripción local. "
                "Esto puede ser más lento."
            )
            device = "cpu"
            compute_type = "int8"
            model = _load_model_with(device, compute_type)
        else:
            # Otro error: lo propagamos
            raise

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=language)

    # Intentamos primero el campo 'text'
    text = (result.get("text") or "").strip()

    # Si no viene, concatenamos los segmentos
    if not text:
        segments = result.get("segments") or []
        pieces = [(seg.get("text") or "").strip() for seg in segments]
        text = " ".join(p for p in pieces if p)

    progress("Transcripción local WhisperX completada.")
    return text.strip()




def diarize_and_transcribe_local(
    audio_path: Path,
    cfg: TranscriberConfig,
    progress: ProgressFn,
) -> str:
    """
    Usa WhisperX para hacer transcripción + diarización local.

    - Intenta primero en CUDA (si está disponible).
    - Si falla por "no CUDA-capable device" o no hay CUDA, cae a CPU.
    - Ajusta compute_type según el dispositivo (float16 en GPU, int8 en CPU).
    - Informa en el log y en la barra de estado qué dispositivo usó.
    """
    if not HAS_WHISPERX:
        raise RuntimeError(
            "La diarización local está activada, pero el paquete 'whisperx' no está instalado.\n"
            "Instálalo con:\n"
            "    pip install whisperx"
        )

    # Importamos torch solo para chequear si hay CUDA
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore

    language = cfg.language or "es"
    model_name = getattr(cfg, "whisperx_model", None) or "large-v2"

    # --- Selección inicial de dispositivo ---
    # Si hay CUDA disponible, intentamos 'cuda'; si no, vamos directo a 'cpu'.
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # compute_type recomendado:
    # - GPU: float16 (rápido)
    # - CPU: int8 (más liviano, evita el error de float16 no soportado)
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "WhisperX: intentando cargar modelo '%s' en dispositivo '%s' (compute_type=%s)",
        model_name,
        device,
        compute_type,
    )
    progress(f"Intentando cargar WhisperX ({model_name}) en dispositivo '{device}'...")

    # --- 1) Cargar modelo ASR con helper y fallback ---
    def _load_model_with(device_to_use: str, compute_type_to_use: str):
        return whisperx.load_model(
            model_name,
            device=device_to_use,
            language=language,
            compute_type=compute_type_to_use,
        )

    try:
        model = _load_model_with(device, compute_type)
    except (RuntimeError, ValueError) as e:
        msg = str(e)
        # Errores típicos: no CUDA o float16 no soportado
        if (
            "no CUDA-capable device is detected" in msg
            or "CUDA" in msg
            or "float16" in msg
        ):
            logger.warning(
                "WhisperX: error en dispositivo '%s' (error: %s). Probando en CPU con int8...",
                device,
                msg,
            )
            progress(
                "CUDA/float16 no disponibles; usando CPU (int8) para diarización. "
                "Esto puede ser más lento."
            )
            device = "cpu"
            compute_type = "int8"
            model = _load_model_with(device, compute_type)
        else:
            # otro error distinto -> lo propagamos
            raise

    logger.info(
        "WhisperX: modelo '%s' cargado en dispositivo '%s' (compute_type=%s)",
        model_name,
        device,
        compute_type,
    )
    progress(f"WhisperX cargado en dispositivo '{device}' (compute_type={compute_type}).")

    # --- 2) Transcripción base ---
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=language)
    segments = result.get("segments", [])
    if not segments:
        progress("AVISO: WhisperX no devolvió segmentos, se devuelve texto vacío.")
        return ""

    # --- 3) Alineación (opcional) ---
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result_aligned = whisperx.align(segments, model_a, metadata, audio, device)
    except Exception as e:
        logger.warning("WhisperX: no se pudo cargar/alinear modelo de alineación: %s", e)
        progress("AVISO: no se pudo hacer alineación fina, se continúa igual.")
        result_aligned = {"segments": segments}

    # --- 4) Diarización (pyannote vía Hugging Face) ---
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Para usar la diarización de WhisperX necesitas definir la variable de entorno\n"
            "HUGGINGFACE_TOKEN con tu token de Hugging Face (pyannote.audio)."
        )

    if DiarizationPipeline is None:
        raise RuntimeError(
            "La versión instalada de whisperx no expone DiarizationPipeline en whisperx.diarize.\n"
            "Verifica que tienes una versión reciente de whisperx (pip install -U whisperx)."
        )

    progress("Cargando pipeline de diarización (pyannote.audio)...")
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token,
        device=device,
    )

    diarize_segments = diarize_model(audio)

    result_diarized = whisperx.assign_word_speakers(
        diarize_segments,
        result_aligned,
    )

    diar_segments = result_diarized.get("segments", [])
    if not diar_segments:
        progress("AVISO: no se generaron segmentos diarizados, se usa transcripción sin speakers.")
        return result.get("text", "")

    # --- 5) Calcular cuánto habla cada SPEAKER_XX ---
    speaker_durations: dict[str, float] = {}
    for seg in diar_segments:
        spk = seg.get("speaker", "SPEAKER")
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start))
        except Exception:
            start, end = 0.0, 0.0
        dur = max(0.0, end - start)
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur

    # Ordenamos por tiempo total de habla (más a menos)
    sorted_speakers = sorted(
        speaker_durations.items(), key=lambda kv: kv[1], reverse=True
    )

    interviewer_name = getattr(cfg, "interviewer_name", "") or ""
    interviewee_name = getattr(cfg, "interviewee_name", "") or ""

    speaker_labels: dict[str, str] = {}

    # Heurística simple:
    # - quien más habla => ENTREVISTADO/A
    # - quien le sigue => ENTREVISTADOR/A
    main_spk = sorted_speakers[0][0] if len(sorted_speakers) >= 1 else None
    second_spk = sorted_speakers[1][0] if len(sorted_speakers) >= 2 else None

    if main_spk is not None:
        if interviewee_name:
            speaker_labels[main_spk] = f"ENTREVISTADO/A ({interviewee_name})"
        else:
            speaker_labels[main_spk] = "ENTREVISTADO/A"

    if second_spk is not None:
        if interviewer_name:
            speaker_labels[second_spk] = f"ENTREVISTADOR/A ({interviewer_name})"
        else:
            speaker_labels[second_spk] = "ENTREVISTADOR/A"

    # El resto de speakers (si los hay) se quedan con su etiqueta original
    for spk in speaker_durations.keys():
        speaker_labels.setdefault(spk, spk)

    # --- 6) Construir texto tipo guion con etiquetas humanas ---
    lines: list[str] = []
    current_speaker_label: str | None = None

    for seg in diar_segments:
        raw_speaker = seg.get("speaker", "SPEAKER")
        speaker_label = speaker_labels.get(raw_speaker, raw_speaker)
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        if speaker_label != current_speaker_label:
            if current_speaker_label is not None:
                lines.append("")  # línea en blanco separando bloques
            lines.append(f"{speaker_label}: {text}")
            current_speaker_label = speaker_label
        else:
            # misma persona, seguimos abajo
            lines.append(text)

    progress("Diarización y transcripción local completadas.")
    return "\n".join(lines)



def run_batch(
    cfg: TranscriberConfig,
    progress: ProgressFn,
    set_progress: Optional[ProgressBarFn] = None,
    on_file_done: Optional[Callable[[Path], None]] = None
):
    if set_progress is None:
        set_progress = lambda x: None  # no-op

    logger.info("=== Inicio de proceso de transcripción por lote ===")
    logger.info(f"Input: {cfg.input_dir} | Output: {cfg.output_dir}")

    if not cfg.api_key:
        raise RuntimeError("No se proporcionó API key de OpenAI.")

    client = OpenAI(api_key=cfg.api_key)

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {cfg.input_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    use_ffmpeg = cfg.use_ffmpeg and check_ffmpeg()
    if cfg.use_ffmpeg and not use_ffmpeg:
        progress("AVISO: se solicitó ffmpeg pero no se encontró. Se continuará sin preprocesamiento.")
    elif use_ffmpeg:
        progress("ffmpeg detectado: se aplicará reducción de ruido y normalización.")

    if cfg.selected_files:
        audio_files = list(cfg.selected_files)
    else:
        audio_files = [
            p for p in cfg.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".mp4"}
        ]

    audio_files.sort()


    if not audio_files:
        progress("No se encontraron archivos de audio en la carpeta de entrada.")
        logger.info("No hay archivos de audio para procesar.")
        set_progress(0.0)
        return

    total_files = len(audio_files)
    progress(f"Se encontraron {total_files} archivos para procesar.")
    set_progress(0.0)

    for idx, audio in enumerate(audio_files, start=1):
        progress(f"\n[{idx}/{total_files}] Procesando: {audio.name}")
        logger.info("Procesando archivo: %s", audio)

        base_name = audio.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # rango de progreso para este archivo
        file_base = (idx - 1) / total_files
        file_span = 1.0 / total_files

        def stage(pct_in_file: float):
            """Actualiza barra dentro del rango de este archivo (0–1)."""
            set_progress(file_base + pct_in_file * file_span)

        try:
            processed_path = cfg.work_dir / f"{base_name}_proc.wav"
            processed_path = preprocess_audio(audio, processed_path, use_ffmpeg, progress)
            stage(0.1)

            # --- Selección de ruta STT según modelo ---
            is_local_whisperx = cfg.stt_model.startswith("whisperx-")

            if is_local_whisperx and cfg.use_diarization:
                # 👉 Modelo local WhisperX + diarización
                progress(
                    "Modelo STT local (WhisperX) seleccionado + diarización local activada. "
                    "Se usará WhisperX para transcribir y etiquetar hablantes."
                )
                raw_text = diarize_and_transcribe_local(processed_path, cfg, progress)
                stage(0.8)

            elif is_local_whisperx:
                # 👉 Modelo local WhisperX sin diarización
                progress(
                    "Modelo STT local (WhisperX) seleccionado sin diarización. "
                    "No se llamará a la API de audio de OpenAI para STT."
                )
                raw_text = transcribe_audio_whisperx_local(processed_path, cfg, progress)
                stage(0.8)

            else:
                # 👉 Modo clásico: segmentación + API de audio de OpenAI
                segments = segment_wav_if_needed(
                    processed_path, base_name, cfg.work_dir, progress
                )

                progress("Enviando a la API de transcripción...")
                stage(0.2)

                raw_parts: list[str] = []

                if segments:
                    per_seg_span = 0.6 / len(segments)
                else:
                    per_seg_span = 0.6

                for i, seg in enumerate(segments):
                    progress(f"Transcribiendo segmento {i+1}/{len(segments)}: {seg.name}")
                    stage(0.2 + per_seg_span * i)

                    segment_text = transcribe_audio(client, cfg, seg, progress)

                    raw_parts.append(
                        f"\n\n[=== SEGMENTO {i+1}/{len(segments)} ===]\n\n{segment_text.strip()}"
                    )

                raw_text = "\n".join(raw_parts).strip()
                stage(0.8)



            raw_out = cfg.output_dir / f"{base_name}.{timestamp}.raw.txt"
            raw_out.write_text(raw_text, encoding="utf-8")
            progress(f"Transcripción cruda guardada en: {raw_out.name}")
            stage(0.82)

            clean_text = raw_text
            did_clean = False
            if cfg.do_clean and cfg.clean_prompt.strip():
                try:
                    clean_text = chat_process(
                        client, cfg, raw_text, cfg.clean_prompt, progress, "limpieza"
                    )
                    did_clean = True
                    stage(0.9)
                except Exception as e:
                    progress(f"ERROR en limpieza, se usará texto crudo. Detalle: {e}")
                    logger.exception("Error en limpieza")
                    clean_text = raw_text

            if cfg.do_clean:
                clean_out = cfg.output_dir / f"{base_name}.{timestamp}.clean.txt"
                clean_out.write_text(clean_text, encoding="utf-8")
                progress(f"Versión limpia guardada en: {clean_out.name}")
                if did_clean:
                    stage(0.92)

            if cfg.do_summary and cfg.summary_prompt.strip():
                try:
                    resumen_text = chat_process(
                        client, cfg, clean_text, cfg.summary_prompt, progress, "resumen"
                    )
                    resumen_out = cfg.output_dir / f"{base_name}.{timestamp}.resumen.txt"
                    resumen_out.write_text(resumen_text, encoding="utf-8")
                    progress(f"Resumen guardado en: {resumen_out.name}")
                    stage(0.98)
                except Exception as e:
                    progress(f"ERROR en resumen. Detalle: {e}")
                    logger.exception("Error en resumen")

            progress(f"Archivo finalizado: {audio.name}")
            stage(1.0)

            if on_file_done:
                on_file_done(audio)

        except Exception as e:
            msg = f"ERROR procesando {audio.name}: {e}"
            progress(msg)
            logger.exception("Error procesando archivo: %s", audio)
            stage(1.0)

    progress("\nProceso completado.")
    set_progress(1.0)
    logger.info("=== Fin del proceso de transcripción por lote ===")



def run_analysis_only(
    cfg: TranscriberConfig,
    progress: ProgressFn,
    set_progress: Optional[ProgressBarFn] = None,
    on_file_done: Optional[Callable[[Path], None]] = None
):
    if set_progress is None:
        set_progress = lambda x: None

    logger.info("=== Inicio de análisis de textos por lote ===")
    logger.info(f"Input textos: {cfg.input_dir} | Output: {cfg.output_dir}")

    if not cfg.api_key:
        raise RuntimeError("No se proporcionó API key de OpenAI.")

    client = OpenAI(api_key=cfg.api_key)

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {cfg.input_dir}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.selected_files:
        text_files = list(cfg.selected_files)
    else:
        text_files = [
            p for p in cfg.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".txt"}
        ]

    text_files.sort()


    if not text_files:
        progress("No se encontraron archivos .txt en la carpeta de entrada.")
        logger.info("No hay archivos de texto para analizar.")
        set_progress(0.0)
        return

    total_files = len(text_files)
    progress(f"Se encontraron {total_files} archivos de texto para analizar.")
    set_progress(0.0)

    for idx, txt in enumerate(text_files, start=1):
        progress(f"\n[{idx}/{total_files}] Analizando: {txt.name}")
        logger.info("Analizando archivo de texto: %s", txt)

        base_name = txt.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_base = (idx - 1) / total_files
        file_span = 1.0 / total_files

        def stage(pct_in_file: float):
            set_progress(file_base + pct_in_file * file_span)

        try:
            raw_text = txt.read_text(encoding="utf-8")
            stage(0.1)

            clean_text = raw_text
            if cfg.do_clean and cfg.clean_prompt.strip():
                try:
                    clean_text = chat_process(
                        client, cfg, raw_text, cfg.clean_prompt, progress, "limpieza"
                    )
                    stage(0.6)
                except Exception as e:
                    progress(f"ERROR en limpieza. Detalle: {e}")
                    logger.exception("Error en limpieza (solo análisis)")
                    clean_text = raw_text

            if cfg.do_clean:
                clean_out = cfg.output_dir / f"{base_name}.{timestamp}.clean.txt"
                clean_out.write_text(clean_text, encoding="utf-8")
                progress(f"Versión limpia guardada en: {clean_out.name}")
                stage(0.7)

            if cfg.do_summary and cfg.summary_prompt.strip():
                try:
                    resumen_text = chat_process(
                        client, cfg, clean_text, cfg.summary_prompt, progress, "resumen"
                    )
                    resumen_out = cfg.output_dir / f"{base_name}.{timestamp}.resumen.txt"
                    resumen_out.write_text(resumen_text, encoding="utf-8")
                    progress(f"Resumen guardado en: {resumen_out.name}")
                    stage(0.95)
                except Exception as e:
                    progress(f"ERROR en resumen. Detalle: {e}")
                    logger.exception("Error en resumen (solo análisis)")

            progress(f"Archivo de texto finalizado: {txt.name}")
            stage(1.0)

            if on_file_done:
                on_file_done(txt)

        except Exception as e:
            msg = f"ERROR procesando {txt.name}: {e}"
            progress(msg)
            logger.exception("Error procesando archivo de texto: %s", txt)
            stage(1.0)

    progress("\nAnálisis de textos completado.")
    set_progress(1.0)
    logger.info("=== Fin del análisis de textos por lote ===")

