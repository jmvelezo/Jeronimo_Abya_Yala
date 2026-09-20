from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_transcriber as core
import credential_store as creds


def _write_wav(path: Path, seconds: float = 0.05) -> None:
    rate = 8000
    frames = max(1, int(rate * seconds))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)


class _FakeKeyring:
    def __init__(self):
        self.data = {}

    def get_password(self, service, account):
        return self.data.get((service, account))

    def set_password(self, service, account, value):
        self.data[(service, account)] = value

    def delete_password(self, service, account):
        self.data.pop((service, account), None)


class Phase2CredentialTests(unittest.TestCase):
    def test_secure_store_roundtrip(self):
        fake = _FakeKeyring()
        with mock.patch.object(creds, "_load_keyring", return_value=fake), \
             mock.patch.dict("os.environ", {}, clear=True):
            creds.set_secure_secret("openai", "sk-test")
            self.assertEqual(creds.get_secure_secret("openai"), "sk-test")
            self.assertEqual(creds.resolve_secret("openai"), "sk-test")
            self.assertEqual(creds.credential_source("openai"), "keyring")
            self.assertTrue(creds.delete_secure_secret("openai"))
            self.assertEqual(creds.resolve_secret("openai"), "")

    def test_explicit_secret_has_priority(self):
        fake = _FakeKeyring()
        with mock.patch.object(creds, "_load_keyring", return_value=fake), \
             mock.patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}, clear=True):
            creds.set_secure_secret("openai", "stored-key")
            self.assertEqual(creds.resolve_secret("openai", "session-key"), "session-key")
            self.assertEqual(creds.credential_source("openai", "session-key"), "session")


class Phase2TempAndFfmpegTests(unittest.TestCase):
    def test_preprocess_without_ffmpeg_uses_original_without_fake_wav(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "entrevista.mp3"
            source.write_bytes(b"not-real-mp3-but-source")
            fake_wav = root / "audio_proc.wav"
            out = core.preprocess_audio(source, fake_wav, False, lambda _m: None)
            self.assertEqual(out, source)
            self.assertFalse(fake_wav.exists())

    def test_preprocess_ffmpeg_failure_returns_original(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "entrevista.mp3"
            source.write_bytes(b"source")
            target = root / "audio_proc.wav"
            with mock.patch.object(core.subprocess, "run", side_effect=core.subprocess.CalledProcessError(1, ["ffmpeg"])):
                out = core.preprocess_audio(source, target, True, lambda _m: None)
            self.assertEqual(out, source)
            self.assertFalse(target.exists())


    def test_preprocess_default_keeps_baseline_ffmpeg_filter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "audio.wav"
            _write_wav(source)
            target = root / "audio_proc.wav"
            with mock.patch.object(core.subprocess, "run") as run_mock:
                out = core.preprocess_audio(source, target, True, lambda _m: None, True)
            self.assertEqual(out, target)
            cmd = run_mock.call_args.args[0]
            self.assertIn("-af", cmd)
            self.assertIn("loudnorm,afftdn=nf=-20", cmd)

    def test_preprocess_can_convert_without_denoise(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "audio.wav"
            _write_wav(source)
            target = root / "audio_proc.wav"
            with mock.patch.object(core.subprocess, "run") as run_mock:
                core.preprocess_audio(source, target, True, lambda _m: None, False)
            cmd = run_mock.call_args.args[0]
            self.assertNotIn("-af", cmd)
            self.assertIn("16000", cmd)

    def test_non_wav_segmentation_does_not_open_as_wave(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "audio.mp3"
            source.write_bytes(b"x")
            with mock.patch.object(core, "get_wav_duration_seconds", side_effect=AssertionError("no debe llamarse")):
                parts = core.segment_wav_if_needed(source, "audio", root, lambda _m: None)
            self.assertEqual(parts, [source])

    def test_run_batch_cleans_per_file_temp_directory_on_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"
            out = root / "out"
            work = root / "work"
            inp.mkdir(); out.mkdir(); work.mkdir()
            audio = inp / "entrevista.wav"
            _write_wav(audio)
            cfg = core.TranscriberConfig(
                api_key="",
                input_dir=inp,
                output_dir=out,
                work_dir=work,
                selected_files=[audio],
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER,
                use_ffmpeg=False,
                do_clean=False,
                do_summary=False,
                export_docx_raw=False,
                export_docx_atlas=False,
                export_rtf_raw=False,
                export_rtf_atlas=False,
            )
            with mock.patch.object(core, "transcribe_audio_local_faster_whisper", return_value="texto correcto"):
                core.run_batch(cfg, lambda _m: None)
            self.assertFalse(any(p.name.startswith(".jeronimo_tmp_") for p in work.iterdir()))
            self.assertEqual(len(list(out.glob("*.raw.txt"))), 1)
            manifest = json.loads(next(out.glob("*.manifest.json")).read_text(encoding="utf-8"))
            self.assertEqual(manifest["temporary_files_policy"], "automatic_cleanup")
            self.assertFalse(manifest["use_ffmpeg"])

    def test_run_batch_cleans_temp_directory_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"; out = root / "out"; work = root / "work"
            inp.mkdir(); out.mkdir(); work.mkdir()
            audio = inp / "entrevista.wav"
            _write_wav(audio)
            cfg = core.TranscriberConfig(
                api_key="",
                input_dir=inp,
                output_dir=out,
                work_dir=work,
                selected_files=[audio],
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER,
                use_ffmpeg=False,
                do_clean=False,
                do_summary=False,
                export_docx_raw=False,
                export_docx_atlas=False,
                export_rtf_raw=False,
                export_rtf_atlas=False,
            )
            with mock.patch.object(core, "transcribe_audio_local_faster_whisper", side_effect=RuntimeError("fallo controlado")):
                core.run_batch(cfg, lambda _m: None)
            self.assertFalse(any(p.name.startswith(".jeronimo_tmp_") for p in work.iterdir()))
            manifest = json.loads(next(out.glob("*.manifest.json")).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "error")


class Phase2PrivacyAndCancellationTests(unittest.TestCase):
    def test_error_sanitizer_redacts_session_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = core.TranscriberConfig(api_key="sk-secret-123", input_dir=root, output_dir=root, work_dir=root)
            text = core._sanitize_error_message("falló sk-secret-123", cfg)
            self.assertNotIn("sk-secret-123", text)
            self.assertIn("[REDACTED]", text)

    def test_privacy_file_ref_does_not_contain_filename(self):
        ref = core._privacy_file_ref(Path("Entrevista_Juan_Perez_01.mp3"))
        self.assertNotIn("Juan", ref)
        self.assertTrue(ref.endswith(".mp3"))

    def test_cancel_before_processing_creates_cancelled_manifest_and_no_temp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"; out = root / "out"; work = root / "work"
            inp.mkdir(); out.mkdir(); work.mkdir()
            audio = inp / "entrevista.wav"
            _write_wav(audio)
            cfg = core.TranscriberConfig(
                api_key="",
                input_dir=inp,
                output_dir=out,
                work_dir=work,
                selected_files=[audio],
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER,
                use_ffmpeg=False,
                do_clean=False,
                do_summary=False,
                export_docx_raw=False,
                export_docx_atlas=False,
                export_rtf_raw=False,
                export_rtf_atlas=False,
                cancel_check=lambda: True,
            )
            core.run_batch(cfg, lambda _m: None)
            self.assertFalse(any(p.name.startswith(".jeronimo_tmp_") for p in work.iterdir()))
            manifest = json.loads(next(out.glob("*.manifest.json")).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
