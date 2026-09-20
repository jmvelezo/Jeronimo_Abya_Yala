from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import core_transcriber as core
import job_engine as jobs
import runtime_isolation as isolation


ASR_BASELINE_HASHES = {
    "transcribe_audio_local_faster_whisper": "c30b4fa79ca7baac2cbdfbdbed22857ab7fe23497a82af7b2a5860ff5d1db1d1",
    "transcribe_audio": "381f1b26bac90fb8eda53138a570b34edaf5ec41ba2b9b3257aa22f77dbf583c",
    "diarize_and_transcribe_local": "91a86a03fcbef536e5a1085bdfae566118ffaaa876b55b426e7f6944d353bbcf",
}


class _Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
    def advance(self, seconds: float):
        self.t += seconds


class Phase5JobTrackerTests(unittest.TestCase):
    def test_eta_uses_observed_runtime_not_hardware_table(self):
        clock = _Clock()
        tracker = jobs.JobTracker(monotonic=clock)
        tracker.start()
        clock.advance(10)
        event = tracker.update(progress=0.25, stage=jobs.STAGE_TRANSCRIBING)
        self.assertIsNotNone(event.eta_seconds)
        self.assertAlmostEqual(event.eta_seconds, 30.0, delta=1.0)

    def test_progress_never_goes_backwards(self):
        tracker = jobs.JobTracker()
        tracker.start()
        tracker.update(progress=0.5)
        event = tracker.update(progress=0.2)
        self.assertEqual(event.progress, 0.5)

    def test_job_payload_does_not_require_interview_content(self):
        job = jobs.Job(kind="transcription")
        payload = job.to_dict()
        self.assertIn("job_id", payload)
        self.assertNotIn("transcript", payload)
        self.assertNotIn("audio", payload)

    def test_complete_sets_terminal_state(self):
        tracker = jobs.JobTracker()
        tracker.start()
        event = tracker.complete()
        self.assertEqual(event.status, jobs.JOB_STATUS_COMPLETED)
        self.assertEqual(event.progress, 1.0)
        self.assertEqual(event.eta_seconds, 0.0)


class Phase5CoreEventTests(unittest.TestCase):
    def test_core_emit_job_callback_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            cfg = core.TranscriberConfig(api_key="", input_dir=p, output_dir=p, work_dir=p)
            cfg.job_event_callback = lambda _e: (_ for _ in ()).throw(RuntimeError("callback"))
            core._emit_job_event(cfg, stage=jobs.STAGE_PREPARING, progress=0.1, message="x")

    def test_default_diarization_path_remains_in_process(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            cfg = core.TranscriberConfig(api_key="", input_dir=p, output_dir=p, work_dir=p)
            with patch.object(core, "diarize_and_transcribe_local", return_value="baseline") as direct, \
                 patch.object(core, "DiarizationWorkerClient") as worker:
                out = core._diarize_via_optional_worker(p / "a.wav", cfg, lambda _m: None)
            self.assertEqual(out, "baseline")
            direct.assert_called_once()
            worker.assert_not_called()

    def test_configured_runtime_uses_worker_without_changing_diarization_function(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            runtime = p / "python.exe"
            runtime.write_text("stub", encoding="utf-8")
            cfg = core.TranscriberConfig(
                api_key="", input_dir=p, output_dir=p, work_dir=p,
                diarization_runtime_python=str(runtime), use_diarization=True,
            )
            fake = unittest.mock.MagicMock()
            fake.run.return_value = {"type": "result", "text": "aislado"}
            with patch.object(core, "DiarizationWorkerClient", return_value=fake), \
                 patch.object(core, "diarize_and_transcribe_local") as direct:
                out = core._diarize_via_optional_worker(p / "a.wav", cfg, lambda _m: None)
            self.assertEqual(out, "aislado")
            direct.assert_not_called()
            self.assertEqual(fake.run.call_args.kwargs["cancel_check"], cfg.cancel_check)

    def test_batch_with_file_error_finishes_job_as_error(self):
        import wave
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "in"; out = root / "out"; work = root / "work"
            inp.mkdir(); out.mkdir(); work.mkdir()
            audio = inp / "entrevista.wav"
            with wave.open(str(audio), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(8000); wf.writeframes(b"\x00\x00" * 100)
            events = []
            cfg = core.TranscriberConfig(
                api_key="", input_dir=inp, output_dir=out, work_dir=work, selected_files=[audio],
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER, use_ffmpeg=False, do_clean=False, do_summary=False,
                export_docx_raw=False, export_docx_atlas=False, export_rtf_raw=False, export_rtf_atlas=False,
                job_event_callback=events.append,
            )
            with patch.object(core, "transcribe_audio_local_faster_whisper", side_effect=RuntimeError("fallo controlado")):
                core.run_batch(cfg, lambda _m: None)
            self.assertEqual(events[-1]["status"], jobs.JOB_STATUS_ERROR)
            self.assertEqual(events[-1]["stage"], jobs.STAGE_ERROR)


class Phase5RuntimeIsolationTests(unittest.TestCase):
    def test_assessment_does_not_force_isolation_without_evidence(self):
        report = isolation.assess_runtime("")
        self.assertFalse(report.isolation_required)
        self.assertFalse(report.configured_worker_available)

    def test_capture_profile_contains_versions_not_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            path = isolation.capture_runtime_profile(Path(td) / "profile.json")
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
            self.assertEqual(payload["purpose"], "diarization_runtime_baseline")
            self.assertNotIn("token", text.lower())
            self.assertNotIn("api_key", text.lower())

    def test_json_worker_protocol_returns_result(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "worker.py"
            script.write_text(textwrap.dedent('''
                import json, sys
                req = json.loads(sys.stdin.readline())
                print(json.dumps({"type":"event","message":"working"}), flush=True)
                print(json.dumps({"type":"result","value":req.get("value")}), flush=True)
            '''), encoding="utf-8")
            events = []
            client = isolation.DiarizationWorkerClient(sys.executable, script)
            result = client.run({"value": 42}, on_event=events.append)
            self.assertEqual(result["value"], 42)
            self.assertEqual(events[0]["message"], "working")

    def test_non_json_stdout_does_not_abort_valid_worker_result(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "worker.py"
            script.write_text(textwrap.dedent('''
                import json, sys
                json.loads(sys.stdin.readline())
                print("aviso externo no JSON", flush=True)
                print(json.dumps({"type":"result","value":7}), flush=True)
            '''), encoding="utf-8")
            client = isolation.DiarizationWorkerClient(sys.executable, script)
            result = client.run({})
            self.assertEqual(result["value"], 7)

    def test_worker_error_preserves_auxiliary_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "worker.py"
            script.write_text(textwrap.dedent('''
                import json, sys
                json.loads(sys.stdin.readline())
                print("mensaje de biblioteca", flush=True)
                print("detalle CUDA en stderr", file=sys.stderr, flush=True)
                print(json.dumps({"type":"error","message":"fallo real"}), flush=True)
                raise SystemExit(1)
            '''), encoding="utf-8")
            client = isolation.DiarizationWorkerClient(sys.executable, script)
            with self.assertRaises(RuntimeError) as ctx:
                client.run({})
            text = str(ctx.exception)
            self.assertIn("fallo real", text)
            self.assertIn("mensaje de biblioteca", text)
            self.assertIn("detalle CUDA", text)

    def test_worker_can_be_preemptively_cancelled(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "worker.py"
            script.write_text(textwrap.dedent('''
                import sys, time
                sys.stdin.readline()
                time.sleep(30)
            '''), encoding="utf-8")
            client = isolation.DiarizationWorkerClient(sys.executable, script)
            with self.assertRaises(InterruptedError):
                client.run({}, cancel_check=lambda: True, poll_interval=0.01)


class Phase5RegressionTests(unittest.TestCase):
    def test_asr_and_diarization_implementations_are_unchanged(self):
        for name, expected in ASR_BASELINE_HASHES.items():
            source = inspect.getsource(getattr(core, name)).encode("utf-8")
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected, name)

    def test_phase5_version_floor(self):
        # FASE 5 froze the jobs/runtime capability; later phases may advance APP_VERSION.
        self.assertTrue(core.APP_VERSION.startswith(("0.6.0-phase5", "0.7.0-phase6", "0.8.0-phase7", "1.0")))


if __name__ == "__main__":
    unittest.main()
