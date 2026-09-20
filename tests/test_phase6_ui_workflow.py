from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

import core_transcriber as core
import ui_workflow as flow

ASR_BASELINE_HASHES = {
    "transcribe_audio_local_faster_whisper": "c30b4fa79ca7baac2cbdfbdbed22857ab7fe23497a82af7b2a5860ff5d1db1d1",
    "transcribe_audio": "381f1b26bac90fb8eda53138a570b34edaf5ec41ba2b9b3257aa22f77dbf583c",
    "diarize_and_transcribe_local": "91a86a03fcbef536e5a1085bdfae566118ffaaa876b55b426e7f6944d353bbcf",
}
ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def _selection(self, root: Path, **kwargs):
        audio = root / "entrevista.wav"
        audio.write_bytes(b"RIFF")
        out = root / "out"
        data = dict(input_path=audio, output_dir=out)
        data.update(kwargs)
        return flow.WorkflowSelection(**data)

    def test_quality_profile_preserves_real_whisperx_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = flow.build_config(self._selection(Path(td), profile=flow.PROFILE_DIARIZATION))
        self.assertTrue(cfg.use_diarization)
        self.assertEqual(cfg.whisperx_model, "large-v2")

    def test_text_none_disables_derived_processing(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = flow.build_config(self._selection(Path(td), text_engine=flow.TEXT_NONE, do_summary=True, do_clean=True))
        self.assertFalse(cfg.do_summary)
        self.assertFalse(cfg.do_clean)

    def test_remote_ollama_is_external(self):
        with tempfile.TemporaryDirectory() as td:
            sel = self._selection(Path(td), text_engine=flow.TEXT_OLLAMA, do_summary=True, ollama_url="http://192.168.1.20:11434")
            privacy = flow.assess_privacy(sel)
        self.assertTrue(privacy.text_leaves_device)

    def test_new_identity_is_visible_and_classic_not_exposed(self):
        source = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        self.assertIn("Jerónimo Abya Yala", source)
        self.assertIn("Nueva desgrabación", source)
        self.assertNotIn('text="Interfaz clásica"', source)

    def test_only_two_batch_entrypoints_exist(self):
        bats = sorted(p.name for p in ROOT.rglob("*.bat") if ".venv" not in p.parts and ".build" not in p.parts)
        self.assertEqual(bats, ["COMPILAR_JERONIMO_ABYA_YALA.bat", "INICIAR_JERONIMO_ABYA_YALA.bat"])


class RegressionTests(unittest.TestCase):
    def test_asr_and_diarization_implementations_remain_unchanged(self):
        for name, expected in ASR_BASELINE_HASHES.items():
            source = inspect.getsource(getattr(core, name)).encode("utf-8")
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
