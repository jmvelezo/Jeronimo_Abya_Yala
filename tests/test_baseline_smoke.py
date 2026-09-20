from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_transcriber as core

TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from compare_transcripts import text_metrics, keyterm_metrics


class CoreBaselineSmokeTests(unittest.TestCase):
    def test_app_version_frozen(self):
        self.assertTrue(bool(core.APP_VERSION))

    def test_default_local_model_frozen(self):
        self.assertEqual(core.LOCAL_FAST_WHISPER_DEFAULT_MODEL, "large-v3")
        self.assertIn("large-v3-turbo", core.LOCAL_FAST_WHISPER_MODELS)

    def test_text_split_preserves_content(self):
        source = "Uno dos tres.\n\nCuatro cinco seis.\n\nSiete ocho nueve."
        chunks = core._split_text_for_llm(source, max_chars=20)
        self.assertGreaterEqual(len(chunks), 2)
        rebuilt = " ".join(" ".join(ch.split()) for ch in chunks)
        for token in ("Uno", "Cuatro", "nueve"):
            self.assertIn(token, rebuilt)

    def test_manifest_privacy_mode_local(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            cfg = core.TranscriberConfig(
                api_key="",
                input_dir=p,
                output_dir=p,
                work_dir=p,
                stt_engine=core.STT_ENGINE_LOCAL_FAST_WHISPER,
                text_engine=core.TEXT_ENGINE_OLLAMA,
            )
            self.assertEqual(core._manifest_privacy_mode(cfg), "local")


class MetricsSmokeTests(unittest.TestCase):
    def test_identical_text_zero_error(self):
        m = text_metrics("Hola, esto es una prueba.", "Hola esto es una prueba")
        self.assertEqual(m["wer"], 0.0)
        self.assertEqual(m["cer"], 0.0)

    def test_deletion_detected(self):
        m = text_metrics("uno dos tres", "uno tres")
        self.assertEqual(m["deletions"], 1)
        self.assertGreater(m["wer"], 0.0)

    def test_keyterm_recall(self):
        m = keyterm_metrics("Freire y Bourdieu", "Freire solamente", ["Freire", "Bourdieu"])
        self.assertEqual(m["reference_occurrences"], 2)
        self.assertEqual(m["hits"], 1)
        self.assertAlmostEqual(m["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
