from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import local_text_runtime

ROOT = Path(__file__).resolve().parents[1]


class OllamaUxContractTests(unittest.TestCase):
    def test_manager_no_longer_blocks_when_private_ollama_is_stopped(self):
        source = (ROOT / "model_manager_ui.py").read_text(encoding="utf-8")
        self.assertNotIn("Ollama no responde. Inícialo o instálalo antes", source)
        self.assertIn("Descargar y usar recomendado", source)
        self.assertIn("Usar recomendado", source)
        self.assertIn("Jerónimo lo iniciará automáticamente", source)
        self.assertIn("auto_recommended", source)

    def test_missing_model_dialog_offers_recommended_path(self):
        source = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        self.assertIn("ask_recommended_model_action", source)
        self.assertIn("recommended_installed=rec_present", source)
        self.assertIn("auto_recommended=True", source)

    def test_onboarding_contains_developer_note_about_local_compute(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn("Nota del desarrollador", source)
        self.assertNotIn("Nota del desarrollador · José Manuel", source)
        self.assertIn("descentralizar esta tecnología", source)
        self.assertIn("modelos de código abierto", source)
        self.assertIn("disipación del calor", source)
        self.assertIn("Con cariño, José Manuel", source)
        self.assertIn("entre 10 y 30 gigas libres", source)
        self.assertIn("tarjetas gráficas NVIDIA", source)

    def test_recommended_download_also_selects_model(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn("_select_recommended_text_model", source)
        self.assertIn("instalado y seleccionado como modelo recomendado", source)


class PrivateOllamaRepairTests(unittest.TestCase):
    def test_ensure_repairs_installed_runtime_after_start_failure(self):
        fake_exe = type("FakePath", (), {"is_file": lambda self: True})()
        expected = {"available": True, "version": local_text_runtime.OLLAMA_RUNTIME_VERSION}
        with patch.object(local_text_runtime, "ollama_executable", return_value=fake_exe), \
             patch.object(local_text_runtime, "start_private_ollama_if_installed", side_effect=RuntimeError("broken")), \
             patch.object(local_text_runtime, "repair_private_ollama", return_value=expected) as repair:
            result = local_text_runtime.ensure_private_ollama()
        self.assertEqual(result, expected)
        repair.assert_called_once()

    def test_repair_removes_runtime_but_preserves_models_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = root / "runtime"
            models = root / "models"
            runtime.mkdir()
            models.mkdir()
            (runtime / "bad.dll").write_text("bad", encoding="utf-8")
            (models / "keep.model").write_text("keep", encoding="utf-8")

            with patch.object(local_text_runtime, "ollama_runtime_dir", return_value=runtime), \
                 patch.object(local_text_runtime, "ollama_models_dir", return_value=models), \
                 patch.object(local_text_runtime, "_stop_owned_process"), \
                 patch.object(local_text_runtime, "install_private_ollama", return_value=runtime / "ollama.exe"), \
                 patch.object(local_text_runtime, "start_private_ollama_if_installed", return_value=True), \
                 patch.object(local_text_runtime, "private_ollama_status", return_value={"available": True}):
                result = local_text_runtime.repair_private_ollama()

            self.assertEqual(result, {"available": True})
            self.assertFalse((runtime / "bad.dll").exists())
            self.assertTrue((models / "keep.model").exists())


if __name__ == "__main__":
    unittest.main()
