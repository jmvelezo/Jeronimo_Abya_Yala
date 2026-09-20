from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import automatic_setup
import onboarding
import team_credentials


class AutomaticSetupTests(unittest.TestCase):
    def _report(self, *, text_model="qwen3:14b", cuda=True, ollama=True):
        return SimpleNamespace(
            recommendations={"text_primary": text_model},
            ollama={"available": ollama},
            torch_cuda_available=cuda,
            ctranslate2_cuda_devices=1 if cuda else 0,
            disk={"free_bytes": 100 * 1024**3, "writable": True},
            hf_cached_models=[],
        )

    @patch("automatic_setup.is_asr_model_cached", return_value=False)
    def test_plan_uses_hardware_recommendation_and_baseline(self, _cached):
        plan = automatic_setup.build_automatic_plan(self._report(), [])
        self.assertEqual(plan.text_model, "qwen3:14b")
        self.assertEqual(plan.asr_model, "large-v2")
        self.assertTrue(plan.cuda)
        self.assertTrue(plan.needs_text_model)
        self.assertTrue(plan.needs_asr_model)

    @patch("automatic_setup.diarization_runtime_status")
    @patch("automatic_setup.is_asr_model_cached", return_value=False)
    def test_plan_marks_heavy_runtime_download_only_when_missing(self, _cached, runtime_status):
        runtime_status.return_value = SimpleNamespace(required=True, ready=False)
        report = self._report()
        report.gpus = [{"name": "NVIDIA"}]
        plan = automatic_setup.build_automatic_plan(report, [])
        self.assertTrue(plan.needs_diarization_runtime)
        self.assertGreaterEqual(plan.diarization_runtime_size_gb, 4.0)

    def test_team_credential_is_not_plaintext_constant(self):
        source = Path(team_credentials.__file__).read_text(encoding="utf-8")
        self.assertIn("_TEAM_HF_BLOB", source)
        self.assertNotIn("TEAM_HF_TOKEN =", source)
        token = team_credentials.get_embedded_team_hf_token()
        self.assertTrue(token.startswith("hf_"))
        self.assertGreater(len(token), 20)

    def test_onboarding_state_persists_mode_without_secrets(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            state = onboarding.complete_onboarding(path=path, setup_mode="automatic")
            raw = path.read_text(encoding="utf-8").lower()
        self.assertEqual(state.setup_mode, "automatic")
        self.assertNotIn("hf_", raw)
        self.assertNotIn("token", raw)

    def test_auto_mode_does_not_configure_external_api(self):
        source = Path("automatic_setup.py").read_text(encoding="utf-8")
        self.assertIn("No configura APIs externas", source)
        self.assertNotIn('set_secure_secret("openai"', source)

    def test_ui_offers_automatic_and_advanced_modes(self):
        source = Path("onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn("Usar configuración automática", source)
        self.assertIn("Configurar manualmente", source)
        self.assertIn("_start_automatic_setup", source)
        self.assertIn("run_automatic_setup", source)

    def test_huggingface_console_progress_is_disabled_for_gui_build(self):
        manager_src = Path("model_manager.py").read_text(encoding="utf-8")
        pyannote_src = Path("pyannote_model_manager.py").read_text(encoding="utf-8")
        for src in (manager_src, pyannote_src):
            self.assertIn("HF_HUB_DISABLE_PROGRESS_BARS", src)
            self.assertIn("disable_progress_bars", src)

    @patch("automatic_setup.set_secure_secret")
    @patch("automatic_setup.get_secure_secret", return_value="")
    def test_team_token_is_migrated_to_secure_store(self, _get, set_secret):
        token, source = automatic_setup.bootstrap_team_hf_credential()
        self.assertTrue(token.startswith("hf_"))
        self.assertEqual(source, "team→keyring")
        set_secret.assert_called_once()
        self.assertEqual(set_secret.call_args.args[0], "huggingface")

    @patch("automatic_setup.benchmark_ollama_model", return_value={"tokens_per_second": 12.5, "gpu_fraction": 1.0})
    @patch("automatic_setup.pull_ollama_model")
    @patch("automatic_setup.ensure_ollama")
    @patch("automatic_setup.ensure_pyannote_model")
    @patch("automatic_setup.download_faster_whisper_model")
    @patch("automatic_setup.pyannote_local_model_status", return_value={"ready": True, "path": "X", "revision": "r"})
    @patch("automatic_setup.ollama_inventory", return_value={"qwen3:14b": {}})
    @patch("automatic_setup.is_asr_model_cached", return_value=True)
    def test_automatic_setup_skips_models_already_present(
        self, _cached, _inv, _py_status, dl_asr, ensure_pyannote, ensure_ollama, pull_text, benchmark
    ):
        result = automatic_setup.run_automatic_setup(self._report())
        self.assertTrue(result["ok"])
        dl_asr.assert_not_called()
        ensure_pyannote.assert_not_called()
        ensure_ollama.assert_called_once()
        pull_text.assert_not_called()
        benchmark.assert_called_once_with("qwen3:14b", automatic_setup.DEFAULT_OLLAMA_URL)

    @patch("automatic_setup.delete_ollama_model")
    @patch("automatic_setup.pull_ollama_model")
    @patch("automatic_setup.ensure_ollama")
    @patch("automatic_setup.ensure_pyannote_model")
    @patch("automatic_setup.download_faster_whisper_model")
    @patch("automatic_setup.pyannote_local_model_status", return_value={"ready": True, "path": "X", "revision": "r"})
    @patch("automatic_setup.ollama_inventory", return_value={"qwen3:14b": {}})
    @patch("automatic_setup.is_asr_model_cached", return_value=True)
    @patch("automatic_setup.benchmark_ollama_model")
    def test_automatic_setup_falls_back_only_on_memory_failure(
        self, benchmark, _cached, _inv, _py_status, dl_asr, ensure_pyannote, ensure_ollama, pull_text, delete_text
    ):
        benchmark.side_effect = [RuntimeError("model requires more system memory"), {"tokens_per_second": 8.0, "gpu_fraction": 0.7}]
        result = automatic_setup.run_automatic_setup(self._report(text_model="qwen3:14b"))
        self.assertEqual(result["text_model"], "qwen3:8b")
        ensure_pyannote.assert_not_called()
        pull_text.assert_called_once()
        self.assertEqual(pull_text.call_args.args[0], "qwen3:8b")
        delete_text.assert_not_called()  # 14B ya existía antes del setup.

    @patch("automatic_setup.benchmark_ollama_model", return_value={"tokens_per_second": 10.0, "gpu_fraction": 0.5})
    @patch("automatic_setup.pull_ollama_model")
    @patch("automatic_setup.ensure_ollama")
    @patch("automatic_setup.validate_diarization_stack", return_value={"type": "result", "whisperx_model_load": True})
    @patch("automatic_setup.resolve_cached_faster_whisper_model", return_value=Path("X") / "large-v2")
    @patch("automatic_setup.download_faster_whisper_model")
    @patch("automatic_setup.ensure_pyannote_model", return_value={"ok": True, "source": "jeronimo-drive", "path": "X"})
    @patch("automatic_setup.pyannote_local_model_status", side_effect=[{"ready": False}, {"ready": False}, {"ready": True, "path": "X"}])
    @patch("automatic_setup.ollama_inventory", return_value={"qwen3:14b": {}})
    @patch("automatic_setup.is_asr_model_cached", return_value=True)
    def test_automatic_setup_prepares_pyannote_before_heavy_runtime(
        self, _cached, _inv, _status, ensure_pyannote, dl_asr, resolve_asr, validate_stack, ensure_ollama, pull_text, benchmark
    ):
        report = self._report()
        with patch("automatic_setup.diarization_runtime_status") as runtime_status, patch("automatic_setup.ensure_diarization_runtime") as ensure_runtime:
            runtime_status.side_effect = [
                SimpleNamespace(required=True, ready=False),
                SimpleNamespace(required=True, ready=True),
                SimpleNamespace(required=True, ready=True, to_dict=lambda: {"ready": True}),
            ]
            result = automatic_setup.run_automatic_setup(report)
        self.assertEqual(result["pyannote_source"], "jeronimo-drive")
        ensure_pyannote.assert_called_once()
        ensure_runtime.assert_called_once()
        resolve_asr.assert_called_once()
        validate_stack.assert_called_once()
        self.assertTrue(result["diarization_stack_probe"]["whisperx_model_load"])


if __name__ == "__main__":
    unittest.main()
