from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_manager
import system_diagnostics


class _FakeResponse:
    def __init__(self, payload: bytes | str | list[bytes]):
        if isinstance(payload, list):
            self.lines = payload
            self.body = b"".join(payload)
        else:
            self.body = payload.encode() if isinstance(payload, str) else payload
            self.lines = [self.body]
        self.stream = io.BytesIO(self.body)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *args):
        return self.stream.read(*args)

    def readline(self, *args):
        return self.stream.readline(*args)


class Phase3DiagnosticsTests(unittest.TestCase):
    def test_recommendation_high_profile_prefers_14b(self):
        report = system_diagnostics.SystemReport(
            platform="Windows", python_version="3.11", cpu_name="CPU", cpu_logical_cores=16,
            ram_total_bytes=32 * 1024**3, ram_available_bytes=20 * 1024**3,
            gpus=[{"name": "GPU", "memory_total_mb": 16384, "memory_free_mb": 12000, "driver_version": "x"}],
            torch_cuda_available=True,
            disk={"free_bytes": 100 * 1024**3, "writable": True},
            ffmpeg={"available": True},
            ollama={"available": True},
        )
        rec = system_diagnostics._recommendations(report)
        self.assertEqual(rec["text_primary"], "qwen3:14b")
        self.assertIn("WhisperX", rec["asr"])

    def test_recommendation_max_profile_can_use_qwen30b(self):
        report = system_diagnostics.SystemReport(
            platform="Windows", python_version="3.11", cpu_name="CPU", cpu_logical_cores=24,
            ram_total_bytes=64 * 1024**3, ram_available_bytes=48 * 1024**3,
            gpus=[{"name": "GPU", "memory_total_mb": 24576, "memory_free_mb": 22000, "driver_version": "x"}],
            torch_cuda_available=True, disk={"free_bytes": 200 * 1024**3, "writable": True},
            ffmpeg={"available": True}, ollama={"available": True},
        )
        rec = system_diagnostics._recommendations(report)
        self.assertEqual(rec["text_primary"], "qwen3:30b")

    def test_ollama_model_alias_does_not_mark_every_qwen_size_installed(self):
        inv = {"qwen3:latest": {"size": 1}}
        self.assertTrue(model_manager.ollama_model_present(inv, "qwen3:8b"))
        self.assertFalse(model_manager.ollama_model_present(inv, "qwen3:14b"))
        self.assertFalse(model_manager.ollama_model_present(inv, "qwen3:30b"))

    def test_recommendation_low_profile_is_conservative(self):
        report = system_diagnostics.SystemReport(
            platform="Windows", python_version="3.11", cpu_name="CPU", cpu_logical_cores=4,
            ram_total_bytes=8 * 1024**3, ram_available_bytes=5 * 1024**3,
            gpus=[], disk={"free_bytes": 20 * 1024**3, "writable": True},
            ffmpeg={"available": True}, ollama={"available": False},
        )
        rec = system_diagnostics._recommendations(report)
        self.assertEqual(rec["text_primary"], "qwen3:4b")
        self.assertTrue(any("Ollama" in w for w in rec["warnings"]))

    def test_diagnostics_checks_write_permission_without_leaving_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(system_diagnostics, "_ollama_status", return_value={"available": False, "models": [], "url": "x"}), \
                 patch.object(system_diagnostics, "_nvidia_gpus", return_value=[]), \
                 patch.object(system_diagnostics, "_tool_version", return_value=(False, "", "")), \
                 patch.object(system_diagnostics, "_hf_cache_models", return_value=[]):
                report = system_diagnostics.run_system_diagnostics(tmp)
            self.assertTrue(report.disk["writable"])
            leftovers = list(Path(tmp).glob(".jeronimo_diag_*"))
            self.assertEqual(leftovers, [])

    def test_format_report_does_not_include_secret(self):
        report = system_diagnostics.SystemReport(
            platform="Windows", python_version="3.11", cpu_name="CPU", cpu_logical_cores=4,
            ram_total_bytes=8 * 1024**3, ram_available_bytes=4 * 1024**3,
            disk={"path": "X", "free_bytes": 10 * 1024**3, "writable": True},
            ffmpeg={"available": False}, ffprobe={"available": False}, ffplay={"available": False},
            ollama={"available": False, "models": [], "executable": ""},
            hf_token_source="keyring", hf_token_validation="configurado",
            recommendations={"text_primary": "qwen3:4b", "text_alternative": "qwen3:1.7b", "text_note": "x", "asr": "x", "warnings": []},
        )
        text = system_diagnostics.format_diagnostics_report(report)
        self.assertIn("keyring", text)
        self.assertNotIn("hf_", text.lower())


class Phase3ModelManagerTests(unittest.TestCase):
    def test_catalog_preserves_real_diarization_baseline_large_v2(self):
        ids = {m.model_id: m for m in model_manager.ASR_MODELS}
        self.assertIn("large-v2", ids)
        self.assertIn("baseline", ids["large-v2"].notes.lower())

    def test_compatibility_high_vram(self):
        report = system_diagnostics.SystemReport(
            platform="Windows", python_version="3.11", cpu_name="CPU", cpu_logical_cores=12,
            ram_total_bytes=32 * 1024**3, ram_available_bytes=20 * 1024**3,
            gpus=[{"memory_total_mb": 16384}],
        )
        spec = next(m for m in model_manager.TEXT_MODELS if m.model_id == "qwen3:14b")
        state, _ = model_manager.compatibility(spec, report)
        self.assertEqual(state, "recomendado")

    def test_ollama_pull_reports_byte_progress(self):
        lines = [
            (json.dumps({"status": "pulling", "completed": 50, "total": 100}) + "\n").encode(),
            (json.dumps({"status": "success", "completed": 100, "total": 100}) + "\n").encode(),
        ]
        events = []
        # Se prueba el protocolo HTTP de pull aislado del runtime privado.
        # La instalación/arranque del runtime administrado tiene su propia capa.
        with patch("urllib.request.urlopen", return_value=_FakeResponse(lines)):
            model_manager.pull_ollama_model(
                "qwen3:4b", base_url="http://127.0.0.1:11434", progress=events.append
            )
        self.assertEqual(events[-1]["fraction"], 1.0)
        self.assertEqual(events[-1]["completed"], 100)

    def test_ollama_inventory_reads_size(self):
        payload = json.dumps({"models": [{"name": "qwen3:8b", "size": 1234}]})
        # Igual que el pull: aquí validamos inventario HTTP sin disparar
        # preflight/instalación del runtime privado durante un unit test.
        with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
            inv = model_manager.ollama_inventory("http://127.0.0.1:11434")
        self.assertTrue("qwen3:8b" in inv)
        self.assertEqual(inv["qwen3:8b"]["size"], 1234)


    def test_asr_download_uses_current_hf_snapshot_api(self):
        spec = next(m for m in model_manager.ASR_MODELS if m.model_id == "small")
        fake_hf = type("FakeHF", (), {})()
        calls = {}
        def snapshot_download(**kwargs):
            calls.update(kwargs)
            return "/tmp/fake_snapshot"
        import types, sys
        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = snapshot_download
        old = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = module
        try:
            path = model_manager.download_faster_whisper_model(spec)
        finally:
            if old is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = old
        self.assertEqual(path, Path("/tmp/fake_snapshot"))
        self.assertEqual(calls, {"repo_id": spec.repo_id})

    def test_delete_asr_only_removes_exact_cache_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hub"
            target = root / "models--Systran--faster-whisper-small"
            target.mkdir(parents=True)
            keep = root / "models--Other--keep"
            keep.mkdir(parents=True)
            spec = next(m for m in model_manager.ASR_MODELS if m.model_id == "small")
            with patch.object(model_manager, "hf_cache_root", return_value=root):
                deleted = model_manager.delete_faster_whisper_model(spec)
            self.assertTrue(deleted)
            self.assertFalse(target.exists())
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
