from __future__ import annotations

import hashlib
import inspect
import unittest
from pathlib import Path
from unittest import mock

import core_transcriber as core
import subprocess_utils

ROOT = Path(__file__).resolve().parents[1]
ASR_BASELINE_HASHES = {
    "transcribe_audio_local_faster_whisper": "c30b4fa79ca7baac2cbdfbdbed22857ab7fe23497a82af7b2a5860ff5d1db1d1",
    "transcribe_audio": "381f1b26bac90fb8eda53138a570b34edaf5ec41ba2b9b3257aa22f77dbf583c",
    "diarize_and_transcribe_local": "91a86a03fcbef536e5a1085bdfae566118ffaaa876b55b426e7f6944d353bbcf",
}


class BuildContractTests(unittest.TestCase):
    def test_build_assets_exist(self):
        for rel in (
            "build/Jeronimo.spec", "build/JeronimoDiarizationWorker.spec",
            "build/build_portable.ps1", "build/bootstrap_local.ps1",
            "build/verify_portable.ps1", "build/requirements-build.txt",
        ):
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_main_executable_uses_project_name_and_excludes_heavy_runtime(self):
        main = (ROOT / "build/Jeronimo.spec").read_text(encoding="utf-8")
        self.assertIn('name="JeronimoAbyaYala"', main)
        self.assertIn('"torch"', main)
        self.assertIn('"whisperx"', main)

    def test_main_gui_runtime_pins_and_collects_pillow(self):
        req = (ROOT / "requirements-base.txt").read_text(encoding="utf-8")
        self.assertIn("Pillow==11.3.0", req)
        spec = (ROOT / "build/Jeronimo.spec").read_text(encoding="utf-8")
        self.assertIn('"PIL"', spec)
        ps = (ROOT / "build/build_portable.ps1").read_text(encoding="utf-8")
        self.assertIn("Validar dependencias GUI main-venv", ps)
        self.assertIn("from PIL import Image", ps)

    def test_compile_uses_local_tested_venv_only(self):
        ps = (ROOT / "build/build_portable.ps1").read_text(encoding="utf-8")
        self.assertIn('.venv\\Scripts\\python.exe', ps)
        self.assertIn("Ejecuta primero INICIAR_JERONIMO_ABYA_YALA.bat", ps)
        self.assertNotIn("Split-Path $Root -Parent", ps)

    def test_default_build_is_thin_base_only(self):
        ps = (ROOT / "build/build_portable.ps1").read_text(encoding="utf-8")
        self.assertIn("Jeronimo_Abya_Yala_BASE_Windows_x64.zip", ps)
        self.assertIn("[switch]$BuildDiarizationComponent", ps)
        self.assertNotIn("Jeronimo_Abya_Yala_COMPLETO_Windows_x64.zip", ps)
        self.assertIn("anti-monolito", ps)
        self.assertIn("runtime_source", ps)

    def test_base_verifier_does_not_require_heavy_worker(self):
        ps = (ROOT / "build/verify_portable.ps1").read_text(encoding="utf-8")
        self.assertNotIn(r'"$P\runtime\diarization\JeronimoDiarizationWorker.exe",', ps)
        self.assertIn("runtime pesado ausente por diseño", ps)

    def test_downloadable_runtime_manager_is_pinned(self):
        src = (ROOT / "diarization_runtime_manager.py").read_text(encoding="utf-8")
        self.assertIn('UV_VERSION = "0.12.17"', src)
        self.assertIn("a252121d5b59398fcb137c6ea448176459a44010f33f67e0072305a637119ca7", src)
        self.assertIn('TORCH_VERSION = "2.8.0"', src)
        self.assertIn('WHISPERX_VERSION = "3.8.6"', src)
        self.assertIn('PYANNOTE_VERSION = "4.0.4"', src)


    def test_windows_child_processes_are_hidden_centrally(self):
        class FakeStartupInfo:
            def __init__(self):
                self.dwFlags = 0
                self.wShowWindow = -1

        with (
            mock.patch.object(subprocess_utils.os, "name", "nt"),
            mock.patch.object(subprocess_utils.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
            mock.patch.object(subprocess_utils.subprocess, "STARTUPINFO", FakeStartupInfo, create=True),
            mock.patch.object(subprocess_utils.subprocess, "STARTF_USESHOWWINDOW", 1, create=True),
            mock.patch.object(subprocess_utils.subprocess, "SW_HIDE", 0, create=True),
        ):
            kwargs = subprocess_utils.hidden_process_kwargs()
        self.assertEqual(kwargs.get("creationflags"), 0x08000000)
        self.assertEqual(kwargs["startupinfo"].dwFlags & 1, 1)
        self.assertEqual(kwargs["startupinfo"].wShowWindow, 0)

    def test_runtime_source_carries_hidden_process_helper(self):
        ps = (ROOT / "build/build_portable.ps1").read_text(encoding="utf-8")
        self.assertIn('"subprocess_utils.py"', ps)
        spec = (ROOT / "build/Jeronimo.spec").read_text(encoding="utf-8")
        self.assertIn("'subprocess_utils'", spec)
        for rel in (
            "core_transcriber.py", "transcript_editor.py", "system_diagnostics.py",
            "portable_runtime.py", "local_text_runtime.py", "runtime_isolation.py",
            "diarization_runtime_manager.py", "jeronimo_app.py",
        ):
            src = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("hidden_process_kwargs", src, rel)

    def test_local_bootstrap_pins_quality_runtime_and_gpu_path(self):
        ps = (ROOT / "build/bootstrap_local.ps1").read_text(encoding="utf-8")
        self.assertIn("torch==2.8.0", ps)
        self.assertIn("cu128", ps)
        diar = (ROOT / "requirements-diarization.txt").read_text(encoding="utf-8")
        self.assertIn("whisperx==3.8.6", diar)
        self.assertIn("pyannote-audio==4.0.4", diar)

    def test_asr_and_diarization_implementations_remain_frozen(self):
        for name, expected in ASR_BASELINE_HASHES.items():
            source = inspect.getsource(getattr(core, name)).encode("utf-8")
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
