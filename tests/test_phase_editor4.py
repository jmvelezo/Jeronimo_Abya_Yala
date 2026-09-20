from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "jeronimo_app.py"


def _extract_result_helpers():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "_is_interview_editor_result",
        "_read_manifest_object",
        "_find_manifest_for_result",
        "_resolve_audio_for_result",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    ns = {"Path": Path, "json": json, "os": os}
    exec(compile(module, str(APP), "exec"), ns)
    return ns


class InterviewEditorPhase4Tests(unittest.TestCase):
    def test_only_raw_txt_is_offered_as_synced_editor_result(self):
        ns = _extract_result_helpers()
        check = ns["_is_interview_editor_result"]
        self.assertTrue(check(Path("entrevista.20260919_180000.raw.txt")))
        self.assertTrue(check(Path("ENTREVISTA.RAW.TXT")))
        self.assertFalse(check(Path("entrevista.clean.txt")))
        self.assertFalse(check(Path("entrevista.resumen.txt")))
        self.assertFalse(check(Path("entrevista.ATLAS.docx")))

    def test_single_file_manifest_resolves_exact_audio_without_guessing(self):
        ns = _extract_result_helpers()
        resolve = ns["_resolve_audio_for_result"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "entrevista A.m4a"
            audio.write_bytes(b"x")
            raw = root / "entrevista_A.20260919_180000.raw.txt"
            raw.write_text("texto", encoding="utf-8")
            manifest = root / "entrevista_A.20260919_180000.manifest.json"
            manifest.write_text(json.dumps({
                "input_file_name": audio.name,
                "generated_files": [raw.name, manifest.name],
            }), encoding="utf-8")
            found, manifest_path = resolve(raw, input_path=audio, selected_files=[audio])
            self.assertEqual(found, audio)
            self.assertEqual(manifest_path, manifest)

    def test_batch_manifest_resolves_audio_from_current_input_directory(self):
        ns = _extract_result_helpers()
        resolve = ns["_resolve_audio_for_result"]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inputs = base / "audios"
            outputs = base / "salida"
            inputs.mkdir()
            outputs.mkdir()
            a = inputs / "uno.wav"
            b = inputs / "dos.mp3"
            a.write_bytes(b"a")
            b.write_bytes(b"b")
            raw = outputs / "dos.20260919_180001.raw.txt"
            raw.write_text("texto", encoding="utf-8")
            manifest = outputs / "dos.20260919_180001.manifest.json"
            manifest.write_text(json.dumps({
                "input_file_name": "dos.mp3",
                "generated_files": [raw.name, manifest.name],
            }), encoding="utf-8")
            found, manifest_path = resolve(raw, input_path=inputs)
            self.assertEqual(found, b)
            self.assertEqual(manifest_path, manifest)

    def test_missing_or_malformed_manifest_never_guesses_audio(self):
        ns = _extract_result_helpers()
        resolve = ns["_resolve_audio_for_result"]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inputs = base / "audios"
            outputs = base / "salida"
            inputs.mkdir()
            outputs.mkdir()
            (inputs / "parecido.wav").write_bytes(b"a")
            raw = outputs / "parecido.20260919_180001.raw.txt"
            raw.write_text("texto", encoding="utf-8")
            found, manifest = resolve(raw, input_path=inputs)
            self.assertIsNone(found)
            self.assertIsNone(manifest)

            bad = outputs / "parecido.20260919_180001.manifest.json"
            bad.write_text("{no es json", encoding="utf-8")
            found, manifest = resolve(raw, input_path=inputs)
            self.assertIsNone(found)
            self.assertIsNone(manifest)

    def test_manifest_with_path_in_input_name_is_rejected(self):
        ns = _extract_result_helpers()
        resolve = ns["_resolve_audio_for_result"]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inputs = base / "audios"
            outputs = base / "salida"
            inputs.mkdir()
            outputs.mkdir()
            raw = outputs / "x.20260919_180001.raw.txt"
            raw.write_text("texto", encoding="utf-8")
            manifest = outputs / "x.20260919_180001.manifest.json"
            manifest.write_text(json.dumps({
                "input_file_name": "../audios/x.wav",
                "generated_files": [raw.name],
            }), encoding="utf-8")
            found, manifest_path = resolve(raw, input_path=inputs)
            self.assertIsNone(found)
            self.assertEqual(manifest_path, manifest)

    def test_results_ui_places_editor_button_beside_open_and_uses_resolver(self):
        src = APP.read_text(encoding="utf-8")
        open_pos = src.index('text="Abrir ↗"')
        editor_pos = src.index('text="Abrir con editor"')
        self.assertLess(open_pos, editor_pos)
        self.assertIn('if _is_interview_editor_result(path):', src)
        self.assertIn('command=lambda p=path: self._open_result_in_interview_editor(p)', src)
        self.assertIn('_resolve_audio_for_result(', src)
        self.assertIn('transcript_path=transcript_path', src)
        self.assertIn('audio_path=audio_path', src)

    def test_maximize_retries_once_after_real_root_map_without_changing_onboarding_contract(self):
        src = APP.read_text(encoding="utf-8")
        self.assertIn('self.bind("<Map>", self._on_main_window_mapped, add="+")', src)
        self.assertIn('self.after(60, self._maximize_after_first_map)', src)
        self.assertIn('self._startup_maximize_after_map_done = True', src)
        self.assertIn('if self._startup_onboarding_required or self._startup_maximize_after_map_done:', src)
        self.assertIn('self.after_idle(self._maximize_main_window)', src)
        self.assertIn('self.withdraw()', src)
        self.assertIn('self.deiconify()', src)


if __name__ == "__main__":
    unittest.main()
