from __future__ import annotations

import ast
import codecs
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITOR = ROOT / "interview_text_editor.py"
APP = ROOT / "jeronimo_app.py"


class InterviewEditorPhase1Tests(unittest.TestCase):
    def test_editor_module_is_independent_from_transcription_pipeline(self):
        tree = ast.parse(EDITOR.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {"core_transcriber", "job_engine", "ui_workflow", "automatic_setup", "onboarding", "onboarding_ui"}
        self.assertTrue(forbidden.isdisjoint(imported))

    def test_sidebar_places_editor_directly_after_results_navigation(self):
        src = APP.read_text(encoding="utf-8")
        results_pos = src.index('("results", "Revisar y exportar")')
        editor_pos = src.index('text="Editor de entrevistas"')
        tools_pos = src.index('tools = [("settings", "Configuración inicial"')
        self.assertLess(results_pos, editor_pos)
        self.assertLess(editor_pos, tools_pos)
        self.assertIn('command=self._open_interview_text_editor', src)

    def test_editor_has_safe_save_and_backup(self):
        src = EDITOR.read_text(encoding="utf-8")
        self.assertIn('backup = path.with_name(path.name + ".bak")', src)
        self.assertIn('shutil.copy2(path, backup)', src)
        self.assertIn('messagebox.askyesnocancel(', src)
        self.assertIn('undo=True', src)
        self.assertIn('self.text.edit_redo()', src)

    def test_editor_preserves_utf8_bom_and_windows_newlines_helpers(self):
        # Replica focalizada de la lógica pura mediante extracción AST para no
        # requerir una sesión gráfica/customtkinter en el runner de pruebas.
        source = EDITOR.read_text(encoding="utf-8")
        tree = ast.parse(source)
        wanted = {"_read_text_file", "_write_text_file"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        module = ast.Module(body=nodes, type_ignores=[])
        ns = {"Path": Path, "codecs": codecs}
        exec(compile(module, str(EDITOR), "exec"), ns)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "entrevista.txt"
            p.write_bytes(codecs.BOM_UTF8 + "línea 1\r\nlínea 2".encode("utf-8"))
            text, encoding, newline = ns["_read_text_file"](p)
            self.assertEqual(text, "línea 1\r\nlínea 2")
            self.assertEqual(encoding, "utf-8-sig")
            self.assertEqual(newline, "\r\n")
            ns["_write_text_file"](p, text + "\ncorrección", encoding, newline)
            raw = p.read_bytes()
            self.assertTrue(raw.startswith(codecs.BOM_UTF8))
            self.assertIn(b"\r\n", raw)


if __name__ == "__main__":
    unittest.main()
