from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InterviewEditorPhase5ReleaseTests(unittest.TestCase):
    def test_windows_builder_requires_and_compiles_document_editor(self):
        ps = (ROOT / "build/build_portable.ps1").read_text(encoding="utf-8")
        self.assertIn('"interview_text_editor.py"', ps)
        compile_line = next(line for line in ps.splitlines() if "-m compileall -q" in line)
        self.assertIn("interview_text_editor.py", compile_line)

    def test_pyinstaller_contract_names_document_editor_explicitly(self):
        spec = (ROOT / "build/Jeronimo.spec").read_text(encoding="utf-8")
        self.assertIn("'interview_text_editor'", spec)

    def test_portable_path_is_bootstrapped_before_document_editor_import(self):
        app = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        bootstrap = app.index("bootstrap_portable_environment()")
        editor_import = app.index("from interview_text_editor import open_interview_text_editor")
        self.assertLess(bootstrap, editor_import)


if __name__ == "__main__":
    unittest.main()
