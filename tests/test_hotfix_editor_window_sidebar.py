from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITOR = ROOT / "interview_text_editor.py"
APP = ROOT / "jeronimo_app.py"


class EditorWindowSidebarHotfixTests(unittest.TestCase):
    def test_editor_opens_maximized_and_remains_resizable(self):
        src = EDITOR.read_text(encoding="utf-8")
        self.assertNotIn("self.withdraw()", src)
        self.assertNotIn("self.deiconify()", src)
        self.assertIn("self.resizable(True, True)", src)
        self.assertIn("self.after(25, self._maximize_editor_window)", src)
        self.assertIn("self.after(180, self._maximize_editor_window)", src)
        self.assertIn('self.wm_state("zoomed")', src)

    def test_editor_is_not_forced_to_transient_window_style(self):
        source = EDITOR.read_text(encoding="utf-8")
        tree = ast.parse(source)
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InterviewTextEditor")
        class_source = ast.get_source_segment(source, klass) or ""
        self.assertNotIn("self.transient(master)", class_source)

    def test_sidebar_footer_is_wrapped_inside_fixed_width(self):
        src = APP.read_text(encoding="utf-8")
        self.assertIn('text="TECNOLOGÍA LOCAL\\nDATOS BAJO TU CONTROL"', src)
        self.assertIn('justify="left"', src)
        self.assertNotIn('text="TECNOLOGÍA LOCAL · DATOS BAJO TU CONTROL"', src)


if __name__ == "__main__":
    unittest.main()
