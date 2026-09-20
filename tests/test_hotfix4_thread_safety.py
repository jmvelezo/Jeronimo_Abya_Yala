from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "jeronimo_app.py"


def _source_tree():
    source = APP.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _find_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"No se encontró {class_name}.{method_name}")


class Hotfix4ThreadSafetyTests(unittest.TestCase):
    def test_job_worker_never_calls_tk_after_directly(self):
        _source, tree = _source_tree()
        worker = _find_method(tree, "JeronimoApp", "_job_worker")
        calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
        direct_after = [
            n for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr in {"after", "after_idle"}
        ]
        self.assertEqual(direct_after, [], "_job_worker no debe llamar after/after_idle desde el worker")

    def test_job_worker_routes_ui_through_dispatch_queue(self):
        source, _tree = _source_tree()
        start = source.index("    def _job_worker")
        end = source.index("    def _remember_last_audio", start)
        body = source[start:end]
        self.assertIn("self._post_ui(self._append_log", body)
        self.assertIn("self._post_ui(mb.showerror", body)
        self.assertIn("self._post_ui(self._finish_job_ui)", body)
        self.assertNotIn("self.after(", body)

    def test_jobtracker_notifications_are_queued_not_sent_to_tk(self):
        source, _tree = _source_tree()
        self.assertIn(
            'JobTracker(Job(kind="transcription"), on_event=lambda event: self._post_ui(self._render_job_event, event))',
            source,
        )
        self.assertNotIn(
            'JobTracker(Job(kind="transcription"), on_event=lambda event: self.after(0, self._render_job_event, event))',
            source,
        )

    def test_all_background_workers_in_app_use_dispatcher(self):
        source, tree = _source_tree()
        for class_name, method_name in [
            ("AdvancedDialog", "_test_provider"),
            ("JeronimoApp", "_load_diagnostics_async"),
        ]:
            method = _find_method(tree, class_name, method_name)
            segment = ast.get_source_segment(source, method) or ""
            self.assertIn("_post_ui", segment)
            # El método puede iniciar el thread; el worker anidado no debe usar after.
            for node in ast.walk(method):
                if isinstance(node, ast.FunctionDef) and node is not method and node.name == "worker":
                    nested = ast.get_source_segment(source, node) or ""
                    self.assertNotIn(".after(", nested)
                    self.assertNotIn(".after_idle(", nested)

    def test_dispatcher_drains_queue_and_reschedules_on_main_loop(self):
        source, tree = _source_tree()
        method = _find_method(tree, "JeronimoApp", "_drain_ui_dispatch_queue")
        segment = ast.get_source_segment(source, method) or ""
        self.assertIn("self._ui_dispatch_queue.get_nowait()", segment)
        self.assertIn("callback(*args, **kwargs)", segment)
        self.assertIn("self.after(40, self._drain_ui_dispatch_queue)", segment)
        self.assertIn("_record_ui_dispatch_exception()", segment)

    def test_faulthandler_persistent_log_is_enabled(self):
        source, _tree = _source_tree()
        self.assertIn('faulthandler.enable(file=stream, all_threads=True)', source)
        self.assertIn('log_dir / "fatal_crash.log"', source)
        self.assertIn("_enable_fatal_crash_log()", source)


if __name__ == "__main__":
    unittest.main()
