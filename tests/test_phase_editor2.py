from __future__ import annotations

import ast
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITOR = ROOT / "interview_text_editor.py"


def _extract_editor_symbols(names: set[str]):
    source = EDITOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in names:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    return compile(module, str(EDITOR), "exec")


class _FakeProc:
    def __init__(self):
        self.running = True
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated += 1
        self.running = False

    def wait(self, timeout=None):
        self.running = False
        return 0

    def kill(self):
        self.killed += 1
        self.running = False


class InterviewEditorPhase2Tests(unittest.TestCase):
    def test_phase2_adds_simple_audio_controls_without_space_shortcut(self):
        src = EDITOR.read_text(encoding="utf-8")
        self.assertIn('text="Abrir audio"', src)
        self.assertIn('text="−5 s"', src)
        self.assertIn('text="+5 s"', src)
        self.assertIn('text="▶ Reproducir"', src)
        self.assertIn('class FFplayTransport:', src)
        self.assertIn('PLAYBACK_TICK_MS = 120', src)
        self.assertNotIn('self.text.bind("<space>"', src.lower())
        self.assertNotIn('self.bind("<space>"', src.lower())

    def test_time_format_and_clamp_are_deterministic(self):
        code = _extract_editor_symbols({"_format_media_time", "_clamp_position"})
        ns = {}
        exec(code, ns)
        self.assertEqual(ns["_format_media_time"](0), "00:00:00")
        self.assertEqual(ns["_format_media_time"](3661.9), "01:01:01")
        self.assertEqual(ns["_clamp_position"](-3, 20), 0.0)
        self.assertEqual(ns["_clamp_position"](25, 20), 20.0)
        self.assertEqual(ns["_clamp_position"](25, 0), 25.0)

    def test_transport_pause_resume_skip_and_close_use_one_process_per_committed_action(self):
        wanted = {"_clamp_position", "FFplayTransport"}
        code = _extract_editor_symbols(wanted)
        clock_value = [100.0]
        commands = []
        processes = []

        def clock():
            return clock_value[0]

        def popen(command, **kwargs):
            commands.append(list(command))
            proc = _FakeProc()
            processes.append(proc)
            return proc

        fake_shutil = types.SimpleNamespace(which=lambda name: f"/runtime/{name}")
        fake_subprocess = types.SimpleNamespace(DEVNULL=-1)
        ns = {
            "Path": Path,
            "time": types.SimpleNamespace(monotonic=clock),
            "shutil": fake_shutil,
            "subprocess": fake_subprocess,
            "hidden_process_kwargs": lambda: {},
        }
        exec(code, ns)
        Transport = ns["FFplayTransport"]

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "entrevista.wav"
            audio.write_bytes(b"fake")
            transport = Transport(clock=clock, popen_factory=popen)
            transport.load(audio, duration=100.0)
            transport.play(10.0)
            self.assertEqual(len(commands), 1)
            self.assertIn("10.000", commands[0])

            clock_value[0] += 3.0
            self.assertAlmostEqual(transport.current_position(), 13.0, places=4)
            transport.pause()
            self.assertAlmostEqual(transport.cursor, 13.0, places=4)
            self.assertFalse(transport.is_playing())

            transport.play()
            self.assertEqual(len(commands), 2)
            self.assertIn("13.000", commands[1])

            transport.skip(5.0)
            self.assertEqual(len(commands), 3)
            self.assertIn("18.000", commands[2])
            self.assertTrue(transport.is_playing())

            transport.close()
            self.assertFalse(transport.is_playing())
            self.assertTrue(any(p.terminated for p in processes))

    def test_slider_drag_only_commits_seek_on_release(self):
        tree = ast.parse(EDITOR.read_text(encoding="utf-8"))
        editor_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InterviewTextEditor")
        methods = {n.name: n for n in editor_cls.body if isinstance(n, ast.FunctionDef)}

        slider_calls = [
            n for n in ast.walk(methods["_on_seek_slider"])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        slider_attrs = {n.func.attr for n in slider_calls}
        self.assertNotIn("play", slider_attrs)
        self.assertNotIn("seek", slider_attrs)

        release_calls = [
            n for n in ast.walk(methods["_on_seek_release"])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        release_attrs = [n.func.attr for n in release_calls]
        self.assertEqual(release_attrs.count("seek"), 1)

    def test_editor_stays_independent_from_transcription_pipeline(self):
        tree = ast.parse(EDITOR.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {"core_transcriber", "job_engine", "ui_workflow", "automatic_setup", "onboarding", "onboarding_ui"}
        self.assertTrue(forbidden.isdisjoint(imported))
        self.assertIn("subprocess_utils", imported)


if __name__ == "__main__":
    unittest.main()
