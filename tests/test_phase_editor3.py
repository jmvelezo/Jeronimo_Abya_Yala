from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITOR = ROOT / "interview_text_editor.py"


def _extract_sync_symbols():
    source = EDITOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted_funcs = {"_timecode_to_seconds", "_parse_timed_turns", "_find_timed_turn"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "_TIMECODE_RANGE_RE" in names:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ns = {"re": re}
    exec(compile(module, str(EDITOR), "exec"), ns)
    return ns


class InterviewEditorPhase3Tests(unittest.TestCase):
    def test_parser_accepts_real_atlas_timecode_variants_and_continuations(self):
        ns = _extract_sync_symbols()
        text = (
            "HABLANTE 1: [T001 00:00:02-00:00:07] Primera frase.\n"
            "Continuación del mismo turno.\n\n"
            "ENTREVISTADO/A (Ana): [00:00:08-00:00:12] Segunda respuesta.\n"
            "Otra línea."
        )
        turns = ns["_parse_timed_turns"](text)
        self.assertEqual(len(turns), 2)
        self.assertEqual((turns[0]["start"], turns[0]["end"]), (2.0, 7.0))
        self.assertEqual((turns[1]["start"], turns[1]["end"]), (8.0, 12.0))
        self.assertEqual(text[turns[0]["char_start"]:turns[0]["char_end"]], text[:turns[1]["char_start"]])
        self.assertIn("Continuación del mismo turno.", text[turns[0]["char_start"]:turns[0]["char_end"]])

    def test_parser_survives_speaker_and_body_edits_because_timecodes_are_authoritative(self):
        ns = _extract_sync_symbols()
        original = "HABLANTE 1: [00:01:10-00:01:20] texto\n\nHABLANTE 2: [00:01:21-00:01:25] respuesta"
        edited = "DOCENTE CORREGIDO: [00:01:10-00:01:20] texto mucho más largo y corregido\n\nALUMNA: [00:01:21-00:01:25] respuesta final"
        a = ns["_parse_timed_turns"](original)
        b = ns["_parse_timed_turns"](edited)
        self.assertEqual([(x["start"], x["end"]) for x in a], [(x["start"], x["end"]) for x in b])
        self.assertNotEqual(a[1]["char_start"], b[1]["char_start"])

    def test_invalid_timecodes_are_not_invented_and_zero_second_rounding_is_visual_only(self):
        ns = _extract_sync_symbols()
        text = (
            "HABLANTE 1: [00:70:00-00:70:03] inválido\n"
            "HABLANTE 2: [00:00:10-00:00:10] muy corto\n"
        )
        turns = ns["_parse_timed_turns"](text)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["start"], 10.0)
        self.assertEqual(turns[0]["end"], 10.0)
        self.assertEqual(turns[0]["effective_end"], 11.0)

    def test_active_turn_lookup_respects_gaps_and_timecode_overlaps(self):
        ns = _extract_sync_symbols()
        turns = ns["_parse_timed_turns"](
            "A: [00:00:02-00:00:05] uno\n\n"
            "B: [00:00:07-00:00:09] dos\n"
        )
        self.assertIsNone(ns["_find_timed_turn"](turns, 1.5))
        self.assertIs(ns["_find_timed_turn"](turns, 2.0), turns[0])
        self.assertIsNone(ns["_find_timed_turn"](turns, 6.0))
        self.assertIs(ns["_find_timed_turn"](turns, 8.0), turns[1])
        self.assertIsNone(ns["_find_timed_turn"](turns, 9.0))

    def test_ui_has_follow_control_and_highlight_without_moving_insertion_cursor(self):
        src = EDITOR.read_text(encoding="utf-8")
        self.assertIn('text="Seguir audio"', src)
        self.assertIn('ACTIVE_TURN_TAG = "audio_active_turn"', src)
        self.assertIn('self.text.tag_add(self.ACTIVE_TURN_TAG', src)
        self.assertIn('self.text.see(start_index)', src)
        self.assertNotIn('mark_set("insert"', src)
        self.assertNotIn("mark_set('insert'", src)

    def test_text_edits_rebuild_sync_map_with_debounce(self):
        tree = ast.parse(EDITOR.read_text(encoding="utf-8"))
        editor_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InterviewTextEditor")
        methods = {n.name: n for n in editor_cls.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("_schedule_sync_rebuild", methods)
        self.assertIn("_rebuild_timed_turns", methods)
        calls = [
            n for n in ast.walk(methods["_on_text_modified"])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertIn("_schedule_sync_rebuild", {n.func.attr for n in calls})
        src = EDITOR.read_text(encoding="utf-8")
        self.assertIn("SYNC_REBUILD_DELAY_MS = 320", src)

    def test_phase3_does_not_add_pipeline_dependencies_or_approximate_timing(self):
        tree = ast.parse(EDITOR.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {"core_transcriber", "job_engine", "ui_workflow", "automatic_setup", "onboarding", "onboarding_ui"}
        self.assertTrue(forbidden.isdisjoint(imported))
        src = EDITOR.read_text(encoding="utf-8").lower()
        self.assertNotIn("sincronización aproximada", src)
        self.assertIn("sin marcas temporales · seguimiento no disponible", src)


if __name__ == "__main__":
    unittest.main()
