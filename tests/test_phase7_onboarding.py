from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import onboarding

ROOT = Path(__file__).resolve().parents[1]


class OnboardingStateTests(unittest.TestCase):
    def test_new_state_starts_at_intro(self):
        with tempfile.TemporaryDirectory() as td:
            state = onboarding.load_state(Path(td) / "settings.json")
        self.assertFalse(state.completed)
        self.assertEqual(state.last_step, "intro")

    def test_older_schema_forces_new_required_onboarding(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            path.write_text(json.dumps({
                "schema": onboarding.STATE_SCHEMA - 1,
                "completed": True,
                "privacy_acknowledged": True,
                "last_step": "done",
            }), encoding="utf-8")
            state = onboarding.load_state(path)
        self.assertFalse(state.completed)
        self.assertEqual(state.last_step, "intro")

    def test_completed_state_contains_no_secrets(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            onboarding.complete_onboarding(path=path)
            raw = path.read_text(encoding="utf-8").lower()
            data = json.loads(raw)
        self.assertTrue(data["completed"])
        for forbidden in ("api_key", "hf_token", "password", "secret", "sk-"):
            self.assertNotIn(forbidden, raw)

    def test_welcome_precedes_privacy_and_contains_project_identity(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('STEP_KEYS = ("intro", "automatic", "privacy"', source)
        self.assertIn("Bienvenido/a a Jerónimo Abya Yala", source)
        self.assertIn("Jóvenes y Escuela Secundaria", source)
        self.assertIn("Instituto de Investigaciones en Ciencias de la Educación (IICE)", source)
        self.assertIn("Facultad de Filosofía y Letras", source)
        self.assertIn("Universidad de Buenos Aires", source)
        self.assertIn("instagram.com/jovenesyescuela", source)
        self.assertIn("Usar configuración automática", source)
        self.assertIn("Configurar manualmente", source)

    def test_privacy_ack_is_checked_on_privacy_step(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('if key == "privacy" and not self._privacy_var.get():', source)
        self.assertNotIn('if key == "welcome" and not self._privacy_var.get():', source)

    def test_recommended_text_model_can_be_downloaded_from_onboarding(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn("Descargar modelo recomendado", source)
        self.assertIn("pull_ollama_model", source)
        self.assertIn("approx_size_gb", source)

    def test_onboarding_pages_are_scrollable_and_navigation_stays_outside(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('frame = ctk.CTkScrollableFrame(', source)
        self.assertIn('self.pages[key] = frame', source)
        self.assertIn('footer = ctk.CTkFrame(main, fg_color="transparent")', source)
        self.assertIn('footer.grid(row=1, column=0', source)
        self.assertIn('self.pages[key].lift()', source)
        self.assertIn('available_height = max(1, sh - 35)', source)
        self.assertIn('available_width = max(1, sw - 20)', source)
        self.assertIn('self._show_step(0)', source)

    def test_developer_note_is_plain_language_and_signed_only_at_end(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('text="Nota del desarrollador"', source)
        self.assertNotIn('Nota del desarrollador · José Manuel', source)
        self.assertIn('entre 10 y 30 gigas libres', source)
        self.assertIn('tarjetas gráficas NVIDIA', source)
        self.assertIn('Con cariño, José Manuel.', source)

    def test_onboarding_is_centered_borderless_and_mandatory(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('self.overrideredirect(True)', source)
        self.assertIn('Proceso inicial obligatorio', source)
        self.assertIn('self.bind("<Alt-F4>"', source)
        self.assertIn('command=self._request_application_exit', source)
        self.assertIn('complete_onboarding(', source)
        self.assertIn('sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()', source)

    def test_context_help_uses_hover_and_click_bubbles(self):
        source = (ROOT / "ui_help.py").read_text(encoding="utf-8")
        self.assertIn('text="?"', source)
        self.assertIn('self.bind("<Enter>"', source)
        self.assertIn('self.bind("<Leave>"', source)
        self.assertIn('command=self._toggle_bubble', source)
        self.assertIn('self._hide()', source)
        self.assertIn('_HIDE_GRACE_MS', source)
        self.assertIn('_pointer_in_help_region', source)
        self.assertIn('top.bind("<Enter>"', source)
        self.assertIn('top.bind("<Leave>"', source)
        self.assertNotIn('_watch_pointer', source)
        self.assertNotIn('_POINTER_POLL_MS', source)
        self.assertNotIn('_pinned', source)
        app = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(app.count('HelpButton('), 12)
        onboarding_ui = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(onboarding_ui.count('HelpBubbleButton('), 12)



if __name__ == "__main__":
    unittest.main()


class StartupRepairContractTests(unittest.TestCase):
    def test_legacy_completed_state_enters_repair_not_full_onboarding(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            path.write_text(json.dumps({
                "schema": onboarding.STATE_SCHEMA - 1,
                "completed": True,
                "privacy_acknowledged": True,
                "last_step": "done",
                "setup_mode": "automatic",
            }), encoding="utf-8")
            with unittest.mock.patch("onboarding.startup_technical_issues", return_value=()):
                decision = onboarding.startup_configuration_decision(path)
        self.assertEqual(decision.mode, "repair")
        self.assertTrue(decision.required)
        self.assertTrue(decision.repair)

    def test_current_completed_state_with_valid_contract_opens_normally(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            onboarding.complete_onboarding(path=path, setup_mode="automatic")
            with unittest.mock.patch("onboarding.startup_technical_issues", return_value=()):
                decision = onboarding.startup_configuration_decision(path)
        self.assertEqual(decision.mode, "none")
        self.assertFalse(decision.required)

    def test_current_completed_state_with_missing_runtime_enters_repair(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            onboarding.complete_onboarding(path=path, setup_mode="automatic")
            with unittest.mock.patch(
                "onboarding.startup_technical_issues",
                return_value=("Falta runtime aislado.",),
            ):
                decision = onboarding.startup_configuration_decision(path)
        self.assertEqual(decision.mode, "repair")
        self.assertIn("Falta runtime aislado.", decision.reasons)

    def test_incomplete_or_unacknowledged_state_still_uses_full_onboarding(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            path.write_text(json.dumps({
                "schema": onboarding.STATE_SCHEMA - 1,
                "completed": False,
                "privacy_acknowledged": False,
            }), encoding="utf-8")
            decision = onboarding.startup_configuration_decision(path)
        self.assertEqual(decision.mode, "full")

    def test_repair_ui_is_automatic_and_does_not_persist_step_before_success(self):
        source = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('self.repair_mode = self.required and self.startup_mode == "repair"', source)
        self.assertIn('self._configure_sidebar_for_mode("repair")', source)
        self.assertIn('self._show_step(STEP_KEYS.index("automatic"))', source)
        self.assertIn('if not self.repair_mode:\n            mark_step(key)', source)
        self.assertIn('"Reparar configuración" if self.repair_mode', source)

    def test_frozen_technical_contract_detects_missing_local_components(self):
        runtime = type("Runtime", (), {"required": True, "ready": False})()
        with unittest.mock.patch("onboarding.is_frozen", return_value=True), \
             unittest.mock.patch("onboarding.diarization_runtime_status", return_value=runtime), \
             unittest.mock.patch("onboarding.is_asr_model_cached", return_value=False), \
             unittest.mock.patch("onboarding.pyannote_local_model_status", return_value={"ready": False}), \
             unittest.mock.patch("onboarding.ollama_executable", return_value=Path("Z:/missing/ollama.exe")):
            issues = onboarding.startup_technical_issues()
        joined = " ".join(issues)
        self.assertIn("runtime aislado", joined)
        self.assertIn("large-v2", joined)
        self.assertIn("Community-1", joined)
        self.assertIn("Ollama", joined)

    def test_source_development_does_not_force_repair_for_external_runtime(self):
        with unittest.mock.patch("onboarding.is_frozen", return_value=False):
            self.assertEqual(onboarding.startup_technical_issues(), ())


def _legacy_advanced_migration_test(self):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "settings.json"
        path.write_text(json.dumps({
            "schema": onboarding.STATE_SCHEMA - 1,
            "completed": True,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "privacy_acknowledged": True,
            "preferred_text_mode": "openai",
            "last_step": "done",
            "setup_mode": "advanced",
        }), encoding="utf-8")
        with unittest.mock.patch("onboarding.startup_technical_issues", return_value=("No debería consultarse",)):
            decision = onboarding.startup_configuration_decision(path)
        migrated = json.loads(path.read_text(encoding="utf-8"))
    self.assertEqual(decision.mode, "none")
    self.assertEqual(migrated["schema"], onboarding.STATE_SCHEMA)
    self.assertTrue(migrated["completed"])
    self.assertEqual(migrated["setup_mode"], "advanced")
    self.assertEqual(migrated["preferred_text_mode"], "openai")

StartupRepairContractTests.test_legacy_advanced_state_migrates_without_forcing_local_stack = _legacy_advanced_migration_test
