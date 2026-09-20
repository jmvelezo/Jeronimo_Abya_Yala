from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WindowBehaviorTests(unittest.TestCase):
    def test_onboarding_remains_above_main_window(self):
        src = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('self.attributes("-topmost", True)', src)
        self.assertIn('self.app.attributes("-disabled", True)', src)
        self.assertIn('def _schedule_front_guard', src)
        self.assertNotIn('self.after(350, lambda: self.attributes("-topmost", False))', src)

    def test_parent_is_restored_when_onboarding_closes(self):
        src = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('self.app.attributes("-disabled", False)', src)
        self.assertIn('def destroy(self):', src)
        self.assertIn('self._unlock_parent_window()', src)

    def test_model_manager_temporarily_releases_modal_lock(self):
        src = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('self._unlock_parent_window()', src)
        self.assertIn('self._lock_parent_window()', src)


    def test_onboarding_has_real_exit_button_with_five_second_failsafe(self):
        src = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        self.assertIn('text="X"', src)
        self.assertIn('command=self._request_application_exit', src)
        self.assertIn('threading.Timer(5.0, self._force_application_exit)', src)
        self.assertIn('os._exit(0)', src)
        self.assertIn('self._automatic_cancel.set()', src)

    def test_gui_executable_guards_missing_standard_streams(self):
        src = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        self.assertIn('def _ensure_gui_standard_streams()', src)
        self.assertIn('getattr(sys, name, None) is not None', src)
        self.assertIn('open(os.devnull, mode, encoding="utf-8")', src)
        self.assertIn('HF_HUB_DISABLE_PROGRESS_BARS', src)

    def test_main_window_opens_maximized(self):
        src = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        self.assertIn('self.after_idle(self._maximize_main_window)', src)
        self.assertIn('self.state("zoomed")', src)

    def test_first_run_hides_main_window_until_onboarding_finishes(self):
        src = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        self.assertIn('startup_decision = startup_configuration_decision()', src)
        self.assertIn('self._startup_onboarding_required = startup_decision.required', src)
        self.assertIn('if self._startup_onboarding_required:', src)
        self.assertIn('self.withdraw()', src)
        self.assertIn('was_startup = bool(getattr(self, "_startup_onboarding_required", False))', src)
        self.assertIn('self.deiconify()', src)
        self.assertIn('self._maximize_main_window()', src)

    def test_completed_onboarding_skips_startup_wizard_and_opens_maximized(self):
        src = (ROOT / "jeronimo_app.py").read_text(encoding="utf-8")
        self.assertIn('if not self._startup_onboarding_required:', src)
        self.assertIn('self.after_idle(self._maximize_main_window)', src)
        self.assertIn('decision = startup_configuration_decision()', src)
        self.assertIn('required = bool(self._startup_onboarding_required or decision.required)', src)

    def test_main_is_revealed_only_after_wizard_is_destroyed(self):
        src = (ROOT / "onboarding_ui.py").read_text(encoding="utf-8")
        automatic = src[src.index('setup_mode="automatic"'):src.index('if key == "privacy"')]
        self.assertLess(automatic.index('self.destroy()'), automatic.index('app.after_idle(callback)'))
        advanced_start = src.index('setup_mode="advanced"')
        advanced = src[advanced_start:advanced_start + 700]
        self.assertLess(advanced.index('self.destroy()'), advanced.index('app.after_idle(callback)'))


if __name__ == "__main__":
    unittest.main()
