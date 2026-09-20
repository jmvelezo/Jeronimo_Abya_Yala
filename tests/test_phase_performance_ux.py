from __future__ import annotations

import unittest
from pathlib import Path

import system_monitor

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "jeronimo_app.py"
EDITOR = ROOT / "interview_text_editor.py"
MONITOR = ROOT / "system_monitor.py"


class PerformanceUxTests(unittest.TestCase):
    def test_editor_is_brought_front_with_temporary_topmost_only(self):
        src = EDITOR.read_text(encoding="utf-8")
        self.assertIn("self.after(260, self._bring_editor_to_front)", src)
        self.assertIn('self.attributes("-topmost", True)', src)
        self.assertIn('self.attributes("-topmost", False)', src)
        self.assertIn("self.focus_force()", src)
        self.assertNotIn("self.transient(master)", src)

    def test_process_has_persistent_performance_notice_and_preflight_dialog(self):
        src = APP.read_text(encoding="utf-8")
        self.assertIn("RECOMENDACIÓN DE RENDIMIENTO", src)
        self.assertIn("PerformanceAdviceDialog(self)", src)
        self.assertIn('text="Iniciar proceso"', src)
        self.assertIn("video, música/streaming, juegos", src)
        # La confirmación debe ocurrir antes de crear el tracker/trabajo.
        self.assertLess(src.index("PerformanceAdviceDialog(self)"), src.index("self._job_tracker = JobTracker"))

    def test_lightweight_monitor_uses_native_windows_and_nvidia_smi_without_psutil(self):
        src = MONITOR.read_text(encoding="utf-8")
        self.assertIn("GetSystemTimes", src)
        self.assertIn("GlobalMemoryStatusEx", src)
        self.assertIn("nvidia-smi", src)
        self.assertIn("utilization.gpu", src)
        self.assertIn("temperature.gpu", src)
        self.assertNotIn("import psutil", src)
        self.assertNotIn("cpu_temperature", src)

    def test_monitor_is_decoupled_and_stopped_when_job_finishes(self):
        src = APP.read_text(encoding="utf-8")
        self.assertIn("SystemResourceMonitor(interval=1.0, gpu_interval=2.0).start()", src)
        self.assertIn("self._stop_resource_monitor()", src)
        self.assertIn("self._render_resource_snapshot", src)
        self.assertIn("TEMP. GPU", src)
        self.assertIn("CPU temp. no estimada", src)

    def test_sidebar_chaos_reigns_links_to_requested_github(self):
        src = APP.read_text(encoding="utf-8")
        self.assertIn('text="Chaos Reigns"', src)
        self.assertIn('webbrowser.open_new_tab("https://github.com/jmvelezo")', src)
        self.assertIn('chaos.bind("<Button-1>"', src)

    def test_portable_builder_names_monitor_explicitly(self):
        ps = (ROOT / "build/build_portable.ps1").read_text(encoding="utf-8")
        spec = (ROOT / "build/Jeronimo.spec").read_text(encoding="utf-8")
        self.assertIn('"system_monitor.py"', ps)
        compile_line = next(line for line in ps.splitlines() if "-m compileall -q" in line)
        self.assertIn("system_monitor.py", compile_line)
        self.assertIn("'system_monitor'", spec)

    def test_monitor_parses_numeric_and_missing_values_safely(self):
        self.assertEqual(system_monitor._number(" 67 "), 67.0)
        self.assertIsNone(system_monitor._number("N/A"))
        used, total, pct = system_monitor.memory_usage()
        if total is not None:
            self.assertGreater(total, 0)
            self.assertIsNotNone(used)
            self.assertIsNotNone(pct)
            self.assertGreaterEqual(float(pct), 0.0)
            self.assertLessEqual(float(pct), 100.0)


if __name__ == "__main__":
    unittest.main()
