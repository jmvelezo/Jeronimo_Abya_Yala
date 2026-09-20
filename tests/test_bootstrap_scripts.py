from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class BootstrapScriptTests(unittest.TestCase):
    def test_only_two_user_bat_launchers(self):
        bats = sorted(p.name for p in ROOT.glob('*.bat'))
        self.assertEqual(bats, ['COMPILAR_JERONIMO_ABYA_YALA.bat', 'INICIAR_JERONIMO_ABYA_YALA.bat'])

    def test_bootstrap_does_not_use_powershell_automatic_args_variable(self):
        text = (ROOT / 'build' / 'bootstrap_local.ps1').read_text(encoding='utf-8-sig')
        self.assertNotIn('[string[]]$Args', text)
        self.assertNotIn('$args =', text.lower())
        self.assertIn('[string[]]$Arguments', text)
        self.assertIn('& $File @Arguments', text)

    def test_bootstrap_invokes_venv_with_m_venv(self):
        text = (ROOT / 'build' / 'bootstrap_local.ps1').read_text(encoding='utf-8-sig')
        self.assertIn('@("-m","venv",$Venv)', text)
        self.assertIn('Verificar entorno virtual', text)

if __name__ == '__main__':
    unittest.main()
