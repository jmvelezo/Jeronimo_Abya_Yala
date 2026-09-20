# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent

datas = []
binaries = []
hiddenimports = []
# Módulos internos críticos: se declaran explícitamente para evitar omisiones
# de análisis estático al empaquetar desde Windows.
hiddenimports += ['portable_runtime', 'pyannote_model_manager', 'automatic_setup', 'core_transcriber', 'credential_store', 'diarization_runtime_manager', 'job_engine', 'local_text_runtime', 'model_manager', 'model_manager_ui', 'onboarding', 'onboarding_ui', 'runtime_isolation', 'subprocess_utils', 'system_diagnostics', 'system_monitor', 'text_providers', 'transcript_editor', 'interview_text_editor', 'ui_help', 'ui_workflow', 'visual_theme']
for package in ("customtkinter", "PIL", "tkinterdnd2", "keyring"):
    d, b, h = collect_all(package)
    datas += d; binaries += b; hiddenimports += h

# Recursos visuales Territorio Vivo (marca, iconos e icono de ejecutable).
datas += [(str(ROOT / "assets"), "assets")]

# El proceso principal NO debe arrastrar Torch/WhisperX. La diarización vive en
# JeronimoDiarizationWorker.exe para aislar dependencias y preservar el baseline.
heavy_excludes = [
    "torch", "torchaudio", "torchvision", "whisperx", "pyannote", "pyannote.audio",
    "lightning", "pytorch_lightning", "speechbrain", "torchmetrics",
]

a = Analysis(
    [str(ROOT / "jeronimo_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "build" / "hooks")],
    excludes=heavy_excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="JeronimoAbyaYala", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    icon=str(ROOT / "assets" / "brand" / "app.ico"),
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name="JeronimoAbyaYala",
)
