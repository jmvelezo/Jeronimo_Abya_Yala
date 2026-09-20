# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent

datas = []
binaries = []
hiddenimports = []
# WhisperX y pyannote utilizan imports dinámicos. Preferimos un worker grande pero
# autocontenido antes que un release que falle sólo en la primera entrevista real.
for package in (
    "whisperx", "pyannote.audio", "faster_whisper", "ctranslate2", "transformers",
    "huggingface_hub", "safetensors", "lightning", "torchmetrics", "keyring",
):
    try:
        d, b, h = collect_all(package)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass
for package in ("whisperx", "pyannote", "transformers"):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

# Torch se deja a los hooks oficiales de PyInstaller/hooks-contrib para recolectar
# DLLs y componentes del runtime instalado que ya fue validado por el usuario.
a = Analysis(
    [str(ROOT / "workers" / "diarization_worker.py")],
    pathex=[str(ROOT)], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "build" / "hooks")], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="JeronimoDiarizationWorker",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, upx_exclude=[], name="JeronimoDiarizationWorker")
