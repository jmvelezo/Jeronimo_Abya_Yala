@echo off
title Instalador de DIARIZACION (WhisperX)
echo ==============================================
echo   Instalador de soporte de diarizacion local
echo   (WhisperX + dependencias pesadas)
echo ==============================================
echo.

REM --- 0) Verificar que exista el entorno virtual .venv ---
if not exist ".venv" (
    echo [ERROR] No se encontro la carpeta ".venv".
    echo Primero ejecuta: instalar_dependencias.bat
    echo para crear el entorno virtual base.
    echo.
    pause
    exit /b 1
)

REM --- 1) Verificar que exista requirements-diarization.txt ---
if not exist "requirements-diarization.txt" (
    echo [ERROR] No se encontro "requirements-diarization.txt" en:
    echo   %cd%
    echo Crea el archivo requirements-diarization.txt con:
    echo   -r requirements.txt
    echo   whisperx
    echo.
    pause
    exit /b 1
)

REM --- 2) Aviso de que esto puede tardar y pesar ---
echo OJO: esto instalara WhisperX, PyTorch y pyannote.audio.
echo En una PC sin GPU Nvidia puede tardar bastante y usar bastante espacio.
echo.
pause

REM --- 3) Actualizar pip dentro del entorno ---
echo Actualizando pip dentro de .venv ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
IF ERRORLEVEL 1 (
    echo [ERROR] No se pudo actualizar pip.
    echo.
    pause
    exit /b 1
)

REM --- 4) Instalar dependencias de diarizacion ---
echo.
echo Instalando dependencias de DIARIZACION desde requirements-diarization.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements-diarization.txt
IF ERRORLEVEL 1 (
    echo [ERROR] Fallo la instalacion de WhisperX o alguna dependencia.
    echo Revisa el log de errores de pip.
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo  DIARIZACION instalada correctamente.
echo
echo  Para usar diarizacion en tu programa:
echo    1) Configura la variable de entorno:
echo         HUGGINGFACE_TOKEN
echo       con tu token de Hugging Face (pyannote.audio).
echo    2) En la GUI, marca:
echo         "Diarizacion local (WhisperX, GPU/CPU)"
echo ==============================================
echo.
pause
