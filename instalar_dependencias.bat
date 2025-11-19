@echo off
title Instalador de dependencias STT (Whisper + GPT)
echo ==============================================
echo   Instalador de dependencias para el proyecto
echo   Desgrabador STT (Whisper + GPT)
echo ==============================================
echo.

REM --- 0) Comprobar que exista requirements.txt ---
if not exist "requirements.txt" (
    echo [ERROR] No se encontro "requirements.txt" en:
    echo   %cd%
    echo Crea el archivo requirements.txt con las dependencias del proyecto.
    echo.
    pause
    exit /b 1
)

REM --- 1) Comprobar que Python este disponible ---
echo Verificando Python...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] No se encontro "python" en el PATH.
    echo Asegurate de tener Python instalado y agregado al PATH.
    echo.
    pause
    exit /b 1
)

REM --- 2) Crear entorno virtual si no existe ---
echo.
if exist ".venv" (
    echo Entorno virtual ".venv" ya existe, se reutilizara.
) else (
    echo Creando entorno virtual ".venv" ...
    python -m venv .venv
    IF ERRORLEVEL 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        echo Revisa tu instalacion de Python.
        echo.
        pause
        exit /b 1
    )
)

REM --- 3) Actualizar pip dentro del entorno ---
echo.
echo Activando entorno virtual y actualizando pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
IF ERRORLEVEL 1 (
    echo [ERROR] No se pudo actualizar pip dentro del entorno virtual.
    echo.
    pause
    exit /b 1
)

REM --- 4) Instalar dependencias desde requirements.txt ---
echo.
echo Instalando dependencias desde requirements.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo [ERROR] Fallo la instalacion de dependencias con pip.
    echo Revisa el contenido de requirements.txt y el mensaje de error.
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo  Dependencias instaladas correctamente.
echo  Para usar el proyecto:
echo    1) (CMD) Activar entorno:  call .venv\Scripts\activate.bat
echo       o lanzar directo:       ".venv\Scripts\python.exe" stt_gui.py
echo    2) Ejecutar GUI:           python stt_gui.py
echo ==============================================
echo.
pause
