$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Section([string]$Text) {
    Write-Host "`n=== $Text ===" -ForegroundColor Cyan
}

function Invoke-Native([string]$File, [string[]]$Arguments, [string]$Title) {
    Write-Host "[$Title]" -ForegroundColor DarkCyan
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $File @Arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) { throw "$Title falló con código $code" }
}

function Test-Python([string]$Exe, [string[]]$Prefix = @()) {
    try {
        $old = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & $Exe @Prefix -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" *> $null
        $code = $LASTEXITCODE
        $ErrorActionPreference = $old
        return ($code -eq 0)
    } catch { return $false }
}

function Resolve-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($v in @("3.10", "3.11", "3.12")) {
            if (Test-Python $py.Source @("-$v")) {
                return @{ File = $py.Source; Prefix = @("-$v"); Label = "py -$v" }
            }
        }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python -and (Test-Python $python.Source)) {
        return @{ File = $python.Source; Prefix = @(); Label = $python.Source }
    }
    $common = @(
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )
    foreach ($p in $common) {
        if ((Test-Path $p) -and (Test-Python $p)) { return @{ File = $p; Prefix = @(); Label = $p } }
    }
    return $null
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (($machine, $user) -join ";")
}

function Test-TorchRuntime([string]$PythonExe, [bool]$RequireCuda) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    if ($RequireCuda) {
        & $PythonExe -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.8.') and torch.cuda.is_available() else 1)" *> $null
    } else {
        & $PythonExe -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.8.') else 1)" *> $null
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    return ($code -eq 0)
}

Write-Host "============================================================" -ForegroundColor White
Write-Host " JERÓNIMO ABYA YALA - PREPARAR E INICIAR" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor White
Write-Host "Carpeta: $Root"
Write-Host "La primera preparación puede tardar y descargar varios GB."
Write-Host "Los modelos de entrevistas/resumen NO se descargan aquí: se eligen desde el programa."

Section "Comprobar Python"
$Base = Resolve-Python
if (-not $Base) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "No se encontró Python 3.10-3.12 ni winget. Instala Python 3.10 x64 y vuelve a ejecutar este BAT."
    }
    Write-Host "Python compatible no encontrado. Se instalará Python 3.10 x64 mediante winget." -ForegroundColor Yellow
    Invoke-Native $winget.Source @("install","-e","--id","Python.Python.3.10","--silent","--accept-package-agreements","--accept-source-agreements") "Instalar Python 3.10"
    Refresh-Path
    $Base = Resolve-Python
    if (-not $Base) { throw "Python se instaló, pero esta consola aún no puede localizarlo. Cierra esta ventana y vuelve a ejecutar el BAT." }
}
Write-Host "Python base: $($Base.Label)" -ForegroundColor Green

Section "Comprobar FFmpeg"
$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host "FFmpeg no encontrado. Se intentará instalar con winget." -ForegroundColor Yellow
        try {
            Invoke-Native $winget.Source @("install","-e","--id","Gyan.FFmpeg","--silent","--accept-package-agreements","--accept-source-agreements") "Instalar FFmpeg"
            Refresh-Path
        } catch {
            Write-Warning $_.Exception.Message
        }
    }
}
$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
if ($ffmpeg) { Write-Host "FFmpeg: $($ffmpeg.Source)" -ForegroundColor Green }
else { Write-Warning "FFmpeg sigue sin detectarse. Jerónimo Abya Yala abrirá, pero algunas operaciones de audio no funcionarán hasta instalarlo." }

Section "Preparar entorno local"
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creando .venv local..."
    $venvArgs = @() + $Base.Prefix + @("-m","venv",$Venv)
    Invoke-Native $Base.File $venvArgs "Crear entorno virtual"
}
if (-not (Test-Path $VenvPython)) { throw "No se creó .venv\Scripts\python.exe" }
Invoke-Native $VenvPython @("-c","import sys; print('Entorno local OK | Python', sys.version.split()[0])") "Verificar entorno virtual"
Invoke-Native $VenvPython @("-m","pip","install","--upgrade","pip","setuptools","wheel") "Actualizar pip"
Invoke-Native $VenvPython @("-m","pip","install","-r","requirements-base.txt","-r","requirements-local.txt") "Dependencias de Jerónimo Abya Yala"

Section "Preparar aceleración de IA"
$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvidia) {
    if (Test-TorchRuntime $VenvPython $true) {
        Write-Host "PyTorch CUDA 2.8 ya está preparado; no se reinstala." -ForegroundColor Green
    } else {
        Write-Host "GPU NVIDIA detectada. Se instalará PyTorch 2.8.0 con CUDA 12.8." -ForegroundColor Green
        try {
            Invoke-Native $VenvPython @("-m","pip","install","--upgrade","--force-reinstall","torch==2.8.0","torchvision==0.23.0","torchaudio==2.8.0","--index-url","https://download.pytorch.org/whl/cu128") "PyTorch CUDA 12.8"
        } catch {
            Write-Warning "Falló la instalación CUDA: $($_.Exception.Message)"
        }
        if (-not (Test-TorchRuntime $VenvPython $true)) {
            Write-Warning "CUDA no quedó operativa. Jerónimo Abya Yala podrá funcionar en CPU; se instalará el runtime CPU para no bloquear las pruebas."
            Invoke-Native $VenvPython @("-m","pip","install","--upgrade","--force-reinstall","torch==2.8.0","torchvision==0.23.0","torchaudio==2.8.0","--index-url","https://download.pytorch.org/whl/cpu") "PyTorch CPU"
        }
    }
} else {
    if (Test-TorchRuntime $VenvPython $false) {
        Write-Host "PyTorch 2.8 ya está preparado; no se reinstala." -ForegroundColor Green
    } else {
        Write-Host "No se detectó nvidia-smi. Se instalará PyTorch CPU." -ForegroundColor Yellow
        Invoke-Native $VenvPython @("-m","pip","install","--upgrade","--force-reinstall","torch==2.8.0","torchvision==0.23.0","torchaudio==2.8.0","--index-url","https://download.pytorch.org/whl/cpu") "PyTorch CPU"
    }
}

Section "Instalar motor de diarización"
$old = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
& $VenvPython -c "import warnings; warnings.filterwarnings('ignore', message='torchcodec is not installed correctly.*', category=UserWarning); import whisperx, pyannote.audio, torch, faster_whisper; import model_manager_ui; assert hasattr(model_manager_ui,'ModelManagerWindow'); from whisperx.diarize import DiarizationPipeline as _DP; import inspect,sys; p=inspect.signature(_DP.__init__).parameters; assert ('token' in p or 'use_auth_token' in p); sys.exit(0 if str(getattr(whisperx,'__version__','')).startswith('3.8.') else 2)" *> $null
$runtimeOk = ($LASTEXITCODE -eq 0); $ErrorActionPreference = $old
if (-not $runtimeOk) {
    Invoke-Native $VenvPython @("-m","pip","install","-r","requirements-diarization.txt") "WhisperX / pyannote"
}

Section "Verificación local"
Invoke-Native $VenvPython @("-m","compileall","-q","jeronimo_app.py","onboarding.py","onboarding_ui.py","model_manager.py","model_manager_ui.py","system_diagnostics.py","ui_workflow.py","text_providers.py","transcript_editor.py","runtime_isolation.py","portable_runtime.py","job_engine.py","core_transcriber.py","workers") "Compilar fuentes"
Invoke-Native $VenvPython @("-c","import warnings; warnings.filterwarnings('ignore', message='torchcodec is not installed correctly.*', category=UserWarning); import customtkinter, tkinterdnd2, keyring, faster_whisper, whisperx, torch, pyannote.audio, model_manager_ui; assert hasattr(model_manager_ui,'ModelManagerWindow'); from whisperx.diarize import DiarizationPipeline as _DP; import inspect; p=inspect.signature(_DP.__init__).parameters; assert ('token' in p or 'use_auth_token' in p); print('Runtime OK | Torch', torch.__version__, '| CUDA', torch.cuda.is_available())") "Probar imports"
if (Test-Path (Join-Path $Root "tests")) {
    Invoke-Native $VenvPython @("-m","unittest","discover","-s","tests","-q") "Pruebas automáticas"
}

Section "Iniciar Jerónimo Abya Yala"
Write-Host "Los modelos grandes se descargarán sólo cuando los elijas desde la configuración inicial." -ForegroundColor Cyan
& $VenvPython (Join-Path $Root "jeronimo_app.py")
$code = $LASTEXITCODE
if ($null -ne $code -and $code -ne 0) { throw "Jerónimo Abya Yala terminó con código $code" }
