param(
    [switch]$BuildDiarizationComponent
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Section([string]$Title) { Write-Host "`n=== $Title ===" -ForegroundColor Cyan }

function Run-Step([string]$Title, [scriptblock]$Action) {
    Section $Title
    $global:LASTEXITCODE = 0
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Action
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) { throw "$Title fallo con codigo $code" }
}

function Invoke-CapturedProcess([string]$FilePath, [string]$Arguments, [string]$StandardInput = "") {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = $Arguments
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = $true
    $psi.CreateNoWindow = $true
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    if (-not [string]::IsNullOrEmpty($StandardInput)) { $proc.StandardInput.WriteLine($StandardInput) }
    $proc.StandardInput.Close()
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    [PSCustomObject]@{ ExitCode=[int]$proc.ExitCode; StdOut=[string]$stdout; StdErr=[string]$stderr }
}

function Require-Source([string]$Rel) {
    $p = Join-Path $Root $Rel
    if (-not (Test-Path $p)) { throw "Falta archivo requerido: $Rel" }
}

function Get-DirectorySizeBytes([string]$Path) {
    $sum = (Get-ChildItem $Path -Recurse -File | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return [int64]0 }
    return [int64]$sum
}

if ($env:OS -ne "Windows_NT") { throw "Este builder debe ejecutarse en Windows." }

Write-Host "============================================================"
Write-Host " JERONIMO ABYA YALA - COMPILAR BASE WINDOWS x64"
Write-Host "============================================================"
Write-Host "Fuentes: $Root"
Write-Host "Modo: BASE LIVIANA; PyTorch/WhisperX/pyannote/modelos NO se empaquetan." -ForegroundColor Yellow

Section "Comprobar fuentes y entorno local probado"
foreach ($rel in @(
    "jeronimo_app.py","automatic_setup.py","core_transcriber.py","credential_store.py","diarization_runtime_manager.py","job_engine.py",
    "local_text_runtime.py","model_manager.py","model_manager_ui.py","onboarding.py","onboarding_ui.py",
    "portable_runtime.py","pyannote_model_manager.py","runtime_isolation.py","subprocess_utils.py","system_diagnostics.py","system_monitor.py","text_providers.py",
    "transcript_editor.py","interview_text_editor.py","ui_help.py","ui_workflow.py","visual_theme.py","workers\diarization_worker.py",
    "build\Jeronimo.spec","build\verify_portable.ps1"
)) { Require-Source $rel }
if ($BuildDiarizationComponent) { Require-Source "build\JeronimoDiarizationWorker.spec" }

$DiarPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $DiarPython)) { throw "Falta .venv. Ejecuta primero INICIAR_JERONIMO_ABYA_YALA.bat para preparar y probar el programa." }
$probe = Invoke-CapturedProcess $DiarPython '"build\probe_diarization_env.py"'
if ($probe.ExitCode -ne 0) { throw "El entorno local no supera el probe WhisperX/Torch. STDERR: $($probe.StdErr)" }
$line = @(($probe.StdOut -split "`r?`n") | Where-Object { $_ -and $_.Trim().StartsWith("{") }) | Select-Object -Last 1
if (-not $line) { throw "El probe de diarizacion no devolvio JSON." }
$probeObj = $line | ConvertFrom-Json
if (-not $probeObj.ok) { throw "El entorno local no esta listo para diarizacion." }
Write-Host "Baseline fuente: WhisperX $($probeObj.whisperx) | Torch $($probeObj.torch) | CUDA $($probeObj.cuda)" -ForegroundColor Green

Run-Step "Pruebas antes de compilar" { & $DiarPython -m unittest discover -s tests -q }
Run-Step "Herramientas de build en entorno probado" { & $DiarPython -m pip install -r build\requirements-build.txt }

Section "Preparar entorno liviano del ejecutable principal"
$MainVenv = Join-Path $Root ".build\main-venv"
$MainPython = Join-Path $MainVenv "Scripts\python.exe"
if (-not (Test-Path $MainPython)) {
    New-Item -ItemType Directory -Force (Split-Path $MainVenv) | Out-Null
    Run-Step "Crear main-venv" { & $DiarPython -m venv $MainVenv }
}
Run-Step "Actualizar main-venv" { & $MainPython -m pip install --upgrade pip setuptools wheel }
Run-Step "Dependencias main-venv" { & $MainPython -m pip install -r requirements-base.txt -r requirements-local.txt -r build\requirements-build.txt }
Run-Step "Validar dependencias GUI main-venv" {
    & $MainPython -c "import PIL, customtkinter, tkinterdnd2; from PIL import Image; print('Pillow ' + PIL.__version__ + ' | CustomTkinter ' + customtkinter.__version__ + ' | GUI imports OK')"
}
Run-Step "Compilar fuentes" {
    & $MainPython -m compileall -q jeronimo_app.py automatic_setup.py core_transcriber.py credential_store.py diarization_runtime_manager.py job_engine.py local_text_runtime.py model_manager.py model_manager_ui.py onboarding.py onboarding_ui.py portable_runtime.py pyannote_model_manager.py runtime_isolation.py subprocess_utils.py system_diagnostics.py system_monitor.py text_providers.py transcript_editor.py interview_text_editor.py ui_help.py ui_workflow.py visual_theme.py workers tools
}

Section "Localizar FFmpeg"
$ffmpegCmd = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
$ffprobeCmd = Get-Command ffprobe.exe -ErrorAction SilentlyContinue
$ffplayCmd = Get-Command ffplay.exe -ErrorAction SilentlyContinue
if (-not $ffmpegCmd -or -not $ffprobeCmd -or -not $ffplayCmd) { throw "Falta ffmpeg/ffprobe/ffplay. Ejecuta primero el BAT de inicio para que intente instalar FFmpeg." }
$PortableFfmpeg = Join-Path $Root ".build\ffmpeg\bin"
New-Item -ItemType Directory -Force $PortableFfmpeg | Out-Null
Copy-Item $ffmpegCmd.Source,$ffprobeCmd.Source,$ffplayCmd.Source -Destination $PortableFfmpeg -Force

Section "Construir ejecutable base"
$Dist = Join-Path $Root ".build\dist"
$Work = Join-Path $Root ".build\work"
Remove-Item $Dist,$Work -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Dist,$Work | Out-Null
Run-Step "Build JeronimoAbyaYala.exe" { & $MainPython -m PyInstaller --noconfirm --clean --distpath $Dist --workpath "$Work\main" build\Jeronimo.spec }

$MainDist = Join-Path $Dist "JeronimoAbyaYala"
if (-not (Test-Path "$MainDist\JeronimoAbyaYala.exe")) { throw "No se genero JeronimoAbyaYala.exe" }

Section "Crear BASE liviana"
$Core = Join-Path $Root ".build\Jeronimo_Abya_Yala_BASE"
Remove-Item $Core -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item $MainDist $Core -Recurse
New-Item -ItemType Directory -Force "$Core\runtime\ffmpeg\bin" | Out-Null
Copy-Item "$PortableFfmpeg\*.exe" "$Core\runtime\ffmpeg\bin\" -Force

# El worker se entrega como fuente diminuta. El Python administrado + Torch +
# WhisperX + pyannote se descargan sólo en el primer setup que requiera hablantes.
$RuntimeSource = Join-Path $Core "runtime_source"
New-Item -ItemType Directory -Force "$RuntimeSource\workers" | Out-Null
foreach ($rel in @(
    "core_transcriber.py","credential_store.py","job_engine.py","local_text_runtime.py",
    "portable_runtime.py","pyannote_model_manager.py","runtime_isolation.py","subprocess_utils.py","text_providers.py"
)) {
    Copy-Item (Join-Path $Root $rel) (Join-Path $RuntimeSource $rel) -Force
}
Copy-Item (Join-Path $Root "workers\diarization_worker.py") "$RuntimeSource\workers\diarization_worker.py" -Force

$MirrorDescriptor = Join-Path $Root "pyannote_mirror_release.json"
if (Test-Path $MirrorDescriptor) {
    Run-Step "Validar descriptor mirror Community-1" {
        & $MainPython -c "from pyannote_model_manager import load_mirror_release; r=load_mirror_release(require_enabled=True); print('Mirror Community-1 OK | revision ' + r.revision[:12] + ' | ' + r.archive_sha256[:12])"
    }
    Copy-Item $MirrorDescriptor (Join-Path $Core "pyannote_mirror_release.json") -Force
    Write-Host "Mirror Community-1: descriptor incluido en la BASE." -ForegroundColor Green
} else {
    Write-Host "AVISO: pyannote_mirror_release.json no existe; Community-1 usara el respaldo oficial de Hugging Face." -ForegroundColor Yellow
}

Remove-Item "$Core\.env" -Force -ErrorAction SilentlyContinue

# Guardia anti-monolito: la BASE nunca debe arrastrar accidentalmente Torch/WhisperX.
$heavyLeaks = Get-ChildItem $Core -Recurse -File | Where-Object {
    $_.FullName -match '(?i)site-packages[\\/](torch|whisperx|pyannote)' -or
    $_.Name -match '(?i)^(torch|whisperx|pyannote).*\.dist-info'
}
if ($heavyLeaks) {
    $names = ($heavyLeaks | Select-Object -First 8 | ForEach-Object { $_.FullName }) -join "`n"
    throw "La BASE contiene dependencias pesadas que deben descargarse bajo demanda:`n$names"
}
$coreBytes = Get-DirectorySizeBytes $Core
$coreMB = [Math]::Round($coreBytes / 1MB, 1)
if ($coreBytes -gt 1.5GB) {
    throw "La BASE ocupa $coreMB MB. Supera el límite anti-monolito de 1536 MB; revisar dependencias antes de distribuir."
}
Write-Host "BASE generada: $coreMB MB sin runtime pesado de diarizacion." -ForegroundColor Green

Section "Autodiagnostico de la BASE"
$env:JERONIMO_PORTABLE_ROOT = $Core
$proc = Start-Process -FilePath "$Core\JeronimoAbyaYala.exe" -ArgumentList @("--self-test-json") -Wait -PassThru -WindowStyle Hidden
if ($proc.ExitCode -ne 0) { throw "JeronimoAbyaYala.exe no supera self-test (codigo $($proc.ExitCode))." }
Remove-Item Env:\JERONIMO_PORTABLE_ROOT -ErrorAction SilentlyContinue
Run-Step "Verificar BASE" { & "$PSScriptRoot\verify_portable.ps1" -PortablePath $Core }

Section "Comprimir BASE"
$CoreZip = Join-Path $Root ".build\Jeronimo_Abya_Yala_BASE_Windows_x64.zip"
Remove-Item $CoreZip -Force -ErrorAction SilentlyContinue
Compress-Archive -Path "$Core\*" -DestinationPath $CoreZip -CompressionLevel Optimal
$f = Get-Item $CoreZip
$mb = [Math]::Round($f.Length / 1MB, 1)
$hash = (Get-FileHash $CoreZip -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "$($f.Name) | $mb MB | SHA256 $hash" -ForegroundColor Green

if ($BuildDiarizationComponent) {
    Section "OPCIONAL mantenedor: compilar componente pesado de diarizacion"
    Write-Host "Este modo NO se usa para el paquete público normal." -ForegroundColor Yellow
    Run-Step "Build componente de diarizacion" { & $DiarPython -m PyInstaller --noconfirm --clean --distpath $Dist --workpath "$Work\diarization" build\JeronimoDiarizationWorker.spec }
    $WorkerDist = Join-Path $Dist "JeronimoDiarizationWorker"
    if (-not (Test-Path "$WorkerDist\JeronimoDiarizationWorker.exe")) { throw "No se genero el worker de diarizacion" }
    $Component = Join-Path $Root ".build\Jeronimo_Abya_Yala_COMPONENTE_DIARIZACION"
    Remove-Item $Component -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force "$Component\runtime\diarization" | Out-Null
    Copy-Item "$WorkerDist\*" "$Component\runtime\diarization\" -Recurse -Force
    $worker = "$Component\runtime\diarization\JeronimoDiarizationWorker.exe"
    $workerProbe = Invoke-CapturedProcess $worker '--probe'
    if ($workerProbe.ExitCode -ne 0) { throw "Worker opcional fallo probe: $($workerProbe.StdOut) $($workerProbe.StdErr)" }
    $ComponentZip = Join-Path $Root ".build\Jeronimo_Abya_Yala_COMPONENTE_DIARIZACION_Windows_x64.zip"
    Remove-Item $ComponentZip -Force -ErrorAction SilentlyContinue
    Compress-Archive -Path "$Component\*" -DestinationPath $ComponentZip -CompressionLevel Optimal
    $cf = Get-Item $ComponentZip
    Write-Host "$($cf.Name) | $([Math]::Round($cf.Length / 1MB, 1)) MB" -ForegroundColor Yellow
}

Write-Host "`nCOMPILACION BASE COMPLETADA" -ForegroundColor Green
Write-Host "Distribui SOLO: $CoreZip"
Write-Host "PyTorch/WhisperX/pyannote y los modelos se descargan desde Jeronimo durante la configuracion inicial." -ForegroundColor Cyan
Write-Host "El espacio pesado aparece solo despues de elegir transcripcion local con hablantes; no dentro del ZIP base." -ForegroundColor Cyan
