param([Parameter(Mandatory=$true)][string]$PortablePath)
$ErrorActionPreference = "Stop"
$P = (Resolve-Path $PortablePath).Path
$required = @(
  "$P\JeronimoAbyaYala.exe",
  "$P\runtime\ffmpeg\bin\ffmpeg.exe",
  "$P\runtime\ffmpeg\bin\ffprobe.exe",
  "$P\runtime\ffmpeg\bin\ffplay.exe",
  "$P\runtime_source\workers\diarization_worker.py",
  "$P\runtime_source\core_transcriber.py"
)
foreach ($item in $required) { if (-not (Test-Path $item)) { throw "Falta: $item" } }
if (Test-Path "$P\.env") { throw "El portable no debe contener .env real." }

# La BASE no debe contener el worker PyInstaller pesado ni paquetes Torch/WhisperX.
if (Test-Path "$P\runtime\diarization\JeronimoDiarizationWorker.exe") {
    throw "La BASE contiene el worker pesado. Debe descargarse/instalarse bajo demanda."
}
$heavy = Get-ChildItem $P -Recurse -File | Where-Object {
    $_.FullName -match '(?i)site-packages[\\/](torch|whisperx|pyannote)'
}
if ($heavy) { throw "La BASE contiene dependencias pesadas de diarización." }

$bytes = (Get-ChildItem $P -Recurse -File | Measure-Object -Property Length -Sum).Sum
if ($bytes -gt 1.5GB) { throw "La BASE supera 1536 MB antes de comprimir." }

$env:JERONIMO_PORTABLE_ROOT = $P
$proc = Start-Process -FilePath "$P\JeronimoAbyaYala.exe" -ArgumentList @("--self-test-json") -Wait -PassThru -WindowStyle Hidden
if ($proc.ExitCode -ne 0) { throw "JeronimoAbyaYala.exe no supera self-test (código $($proc.ExitCode))." }
Remove-Item Env:\JERONIMO_PORTABLE_ROOT -ErrorAction SilentlyContinue
Write-Host "BASE estructuralmente OK; runtime pesado ausente por diseño." -ForegroundColor Green
