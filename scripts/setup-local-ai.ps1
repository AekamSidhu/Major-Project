# Creates the OpenMeet local AI sidecar virtual environment (local-ai/.venv)
# and installs faster-whisper + fastembed. Everything stays on this device.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup-local-ai.ps1 [-Python 3.11] [-Dev] [-DownloadModels]

param(
    [string]$Python = "",
    [switch]$Dev,
    [switch]$DownloadModels
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root "local-ai"
$venv = Join-Path $dir ".venv"

function Find-Python {
    param([string]$Requested)
    $candidates = @()
    if ($Requested) { $candidates += , @("py", "-$Requested") }
    # faster-whisper / onnxruntime wheels are most reliable on 3.10 - 3.12.
    foreach ($v in @("3.12", "3.11", "3.10")) { $candidates += , @("py", "-$v") }
    $candidates += , @("python")
    foreach ($c in $candidates) {
        try {
            $exe = $c[0]
            $args = @()
            if ($c.Length -gt 1) { $args = $c[1..($c.Length - 1)] }
            $ver = & $exe @args -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver) {
                $parts = $ver.Trim().Split(".")
                if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10 -and [int]$parts[1] -le 12) {
                    return , $c
                }
            }
        } catch { }
    }
    return $null
}

$py = Find-Python -Requested $Python
if (-not $py) {
    Write-Error "Python 3.10 - 3.12 was not found. Install it from https://www.python.org/downloads/ and re-run."
}
$pyExe = $py[0]
$pyArgs = @()
if ($py.Length -gt 1) { $pyArgs = $py[1..($py.Length - 1)] }

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "Creating virtual environment in $venv"
    & $pyExe @pyArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create the virtual environment." }
}

$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
$req = if ($Dev) { "requirements-dev.txt" } else { "requirements.txt" }
& $venvPython -m pip install -r (Join-Path $dir $req)
if ($LASTEXITCODE -ne 0) { Write-Error "Dependency installation failed." }

if ($DownloadModels) {
    & $venvPython (Join-Path $dir "server.py") --download-models
}

Write-Host ""
Write-Host "Local AI is installed. OpenMeet starts it automatically when a meeting needs it,"
Write-Host "or run it yourself:  $venvPython $(Join-Path $dir 'server.py')"
