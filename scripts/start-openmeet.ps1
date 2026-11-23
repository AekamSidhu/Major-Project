# Start OpenMeet on Windows in one command: the local AI sidecar (when it is
# installed and not already running), then the desktop app. When the app is
# closed (or Ctrl+C is pressed), the sidecar this script started is stopped too.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\start-openmeet.ps1 [-Dev] [-NoSidecar] [-SidecarPort 8765]
#   -Dev          run `npx tauri dev` (Vite + the app from source) instead of the built exe
#   -NoSidecar    do not start the sidecar (transcription and embeddings then need another server)
#   -SidecarPort  port of the sidecar (must match the URL in Settings; default 8765)
# First time? Run scripts\setup-local-ai.ps1 once, and `npm ci` in the repository root.

param(
    [switch]$Dev,
    [switch]$NoSidecar,
    [int]$SidecarPort = 8765
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$localAi = Join-Path $root "local-ai"
$sidecarPython = Join-Path $localAi ".venv\Scripts\python.exe"
$sidecarUrl = "http://127.0.0.1:$SidecarPort"

# The built app: CARGO_TARGET_DIR if set (as in a shared build folder), else the repository's target folder.
$targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root "target" }
$exe = Join-Path $targetDir "release\openmeet.exe"

$sidecar = $null

function Test-Sidecar {
    try {
        $health = Invoke-RestMethod -Uri "$sidecarUrl/health" -TimeoutSec 2
        return ($health.status -eq "ok")
    } catch {
        return $false
    }
}

function Stop-Sidecar {
    # The venv's python.exe starts the interpreter as a child, so stop the whole tree.
    if ($script:sidecar -and -not $script:sidecar.HasExited) {
        & taskkill.exe /PID $script:sidecar.Id /T /F | Out-Null
        Write-Host "Stopped the local AI sidecar that this script started."
    }
}

try {
    if ($NoSidecar) {
        Write-Host "Skipping the local AI sidecar (-NoSidecar)."
    } elseif (Test-Sidecar) {
        Write-Host "Local AI is already running at $sidecarUrl; OpenMeet will use it."
    } elseif (-not (Test-Path $sidecarPython)) {
        Write-Host "Local AI is not installed (no local-ai\.venv). Transcription and embeddings that need it will not work." -ForegroundColor Yellow
        Write-Host "To install it, run: powershell -ExecutionPolicy Bypass -File scripts\setup-local-ai.ps1" -ForegroundColor Yellow
    } else {
        Write-Host "Starting the local AI sidecar on $sidecarUrl (loading models, this can take a minute)..."
        $env:PYTHONUNBUFFERED = "1"
        $sidecar = Start-Process -FilePath $sidecarPython `
            -ArgumentList @("server.py", "--host", "127.0.0.1", "--port", "$SidecarPort") `
            -WorkingDirectory $localAi -WindowStyle Hidden -PassThru
        $ready = $false
        for ($i = 0; $i -lt 120; $i++) {
            if (Test-Sidecar) { $ready = $true; break }
            if ($sidecar.HasExited) { break }
            Start-Sleep -Seconds 1
        }
        if (-not $ready) {
            Stop-Sidecar
            throw "The local AI sidecar did not start on $sidecarUrl. Check that the port is free, then see the troubleshooting section of docs\RUNNING.md."
        }
        Write-Host "Local AI is ready."
    }

    if ($Dev) {
        Write-Host "Starting OpenMeet from source (npx tauri dev). Close the window or press Ctrl+C to stop."
        Push-Location $root
        try { & npx tauri dev } finally { Pop-Location }
    } elseif (Test-Path $exe) {
        Write-Host "Starting OpenMeet: $exe"
        $app = Start-Process -FilePath $exe -PassThru
        Write-Host "OpenMeet is running. Closing its window hides it to the tray; use the tray menu to quit."
        $app.WaitForExit()
    } else {
        throw "No built app at $exe. Build it with: npx tauri build (see docs\RUNNING.md), or start from source with -Dev."
    }
} finally {
    Stop-Sidecar
}
