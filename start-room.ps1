# Start the OpenMeet Room. A shareable Cloudflare link is created automatically.
#   .\start-room.ps1                             # Groq transcription + Gemini follow-ups + tunnel (default)
#   .\start-room.ps1 -Transcription local        # local Whisper instead of Groq
#   .\start-room.ps1 -Followups rules            # simple-rules follow-ups (no AI model)
#   .\start-room.ps1 -NoTunnel                   # LAN only (same Wi-Fi, no Cloudflare)
param(
    [ValidateSet("rules", "local", "online")] [string]$Followups = "online",
    [ValidateSet("local", "groq")] [string]$Transcription = "groq",
    [int]$Port = 8443,
    [switch]$NoTunnel
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# A VPN (e.g. Proton's ProTUN) drops traffic from other devices on the Wi-Fi and breaks the video link.
$vpn = Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object {
    $_.Status -eq "Up" -and ($_.Name -match "TUN|VPN|WireGuard|Proton" -or $_.InterfaceDescription -match "TAP|Wintun|WireGuard|VPN|Proton")
}
if ($vpn) {
    Write-Host "`nA VPN is connected: $($vpn.Name -join ', ')" -ForegroundColor Yellow
    Write-Host "Other devices usually can't reach this computer while it's on. Disconnect it, or enable 'Allow LAN connections'." -ForegroundColor Yellow
    $answer = Read-Host "Continue anyway? (y/N)"
    if ($answer -ne "y") { exit 1 }
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Port $Port is already in use (process $($existing[0].OwningProcess)). Stop it first." -ForegroundColor Red
    exit 1
}

# Build the argument list: the tunnel is on by default, --no-tunnel disables it.
$pyArgs = @("room_server.py", "--followups", $Followups, "--transcription", $Transcription, "--port", $Port)
if ($NoTunnel) { $pyArgs += "--no-tunnel" }

$env:PYTHONUNBUFFERED = "1"
$server = Start-Process python -ArgumentList $pyArgs `
    -WorkingDirectory (Join-Path $root "local-ai") -NoNewWindow -PassThru

Write-Host "Starting the Room (loading models and tunnel)..."
$url = "https://localhost:$Port/host"
for ($i = 0; $i -lt 120; $i++) {
    if ($server.HasExited) { Write-Host "The Room server stopped. See the output above." -ForegroundColor Red; exit 1 }
    $code = & curl.exe -sk -o NUL -w "%{http_code}" $url 2>$null
    if ($code -eq "200") { break }
    Start-Sleep -Seconds 1
}

Write-Host "`nRoom ready. Opening $url" -ForegroundColor Green
Write-Host "  1. Accept the certificate warning once (Advanced -> Continue)."
Write-Host "  2. Copy the invite link from the host page and send it to the candidate."
if (-not $NoTunnel) {
    Write-Host "     The link works from any device on any network (powered by Cloudflare Tunnel)."
}
Write-Host "  Ctrl+C here stops the Room.`n"
Start-Process $url
Wait-Process -Id $server.Id
