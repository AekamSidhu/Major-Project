# Start the OpenMeet Room. An invite link is created automatically.
#   .\start-room.ps1                       # local only: local Whisper, simple-rules follow-ups, same Wi-Fi
#   .\start-room.ps1 -Followups local      # Ollama follow-ups (still on this computer)
#   .\start-room.ps1 -Tunnel               # also a public Cloudflare link (anyone with it can reach the join page)
#   .\start-room.ps1 -Transcription groq   # candidate's audio goes to Groq for transcription
#   .\start-room.ps1 -Followups online     # interviewer's questions and answer text go to Google Gemini
#   .\start-room.ps1 -Cloud                # the demo setup: -Tunnel -Transcription groq -Followups online
#   .\start-room.ps1 -NoTunnel             # no-op (the tunnel is off unless -Tunnel or -Cloud is given)
# Anything that goes online is printed before the room starts.
param(
    [ValidateSet("rules", "local", "online")] [string]$Followups = "rules",
    [ValidateSet("local", "groq")] [string]$Transcription = "local",
    [switch]$Tunnel,
    [switch]$Cloud,
    [int]$Port = 8443,
    [switch]$NoTunnel
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# -Cloud turns on the demo setup; an explicit -Followups or -Transcription still wins.
if ($Cloud) {
    $Tunnel = $true
    if (-not $PSBoundParameters.ContainsKey("Followups")) { $Followups = "online" }
    if (-not $PSBoundParameters.ContainsKey("Transcription")) { $Transcription = "groq" }
}

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

# What leaves this computer, printed before anything starts.
$online = @()
if ($Transcription -eq "groq") {
    $online += "  - The candidate's voice (audio, in short segments) goes to Groq (whisper-large-v3) for transcription."
}
if ($Followups -eq "online") {
    $online += "  - The interviewer's questions and the text of the answers (no audio or video; names and contact details removed) go to Google Gemini for follow-up questions and reference answers."
}
if ($Tunnel) {
    $online += "  - Anyone with the invite link can reach this room's join page (through Cloudflare Tunnel)."
}
if ($Cloud) { Write-Host "`nCLOUD MODE: parts of this room run online." -ForegroundColor Yellow }
if ($online.Count -gt 0) {
    Write-Host "`nGoes online:" -ForegroundColor Yellow
    $online | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
} else {
    Write-Host "`nNothing goes online: transcription, follow-ups and the link stay on this computer and this Wi-Fi."
}

# Build the argument list. Local is the default; nothing is sent online unless asked for.
$pyArgs = @("room_server.py", "--followups", $Followups, "--transcription", $Transcription, "--port", $Port)
if ($Tunnel) { $pyArgs += "--tunnel" }

$env:PYTHONUNBUFFERED = "1"
$server = Start-Process python -ArgumentList $pyArgs `
    -WorkingDirectory (Join-Path $root "local-ai") -NoNewWindow -PassThru

Write-Host "Starting the Room (loading models$(if ($Tunnel) { ' and tunnel' }))..."
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
if (-not $Tunnel) {
    Write-Host "     The link works from devices on this Wi-Fi. Use -Tunnel for other networks."
}
Write-Host "  Ctrl+C here stops the Room.`n"
Start-Process $url
Wait-Process -Id $server.Id
