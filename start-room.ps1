# Start the OpenMeet Room. An invite link is created automatically.
#   .\start-room.ps1                       # simple-rules follow-ups, LAN only (same Wi-Fi)
#   .\start-room.ps1 -Followups local      # Ollama follow-ups, LAN only
#   .\start-room.ps1 -Tunnel               # also a public Cloudflare link (off by default)
#   .\start-room.ps1 -NoTunnel             # no-op, kept so old commands still work
# With -Tunnel, anyone with the link can reach the room's join page.
param(
    [ValidateSet("rules", "local", "online")] [string]$Followups = "rules",
    [int]$Port = 8443,
    [switch]$Tunnel,
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

# Build the argument list: the tunnel is off unless -Tunnel is given.
$pyArgs = @("room_server.py", "--followups", $Followups, "--port", $Port)
if ($Tunnel) { $pyArgs += "--tunnel" }

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
if ($Tunnel) {
    Write-Host "     Anyone with the link can reach this room's join page (through Cloudflare Tunnel)." -ForegroundColor Yellow
} else {
    Write-Host "     The link works from devices on this Wi-Fi. Use -Tunnel for other networks."
}
Write-Host "  Ctrl+C here stops the Room.`n"
Start-Process $url
Wait-Process -Id $server.Id
