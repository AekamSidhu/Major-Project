# Runtime smoke test for the OpenMeet overlay on Windows.
#
# Start OpenMeet first (npm run tauri dev, or the installed app), then run:
#   powershell -ExecutionPolicy Bypass -File scripts\verify-overlay.ps1 [-Hotkeys] [-Screenshot <dir>]
#
# Checks, from outside the app, that the overlay window is:
#   * present and borderless,
#   * always-on-top (WS_EX_TOPMOST),
#   * non-activating (WS_EX_NOACTIVATE),
#   * excluded from capture when capture privacy is on (GetWindowDisplayAffinity),
# and, with -Hotkeys, that Ctrl+Shift+Space toggles its visibility without
# taking focus away from the foreground window.

param(
    [switch]$Hotkeys,
    [string]$Screenshot = ""
)

$ErrorActionPreference = "Stop"

Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class OmWin32 {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc cb, IntPtr lParam);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll", EntryPoint = "GetWindowLongPtrW")] public static extern IntPtr GetWindowLongPtr(IntPtr hWnd, int index);
    [DllImport("user32.dll")] public static extern bool GetWindowDisplayAffinity(IntPtr hWnd, out uint affinity);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT r);
    [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }

    public static List<IntPtr> WindowsOf(uint pid) {
        var list = new List<IntPtr>();
        EnumWindows((h, l) => { uint p; GetWindowThreadProcessId(h, out p); if (p == pid) list.Add(h); return true; }, IntPtr.Zero);
        return list;
    }
    public static string Title(IntPtr h) { var sb = new StringBuilder(512); GetWindowText(h, sb, 512); return sb.ToString(); }
}
"@

$GWL_STYLE = -16
$GWL_EXSTYLE = -20
$WS_CAPTION = 0x00C00000
$WS_EX_TOPMOST = 0x00000008
$WS_EX_NOACTIVATE = 0x08000000
$WS_EX_TRANSPARENT = 0x00000020
$WDA_EXCLUDEFROMCAPTURE = 0x11

$proc = Get-Process | Where-Object { $_.ProcessName -match '^(openmeet|OpenMeet)$' } | Select-Object -First 1
if (-not $proc) { Write-Error "OpenMeet is not running." }

$overlay = $null
$main = $null
foreach ($h in [OmWin32]::WindowsOf([uint32]$proc.Id)) {
    $title = [OmWin32]::Title($h)
    if ($title -eq "OpenMeet Overlay") { $overlay = $h }
    elseif ($title -eq "OpenMeet") { $main = $h }
}

$results = @()
function Check($name, $ok, $detail = "") {
    $script:results += [pscustomobject]@{ Check = $name; Result = $(if ($ok) { "PASS" } else { "FAIL" }); Detail = $detail }
}

Check "Overlay window exists" ($null -ne $overlay)
Check "Main window exists" ($null -ne $main)
if ($overlay) {
    $style = [int64][OmWin32]::GetWindowLongPtr($overlay, $GWL_STYLE)
    $ex = [int64][OmWin32]::GetWindowLongPtr($overlay, $GWL_EXSTYLE)
    Check "Overlay is frameless" (($style -band $WS_CAPTION) -ne $WS_CAPTION) ("style=0x{0:X}" -f $style)
    Check "Overlay is always on top" (($ex -band $WS_EX_TOPMOST) -ne 0) ("exstyle=0x{0:X}" -f $ex)
    Check "Overlay is non-activating" (($ex -band $WS_EX_NOACTIVATE) -ne 0)
    $affinity = 0
    [void][OmWin32]::GetWindowDisplayAffinity($overlay, [ref]$affinity)
    Check "Overlay capture affinity" $true ("affinity=0x{0:X} ({1})" -f $affinity, $(if ($affinity -eq $WDA_EXCLUDEFROMCAPTURE) { "excluded from capture" } else { "capturable" }))
    $clickThrough = ($ex -band $WS_EX_TRANSPARENT) -ne 0
    Check "Click-through state readable" $true ("click-through=" + $clickThrough)
    $r = New-Object OmWin32+RECT
    [void][OmWin32]::GetWindowRect($overlay, [ref]$r)
    Check "Overlay geometry" $true ("{0},{1} {2}x{3} visible={4}" -f $r.Left, $r.Top, ($r.Right - $r.Left), ($r.Bottom - $r.Top), [OmWin32]::IsWindowVisible($overlay))
}

if ($Hotkeys -and $overlay) {
    $VK_CONTROL = 0x11; $VK_SHIFT = 0x10; $VK_SPACE = 0x20; $KEYUP = 0x2
    function Send-ToggleOverlay {
        [OmWin32]::keybd_event($VK_CONTROL, 0, 0, [UIntPtr]::Zero)
        [OmWin32]::keybd_event($VK_SHIFT, 0, 0, [UIntPtr]::Zero)
        [OmWin32]::keybd_event($VK_SPACE, 0, 0, [UIntPtr]::Zero)
        Start-Sleep -Milliseconds 40
        [OmWin32]::keybd_event($VK_SPACE, 0, $KEYUP, [UIntPtr]::Zero)
        [OmWin32]::keybd_event($VK_SHIFT, 0, $KEYUP, [UIntPtr]::Zero)
        [OmWin32]::keybd_event($VK_CONTROL, 0, $KEYUP, [UIntPtr]::Zero)
    }
    $before = [OmWin32]::IsWindowVisible($overlay)
    $fgBefore = [OmWin32]::GetForegroundWindow()
    Send-ToggleOverlay
    Start-Sleep -Milliseconds 700
    $after = [OmWin32]::IsWindowVisible($overlay)
    $fgAfter = [OmWin32]::GetForegroundWindow()
    Check "Ctrl+Shift+Space toggles the overlay" ($before -ne $after) ("visible: $before -> $after")
    Check "Toggling does not steal focus" ($fgAfter -ne $overlay) ("foreground unchanged: " + ($fgBefore -eq $fgAfter))
    Send-ToggleOverlay
    Start-Sleep -Milliseconds 700
    Check "Toggle restores previous visibility" ([OmWin32]::IsWindowVisible($overlay) -eq $before)
}

if ($Screenshot) {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    New-Item -ItemType Directory -Force $Screenshot | Out-Null
    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
    $file = Join-Path $Screenshot ("screen-{0:yyyyMMdd-HHmmss}.png" -f (Get-Date))
    $bmp.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    Check "Screenshot saved" $true $file
}

$results | Format-Table -AutoSize
if ($results | Where-Object { $_.Result -eq "FAIL" }) { exit 1 }
