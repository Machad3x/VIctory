# Automated game test: launch, press Enter from 15 s to 20 s, watch, stop, summarize.
# Usage (from the worktree): .\autotest.ps1 -Name gpu65 [-Seconds 360] [-Warm] [-Skip hash,hash] [-Dred] [-Loop n] [-Console series-s]
param(
    [Parameter(Mandatory)][string]$Name,
    [int]$Seconds = 300,
    [int]$ShotEvery = 60,
    [switch]$Warm,
    [string]$Skip = '',
    [switch]$Dred,
    [switch]$Timing,
    [int]$Loop = 4096,
    [ValidateSet('series-x', 'series-s')][string]$Console = 'series-x',
    [string]$Game = $env:VICTORY_GAME_DIR
)

if (-not $Game) { throw 'Pass -Game <dir> or set VICTORY_GAME_DIR.' }
$wt = $PSScriptRoot
$out = Join-Path $wt "target\autotest\$Name"
New-Item -ItemType Directory -Force $out | Out-Null
$exe = Join-Path $wt 'target\diag\victory.exe'

$env:VICTORY_GAME_DIR = $Game
$env:VICTORY_KEYS = (0..10 | ForEach-Object { '{0}:Enter' -f (15 + $_ * 0.5).ToString([cultureinfo]::InvariantCulture) }) -join ','
$env:VICTORY_SHADER_LOOP_LIMIT = "$Loop"
$env:VICTORY_SUBMIT_MAX_RUNS = '1'
# Title buffers in video memory through the resizable BAR: those shaders read
# (#2120) and those they write, with a mirror the CPU reads (#2130).
$env:VICTORY_GPU_UPLOAD = '1'
$env:VICTORY_GPU_UPLOAD_WRITES = '1'
if ($Dred) { $env:VICTORY_D3D12_DRED = '1'; $env:VICTORY_GPU_TIMING = '1' }
else { Remove-Item Env:VICTORY_D3D12_DRED, Env:VICTORY_GPU_TIMING -ErrorAction SilentlyContinue }
# GPU time per pipeline in the log (each list submitted on its own: slower).
if ($Timing) { $env:VICTORY_GPU_TIMING = '1' }
if ($Skip) { $env:VICTORY_SKIP_PIPELINES = $Skip } else { Remove-Item Env:VICTORY_SKIP_PIPELINES -ErrorAction SilentlyContinue }

if ($Warm) {
    $files = 'shaders.rpf', 'game_scarlett_master.exe', 'XboxBvhBuilder.dll', 'amd_fidelityfx_upscaler_xbox.dll' |
        ForEach-Object { Join-Path $Game $_ }
    & $exe translate --cache @files 2>&1 | Select-Object -Last 1
}

Add-Type -AssemblyName System.Drawing
Add-Type 'using System; using System.Runtime.InteropServices; public class Win { public struct RECT { public int Left, Top, Right, Bottom; } [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r); [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint f); }'

function Save-Shot($p, $path) {
    $p.Refresh()
    if ($p.MainWindowHandle -eq 0) { return }
    $r = New-Object Win+RECT
    [void][Win]::GetWindowRect($p.MainWindowHandle, [ref]$r)
    $w = $r.Right - $r.Left; $h = $r.Bottom - $r.Top
    if ($w -le 0 -or $h -le 0) { return }
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $dc = $gfx.GetHdc(); [void][Win]::PrintWindow($p.MainWindowHandle, $dc, 2); $gfx.ReleaseHdc($dc)
    $small = New-Object System.Drawing.Bitmap $bmp, 960, ([int](960 * $h / $w))
    $small.Save($path)
    $gfx.Dispose(); $bmp.Dispose(); $small.Dispose()
}

$log = Join-Path $out 'err.log'
$p = Start-Process $exe -ArgumentList @('run', (Join-Path $Game 'game_scarlett_master.exe'), '--execute', '--console', $Console) -WorkingDirectory $wt -PassThru -RedirectStandardError $log -RedirectStandardOutput (Join-Path $out 'out.log')
$start = Get-Date
$next = $ShotEvery
while (-not $p.HasExited -and ((Get-Date) - $start).TotalSeconds -lt $Seconds) {
    Start-Sleep -Seconds 3
    if (((Get-Date) - $start).TotalSeconds -ge $next) {
        Save-Shot $p (Join-Path $out ('shot-{0:000}.png' -f $next))
        $next += $ShotEvery
    }
}
$alive = -not $p.HasExited
if ($alive) { Save-Shot $p (Join-Path $out 'shot-end.png'); Stop-Process -Id $p.Id -Force -Confirm:$false }
$ran = [int]((Get-Date) - $start).TotalSeconds

$t = Get-Content $log -Encoding UTF8 | ForEach-Object { $_ -replace "$([char]27)\[[0-9;]*m" }
$cut = { param($s, $n) $s.Substring(0, [Math]::Min($n, $s.Length)) }
"result: alive=$alive after ${ran}s"
$t | Select-String '^VIctory: ' | Select-Object -First 1 | ForEach-Object { & $cut $_.Line 600 }
($t | Select-String 'frame=(\d+)' | Select-Object -Last 1).Matches.Value
$t | Select-String ' fps, ' | Select-Object -Last 2 | ForEach-Object { & $cut ($_.Line -replace '^.*victory_gpu::host: ') 300 }
'slow submissions > 1 s: ' + @($t | Select-String 'slow host submission ms=\d{4,}').Count
'untranslated, by reason:'
$t | Select-String 'does not translate.*error=(.*?) cached' |
    ForEach-Object { $_.Matches[0].Groups[1].Value -replace '^.*GFX10 code at 0x[0-9a-f]+: ' } |
    Group-Object | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { '  {0} {1}' -f $_.Count, $_.Name }
$fail = $t | Select-String '^VIctory: host GPU failure' | Select-Object -First 1
if ($fail -and $fail.Line -match 'a list stopped at command.*') { 'hang: ' + (& $cut $Matches[0] 700) }
$first = $t | Select-String '^(\S+Z).*first bind' | Select-Object -First 1
if ($first) { 'first draw after {0:N0}s' -f ([datetimeoffset]::Parse($first.Matches[0].Groups[1].Value) - [datetimeoffset]$start).TotalSeconds }
"log: $((Get-Item $log).Length / 1MB -as [int]) MB; shots: $out"
