# Automated game test with the repository's NVIDIA profile (scripts\play-foxhound.ps1 switches), diag build:
# launch, press Enter from 15 s to 20 s, watch, stop, summarize.
# Usage (from the worktree): .\autotest.ps1 -Name tn1a [-Seconds 300] [-ShotEvery 150] [-Warm] [-SeriesS]
#   [-More @{VICTORY_X='1'}] [-Skip hash,hash] [-Dred] [-Timing] [-Loop n]
param(
    [Parameter(Mandatory)][string]$Name,
    [int]$Seconds = 300,
    [int]$ShotEvery = 150,
    # `victory preload` (every shader, then the pipelines written down by earlier runs) before the game.
    [switch]$Warm,
    [switch]$SeriesS,
    # Extra or overriding switches, passed on to play-foxhound.ps1.
    [hashtable]$More = @{},
    [string]$Skip = '',
    [switch]$Dred,
    [switch]$Timing,
    # Diag builds only; the value keys the shader cache, so it is left unset unless asked.
    [int]$Loop = 0,
    [string]$Game = $env:VICTORY_GAME_DIR
)

if (-not $Game) { throw 'Pass -Game <dir> or set VICTORY_GAME_DIR.' }
$wt = $PSScriptRoot
$out = Join-Path $wt "target\autotest\$Name"
New-Item -ItemType Directory -Force $out | Out-Null
$exe = Join-Path $wt 'target\diag\victory.exe'
$gameExe = Join-Path $Game 'game_scarlett_master.exe'

# The launcher's switches (Series X, 1440, DLSS when present, frame generation off, no FPS cap), set here.
& (Join-Path $wt 'scripts\play-foxhound.ps1') -Exe $gameExe -Build diag -SeriesS:$SeriesS -NoFrameGenReady -More $More -EnvOnly
$Console = if ($SeriesS) { 'series-s' } else { 'series-x' }

# Run from one fixed path: the NVIDIA shader cache is keyed by exe path, so it stays warm across worktrees.
$run = Join-Path $env:LOCALAPPDATA 'victory\run'
New-Item -ItemType Directory -Force $run | Out-Null
Get-ChildItem (Split-Path $exe) -File | Where-Object { $_.Name -in 'victory.exe', 'victory.pdb' -or $_.Extension -eq '.dll' } | ForEach-Object {
    $dst = Join-Path $run $_.Name
    # Any timestamp change, not only newer: an older build from another worktree must still replace the copy.
    if (-not (Test-Path $dst) -or $_.LastWriteTimeUtc -ne (Get-Item $dst).LastWriteTimeUtc) { Copy-Item $_.FullName $dst -Force }
}
$exe = Join-Path $run 'victory.exe'

$env:VICTORY_GAME_DIR = $Game
# Work on this base is done with the title's ray tracing on; -More @{VICTORY_BVH_BUILDER=''} turns it off.
if (-not $More.ContainsKey('VICTORY_BVH_BUILDER')) { $env:VICTORY_BVH_BUILDER = '1' }
elseif (-not $More['VICTORY_BVH_BUILDER']) { Remove-Item Env:VICTORY_BVH_BUILDER -ErrorAction SilentlyContinue }
# No pipeline is left out in our runs (the launcher leaves two out); -More @{VICTORY_LEAVE_OUT='...'} only by hand.
if (-not $More.ContainsKey('VICTORY_LEAVE_OUT')) { Remove-Item Env:VICTORY_LEAVE_OUT -ErrorAction SilentlyContinue }
$env:VICTORY_KEYS =(0..10 | ForEach-Object { '{0}:Enter' -f (15 + $_ * 0.5).ToString([cultureinfo]::InvariantCulture) }) -join ','
if ($Loop -gt 0) { $env:VICTORY_SHADER_LOOP_LIMIT = "$Loop" }
if ($Dred) { $env:VICTORY_D3D12_DRED = '1'; $env:VICTORY_GPU_TIMING = '1' }
# GPU time per pipeline in the log (each list submitted on its own: slower).
if ($Timing) { $env:VICTORY_GPU_TIMING = '1' }
if ($Skip) { $env:VICTORY_SKIP_PIPELINES = $Skip }
(Get-ChildItem Env:VICTORY_* | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' ' |
    Set-Content (Join-Path $out 'switches.txt')

if ($Warm) {
    $began = Get-Date
    Push-Location $wt
    try { & $exe preload $Game 2>&1 | Select-Object -Last 3 } finally { Pop-Location }
    'preload: {0:N1} min' -f ((Get-Date) - $began).TotalMinutes
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
$ran = [int]((Get-Date) - $start).TotalSeconds
if ($alive) {
    Save-Shot $p (Join-Path $out 'shot-end.png')
    # Ask to close first so an in-progress pipeline-library save can finish; kill after 30 s.
    if (-not ($p.CloseMainWindow() -and $p.WaitForExit(30000))) { Stop-Process -Id $p.Id -Force -Confirm:$false }
}

$t = Get-Content $log -Encoding UTF8 | ForEach-Object { $_ -replace "$([char]27)\[[0-9;]*m" }
$cut = { param($s, $n) $s.Substring(0, [Math]::Min($n, $s.Length)) }
"result: alive=$alive after ${ran}s"
$t | Select-String '^VIctory: ' | Select-Object -First 1 | ForEach-Object { & $cut $_.Line 600 }
($t | Select-String 'frame=(\d+)' | Select-Object -Last 1).Matches.Value
$t | Select-String ' fps, ' | Select-Object -Last 2 | ForEach-Object { & $cut ($_.Line -replace '^.*victory_gpu::host: ') 300 }
'refused or left out: ' + (@($t | Select-String 'left out|refus|stays off|does not translate|E_NOTIMPL|without a host pipeline \(' |
    Where-Object { $_.Line -notmatch 'dropped 0 without a host pipeline \(0 being made\) and 0 indirect' }).Count) + ' lines'
$t | Select-String 'left out|refus|stays off|does not translate|E_NOTIMPL' | ForEach-Object { ($_.Line -replace '^.*?(WARN|INFO|ERROR)\s+', '') -replace '\s+\S+=.*$', '' } |
    Group-Object | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { '  {0} {1}' -f $_.Count, (& $cut $_.Name 160) }
'slow submissions > 1 s: ' +@($t | Select-String 'slow host submission ms=\d{4,}').Count
'untranslated, by reason:'
$t | Select-String 'does not translate.*error=(.*?) cached' |
    ForEach-Object { $_.Matches[0].Groups[1].Value -replace '^.*GFX10 code at 0x[0-9a-f]+: ' } |
    Group-Object | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { '  {0} {1}' -f $_.Count, $_.Name }
$fail = $t | Select-String '^VIctory: host GPU failure' | Select-Object -First 1
if ($fail -and $fail.Line -match 'a list stopped at command.*') { 'hang: ' + (& $cut $Matches[0] 700) }
$first = $t | Select-String '^(\S+Z).*first bind' | Select-Object -First 1
if ($first) { 'first draw after {0:N0}s' -f ([datetimeoffset]::Parse($first.Matches[0].Groups[1].Value) - [datetimeoffset]$start).TotalSeconds }
"log: $((Get-Item $log).Length / 1MB -as [int]) MB; shots: $out"
