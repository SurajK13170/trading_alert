# =============================================================================
#  BANKNIFTY AI TRADING SYSTEM -- STOP ALL
#  ----------------------------------------------------------------------------
#  Graceful shutdown of services launched by start_all.ps1
#  Uses .running_pids file to target only OUR processes
#  (won't kill other Python/Node processes running on the system)
# =============================================================================

$root      = $PSScriptRoot
$pids_file = Join-Path $root ".running_pids"

if (-not (Test-Path $pids_file)) {
    Write-Host "[WARN] .running_pids file nahi mili -- start_all.ps1 se kuch start nahi hua tha?" -ForegroundColor Yellow
    Write-Host ""
    $force = Read-Host "Force-kill SAARE python+node processes? (y/n)"
    if ($force -eq "y") {
        Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
        Get-Process node   -ErrorAction SilentlyContinue | Stop-Process -Force
        Write-Host "[OK] Sab python + node processes kill kar diye." -ForegroundColor Green
    }
    exit 0
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  STOPPING BANKNIFTY AI TRADING SYSTEM" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

$lines = Get-Content $pids_file
$total = $lines.Count
$killed = 0
$missing = 0

foreach ($line in $lines) {
    $parts = $line -split "\|"
    $pid_id = [int]$parts[0]
    $title  = if ($parts.Length -gt 1) { $parts[1] } else { "Service" }

    $proc = Get-Process -Id $pid_id -ErrorAction SilentlyContinue
    if ($proc) {
        try {
            # Stop launcher window AND its children (the actual python/node process)
            $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $pid_id" -ErrorAction SilentlyContinue
            foreach ($c in $children) {
                Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue
            }
            Stop-Process -Id $pid_id -Force
            Write-Host "  [STOP] $title (PID $pid_id)" -ForegroundColor Green
            $killed++
        } catch {
            Write-Host "  [ERR] $title (PID $pid_id) -- $_" -ForegroundColor Red
        }
    } else {
        Write-Host "  [MISS] $title (PID $pid_id) -- already exited" -ForegroundColor Gray
        $missing++
    }
}

# Cleanup
Remove-Item $pids_file -Force

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "  Stopped: $killed / $total  |  Already exited: $missing" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
