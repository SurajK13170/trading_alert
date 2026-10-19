# =============================================================================
#  BANKNIFTY AI TRADING SYSTEM -- START ALL
#  ----------------------------------------------------------------------------
#  Ek hi command se saari engines + backend + frontend chala deta hai.
#  Har service apne NAYE PowerShell window me kholta hai (title se identify kar).
#
#  Usage:
#    .\start_all.ps1            # sab start
#    .\start_all.ps1 -SkipUI    # bas backend + engines, frontend skip
#
#  PIDs saved to .running_pids -- stop_all.ps1 use karke graceful shutdown
# =============================================================================

param(
    [switch]$SkipUI,
    [switch]$SkipTickCollector
)

$ErrorActionPreference = "Stop"
$root      = $PSScriptRoot
$ai_dir    = Join-Path $root "ai_engine"
$be_dir    = Join-Path $root "backend_manager"
$fe_dir    = Join-Path $root "frontend_manager"
$venv_act  = Join-Path $root ".venv\Scripts\Activate.ps1"
$pids_file = Join-Path $root ".running_pids"

# Sanity checks
if (-not (Test-Path $venv_act)) {
    Write-Host "[ERR] .venv not found at $venv_act" -ForegroundColor Red
    Write-Host "      Create venv first: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path (Join-Path $ai_dir "trading_model.pkl"))) {
    Write-Host "[ERR] trading_model.pkl not found in ai_engine/" -ForegroundColor Red
    Write-Host "      Train model first: python model_trainer.py" -ForegroundColor Yellow
    exit 1
}

# Clear any old pid file
if (Test-Path $pids_file) { Remove-Item $pids_file -Force }

# Check Redis
$redis_running = $false
try {
    $redis_test = & redis-cli ping 2>&1
    if ($redis_test -match "PONG") { $redis_running = $true }
} catch {}
if (-not $redis_running) {
    Write-Host "[WARN] Redis pe PONG nahi mila. Redis chal raha hai? Pehle start kar:" -ForegroundColor Yellow
    Write-Host "       redis-server" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue without Redis check? (y/n)"
    if ($continue -ne "y") { exit 1 }
} else {
    Write-Host "[OK] Redis is up" -ForegroundColor Green
}

# Helper: open a new PowerShell window with title + venv + command
function Start-Window {
    param(
        [string]$Title,
        [string]$WorkDir,
        [string]$Cmd,
        [string]$Color = "Green"
    )
    Write-Host "  Starting $Title..." -ForegroundColor $Color
    # Build the command that the new window will run
    $inner = @"
`$Host.UI.RawUI.WindowTitle = '$Title'
Set-Location '$WorkDir'
& '$venv_act'
Write-Host '=================================================' -ForegroundColor Cyan
Write-Host ' $Title' -ForegroundColor Cyan
Write-Host '=================================================' -ForegroundColor Cyan
Write-Host ''
$Cmd
"@
    $proc = Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", $inner
    ) -PassThru
    Add-Content -Path $pids_file -Value "$($proc.Id)|$Title"
    Start-Sleep -Milliseconds 800
}

# Pretty banner
Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  BANKNIFTY AI TRADING SYSTEM -- BOOT SEQUENCE" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

$step = 1
$total = 6
if ($SkipTickCollector) { $total-- }
if ($SkipUI) { $total-- }

# 1. Tick Collector (sabse pehle -- Angel session le lega)
if (-not $SkipTickCollector) {
    Write-Host "[$step/$total] Tick Collector (Angel WebSocket)" -ForegroundColor Yellow
    Start-Window "[BNF] Tick Collector" $ai_dir "python tick_collector.py"
    $step++
    Start-Sleep -Seconds 4   # Angel auth ko time
}

# 2. Quant Engine
Write-Host "[$step/$total] Quant Engine (indicators every 30s)" -ForegroundColor Yellow
Start-Window "[BNF] Quant Engine" $ai_dir "python indian_quant_engine.py"
$step++
Start-Sleep -Seconds 3

# 3. AI Engine
Write-Host "[$step/$total] AI Engine (signal generation)" -ForegroundColor Yellow
Start-Window "[BNF] AI Engine" $ai_dir "python indian_ai_engine.py"
$step++
Start-Sleep -Seconds 3

# 4. Execution Manager
Write-Host "[$step/$total] Execution Manager (trade execution)" -ForegroundColor Yellow
Start-Window "[BNF] Execution Manager" $ai_dir "python indian_execution_manager.py"
$step++
Start-Sleep -Seconds 2

# 5. Backend (Express API)
Write-Host "[$step/$total] Backend (port 3100)" -ForegroundColor Yellow
Start-Window "[BNF] Backend" $be_dir "node app.js"
$step++
Start-Sleep -Seconds 2

# 6. Frontend (Vite dev)
if (-not $SkipUI) {
    Write-Host "[$step/$total] Frontend (port 5174)" -ForegroundColor Yellow
    Start-Window "[BNF] Frontend" $fe_dir "npm run dev"
    Start-Sleep -Seconds 3
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "  ALL SERVICES LAUNCHED -- check the new windows" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard : http://localhost:5174" -ForegroundColor Cyan
Write-Host "  Backend   : http://localhost:3100" -ForegroundColor Cyan
Write-Host ""
Write-Host "  PIDs saved to:  $pids_file" -ForegroundColor Gray
Write-Host "  To stop all  :  .\stop_all.ps1" -ForegroundColor Gray
Write-Host ""

# Optionally auto-open dashboard
if (-not $SkipUI) {
    Start-Sleep -Seconds 4   # give Vite time to be ready
    Start-Process "http://localhost:5174"
}
