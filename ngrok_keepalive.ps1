# Keeps the ngrok tunnel to the dashboard running.
# Started at logon by the "Nifty50 ngrok" scheduled task; restarts ngrok if it exits.
# Domain + password live in the ngrok config (%LOCALAPPDATA%\ngrok\ngrok.yml), not here.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = Join-Path $root "logs\ngrok.log"
$cfg  = Join-Path $env:LOCALAPPDATA "ngrok\ngrok.yml"
$ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $ngrok) {
    $ngrok = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ngrok.Ngrok_*\ngrok.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

function Log($msg) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" }

if (Select-String -Path $cfg -Pattern "YAHAN_APNA_PASSWORD" -Quiet) {
    Log "Password not set in ngrok.yml -- waiting."
    while (Select-String -Path $cfg -Pattern "YAHAN_APNA_PASSWORD" -Quiet) { Start-Sleep -Seconds 60 }
}

while ($true) {
    if (-not $ngrok -or -not (Test-Path $ngrok)) {
        Log "ngrok.exe not found (removed by Defender?) -- retrying in 60s."
        Start-Sleep -Seconds 60
        continue
    }
    if (Get-Process ngrok -ErrorAction SilentlyContinue) {
        # Another ngrok (e.g. started by hand) already holds the domain.
        Start-Sleep -Seconds 30
        continue
    }
    Log "Starting ngrok tunnel."
    & $ngrok start nifty --log=false
    Log "ngrok exited (code $LASTEXITCODE) -- restarting in 10s."
    Start-Sleep -Seconds 10
}
