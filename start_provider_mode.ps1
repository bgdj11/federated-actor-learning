# Federated Learning - Provider Mode Startup Script
# This script starts all 4 components in separate PowerShell windows

$projectRoot = $PSScriptRoot
$venvActivate = Join-Path $projectRoot "venv\Scripts\Activate.ps1"


Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting Federated Learning System (Provider Mode)" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Cleanup any existing Python processes
Write-Host "Cleaning up existing Python processes..." -ForegroundColor Yellow
$existing = Get-Process python -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "Cleaned up $($existing.Count) Python process(es)." -ForegroundColor Green
} else {
    Write-Host "No existing Python processes found." -ForegroundColor Green
}
Write-Host ""

# 1. Start Provider (main orchestrator)
Write-Host "[1/4] Starting Provider on port 5001..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$venvActivate'; cd '$projectRoot'; python fl/run_scripts/run_provider.py --port 5001 --rounds 5 --workers 3"

Start-Sleep -Seconds 3

# 2. Start Aggregator
Write-Host "[2/4] Starting Aggregator on port 5000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$venvActivate'; cd '$projectRoot'; python fl/run_scripts/run_aggregator.py --port 5000 --provider-host localhost --provider-port 5001"

Start-Sleep -Seconds 3

# 3. Start Evaluator
Write-Host "[3/4] Starting Evaluator on port 5010..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$venvActivate'; cd '$projectRoot'; python fl/run_scripts/run_evaluator.py --provider-host localhost --provider-port 5001 --port 5010"

Start-Sleep -Seconds 3

# 4. Start Workers (A, B, C)
Write-Host "[4/4] Starting Workers on port 5002..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$venvActivate'; cd '$projectRoot'; python fl/run_scripts/run_workers.py --port 5002 --provider-host localhost --provider-port 5001 --epochs 3"

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "All components started!" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor the 4 PowerShell windows for logs." -ForegroundColor Yellow
Write-Host "To stop: Press Ctrl+C in each window or run cleanup_processes.ps1" -ForegroundColor Yellow
Write-Host ""
