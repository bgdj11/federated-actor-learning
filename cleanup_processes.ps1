# Cleanup all Python processes
# Run this script before starting the FL system if you encounter port binding errors

Write-Host "===============================================" -ForegroundColor Yellow
Write-Host "Cleaning up Python processes..." -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Yellow
Write-Host ""

$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue

if ($pythonProcesses) {
    Write-Host "Found $($pythonProcesses.Count) Python process(es). Terminating..." -ForegroundColor Cyan
    $pythonProcesses | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "All Python processes terminated." -ForegroundColor Green
} else {
    Write-Host "No Python processes found." -ForegroundColor Green
}

Write-Host ""
Write-Host "Checking for remaining processes..." -ForegroundColor Cyan
$remaining = Get-Process python -ErrorAction SilentlyContinue

if ($remaining) {
    Write-Host "WARNING: Some processes are still running!" -ForegroundColor Red
    $remaining | Format-Table -AutoSize
} else {
    Write-Host "All clear! You can now start the FL system." -ForegroundColor Green
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Yellow
Write-Host "Done!" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Yellow
