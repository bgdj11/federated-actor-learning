Write-Host "===============================================" -ForegroundColor Yellow
Write-Host "Cleaning up Python processes and FL ports..." -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Yellow
Write-Host ""

function Get-ListeningProcessId([int]$Port) {
    try {
        $c = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($c) { return [int]$c.OwningProcess }
    } catch {}
    return $null
}

function Get-ProcessNameSafe([int]$ProcessId) {
    try { return (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue).Name } catch {}
    return $null
}

function Stop-PythonOnPort([int]$Port) {
    $procId = Get-ListeningProcessId $Port
    if (-not $procId) { return }

    $procName = Get-ProcessNameSafe $procId
    Write-Host "[INFO] Port $Port is in use by PID $procId ($procName)" -ForegroundColor Cyan

    if ($procName -and ($procName -ieq "python" -or $procName -ieq "pythonw")) {
        Write-Host "[INFO] Stopping $procName PID $procId to free port $Port..." -ForegroundColor Yellow
        try { Stop-Process -Id $procId -Force -ErrorAction Stop } catch {}
    } else {
        Write-Host "[WARN] Not stopping PID $procId ($procName). If you want to free it manually:" -ForegroundColor Yellow
        Write-Host "       netstat -ano | findstr :$Port" -ForegroundColor Yellow
        Write-Host "       taskkill /PID $procId /F" -ForegroundColor Yellow
    }
}

$pyProcs = Get-Process python, pythonw, python3, python3.10 -ErrorAction SilentlyContinue
if ($pyProcs) {
    Write-Host "Found $($pyProcs.Count) Python process(es). Terminating..." -ForegroundColor Cyan
    $pyProcs | Stop-Process -Force
    Start-Sleep -Seconds 1
    Write-Host "Python processes terminated." -ForegroundColor Green
} else {
    Write-Host "No python/pythonw processes found." -ForegroundColor Green
}


$portsToCheck = @(
    5000, 5001, 5002, 5010,
    5100, 5110, 5111, 5112, 
    5200, 5210, 5211, 5212 
)
foreach ($p in $portsToCheck) {
    Stop-PythonOnPort $p
}

Write-Host ""
Write-Host "Checking remaining listeners on common ports..." -ForegroundColor Cyan
foreach ($p in $portsToCheck) {
    $procId = Get-ListeningProcessId $p
    if ($procId) {
        $procName = Get-ProcessNameSafe $procId
        Write-Host "[WARN] Still listening: port $p -> PID $procId ($procName)" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] Port $p is free." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Yellow
Write-Host "Done!" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Yellow
