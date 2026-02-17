$projectRoot = $PSScriptRoot
$venvActivate = Join-Path $projectRoot "venv\Scripts\Activate.ps1"

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
    if (-not $procId) { return $true }

    $procName = Get-ProcessNameSafe $procId
    Write-Host "[WARN] Port $Port is in use by PID $procId ($procName)" -ForegroundColor Yellow

    if ($procName -and ($procName -ieq "python" -or $procName -ieq "pythonw")) {
        Write-Host "[INFO] Stopping $procName (PID $procId) to free port $Port..." -ForegroundColor Yellow
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Start-Sleep -Milliseconds 500
        } catch {
            Write-Host "[WARN] Could not stop PID $procId. Will pick another port." -ForegroundColor Yellow
            return $false
        }
        return $true
    }

    # Not python/pythonw -> don't kill, just signal "not free"
    return $false
}

function Test-PortFree([int]$Port) {
    return -not (Get-ListeningProcessId $Port)
}

function Get-FreePort([int]$Start, [int]$MaxTries = 2000) {
    for ($p = $Start; $p -lt ($Start + $MaxTries); $p++) {
        if (Test-PortFree $p) { return $p }
    }
    throw "Could not find a free port starting from $Start"
}

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting Federated Learning (Gossip P2P Mode)" -ForegroundColor Cyan
Write-Host "AUTONOMOUS - NO COORDINATOR NEEDED!" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Cleanup any existing Python processes (best-effort)
Write-Host "Cleaning up existing Python processes..." -ForegroundColor Yellow
$existing = Get-Process python, pythonw -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "Cleaned up $($existing.Count) Python process(es)." -ForegroundColor Green
} else {
    Write-Host "No existing Python processes found." -ForegroundColor Green
}
Write-Host ""

# Fresh-start gossip persistence by default to avoid stale restored rounds.
$gossipDbPath = Join-Path $projectRoot "storage\gossip.db"
if (Test-Path $gossipDbPath) {
    Write-Host "[INFO] Removing existing gossip DB for fresh start: $gossipDbPath" -ForegroundColor Yellow
    try {
        Remove-Item -Path $gossipDbPath -Force -ErrorAction Stop
        Write-Host "[OK] Removed old gossip DB." -ForegroundColor Green
    } catch {
        Write-Host "[WARN] Could not remove gossip DB: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
Write-Host ""

$reporterPort = 5200
if (-not (Stop-PythonOnPort $reporterPort)) {
    $reporterPort = Get-FreePort 5200
    Write-Host "[INFO] Using alternative reporter port: $reporterPort" -ForegroundColor Yellow
}

Write-Host "[0/3] Starting optional Reporter on port $reporterPort (monitoring only)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$venvActivate'; cd '$projectRoot'; python fl/run_scripts/run_gossip_reporter.py --port $reporterPort --report-interval 5 --startup-delay 2"
Start-Sleep -Seconds 2

$regions = @("A", "B", "C")
$peerPorts = @()
$nextPreferredPort = 5210

for ($i = 0; $i -lt 3; $i++) {
    $peerName = "peer-$($regions[$i])"
    $region = $regions[$i]
    $seedPort = $null
    if ($peerPorts.Count -gt 0) {
        $seedPort = $peerPorts[$peerPorts.Count - 1]
    }

    $started = $false
    $attempt = 0
    $peerPort = $null

    while (-not $started -and $attempt -lt 5) {
        $attempt += 1
        $peerPort = Get-FreePort $nextPreferredPort
        $nextPreferredPort = $peerPort + 1

        # Best-effort: if a python process grabbed this port right now, free it.
        $null = Stop-PythonOnPort $peerPort
        if (-not (Test-PortFree $peerPort)) {
            Write-Host "[WARN] Port $peerPort became busy before launch. Retrying..." -ForegroundColor Yellow
            continue
        }

        if ($seedPort) {
            Write-Host "[$(1+$i)/3] Starting '$peerName' (region $region) on port $peerPort; seed=$seedPort (attempt $attempt)..." -ForegroundColor Green
        } else {
            Write-Host "[$(1+$i)/3] Starting '$peerName' (region $region) on port $peerPort; no seed for bootstrap (attempt $attempt)..." -ForegroundColor Green
        }

        $cmdActivate = $venvActivate -replace '\\', '\\'
        $cmdRoot = $projectRoot -replace '\\', '\\'

        $peerCmd = "python fl/run_scripts/run_gossip_peer.py '$peerName' $region --port $peerPort --local-epochs 1 --max-rounds 20 --reporter localhost:$reporterPort"
        if ($seedPort) {
            $peerCmd += " --peers localhost:$seedPort"
        }

        $scriptText = @"
. '$cmdActivate'
cd '$cmdRoot'
$peerCmd
"@

        Start-Process powershell -ArgumentList "-NoExit", "-Command", $scriptText
        Start-Sleep -Seconds 4

        $peerPid = Get-ListeningProcessId $peerPort
        if ($peerPid) {
            $peerProcName = Get-ProcessNameSafe $peerPid
            Write-Host "[INFO] Peer '$peerName' now listening on $peerPort (PID $peerPid, $peerProcName)." -ForegroundColor Cyan
            $started = $true
        } else {
            Write-Host "[WARN] Peer '$peerName' did not bind to port $peerPort. Retrying with another port..." -ForegroundColor Yellow
            Start-Sleep -Seconds 1
        }
    }

    if (-not $started) {
        throw "Failed to start peer '$peerName' after multiple attempts."
    }

    $peerPorts += $peerPort
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "All Gossip P2P Peers started AUTONOMOUSLY!" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Reporter (optional): http://localhost:$reporterPort - monitors progress" -ForegroundColor Cyan
Write-Host "Peers (AUTONOMOUS): $($peerPorts[0]) (peer-A), $($peerPorts[1]) (peer-B), $($peerPorts[2]) (peer-C)" -ForegroundColor Cyan
Write-Host ""
