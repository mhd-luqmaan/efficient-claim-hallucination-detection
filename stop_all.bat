@echo off
title NeuralFact - Stop All Running Servers
color 0C

echo ===============================================================================
echo                NEURALFACT: STOPPING ALL RUNNING SERVERS
echo ===============================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ports = @(8501, 8080); " ^
    "$found = $false; " ^
    "foreach ($p in $ports) { " ^
    "    $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; " ^
    "    if ($conns) { " ^
    "        $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique; " ^
    "        foreach ($id in $pids) { " ^
    "            if ($id -gt 0) { " ^
    "                $name = (Get-Process -Id $id -ErrorAction SilentlyContinue).ProcessName; " ^
    "                Write-Host ('[*] Terminating ' + $name + ' (PID: ' + $id + ') on port ' + $p + '...'); " ^
    "                Stop-Process -Id $id -Force -ErrorAction SilentlyContinue; " ^
    "                $found = $true; " ^
    "            } " ^
    "        } " ^
    "    } " ^
    "} " ^
    "if ($found) { " ^
    "    Write-Host ''; " ^
    "    Write-Host '[V] Successfully terminated servers on ports 8501 and 8080.' -ForegroundColor Green; " ^
    "} else { " ^
    "    Write-Host '[i] No servers currently active on ports 8501 or 8080.' -ForegroundColor Yellow; " ^
    "}"

echo.
echo ===============================================================================
pause
