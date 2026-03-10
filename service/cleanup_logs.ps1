# ================================================================
# Market Bot v26 — Log Cleanup (run weekly via Task Scheduler)
#
# Retention policy:
#   - Bot logs (bot_*.log): keep 30 days
#   - NSSM service logs (service_std*.log*): keep 7 days
#   - Health check logs (health_check.log): keep 30 days
# ================================================================

$logDir = "C:\Users\mattg\Downloads\Claude\logs"

if (-not (Test-Path $logDir)) {
    Write-Host "Log directory not found: $logDir"
    exit
}

$now = Get-Date

# Bot daily logs: keep 30 days
$botLogs = Get-ChildItem $logDir -Filter "bot_*.log" -ErrorAction SilentlyContinue
$removed = 0
foreach ($log in $botLogs) {
    if ($log.LastWriteTime -lt $now.AddDays(-30)) {
        Remove-Item $log.FullName -Force
        $removed++
    }
}
if ($removed -gt 0) {
    Write-Host "Removed $removed bot log(s) older than 30 days"
}

# NSSM service logs: keep 7 days
$svcLogs = Get-ChildItem $logDir -Filter "service_std*.log*" -ErrorAction SilentlyContinue
$removed = 0
foreach ($log in $svcLogs) {
    if ($log.LastWriteTime -lt $now.AddDays(-7)) {
        Remove-Item $log.FullName -Force
        $removed++
    }
}
if ($removed -gt 0) {
    Write-Host "Removed $removed service log(s) older than 7 days"
}

# Health check log: truncate if over 30 days old
$hcLog = Join-Path $logDir "health_check.log"
if (Test-Path $hcLog) {
    $hcAge = ($now - (Get-Item $hcLog).LastWriteTime).TotalDays
    $hcSize = (Get-Item $hcLog).Length / 1MB
    if ($hcAge -gt 30 -or $hcSize -gt 5) {
        # Keep last 100 lines, remove the rest
        $tail = Get-Content $hcLog -Tail 100
        $tail | Set-Content $hcLog -Force
        Write-Host "Trimmed health_check.log to last 100 lines"
    }
}

Write-Host "Log cleanup complete."
