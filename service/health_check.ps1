# ================================================================
# Market Bot v26 — Health Check (run every 15 min via Task Scheduler)
#
# Checks:
#   1. Service running? If not, restart it.
#   2. Log file fresh? If stale >45 min, alert.
#   3. Error storm? If 5+ [ERROR] lines in last 20, alert.
#
# Alerts via Telegram (reads credentials from .env)
# ================================================================

$botDir = "C:\Users\mattg\Downloads\Claude"
$logDir = Join-Path $botDir "logs"
$envFile = Join-Path $botDir ".env"
$today = Get-Date -Format "yyyyMMdd"
$logFile = Join-Path $logDir "bot_$today.log"

$healthy = $true
$issues = @()

# --- Check 1: Service running? ---
$svc = Get-Service -Name "MarketBot" -ErrorAction SilentlyContinue
if ($null -eq $svc) {
    $issues += "MarketBot service not installed"
    $healthy = $false
}
elseif ($svc.Status -ne "Running") {
    $issues += "Service is $($svc.Status) -- restarting"
    $healthy = $false
    try {
        Start-Service -Name "MarketBot" -ErrorAction Stop
    }
    catch {
        $issues += "Failed to restart: $_"
    }
}

# --- Check 2: Log file freshness ---
if (Test-Path $logFile) {
    $lastWrite = (Get-Item $logFile).LastWriteTime
    $ageMinutes = [int]((Get-Date) - $lastWrite).TotalMinutes
    if ($ageMinutes -gt 45) {
        $issues += "Log stale: last written $ageMinutes min ago"
        $healthy = $false
    }
}
else {
    # No log today could be normal if bot just started or it's very early
    $hour = (Get-Date).Hour
    if ($hour -ge 10) {
        $issues += "No log file for today (after 10 AM)"
        $healthy = $false
    }
}

# --- Check 3: Error storm ---
if (Test-Path $logFile) {
    $tail = Get-Content $logFile -Tail 20 -ErrorAction SilentlyContinue
    if ($tail) {
        $errorCount = ($tail | Where-Object { $_ -match "\[ERROR\]" }).Count
        if ($errorCount -ge 5) {
            $issues += "High error rate: $errorCount errors in last 20 lines"
            $healthy = $false
        }
    }
}

# --- Send Telegram alert if unhealthy ---
if (-not $healthy) {
    # Read Telegram credentials from .env
    $token = $null
    $chatId = $null
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match "^TELEGRAM_TOKEN=(.+)$") { $token = $Matches[1].Trim() }
            if ($_ -match "^TELEGRAM_CHAT_ID=(.+)$") { $chatId = $Matches[1].Trim() }
        }
    }

    if ($token -and $chatId -and $token.Length -gt 20) {
        $timestamp = Get-Date -Format "HH:mm"
        $msg = "*HEALTH CHECK [$timestamp]*`n" + ($issues -join "`n")
        $body = @{
            chat_id    = $chatId
            text       = $msg
            parse_mode = "Markdown"
        }
        try {
            Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" -Method Post -Body $body -TimeoutSec 10 | Out-Null
        }
        catch {
            # Log locally if Telegram also fails
            $errorMsg = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [HEALTH_CHECK] Alert failed: $_ | Issues: $($issues -join '; ')"
            Add-Content -Path (Join-Path $logDir "health_check.log") -Value $errorMsg
        }
    }
    else {
        # No Telegram credentials -- log locally
        $errorMsg = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [HEALTH_CHECK] $($issues -join '; ')"
        Add-Content -Path (Join-Path $logDir "health_check.log") -Value $errorMsg
    }
}
