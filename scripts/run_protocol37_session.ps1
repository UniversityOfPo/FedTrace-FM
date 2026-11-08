# Protocol 37 session runner (Amendment 2: the PI splits the 15 runs into three ~10-hour sessions).
# Same behaviour as run_protocol37.ps1, but preflight and execution cover only the runs given for this session.
#   powershell -ExecutionPolicy Bypass -File scripts/run_protocol37_session.ps1 K10_clean_d123 K10_badnets_d123 K10_drift_d123
param([Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Runs)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
$logDir = "logs_rev\protocol37"
New-Item -ItemType Directory -Force $logDir | Out-Null
$status = "$logDir\runner_status.log"
function Log($msg) { Add-Content -Path $status -Value ("{0:yyyy-MM-ddTHH:mm:ss} {1}" -f (Get-Date), $msg) -Encoding utf8 }

try {
    if (Test-Path "$logDir\STOP") { Remove-Item "$logDir\STOP"; Log "SESSION: removed previous session's STOP file" }
    $tag = ($Runs -join "_")
    Log "SESSION start: $($Runs -join ' ')"
    cmd /c "$py -W ignore -m src.rev.preflight --runs $($Runs -join ' ') --out $logDir\preflight_$tag.json > $logDir\preflight_$tag.console.log 2>&1"
    if ($LASTEXITCODE -ne 0) { Log "PREFLIGHT FAIL (exit $LASTEXITCODE); nothing launched"; exit 1 }
    Log "PREFLIGHT PASS"
    foreach ($r in $Runs) {
        if (Test-Path "$logDir\STOP") { Log "STOP file present; halting before $r"; exit 2 }
        Log "START $r"
        cmd /c "$py -m src.rev.run_rev_experiment --config configs/rev/$r.rev.yaml > $logDir\$r.console.log 2>&1"
        if ($LASTEXITCODE -ne 0) { Log "FAIL $r (exit $LASTEXITCODE); sequence stopped"; exit 1 }
        Log "DONE $r"
    }
    Log "SESSION DONE: $($Runs -join ' ')"
} catch {
    Log "RUNNER ERROR: $($_.Exception.Message)"
    exit 3
}
