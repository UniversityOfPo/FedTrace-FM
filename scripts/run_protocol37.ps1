# Protocol 37 sequential runner (paper_plan/37_extension_protocol.md section 2).
# Runs the 15 extension runs in the frozen order; each starts only after the previous one exits 0.
# Stops on the first failure, or before the next run if logs_rev/protocol37/STOP exists.
#   powershell -ExecutionPolicy Bypass -File scripts/run_protocol37.ps1
# Native commands are launched through cmd /c so that Python warnings on stderr are plain log text; in Windows
# PowerShell 5.1 a redirected native stderr line becomes an error record (Amendment 1 of protocol 37).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
$logDir = "logs_rev\protocol37"
New-Item -ItemType Directory -Force $logDir | Out-Null
$status = "$logDir\runner_status.log"

$runs = @(
    "W3_clean", "W3_badnets", "W3_drift",
    "W4_clean", "W4_badnets", "W4_drift",
    "W5_clean", "W5_badnets", "W5_drift",
    "K10_clean_d123", "K10_badnets_d123", "K10_drift_d123",
    "K10_clean_d2026", "K10_badnets_d2026", "K10_drift_d2026"
)

function Log($msg) { Add-Content -Path $status -Value ("{0:yyyy-MM-ddTHH:mm:ss} {1}" -f (Get-Date), $msg) -Encoding utf8 }

try {
    Log "PREFLIGHT start"
    cmd /c "$py -W ignore -m src.rev.preflight --runs $($runs -join ' ') --out $logDir\preflight.json > $logDir\preflight.console.log 2>&1"
    if ($LASTEXITCODE -ne 0) { Log "PREFLIGHT FAIL (exit $LASTEXITCODE); nothing launched"; exit 1 }
    Log "PREFLIGHT PASS"

    foreach ($r in $runs) {
        if (Test-Path "$logDir\STOP") { Log "STOP file present; halting before $r"; exit 2 }
        Log "START $r"
        cmd /c "$py -m src.rev.run_rev_experiment --config configs/rev/$r.rev.yaml > $logDir\$r.console.log 2>&1"
        if ($LASTEXITCODE -ne 0) { Log "FAIL $r (exit $LASTEXITCODE); sequence stopped"; exit 1 }
        Log "DONE $r"
    }
    Log "ALL 15 RUNS DONE"
} catch {
    Log "RUNNER ERROR: $($_.Exception.Message)"
    exit 3
}
