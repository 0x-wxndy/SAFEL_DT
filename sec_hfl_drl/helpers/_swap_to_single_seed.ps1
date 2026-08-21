param(
    [int]$TargetRounds = 300
)
# Watch results/runs/sweep_headline/all__d3qn__seed0000.jsonl until it has
# $TargetRounds lines, then kill all python processes and relaunch the
# sweep with --seeds 0 --skip-existing.

$ErrorActionPreference = "Stop"
$tracePath = "results/runs/sweep_headline/all__d3qn__seed0000.jsonl"

Write-Host "[swap] watching $tracePath until $TargetRounds rounds complete..."
while ($true) {
    if (Test-Path $tracePath) {
        $count = (Get-Content $tracePath | Measure-Object -Line).Lines
        Write-Host "[swap] $((Get-Date).ToString('HH:mm:ss'))  current rounds=$count"
        if ($count -ge $TargetRounds) {
            break
        }
    } else {
        Write-Host "[swap] trace not present yet"
    }
    Start-Sleep -Seconds 90
}

Write-Host "[swap] current cell complete. Killing python..."
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

Write-Host "[swap] relaunching sweep with --seeds 0 --skip-existing"
& .\.venv\Scripts\python.exe scripts/run_sweep.py `
    --dataset nbaiot --mode multi --rounds 300 `
    --clients 30 --fogs 3 `
    --seeds 0 `
    --fog-policies all,random,heuristic,sac `
    --cloud-policies static,round_robin,d3qn `
    --attack mixed `
    --attack-stepwise "0:0.10,100:0.15,200:0.20,250:0.25" `
    --mixed-gamma-range 10 50 `
    --adversary-features `
    --sac-heuristic-hint `
    --encryption plain `
    --skip-existing `
    --out results/runs/sweep_headline
Write-Host "[swap] sweep exited with code $LASTEXITCODE"
