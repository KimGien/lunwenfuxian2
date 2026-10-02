param(
    [ValidateSet('check','prepare','test','train')]
    [string]$Action = 'check',
    [string]$Experiment = 'TC-01-37-fulltext',
    [int]$Seed = 11,
    [int]$Fold = 1,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$pythonPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../S00/.venv/Scripts/python.exe'))
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The prepared S00 Python environment is missing.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    switch ($Action) {
        'check' { & $pythonPath -B -m tridep_ccml.readiness }
        'prepare' { & $pythonPath -B -m tridep_ccml.prepare export }
        'test' { & $pythonPath -B -m unittest discover -s tridep_ccml/tests -v }
        'train' {
            $arguments = @('-B', '-m', 'tridep_ccml.train', '--experiment', $Experiment, '--seed', $Seed, '--fold', $Fold)
            if ($Execute) { $arguments += '--execute' }
            & $pythonPath @arguments
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "TC $Action failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
