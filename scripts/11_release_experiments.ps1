$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "$PWD\src"
# Run in an environment with the project dependencies installed. Data preparation uses configs/base.yaml separately.
foreach ($command in @('validate-models', 'train-ranker', 'train-single-rankers', 'optimize-fusion', 'report', 'test')) {
    if ($command -eq 'test') { python -m gamefeed.cli test }
    else { python -m gamefeed.cli $command --config configs/release.yaml }
    if ($LASTEXITCODE -ne 0) { throw "Failed: $command" }
}
python -m gamefeed.cli analyze-groups --config configs/release.yaml --method selected --split test
if ($LASTEXITCODE -ne 0) { throw 'Failed: analyze-groups' }
