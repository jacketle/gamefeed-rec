$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli train-ranker --config (Join-Path $PSScriptRoot "..\configs\base.yaml")

