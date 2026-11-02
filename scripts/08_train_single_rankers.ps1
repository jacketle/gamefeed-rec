$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli train-single-rankers --config (Join-Path $PSScriptRoot "..\configs\base.yaml")
