$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli inspect --config (Join-Path $PSScriptRoot "..\configs\base.yaml")

