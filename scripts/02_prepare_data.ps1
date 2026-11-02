$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli prepare --config (Join-Path $PSScriptRoot "..\configs\base.yaml")

