$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli fusion --config (Join-Path $PSScriptRoot "..\configs\base.yaml") --split dev
