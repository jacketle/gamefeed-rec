$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli itemcf --config (Join-Path $PSScriptRoot "..\configs\base.yaml") --split dev
