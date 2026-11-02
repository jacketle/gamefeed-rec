$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli demo --config (Join-Path $PSScriptRoot "..\configs\base.yaml") --top-k 10
