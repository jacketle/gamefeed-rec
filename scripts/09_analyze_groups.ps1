$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli analyze-groups --config (Join-Path $PSScriptRoot "..\configs\base.yaml") --method fusion --split test
