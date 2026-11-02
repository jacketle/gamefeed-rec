param([ValidateSet("dev", "test")][string]$Split = "dev")
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
python -m gamefeed.cli popularity --config (Join-Path $PSScriptRoot "..\configs\base.yaml") --split $Split

