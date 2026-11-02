$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "$PWD\src"
python -m gamefeed.cli serve --config configs/release.yaml --port 8765
