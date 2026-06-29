$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:JUPYTER_CONFIG_DIR = Join-Path $root ".jupyter"
$env:JUPYTER_DATA_DIR = Join-Path $root ".jupyter-data"
$env:JUPYTER_RUNTIME_DIR = Join-Path $root ".jupyter-runtime"

New-Item -ItemType Directory -Force $env:JUPYTER_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force $env:JUPYTER_DATA_DIR | Out-Null
New-Item -ItemType Directory -Force $env:JUPYTER_RUNTIME_DIR | Out-Null

& ".\.venv\Scripts\python.exe" -m jupyter lab

