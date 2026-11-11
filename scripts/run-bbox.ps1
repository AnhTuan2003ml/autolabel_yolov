$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m yolo_agent.infer --source images --output runs\bbox --mode bbox --model yolo11x.pt
