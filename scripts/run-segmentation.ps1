$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m yolo_agent.infer --source images --output runs\seg --mode segmentation --model yolo11x-seg.pt
