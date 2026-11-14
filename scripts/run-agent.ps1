$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$url = Read-Host 'CVAT job URL'
$organization = Read-Host 'Organization [cohort-5a]'
if ([string]::IsNullOrWhiteSpace($organization)) { $organization = 'cohort-5a' }
$mode = Read-Host 'Mode: bbox, segmentation, tracking or skeleton [bbox]'
if ([string]::IsNullOrWhiteSpace($mode)) { $mode = 'bbox' }
if ($mode -notin @('bbox', 'segmentation', 'tracking', 'skeleton')) { throw 'Mode must be bbox, segmentation, tracking or skeleton' }
$points = Read-Host 'Maximum polygon points [80]'
if ([string]::IsNullOrWhiteSpace($points)) { $points = '80' }
python -m cvat_agent.job_runner $url $organization yolo $mode runs $points
