$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$url = Read-Host 'CVAT job URL'
$organization = Read-Host 'Organization [cohort-5a]'
if ([string]::IsNullOrWhiteSpace($organization)) { $organization = 'cohort-5a' }
$mode = Read-Host 'Mode: bbox or segmentation [bbox]'
if ([string]::IsNullOrWhiteSpace($mode)) { $mode = 'bbox' }
if ($mode -notin @('bbox', 'segmentation')) { throw 'Mode must be bbox or segmentation' }
$points = Read-Host 'Maximum polygon points [80]'
if ([string]::IsNullOrWhiteSpace($points)) { $points = '80' }
python -m cvat_agent.job_runner $url $organization yolo $mode runs $points
