param([switch]$Overwrite)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $ProjectRoot 'Config\project_config.json'
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$Marine = $Config.copernicus_marine
$Bbox = $Config.site_bbox_wgs84
$MinLon = [double]$Bbox[0]
$MinLat = [double]$Bbox[1]
$MaxLon = [double]$Bbox[2]
$MaxLat = [double]$Bbox[3]
$OutDir = Join-Path $ProjectRoot 'Data\Copernicus_Marine'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$CopernicusMarineExe = Join-Path $ProjectRoot '.venv\Scripts\copernicusmarine.exe'
if (-not (Test-Path -LiteralPath $CopernicusMarineExe)) {
    throw "Copernicus Marine Toolbox not found in the project's .venv. Run: & .\.venv\Scripts\python.exe -m pip install copernicusmarine"
}

Write-Host 'Checking Copernicus Marine login status...'
& $CopernicusMarineExe login --check-credentials-valid
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Log in using the interactive prompt. The script does not store credentials.'
    & $CopernicusMarineExe login
    if ($LASTEXITCODE -ne 0) { throw 'Copernicus Marine login failed.' }
}

function Invoke-CmemsSubset {
    param(
        [string]$DatasetId,
        [string[]]$Variables,
        [string]$OutputName,
        [switch]$UseDepth
    )
    $OutputPath = Join-Path $OutDir $OutputName
    if ((Test-Path -LiteralPath $OutputPath) -and -not $Overwrite) {
        Write-Host "ALREADY AVAILABLE; skipped: $OutputPath"
        return
    }
    $ArgsForRequest = @(
        'subset', '--dataset-id', $DatasetId,
        '--start-datetime', [string]$Marine.start_datetime,
        '--end-datetime', [string]$Marine.end_datetime,
        '--minimum-longitude', "$MinLon", '--maximum-longitude', "$MaxLon",
        '--minimum-latitude', "$MinLat", '--maximum-latitude', "$MaxLat",
        '--output-directory', $OutDir, '--output-filename', $OutputName
    )
    foreach ($Variable in $Variables) { $ArgsForRequest += @('--variable', $Variable) }
    if ($UseDepth) {
        $ArgsForRequest += @('--minimum-depth', [string]$Marine.minimum_depth_m,
                             '--maximum-depth', [string]$Marine.maximum_depth_m)
    }
    if ($Overwrite) { $ArgsForRequest += '--overwrite' }
    Write-Host "`nDry run: $DatasetId -> $OutputName"
    & $CopernicusMarineExe @ArgsForRequest --dry-run
    if ($LASTEXITCODE -ne 0) { throw "Dry run failed for $DatasetId. Review the dataset id, dates, variables, and account." }
    if ((Read-Host 'If the planned subset and size are acceptable, type DOWNLOAD') -cne 'DOWNLOAD') {
        Write-Host "Skipped by user: $OutputName"
        return
    }
    & $CopernicusMarineExe @ArgsForRequest
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $DatasetId." }
    Write-Host "Saved subset: $OutputPath"
}

Write-Host "Study box (approximate, from existing EMODnet extract): $MinLon,$MinLat to $MaxLon,$MaxLat"
Write-Host "Time range: $($Marine.start_datetime) to $($Marine.end_datetime)"
Write-Host 'The IBI grid is regional/coarse; its currents are boundary context and do not resolve surf-zone currents.'
Write-Host 'The existing local bathymetry and its port hydrographic zero are not converted by this download.'

Invoke-CmemsSubset -DatasetId $Marine.wave_dataset_id -Variables @($Marine.wave_variables) `
    -OutputName 'IBI_wave_Anza_subset.nc'
Invoke-CmemsSubset -DatasetId $Marine.current_dataset_id -Variables @($Marine.current_variables) `
    -OutputName 'IBI_currents_surface_Anza_subset.nc' -UseDepth

Write-Host "`nCopernicus Marine download step finished. Inspect files in $OutDir before running later processing."
