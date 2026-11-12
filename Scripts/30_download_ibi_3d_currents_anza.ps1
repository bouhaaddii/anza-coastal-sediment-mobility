param([switch]$Overwrite)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $ProjectRoot 'Config\project_config.json'
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$Bbox = $Config.site_bbox_wgs84
$MinLon = [double]$Bbox[0]
$MinLat = [double]$Bbox[1]
$MaxLon = [double]$Bbox[2]
$MaxLat = [double]$Bbox[3]
$StartDate = '2022-10-01T00:00:00'
$EndDate = '2023-03-31T23:00:00'
$CurrentConfig = $Config.copernicus_marine.current_3d
$DatasetId = [string]$CurrentConfig.dataset_id
$ProductId = [string]$CurrentConfig.product_id
$Variables = @($CurrentConfig.variables | ForEach-Object { [string]$_ })
$MinimumDepth = [double]$CurrentConfig.minimum_depth_m
$MaximumDepth = [double]$CurrentConfig.maximum_depth_m
$OutputDir = Join-Path $ProjectRoot 'Data\Copernicus_Marine\Currents_3D'
$OutputName = 'IBI_currents_3D_daily_Anza_0_30m_20221001_20230331.nc'
$OutputPath = Join-Path $OutputDir $OutputName
$CopernicusMarineExe = Join-Path $ProjectRoot '.venv\Scripts\copernicusmarine.exe'

if (-not (Test-Path -LiteralPath $CopernicusMarineExe -PathType Leaf)) {
    throw "Copernicus Marine Toolbox is not installed in this project environment. Install it with: & .\.venv\Scripts\python.exe -m pip install copernicusmarine"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if ((Test-Path -LiteralPath $OutputPath -PathType Leaf) -and -not $Overwrite) {
    Write-Host "ALREADY AVAILABLE; skipped: $OutputPath"
    Write-Host 'Use -Overwrite only if you intend to replace the existing subset.'
    exit 0
}

Write-Host "Product: $ProductId"
Write-Host "Dataset: $DatasetId"
Write-Host "Variables: $($Variables -join ', ') (eastward and northward current velocity)"
Write-Host "Dates (UTC): $StartDate to $EndDate"
Write-Host "Depth request: $MinimumDepth to $MaximumDepth m; approximate box: $MinLon,$MinLat to $MaxLon,$MaxLat"
Write-Host 'This is a regional daily 3D reanalysis (~0.028 degrees), not a surf-zone-resolving current model.'
Write-Host 'The existing surface-current file is not overwritten.'

Write-Host 'Checking Copernicus Marine login status...'
& $CopernicusMarineExe login --check-credentials-valid
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Log in interactively. Credentials are not stored by this script.'
    & $CopernicusMarineExe login
    if ($LASTEXITCODE -ne 0) { throw 'Copernicus Marine login failed.' }
}

$RequestArgs = @(
    'subset', '--dataset-id', $DatasetId,
    '--start-datetime', $StartDate, '--end-datetime', $EndDate,
    '--minimum-longitude', "$MinLon", '--maximum-longitude', "$MaxLon",
    '--minimum-latitude', "$MinLat", '--maximum-latitude', "$MaxLat",
    '--minimum-depth', "$MinimumDepth", '--maximum-depth', "$MaximumDepth",
    '--variable', $Variables[0], '--variable', $Variables[1],
    '--output-directory', $OutputDir, '--output-filename', $OutputName
)
if ($Overwrite) { $RequestArgs += '--overwrite' }

Write-Host ''
Write-Host 'Dry run: check the selected depth levels, dates, variables, and transfer size.'
& $CopernicusMarineExe @RequestArgs --dry-run
if ($LASTEXITCODE -ne 0) {
    throw 'Dry run failed. No download was started. Check dataset access and the requested depth range.'
}
if ((Read-Host 'If this subset is correct, type DOWNLOAD') -cne 'DOWNLOAD') {
    Write-Host 'Cancelled; no file was downloaded.'
    exit 0
}

& $CopernicusMarineExe @RequestArgs
if ($LASTEXITCODE -ne 0) { throw 'Copernicus Marine download failed.' }
if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
    throw "Expected output was not found: $OutputPath"
}

$Manifest = [ordered]@{
    product_id = $ProductId
    dataset_id = $DatasetId
    doi = [string]$CurrentConfig.doi
    variables = $Variables
    start_datetime_utc = $StartDate
    end_datetime_utc = $EndDate
    requested_depth_m = @($MinimumDepth, $MaximumDepth)
    bbox_wgs84_approximate = @($MinLon, $MinLat, $MaxLon, $MaxLat)
    temporal_resolution = 'daily mean'
    nominal_horizontal_resolution_degrees = 0.028
    limitation = 'Regional reanalysis; does not resolve surf-zone and rip currents. Current profiles must be audited against model bathymetry and local survey depths before any bed-stress calculation.'
    downloaded_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    file = $OutputName
    file_size_bytes = (Get-Item -LiteralPath $OutputPath).Length
}
$ManifestPath = Join-Path $OutputDir 'IBI_currents_3D_daily_Anza_download_manifest.json'
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
Write-Host "`nDownloaded: $OutputPath"
Write-Host "Manifest: $ManifestPath"
Write-Host 'Next: inspect the vertical levels and nearest wet-cell/model-depth support before calculating combined wave-current stress.'
