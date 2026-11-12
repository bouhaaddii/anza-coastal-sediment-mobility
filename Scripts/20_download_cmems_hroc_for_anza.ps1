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

$OutDir = Join-Path $ProjectRoot 'Data\Copernicus_Marine\Ocean_Colour_HR'
$OutputName = 'Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc'
$OutputPath = Join-Path $OutDir $OutputName
$ProductId = 'OCEANCOLOUR_IBI_BGC_HR_L3_NRT_009_204'
$DatasetId = 'cmems_obs_oc_ibi_bgc_tur-spm-chl_nrt_l3-hr-mosaic_P1D-m'
$StartDate = '2022-11-01T00:00:00'
$EndDate = '2023-03-31T23:59:59'
$Variables = @('SPM', 'TUR')

$CopernicusMarineExe = Join-Path $ProjectRoot '.venv\Scripts\copernicusmarine.exe'
if (-not (Test-Path -LiteralPath $CopernicusMarineExe -PathType Leaf)) {
    throw "Copernicus Marine Toolbox is missing. Install it with: & .\.venv\Scripts\python.exe -m pip install copernicusmarine"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ((Test-Path -LiteralPath $OutputPath -PathType Leaf) -and -not $Overwrite) {
    Write-Host "ALREADY AVAILABLE; skipped: $OutputPath"
    Write-Host 'Use -Overwrite only if you intend to replace this subset.'
    exit 0
}

Write-Host "Product: $ProductId"
Write-Host "Dataset: $DatasetId"
Write-Host "Variables: $($Variables -join ', ')"
Write-Host "Dates: $StartDate to $EndDate"
Write-Host "Approximate project box: $MinLon,$MinLat to $MaxLon,$MaxLat"
Write-Host 'Sentinel-2-derived product at 100 m; comparison product, not independent in-situ validation.'

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
    '--output-directory', $OutDir, '--output-filename', $OutputName
)
foreach ($Variable in $Variables) { $RequestArgs += @('--variable', $Variable) }
if ($Overwrite) { $RequestArgs += '--overwrite' }

Write-Host ''
Write-Host 'Checking the request before downloading...'
& $CopernicusMarineExe @RequestArgs --dry-run
if ($LASTEXITCODE -ne 0) {
    throw 'Dry run failed. Check dataset, dates, variables, login, and area. No download was started.'
}

$Confirmation = Read-Host 'If the planned subset is correct, type DOWNLOAD'
if ($Confirmation -cne 'DOWNLOAD') {
    Write-Host 'Download cancelled; no subset was downloaded.'
    exit 0
}

& $CopernicusMarineExe @RequestArgs
if ($LASTEXITCODE -ne 0) { throw 'Copernicus Marine subset download failed.' }
if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
    throw "Expected output file was not found: $OutputPath"
}

$Manifest = [ordered]@{
    product_id = $ProductId
    dataset_id = $DatasetId
    doi = '10.48670/moi-00107'
    variables_requested = $Variables
    start_datetime_utc = $StartDate
    end_datetime_utc = $EndDate
    bbox_wgs84_approximate = @($MinLon, $MinLat, $MaxLon, $MaxLat)
    nominal_resolution_m = 100
    temporal_product = 'Daily Level-3 Sentinel-2/MSI ocean-colour retrieval'
    limitation = 'Satellite-derived TUR/SPM; not independent local in-situ validation.'
    downloaded_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    file = $OutputName
    file_size_bytes = (Get-Item -LiteralPath $OutputPath).Length
}
$ManifestPath = Join-Path $OutDir 'Anza_CMEMS_HR_OC_download_manifest.json'
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding utf8

Write-Host ''
Write-Host "Downloaded subset: $OutputPath"
Write-Host "Manifest: $ManifestPath"
Write-Host 'Inspect NetCDF dates, variables, flags, fill values and valid pixels before scientific use.'

