$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$InputNetCDF = Join-Path $ProjectRoot 'Data\Copernicus_Marine\Ocean_Colour_HR\Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc'
$ScriptNames = @(
    '21_audit_cmems_hr_oc.py',
    '22_audit_cmems_water_support.py',
    '23_compare_cmems_with_s2_ndti.py',
    '24_test_bathymetric_confounding.py',
    '25_spatial_block_sensitivity_depth.py'
)

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python de l'environnement du projet introuvable : $PythonExe"
}
if (-not (Test-Path -LiteralPath $InputNetCDF -PathType Leaf)) {
    throw "Sous-ensemble HR-OC introuvable : $InputNetCDF"
}
foreach ($ScriptName in $ScriptNames) {
    $ScriptPath = Join-Path $PSScriptRoot $ScriptName
    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        throw "Script requis introuvable : $ScriptPath"
    }
}

Set-Location -LiteralPath $ProjectRoot
Write-Host 'Pipeline HR-OC Anza: steps 21 to 25.'
Write-Host 'No download is performed; scripts read the source data and rewrite analysis outputs.'

foreach ($ScriptName in $ScriptNames) {
    Write-Host "`n===== $ScriptName =====" -ForegroundColor Cyan
    $ScriptPath = Join-Path $PSScriptRoot $ScriptName
    & $PythonExe $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Échec de $ScriptName (code de sortie $LASTEXITCODE). Le pipeline est arrêté."
    }
}

Write-Host "`nPipeline completed. Results: $(Join-Path $ProjectRoot 'Results\CMEMS_Ocean_Colour_Audit')" -ForegroundColor Green
