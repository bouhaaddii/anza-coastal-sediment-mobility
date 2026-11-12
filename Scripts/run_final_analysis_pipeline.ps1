$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RequiredInputs = @(
    'Data\Local_Bathymetry\Anza_Agadir\Anza_Agadir_bathy_points_candidate_utm29.csv',
    'Data\Copernicus_Marine\IBI_wave_Anza_subset.nc',
    'Data\Copernicus_Marine\Currents_3D\IBI_currents_3D_daily_Anza_0_30m_20221001_20230331.nc',
    'Data\Copernicus_Marine\Ocean_Colour_HR\Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc',
    'Results\Currents_3D_QC\IBI_3D_current_support_audit.json',
    'Results\scene_forcing_and_quality.csv',
    'Results\Sediment_Proxy\Spatial_Support\ndti_spatial_support_by_scene.csv',
    'Results\CMEMS_Ocean_Colour_Audit\cmems_hr_oc_water_support_by_date.csv'
)
$ScriptNames = @(
    '21_audit_cmems_hr_oc.py',
    '22_audit_cmems_water_support.py',
    '23_compare_cmems_with_s2_ndti.py',
    '24_test_bathymetric_confounding.py',
    '25_spatial_block_sensitivity_depth.py',
    '26_within_cell_wave_response.py',
    '27_wave_response_robustness.py',
    '28_wave_lag_response.py',
    '29_redraw_wave_window_figure.py',
    '31_audit_ibi_3d_current_support.py',
    '32_estimate_sediment_mobility_scenarios.py',
    '33_link_mobility_to_hr_oc_spm.py',
    '34_redraw_mobility_hr_oc_figure.py',
    '36_temporal_mobility_persistence.py',
    '37_satellite_scene_sampling_representativeness.py',
    '39_continuous_mobility_sampling_sensitivity.py',
    '40_make_manuscript_workflow_figure.py',
    '38_generate_final_manifest.py'
)

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Project Python environment not found: $PythonExe"
}
foreach ($RelativePath in $RequiredInputs) {
    $Path = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required pre-downloaded input is missing: $Path"
    }
}
foreach ($ScriptName in $ScriptNames) {
    $ScriptPath = Join-Path $PSScriptRoot $ScriptName
    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        throw "Required script is missing: $ScriptPath"
    }
}

Set-Location -LiteralPath $ProjectRoot
Write-Host 'FINAL LOCAL ANALYSIS REFRESH' -ForegroundColor Cyan
Write-Host 'This sequence performs no download. It rebuilds the selected analyses and figures from files already in Data/.'
Write-Host 'Previously completed Sentinel-2 preprocessing and downloads are not repeated.'

$Stage = 0
foreach ($ScriptName in $ScriptNames) {
    $Stage++
    $ScriptPath = Join-Path $PSScriptRoot $ScriptName
    Write-Host "`n===== FINAL STAGE $Stage/$($ScriptNames.Count): $ScriptName =====" -ForegroundColor Cyan
    & $PythonExe $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline stopped at $ScriptName with exit code $LASTEXITCODE. Earlier outputs remain available."
    }
}

Write-Host "`nFinal local analysis refresh complete." -ForegroundColor Green
Write-Host (Join-Path $ProjectRoot 'Results\Final_Pipeline')

