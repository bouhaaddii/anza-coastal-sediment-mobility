# Anza Coastal Sediment Mobility Study

Research materials for the Anza / Agadir coastal sediment-mobility screening study in Morocco. This repository provides the scripts, selected input data, and saved outputs needed to inspect and reproduce the analyses described in the project.

## Scope and interpretation

This is a site-specific screening and association study for October 2022–March 2023. It combines Sentinel-2 observations, regional Copernicus Marine forcing, public bathymetry products, a local hydrographic survey, and scenario-based sediment-mobility calculations.

The mobility maps are conditional scenarios, not measurements of sediment flux, seabed change, or suspended-sediment concentration. The median grain size (D50) is not measured locally and is varied as a sensitivity input. Regional wave/current products do not resolve surf-zone hydrodynamics. Copernicus Marine high-resolution ocean-colour fields are Sentinel-derived and are not independent in-situ validation. The vertical reference of local soundings is the site-local hydrographic zero of the Port of Agadir; no absolute vertical datum correction is applied. The two-dimensional horizontal transformation reports 7 m nominal operation accuracy, not a local control-point residual.

## Repository contents

- `Scripts/`: acquisition, preprocessing, quality checks, analyses, and figure-generation code used in this study. The manuscript and editorial document-generation scripts are excluded.
- `Config/project_config.json`: portable project configuration and survey-reference metadata.
- `Data/Local_Bathymetry/Anza_Agadir/`: local survey point files used by the analysis. The original DWG plan is not included.
- `Data/Sentinel2/Processed/`: the 11 selected 2022–2023 scenes as clipped reflectance and QA rasters. The unrelated 2015 test scene, full SAFE archives, and original product ZIPs are not included.
- `Data/Copernicus_Marine/`, `Data/Open_Bathymetry/`, `Data/Tide_Model/GOT5.6/`, and `Data/External_Sediment/`: model/product inputs and source metadata used for context and sensitivity analyses.
- `Results/`: saved tables, intermediate analysis products, maps, figures, captions, and run metadata. Editorial drafts and author correspondence are excluded.
- `References/references.bib`: bibliographic entries used by the project.
- `SHA256SUMS.csv`: SHA-256 checksums for repository files, excluding Git internals and the checksum file itself.
- `RIGHTS_AND_ATTRIBUTION.md`: source acknowledgments and reuse notes.

The original Sentinel-2 SAFE products and duplicate product ZIP archives would add roughly 20 GB. The derived inputs actually consumed by the analysis are included; the scene catalogue and selection files identify the products. Users who need to rebuild preprocessing from raw products can use `Scripts/02_download_sentinel2_selected.py` with their own Copernicus Data Space credentials, then run `Scripts/04_process_sentinel2.py`. Credentials are requested interactively and must never be committed.

## Reproduce the saved analysis

On Windows with PowerShell, from the repository root:

```powershell
.\Install_Environment.ps1
.\Scripts\run_final_analysis_pipeline.ps1
```

The pipeline rebuilds the selected audits, analyses, tables, and figures from the local files already present in `Data/`; it does not download data. It stops if a required input is missing and preserves prior outputs if a stage fails. For targeted analyses, run the corresponding numbered Python script from the repository root, for example:

```powershell
.\.venv\Scripts\python.exe .\Scripts\32_estimate_sediment_mobility_scenarios.py
```

The download scripts are optional and only needed to refresh source products. Copernicus Marine downloads require a separately configured account. Do not place passwords, access tokens, or `.env` files in this repository.

## Data handling

Local survey coordinates are supplied in the original point file and in a transformed candidate-coordinate file. The scripts and metadata distinguish original from transformed coordinates. No vertical shift is applied. Consult `Config/project_config.json`, the data-source manifests, and the audit outputs before interpreting elevations or combining products.

Use the saved out-of-sample or sensitivity outputs for the scope stated by each analysis. Do not interpret between-product agreement as independent validation, or scenario mobility frequency as observed transport.