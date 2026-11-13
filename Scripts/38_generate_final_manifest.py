"""Create a provenance inventory for the completed Anza analysis package.

The script reads file metadata and software versions only. It does not alter
input data or recalculate scientific results. Files larger than the configured
hash limit are inventoried by path and size without a content hash.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Final_Pipeline"
HASH_LIMIT_BYTES = 16 * 1024 * 1024
PACKAGE_NAMES = (
    "numpy", "pandas", "matplotlib", "rasterio", "xarray", "pyTMD",
    "scikit-learn", "pyproj", "copernicusmarine", "scipy", "netCDF4",
)

KEY_INPUTS = (
    "Config/project_config.json",
    "requirements.txt",
    "Data/Local_Bathymetry/Anza_Agadir/Anza_Agadir_bathy_points.txt",
    "Data/Local_Bathymetry/Anza_Agadir/Anza_Agadir_bathy_points_candidate_utm29.csv",
    "Data/Copernicus_Marine/IBI_wave_Anza_subset.nc",
    "Data/Copernicus_Marine/Currents_3D/IBI_currents_3D_daily_Anza_0_30m_20221001_20230331.nc",
    "Data/Copernicus_Marine/Ocean_Colour_HR/Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc",
    "Results/Currents_3D_QC/IBI_3D_current_support_audit.json",
    "Results/scene_forcing_and_quality.csv",
    "Results/Sediment_Proxy/Spatial_Support/ndti_spatial_support_by_scene.csv",
    "Results/CMEMS_Ocean_Colour_Audit/cmems_hr_oc_water_support_by_date.csv",
    "Results/Sediment_Mobility_Scenarios/mobility_daily_by_depth_grain_datum_scenario.csv.gz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, category: str) -> dict[str, object]:
    size = path.stat().st_size
    digest = sha256_file(path) if size <= HASH_LIMIT_BYTES else ""
    status = "sha256_recorded" if digest else "large_file_hash_skipped"
    return {
        "category": category,
        "path_relative": path.relative_to(ROOT).as_posix(),
        "size_bytes": size,
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "sha256": digest,
        "hash_status": status,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> None:
    config_path = ROOT / "Config" / "project_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    missing: list[str] = []
    seen: set[Path] = set()

    for relative in KEY_INPUTS:
        path = ROOT / Path(relative)
        if not path.is_file():
            missing.append(relative)
            continue
        records.append(file_record(path, "key_input"))
        seen.add(path.resolve())

    for path in sorted((ROOT / "Scripts").glob("*.py")) + sorted((ROOT / "Scripts").glob("*.ps1")):
        if path.resolve() not in seen:
            records.append(file_record(path, "analysis_code"))
            seen.add(path.resolve())

    for folder_name in ("Config", "Data", "Results"):
        folder = ROOT / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or OUT in path.parents or path.resolve() in seen:
                continue
            records.append(file_record(path, "project_result_or_metadata"))
            seen.add(path.resolve())

    inventory_path = OUT / "final_file_inventory.csv"
    pd.DataFrame(records).to_csv(inventory_path, index=False, encoding="utf-8-sig")
    manifest = {
        "project": config.get("project_name"),
        "site": config.get("site_name"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": package_versions(),
        "key_inputs": list(KEY_INPUTS),
        "required_key_inputs_missing": missing,
        "hash_policy": f"SHA-256 recorded for files up to {HASH_LIMIT_BYTES} bytes; larger files are listed with size and modification time only.",
        "inventory_csv": inventory_path.relative_to(ROOT).as_posix(),
        "scientific_scope": "Anza open-data optical-product comparison and model-conditional bed-mobility screening; not an in-situ SSC validation, sediment-flux model, or validated morphodynamic simulation.",
        "known_data_limits": [
            "The local soundings retain the Port of Agadir hydrographic zero; the vertical tie to external bathymetry is unverified.",
            "D50 values are scenarios; no grain-size measurements from the 2022 survey are present.",
            "Copernicus IBI currents are regional daily means and do not resolve the surf zone.",
            "The HR-OC product is satellite-derived and is not independent in-situ validation.",
            "Product cloud cover is tile-level; it is not a water-only cloud fraction for the study polygon.",
            "The 11 processed scene subset is small and temporally uneven; month composition explains most of its difference from the full period in the model-conditional mobility indicator.",
        ],
    }
    manifest_path = OUT / "final_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing:
        print("Manifest created, but required key inputs are missing:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(2)
    print("Final provenance manifest created; no input or analysis outputs were modified.")
    print(f"File inventory: {inventory_path}")
    print(f"Run manifest: {manifest_path}")
    print(f"Inventory entries: {len(records)}")


if __name__ == "__main__":
    main()

