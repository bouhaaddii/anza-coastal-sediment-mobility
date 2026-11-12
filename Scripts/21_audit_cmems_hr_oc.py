from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Data" / "Copernicus_Marine" / "Ocean_Colour_HR" / "Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc"
SELECTION = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
OUT = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def find_coord(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    names = {name.lower(): name for name in list(ds.coords) + list(ds.dims)}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


def pick_var(ds: xr.Dataset, requested: str) -> str | None:
    for name in ds.data_vars:
        if name.lower() == requested.lower():
            return name
    for name in ds.data_vars:
        if requested.lower() in name.lower():
            return name
    return None


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"CMEMS NetCDF subset not found: {INPUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    print("Auditing the downloaded Copernicus Marine HR-OC subset.")
    print("The file is read only; no resampling, masking, or source-data change is performed.")
    print(f"Input: {INPUT}")
    print(f"File size: {INPUT.stat().st_size:,} bytes")

    with xr.open_dataset(INPUT, decode_cf=True, mask_and_scale=True) as ds:
        time_name = find_coord(ds, ("time", "datetime"))
        lat_name = find_coord(ds, ("latitude", "lat", "y"))
        lon_name = find_coord(ds, ("longitude", "lon", "x"))
        if not time_name:
            raise RuntimeError(f"No recognizable time coordinate. Available coordinates: {list(ds.coords)}")
        if not lat_name or not lon_name:
            raise RuntimeError(f"Could not identify latitude/longitude. Coordinates: {list(ds.coords)}")

        times = pd.to_datetime(np.asarray(ds[time_name].values).reshape(-1), utc=True, errors="coerce")
        if times.isna().any():
            raise RuntimeError("At least one time coordinate could not be decoded as a date.")
        available_days = set(times.strftime("%Y-%m-%d"))

        lat = np.asarray(ds[lat_name].values, dtype=float)
        lon = np.asarray(ds[lon_name].values, dtype=float)
        print(f"Dimensions: {dict(ds.sizes)}")
        print(f"Coordinates: time={time_name}, latitude={lat_name}, longitude={lon_name}")
        print(f"Available daily time steps: {len(times)}")
        print(f"Date range: {times.min().isoformat()} to {times.max().isoformat()}")
        print(f"Longitude range: {np.nanmin(lon):.6f} to {np.nanmax(lon):.6f}")
        print(f"Latitude range: {np.nanmin(lat):.6f} to {np.nanmax(lat):.6f}")
        print(f"Data variables: {list(ds.data_vars)}")

        variable_rows: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        chosen: dict[str, str | None] = {
            "SPM": pick_var(ds, "SPM"),
            "TUR": pick_var(ds, "TUR"),
        }
        if any(name is None for name in chosen.values()):
            missing = [key for key, value in chosen.items() if value is None]
            raise RuntimeError(
                f"Requested variables not found: {missing}. Actual data variables: {list(ds.data_vars)}"
            )

        for target, name in chosen.items():
            da = ds[name]
            attrs = {k: json_safe(v) for k, v in da.attrs.items()}
            variable_rows.append({
                "requested_name": target,
                "netcdf_name": name,
                "dimensions": ",".join(da.dims),
                "units": attrs.get("units", ""),
                "long_name": attrs.get("long_name", ""),
                "standard_name": attrs.get("standard_name", ""),
                "attributes_json": json.dumps(attrs, ensure_ascii=False),
            })

            for index, timestamp in enumerate(times):
                slice_da = da.isel({time_name: index}) if time_name in da.dims else da
                values = np.asarray(slice_da.values, dtype=np.float64)
                finite = np.isfinite(values)
                valid = values[finite]
                row: dict[str, Any] = {
                    "date_utc": timestamp.strftime("%Y-%m-%d"),
                    "variable": target,
                    "netcdf_variable": name,
                    "n_grid_cells": int(values.size),
                    "n_finite": int(valid.size),
                    "finite_fraction": float(valid.size / values.size) if values.size else np.nan,
                    "units": attrs.get("units", ""),
                    "median": float(np.nanmedian(valid)) if valid.size else np.nan,
                    "p05": float(np.nanpercentile(valid, 5)) if valid.size else np.nan,
                    "p95": float(np.nanpercentile(valid, 95)) if valid.size else np.nan,
                    "minimum": float(np.nanmin(valid)) if valid.size else np.nan,
                    "maximum": float(np.nanmax(valid)) if valid.size else np.nan,
                }
                daily_rows.append(row)

        flag_names = [
            name for name in ds.data_vars
            if any(token in name.lower() for token in ("flag", "quality", "mask", "wqsf", "l2_flags"))
        ]
        flag_rows = []
        for name in flag_names:
            flag_rows.append({
                "variable": name,
                "dimensions": ",".join(ds[name].dims),
                "attributes_json": json.dumps(
                    {k: json_safe(v) for k, v in ds[name].attrs.items()},
                    ensure_ascii=False,
                ),
            })

        global_attrs = {k: json_safe(v) for k, v in ds.attrs.items()}
        metadata = {
            "input_file": str(INPUT),
            "file_size_bytes": INPUT.stat().st_size,
            "dataset_dimensions": dict(ds.sizes),
            "coordinate_names": {"time": time_name, "latitude": lat_name, "longitude": lon_name},
            "time_range_utc": [times.min().isoformat(), times.max().isoformat()],
            "time_steps": len(times),
            "longitude_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
            "latitude_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
            "data_variables": list(ds.data_vars),
            "requested_variables": chosen,
            "quality_flag_variables_detected": flag_names,
            "quality_interpretation": (
                "The downloaded subset requested only SPM and TUR and contains no separate per-pixel quality variable. "
                "This does not establish that the HR-OC processing chain applied no quality screening; consult the official QUID."
            ),
            "global_attributes": global_attrs,
            "interpretation_limit": (
                "CMEMS SPM/TUR are satellite-derived products from Sentinel-2/MSI; "
                "they are not local in-situ observations or independent validation."
            ),
            "spatial_note": (
                "The downloaded rectangular subset may include land and invalid water pixels. "
                "Finite-data fractions are not water-only coverage fractions."
            ),
        }

    pd.DataFrame(variable_rows).to_csv(OUT / "cmems_hr_oc_variable_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_rows).to_csv(OUT / "cmems_hr_oc_daily_validity_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flag_rows, columns=["variable", "dimensions", "attributes_json"]).to_csv(
        OUT / "cmems_hr_oc_quality_flag_inventory.csv", index=False, encoding="utf-8-sig"
    )

    scene_matches: list[dict[str, Any]] = []
    if SELECTION.is_file():
        scenes = pd.read_csv(SELECTION, dtype=str, keep_default_na=False)
        selected = scenes[scenes["download_selected"].str.upper().eq("YES")]
        for _, row in selected.iterrows():
            acquisition = pd.to_datetime(row["acquisition_utc"], utc=True, errors="coerce")
            day = acquisition.strftime("%Y-%m-%d") if not pd.isna(acquisition) else ""
            scene_matches.append({
                "product_name": row.get("product_name", ""),
                "acquisition_utc": row.get("acquisition_utc", ""),
                "acquisition_day_utc": day,
                "cmems_daily_product_available": day in available_days,
                "note": "Same calendar day does not prove the CMEMS composite used this exact Sentinel-2 granule.",
            })
    pd.DataFrame(scene_matches).to_csv(
        OUT / "cmems_hr_oc_selected_scene_date_match.csv", index=False, encoding="utf-8-sig"
    )

    (OUT / "cmems_hr_oc_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    notes = [
        "CMEMS HR-OC subset audit",
        f"Variables requested and found: SPM={chosen['SPM']}; TUR={chosen['TUR']}",
        f"Available daily time steps: {len(times)}",
        f"Separate quality-like variables returned in this SPM/TUR subset: {flag_names if flag_names else 'none detected by name'}",
        "The requested subset contains SPM and TUR only. No standalone per-pixel quality field was returned; this does not mean the processor used no quality screening.",
        "Consult the official HR-OC QUID for processor screening, masking, known limitations, and variable-specific interpretation.",
        "Finite fractions cover the rectangular subset, which includes land; they are not water-only coverage.",
        "CMEMS SPM/TUR are Sentinel-2-derived products, not independent local field validation.",
        "Exact date equality does not establish that the CMEMS daily product used the same granule as the local L2A image.",
        "No resampling, calibration, model fitting, datum conversion, or modification of source data was performed.",
    ]
    (OUT / "cmems_hr_oc_audit_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"SPM variable: {chosen['SPM']}; TUR variable: {chosen['TUR']}")
    print(f"Separate quality-like variables returned in this SPM/TUR subset: {flag_names if flag_names else 'none detected by name'}")
    print(f"Audit outputs: {OUT}")
    print("No standalone per-pixel quality field was returned; consult the official QUID and audit water-only coverage before comparison.")


if __name__ == "__main__":
    main()

