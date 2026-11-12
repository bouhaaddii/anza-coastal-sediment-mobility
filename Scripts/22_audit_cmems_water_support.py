from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject


ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / "Data" / "Copernicus_Marine" / "Ocean_Colour_HR" / "Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc"
WATER_CLASS_PATH = ROOT / "Results" / "Marine_Mask" / "S2_water_stability_class_10m.tif"
SCENE_PATH = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
OUT = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit"
WATER_THRESHOLDS = (0.25, 0.50, 0.75)
PRIMARY_THRESHOLD = 0.50


def find_coord(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    names = {name.lower(): name for name in list(ds.coords) + list(ds.dims)}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


def regular_step(values: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"The {name} coordinate must contain at least two finite values.")
    diffs = np.diff(values)
    step = float(np.median(diffs))
    if step == 0 or not np.allclose(diffs, step, rtol=1e-5, atol=1e-10):
        raise ValueError(f"The {name} grid is not regularly spaced; refusing approximate resampling.")
    return abs(step)


def fraction_on_product_grid(
    source_values: np.ndarray,
    src_transform: Affine,
    src_crs: str,
    dst_transform: Affine,
    dst_crs: str,
    shape: tuple[int, int],
) -> np.ndarray:
    dest = np.zeros(shape, dtype=np.float32)
    reproject(
        source=source_values.astype(np.float32, copy=False),
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.average,
        init_dest_nodata=False,
    )
    return np.clip(dest, 0.0, 1.0)


def read_selected_scenes() -> pd.DataFrame:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Sentinel-2 scene selection file not found: {SCENE_PATH}")
    table = pd.read_csv(SCENE_PATH, dtype=str, keep_default_na=False)
    required = {"download_selected", "product_name", "acquisition_utc"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns in scene selection file: {sorted(missing)}")
    selected = table[table["download_selected"].str.upper().eq("YES")].copy()
    selected["date_utc"] = pd.to_datetime(selected["acquisition_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if selected["date_utc"].isna().any():
        raise ValueError("At least one selected scene has an unreadable acquisition date.")
    return selected


def main() -> None:
    for path in (NC_PATH, WATER_CLASS_PATH, SCENE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")
    OUT.mkdir(parents=True, exist_ok=True)

    print("Auditing CMEMS HR-OC coverage over the multi-scene Sentinel-2 stable-water core.")
    print("The NetCDF, Sentinel-2 water-class raster, and survey are read only.")
    print("Water-cell thresholds 0.25, 0.50, and 0.75 are reported as a boundary sensitivity check.")

    with xr.open_dataset(NC_PATH, decode_cf=True, mask_and_scale=True) as ds:
        t_name = find_coord(ds, ("time", "datetime"))
        y_name = find_coord(ds, ("latitude", "lat"))
        x_name = find_coord(ds, ("longitude", "lon"))
        if not t_name or not y_name or not x_name:
            raise ValueError(f"Could not identify time/latitude/longitude coordinates: {list(ds.coords)}")
        if "SPM" not in ds.data_vars or "TUR" not in ds.data_vars:
            raise ValueError(f"Expected SPM and TUR variables. Found: {list(ds.data_vars)}")

        # Standardize to north-up rows and west-to-east columns before raster alignment.
        ds = ds.sortby(y_name, ascending=False).sortby(x_name, ascending=True)
        lats = np.asarray(ds[y_name].values, dtype=np.float64)
        lons = np.asarray(ds[x_name].values, dtype=np.float64)
        dx = regular_step(lons, "longitude")
        dy = regular_step(lats, "latitude")
        lat_step = float(np.median(np.diff(lats)))
        if lat_step >= 0:
            raise ValueError("Latitude ordering did not become north-to-south after sorting.")
        dst_transform = Affine(dx, 0.0, float(lons.min() - dx / 2), 0.0, -dy, float(lats.max() + dy / 2))
        dst_crs = "EPSG:4326"
        target_shape = (lats.size, lons.size)
        times = pd.to_datetime(np.asarray(ds[t_name].values).reshape(-1), utc=True, errors="coerce")
        if times.isna().any():
            raise ValueError("The NetCDF contains an invalid time coordinate.")

        with rasterio.open(WATER_CLASS_PATH) as mask_ds:
            if mask_ds.crs is None:
                raise ValueError("The Sentinel-2 water-class raster has no CRS.")
            classes = mask_ds.read(1, masked=True)
            values = np.asarray(classes.filled(0), dtype=np.uint8)
            class_fractions: dict[int, np.ndarray] = {}
            for code in (1, 2, 3):
                source_binary = (values == code).astype(np.float32)
                class_fractions[code] = fraction_on_product_grid(
                    source_binary,
                    mask_ds.transform,
                    mask_ds.crs,
                    dst_transform,
                    dst_crs,
                    target_shape,
                )

        selected_scenes = read_selected_scenes()
        selected_days = set(selected_scenes["date_utc"].dropna())
        output_rows: list[dict[str, object]] = []
        primary_rows: list[dict[str, object]] = []

        for ti, timestamp in enumerate(times):
            date = timestamp.strftime("%Y-%m-%d")
            for threshold in WATER_THRESHOLDS:
                water_cells = class_fractions[3] >= threshold
                n_water = int(water_cells.sum())
                for variable in ("SPM", "TUR"):
                    da = ds[variable]
                    array = np.asarray(da.isel({t_name: ti}).transpose(y_name, x_name).values, dtype=np.float64)
                    attrs = dict(da.attrs)
                    valid = np.isfinite(array)
                    valid_min = attrs.get("valid_min")
                    valid_max = attrs.get("valid_max")
                    if valid_min is not None:
                        valid &= array >= float(valid_min)
                    if valid_max is not None:
                        valid &= array <= float(valid_max)
                    valid_water = valid & water_cells
                    sample = array[valid_water]
                    count = int(sample.size)
                    row: dict[str, object] = {
                        "date_utc": date,
                        "is_selected_s2_acquisition_day": date in selected_days,
                        "variable": variable,
                        "units": attrs.get("units", ""),
                        "stable_water_fraction_threshold": threshold,
                        "stable_water_cmems_cells": n_water,
                        "valid_water_cmems_cells": count,
                        "valid_fraction_within_water_cells": count / n_water if n_water else np.nan,
                        "rectangular_grid_cells": int(array.size),
                        "valid_min_applied": valid_min if valid_min is not None else "",
                        "valid_max_applied": valid_max if valid_max is not None else "",
                        "median": float(np.median(sample)) if count else np.nan,
                        "p25": float(np.percentile(sample, 25)) if count else np.nan,
                        "p75": float(np.percentile(sample, 75)) if count else np.nan,
                        "p95": float(np.percentile(sample, 95)) if count else np.nan,
                        "maximum": float(np.max(sample)) if count else np.nan,
                    }
                    output_rows.append(row)
                    if threshold == PRIMARY_THRESHOLD:
                        primary_rows.append(row)

    all_rows = pd.DataFrame(output_rows)
    all_rows.to_csv(OUT / "cmems_hr_oc_water_support_by_date.csv", index=False, encoding="utf-8-sig")

    primary = pd.DataFrame(primary_rows)
    scene_rows: list[dict[str, object]] = []
    for _, scene in selected_scenes.iterrows():
        date_rows = primary[primary["date_utc"].eq(scene["date_utc"])]
        if date_rows.empty:
            scene_rows.append({
                "product_name": scene["product_name"],
                "acquisition_utc": scene["acquisition_utc"],
                "date_utc": scene["date_utc"],
                "cmems_same_date_available": False,
                "note": "No matching daily CMEMS date; same-day presence would not prove identical source granule.",
            })
        else:
            for _, row in date_rows.iterrows():
                scene_rows.append({
                    "product_name": scene["product_name"],
                    "acquisition_utc": scene["acquisition_utc"],
                    "date_utc": scene["date_utc"],
                    "cmems_same_date_available": True,
                    **row.to_dict(),
                    "note": "Same calendar day is not proof of identical Sentinel-2 granule; CMEMS is not independent field validation.",
                })
    selected_summary = pd.DataFrame(scene_rows)
    selected_summary.to_csv(OUT / "cmems_hr_oc_selected_scene_water_support.csv", index=False, encoding="utf-8-sig")

    # Audit figure: temporal availability and water-only valid support; daily values are descriptive.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.4, 8.2), sharex=True, constrained_layout=True)
    dates = pd.to_datetime(primary[primary["variable"].eq("SPM")]["date_utc"])
    spm = primary[primary["variable"].eq("SPM")].set_index("date_utc")
    tur = primary[primary["variable"].eq("TUR")].set_index("date_utc")
    spm = spm.reindex(dates.dt.strftime("%Y-%m-%d"))
    tur = tur.reindex(dates.dt.strftime("%Y-%m-%d"))
    valid_fraction = spm["valid_fraction_within_water_cells"].to_numpy(dtype=float)
    valid_cells = spm["valid_water_cmems_cells"].to_numpy(dtype=float)
    ax1.plot(dates, valid_fraction, color="#007C83", lw=1.7, label="Valid CMEMS cells / stable-water cells")
    ax1.scatter(dates, valid_fraction, c=np.where(np.isin(dates.dt.strftime("%Y-%m-%d"), list(selected_days)), "#E07A32", "#007C83"), s=25, zorder=3)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_ylabel("Valid water-cell fraction")
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(frameon=False, loc="upper left")
    ax1b = ax1.twinx()
    ax1b.bar(dates, valid_cells, width=0.8, color="#A9C8C8", alpha=0.35, label="Valid cells (count)")
    ax1b.set_ylabel("Valid CMEMS water cells")

    ax2.plot(dates, spm["median"].to_numpy(dtype=float), color="#2878B5", lw=1.5, label="SPM median (g m$^{-3}$)")
    ax2.fill_between(dates, spm["p25"].to_numpy(dtype=float), spm["p75"].to_numpy(dtype=float), color="#2878B5", alpha=0.14, linewidth=0)
    ax2b = ax2.twinx()
    ax2b.plot(dates, tur["median"].to_numpy(dtype=float), color="#C45A3A", lw=1.5, label="TUR median (FNU)")
    ax2b.fill_between(dates, tur["p25"].to_numpy(dtype=float), tur["p75"].to_numpy(dtype=float), color="#C45A3A", alpha=0.12, linewidth=0)
    for day in sorted(selected_days):
        dt = pd.Timestamp(day)
        ax1.axvline(dt, color="#E07A32", lw=0.5, alpha=0.24)
        ax2.axvline(dt, color="#E07A32", lw=0.5, alpha=0.24)
    ax2.set_ylabel("SPM median and IQR (g m$^{-3}$)", color="#2878B5")
    ax2b.set_ylabel("TUR median and IQR (FNU)", color="#C45A3A")
    ax2.grid(axis="y", alpha=0.22)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.set_xlabel("CMEMS daily composite date (UTC)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", ncol=2)
    fig.suptitle("CMEMS HR-OC availability over the Sentinel-2 stable-water core", fontsize=13, weight="bold")
    fig.text(0.5, -0.005,
             "Water cells require ≥50% stable-water fraction. Shading is the spatial IQR, not a confidence interval. "
             "Daily valid footprints vary; same-date CMEMS and MSI records are not assumed to be the same granule.",
             ha="center", va="top", fontsize=8.4)
    fig.savefig(OUT / "Figure_CMEMS_water_support_audit.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_CMEMS_water_support_audit.pdf", bbox_inches="tight")
    plt.close(fig)

    counts = primary[(primary["variable"].eq("SPM")) & primary["is_selected_s2_acquisition_day"]]
    note_lines = [
        "CMEMS HR-OC stable-water support audit",
        "Primary water-cell definition: at least 50% of the CMEMS grid-cell area overlaps the exploratory Sentinel-2 stable-water class.",
        "Sensitivity thresholds: 25%, 50%, and 75%; all are reported in cmems_hr_oc_water_support_by_date.csv.",
        "Class 3 is an SCL-derived persistent-water screening mask, not a surveyed shoreline or authoritative marine boundary.",
        "The NetCDF subset contains SPM and TUR only; it contains no separate pixel-level quality flag variable.",
        "Do not interpret the absence of a returned flag variable as absence of processor quality screening. Consult the official HR-OC QUID and product documentation.",
        "SPM and TUR are satellite-derived products and are not independent in-situ validation of the Sentinel-2 NDTI proxy.",
        "Selected dates represent same-calendar-day comparisons only; they do not establish that HR-OC used the identical MSI granule.",
        f"Selected S2 dates with at least 2000 finite SPM cells over primary water mask: {int((counts['valid_water_cmems_cells'] >= 2000).sum())} / {len(counts)}.",
        "Median/IQR summaries use a date-varying valid footprint; they are descriptive and should not be treated as fixed-pixel temporal change.",
        "No calibration, regression, significance test, resampling of source rasters, or sediment-transport rate was calculated.",
    ]
    (OUT / "cmems_hr_oc_water_support_notes_FR.txt").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    (OUT / "Figure_CMEMS_water_support_audit_caption_EN.txt").write_text(
        "Figure. Daily valid-cell support and HR-OC SPM/TUR distributions over the Sentinel-2 stable-water core at Anza. "
        "A CMEMS cell is included when at least half its area overlaps the exploratory stable-water mask; 25% and 75% "
        "threshold sensitivities are retained in the accompanying table. The upper panel reports valid-water-cell "
        "coverage and counts, with selected Sentinel-2 acquisition dates highlighted. The lower panel shows daily "
        "SPM and TUR medians with spatial interquartile ranges. Date-specific valid footprints vary, and a same-day "
        "match does not establish use of the same MSI granule or constitute in-situ validation.\n",
        encoding="utf-8",
    )

    print("Water-only coverage audit complete; source data were not changed.")
    print(f"Selected Sentinel-2 acquisition days: {len(selected_scenes)}")
    print(f"Selected dates with >=2,000 finite SPM cells in the primary water mask: {(counts['valid_water_cmems_cells'] >= 2000).sum()}/{len(counts)}")
    print(f"Tables, notes, and figure: {OUT}")
    print("Use the water-support table before any date-to-date comparison; this is product consistency analysis, not independent validation.")


if __name__ == "__main__":
    main()
