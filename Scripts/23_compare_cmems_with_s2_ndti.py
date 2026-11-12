from __future__ import annotations

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
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / "Data" / "Copernicus_Marine" / "Ocean_Colour_HR" / "Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc"
WATER_CLASS_PATH = ROOT / "Results" / "Marine_Mask" / "S2_water_stability_class_10m.tif"
SCENE_PATH = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
NDTI_DIR = ROOT / "Results" / "Sediment_Proxy"
OUT = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit"

WATER_AREA_FRACTION_MIN = 0.50
NDTI_VALID_FRACTION_MIN = 0.25
NDTI_SUPPORT_SENSITIVITY = (0.10, 0.25, 0.50)
MIN_CMEMS_WATER_CELLS = 1000
BLOCK_SIZE_M = 500.0
BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20261111
DISPLAY_QUANTILES = (2.0, 98.0)


def find_coord(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    names = {name.lower(): name for name in list(ds.coords) + list(ds.dims)}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


def regular_step(values: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    diffs = np.diff(values)
    step = float(np.median(diffs))
    if len(values) < 2 or step == 0 or not np.allclose(diffs, step, rtol=1e-5, atol=1e-10):
        raise ValueError(f"The {name} coordinate is not a regular grid; refusing approximate alignment.")
    return abs(step)


def average_to_product_grid(
    source: np.ndarray,
    src_transform: Affine,
    src_crs: object,
    dst_transform: Affine,
    shape: tuple[int, int],
    *,
    src_nodata: float | None = None,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    result = np.full(shape, dst_nodata, dtype=np.float32)
    reproject(
        source=source.astype(np.float32, copy=False),
        destination=result,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    return result


def load_scenes() -> pd.DataFrame:
    scenes = pd.read_csv(SCENE_PATH, dtype=str, keep_default_na=False)
    required = {"download_selected", "product_name", "acquisition_utc"}
    if not required.issubset(scenes.columns):
        raise ValueError(f"Selection file must contain: {sorted(required)}")
    scenes = scenes[scenes["download_selected"].str.upper().eq("YES")].copy()
    scenes["date_utc"] = pd.to_datetime(scenes["acquisition_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if scenes["date_utc"].isna().any():
        raise ValueError("At least one selected Sentinel-2 acquisition date is invalid.")
    return scenes.sort_values("date_utc").reset_index(drop=True)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def block_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, int]:
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 10 or len(x) < 50:
        return np.nan, np.nan, len(unique_blocks)
    grouped = {block: np.flatnonzero(block_ids == block) for block in unique_blocks}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        picked = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        indices = np.concatenate([grouped[block] for block in picked])
        rho = safe_spearman(x[indices], y[indices])
        if np.isfinite(rho):
            estimates.append(rho)
    if len(estimates) < 50:
        return np.nan, np.nan, len(unique_blocks)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), len(unique_blocks)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_median": float(np.median(values)) if values.size else np.nan,
        f"{prefix}_p25": float(np.percentile(values, 25)) if values.size else np.nan,
        f"{prefix}_p75": float(np.percentile(values, 75)) if values.size else np.nan,
        f"{prefix}_p95": float(np.percentile(values, 95)) if values.size else np.nan,
    }


def main() -> None:
    for path in (NC_PATH, WATER_CLASS_PATH, SCENE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes()
    selected_by_date = {row["date_utc"]: row for _, row in scenes.iterrows()}
    rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    eligible_arrays: dict[str, dict[str, np.ndarray]] = {}
    grid_info: dict[str, object] = {}

    print("Stage 8: paired spatial comparison of same-date Sentinel-2 NDTI and CMEMS HR-OC SPM/TUR.")
    print("NDTI is area-averaged to the CMEMS grid; only the stable-water core is retained.")
    print("The same-day product is not assumed to use the same Sentinel-2 granule and is not in-situ validation.")

    with xr.open_dataset(NC_PATH, decode_cf=True, mask_and_scale=True) as ds:
        t_name = find_coord(ds, ("time", "datetime"))
        y_name = find_coord(ds, ("latitude", "lat"))
        x_name = find_coord(ds, ("longitude", "lon"))
        if not t_name or not y_name or not x_name:
            raise ValueError(f"Could not identify NetCDF coordinates: {list(ds.coords)}")
        if not {"SPM", "TUR"}.issubset(ds.data_vars):
            raise ValueError(f"Expected SPM and TUR. Found {list(ds.data_vars)}")
        ds = ds.sortby(y_name, ascending=False).sortby(x_name, ascending=True)
        lats = np.asarray(ds[y_name].values, dtype=float)
        lons = np.asarray(ds[x_name].values, dtype=float)
        dx = regular_step(lons, "longitude")
        dy = regular_step(lats, "latitude")
        if np.median(np.diff(lats)) >= 0:
            raise ValueError("Latitude coordinate is not ordered north-to-south.")
        shape = (len(lats), len(lons))
        dst_transform = Affine(dx, 0, float(lons.min() - dx / 2), 0, -dy, float(lats.max() + dy / 2))
        times = pd.to_datetime(np.asarray(ds[t_name].values).reshape(-1), utc=True, errors="coerce")
        if times.isna().any():
            raise ValueError("Invalid CMEMS time coordinate.")
        date_to_index = {time.strftime("%Y-%m-%d"): i for i, time in enumerate(times)}

        with rasterio.open(WATER_CLASS_PATH) as water_ds:
            class_data = np.asarray(water_ds.read(1, masked=True).filled(0), dtype=np.uint8)
            water_fraction = average_to_product_grid(
                (class_data == 3).astype(np.float32), water_ds.transform, water_ds.crs, dst_transform, shape,
                src_nodata=None, dst_nodata=0.0,
            )
        stable_water = water_fraction >= WATER_AREA_FRACTION_MIN

        # Set the independent spatial-block IDs once for every product date.
        lon2d, lat2d = np.meshgrid(lons, lats)
        to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
        x_m, y_m = to_metric.transform(lon2d, lat2d)
        bx = np.floor((x_m - np.nanmin(x_m)) / BLOCK_SIZE_M).astype(np.int64)
        by = np.floor((y_m - np.nanmin(y_m)) / BLOCK_SIZE_M).astype(np.int64)
        block_grid = bx + (by * (int(np.nanmax(bx)) + 1))

        for scene_i, (date, scene) in enumerate(selected_by_date.items(), start=1):
            if date not in date_to_index:
                sensitivity_rows.append({
                    "date_utc": date,
                    "product_name": scene["product_name"],
                    "status": "CMEMS daily date unavailable",
                })
                continue
            ndti_path = NDTI_DIR / f"{scene['product_name']}_NDTI_relative_proxy.tif"
            if not ndti_path.is_file():
                sensitivity_rows.append({
                    "date_utc": date,
                    "product_name": scene["product_name"],
                    "status": f"NDTI raster missing: {ndti_path.name}",
                })
                continue

            with rasterio.open(ndti_path) as ndti_ds:
                ndti_masked = ndti_ds.read(1, masked=True).astype(np.float32)
                ndti_values = np.asarray(ndti_masked.filled(-9999.0), dtype=np.float32)
                ndti_source_valid = (~np.ma.getmaskarray(ndti_masked)) & np.isfinite(ndti_values) & (ndti_values != -9999.0)
                ndti_mean = average_to_product_grid(
                    ndti_values, ndti_ds.transform, ndti_ds.crs, dst_transform, shape,
                    src_nodata=-9999.0, dst_nodata=np.nan,
                )
                ndti_fraction = average_to_product_grid(
                    ndti_source_valid.astype(np.float32), ndti_ds.transform, ndti_ds.crs, dst_transform, shape,
                    src_nodata=None, dst_nodata=0.0,
                )

            ti = date_to_index[date]
            product_values: dict[str, np.ndarray] = {}
            valid_product: dict[str, np.ndarray] = {}
            for variable in ("SPM", "TUR"):
                da = ds[variable]
                values = np.asarray(da.isel({t_name: ti}).transpose(y_name, x_name).values, dtype=np.float64)
                valid = np.isfinite(values)
                if da.attrs.get("valid_min") is not None:
                    valid &= values >= float(da.attrs["valid_min"])
                if da.attrs.get("valid_max") is not None:
                    valid &= values <= float(da.attrs["valid_max"])
                product_values[variable] = values
                valid_product[variable] = valid

            joint_product_valid = valid_product["SPM"] & valid_product["TUR"]
            cmems_water_n = int(np.count_nonzero(stable_water & joint_product_valid))
            primary_pair = stable_water & joint_product_valid & (ndti_fraction >= NDTI_VALID_FRACTION_MIN) & np.isfinite(ndti_mean)
            primary_n = int(primary_pair.sum())
            for ndti_threshold in NDTI_SUPPORT_SENSITIVITY:
                support_mask = stable_water & joint_product_valid & (ndti_fraction >= ndti_threshold) & np.isfinite(ndti_mean)
                sensitivity_rows.append({
                    "date_utc": date,
                    "product_name": scene["product_name"],
                    "ndti_valid_source_area_fraction_threshold": ndti_threshold,
                    "stable_water_cmems_cells": int(stable_water.sum()),
                    "cmems_joint_spm_tur_valid_water_cells": cmems_water_n,
                    "paired_cells_after_ndti_support": int(support_mask.sum()),
                    "eligible_for_primary_association": cmems_water_n >= MIN_CMEMS_WATER_CELLS,
                    "status": "included" if cmems_water_n >= MIN_CMEMS_WATER_CELLS else f"low CMEMS support (<{MIN_CMEMS_WATER_CELLS} water cells)",
                })

            if cmems_water_n < MIN_CMEMS_WATER_CELLS:
                print(f"  {scene_i}/{len(scenes)} {date}: CMEMS water support={cmems_water_n}; descriptive only, excluded from primary association")
                continue
            if primary_n < 50:
                print(f"  {scene_i}/{len(scenes)} {date}: only {primary_n} paired cells after NDTI support; insufficient for association")
                continue

            block_ids = block_grid[primary_pair]
            x = ndti_mean[primary_pair].astype(np.float64)
            date_bundle = {
                "pair_mask": primary_pair,
                "ndti": ndti_mean,
                "ndti_fraction": ndti_fraction,
                "SPM": product_values["SPM"],
                "TUR": product_values["TUR"],
            }
            eligible_arrays[date] = date_bundle
            summary: dict[str, object] = {
                "date_utc": date,
                "product_name": scene["product_name"],
                "acquisition_utc": scene["acquisition_utc"],
                "cmems_day_is_same_s2_calendar_day": True,
                "cmems_water_cells_joint_valid": cmems_water_n,
                "paired_spatial_cells": primary_n,
                "ndti_valid_area_fraction_threshold": NDTI_VALID_FRACTION_MIN,
                "stable_water_area_fraction_threshold": WATER_AREA_FRACTION_MIN,
            }
            summary.update(summarize(x, "ndti"))
            for variable in ("SPM", "TUR"):
                y = product_values[variable][primary_pair]
                rho = safe_spearman(x, y)
                low, high, nblocks = block_bootstrap_ci(
                    x, y, block_ids, seed=RANDOM_SEED + scene_i * 31 + (0 if variable == "SPM" else 1)
                )
                summary.update(summarize(y, variable.lower()))
                summary[f"{variable.lower()}_spearman_rho"] = rho
                summary[f"{variable.lower()}_block_bootstrap_95ci_low"] = low
                summary[f"{variable.lower()}_block_bootstrap_95ci_high"] = high
                summary[f"{variable.lower()}_spatial_blocks_500m"] = nblocks
            rows.append(summary)
            print(f"  {scene_i}/{len(scenes)} {date}: CMEMS water cells={cmems_water_n:,}; paired cells={primary_n:,}")

        grid_info = {
            "longitude": lons,
            "latitude": lats,
            "stable_water_fraction": water_fraction,
            "block_grid": block_grid,
        }

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT / "cmems_s2_ndti_paired_support_sensitivity.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT / "cmems_s2_ndti_spatial_association_by_date.csv", index=False, encoding="utf-8-sig")

    sample_rows: list[dict[str, object]] = []
    for date, bundle in eligible_arrays.items():
        mask = bundle["pair_mask"]
        rr, cc = np.where(mask)
        for r, c in zip(rr, cc):
            sample_rows.append({
                "date_utc": date,
                "row": int(r),
                "column": int(c),
                "longitude": float(grid_info["longitude"][c]),
                "latitude": float(grid_info["latitude"][r]),
                "stable_water_fraction": float(grid_info["stable_water_fraction"][r, c]),
                "ndti_valid_source_area_fraction": float(bundle["ndti_fraction"][r, c]),
                "ndti_cell_mean": float(bundle["ndti"][r, c]),
                "spm_g_m3": float(bundle["SPM"][r, c]),
                "turbidity_fnu": float(bundle["TUR"][r, c]),
                "spatial_block_500m": int(grid_info["block_grid"][r, c]),
            })
    paired_df = pd.DataFrame(sample_rows)
    paired_df.to_csv(OUT / "cmems_s2_ndti_paired_cells.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")

    # Fixed-cell sensitivity: retain only grid cells paired on every eligible acquisition date.
    common_mask: np.ndarray | None = None
    for bundle in eligible_arrays.values():
        common_mask = bundle["pair_mask"].copy() if common_mask is None else common_mask & bundle["pair_mask"]
    common_n = int(common_mask.sum()) if common_mask is not None else 0
    common_rows: list[dict[str, object]] = []
    if common_mask is not None and common_n:
        for date, bundle in eligible_arrays.items():
            x = bundle["ndti"][common_mask].astype(np.float64)
            row: dict[str, object] = {"date_utc": date, "fixed_common_cells": common_n}
            for variable in ("SPM", "TUR"):
                y = bundle[variable][common_mask].astype(np.float64)
                bids = grid_info["block_grid"][common_mask]
                rho = safe_spearman(x, y)
                low, high, nblocks = block_bootstrap_ci(
                    x, y, bids, seed=RANDOM_SEED + 900 + len(common_rows) * 13 + (0 if variable == "SPM" else 1)
                )
                row[f"{variable.lower()}_median"] = float(np.median(y))
                row[f"{variable.lower()}_spearman_rho"] = rho
                row[f"{variable.lower()}_block_bootstrap_95ci_low"] = low
                row[f"{variable.lower()}_block_bootstrap_95ci_high"] = high
                row[f"{variable.lower()}_spatial_blocks_500m"] = nblocks
            row["ndti_median"] = float(np.median(x))
            common_rows.append(row)
    pd.DataFrame(common_rows).to_csv(OUT / "cmems_s2_ndti_fixed_common_support.csv", index=False, encoding="utf-8-sig")

    # Figure 1: association by date with a spatial-block bootstrap interval.
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(10.8, max(4.3, 0.48 * len(summary_df))), constrained_layout=True)
        ordered = summary_df.sort_values("date_utc").reset_index(drop=True)
        ybase = np.arange(len(ordered), dtype=float)
        for offset, variable, color, label in ((-0.10, "spm", "#2878B5", "SPM"), (0.10, "tur", "#C45A3A", "TUR")):
            rho = ordered[f"{variable}_spearman_rho"].to_numpy(dtype=float)
            low = ordered[f"{variable}_block_bootstrap_95ci_low"].to_numpy(dtype=float)
            high = ordered[f"{variable}_block_bootstrap_95ci_high"].to_numpy(dtype=float)
            good = np.isfinite(rho)
            xerr = np.vstack([np.maximum(0, rho[good] - low[good]), np.maximum(0, high[good] - rho[good])])
            ax.errorbar(rho[good], ybase[good] + offset, xerr=xerr, fmt="o", markersize=5, capsize=2.5,
                        color=color, ecolor=color, linewidth=1.2, label=label)
        ax.axvline(0, color="#666666", lw=0.9, linestyle="--")
        ax.set_yticks(ybase, ordered["date_utc"].tolist())
        ax.set_xlim(-1.02, 1.02)
        ax.set_xlabel("Spatial Spearman correlation with area-averaged S2 NDTI")
        ax.set_ylabel("Same-calendar-day product pair (UTC)")
        ax.grid(axis="x", alpha=0.22)
        ax.legend(frameon=False, ncol=2, loc="lower right")
        fig.suptitle("Spatial correspondence between Sentinel-2 NDTI and CMEMS HR-OC", fontsize=13, weight="bold")
        fig.text(0.5, -0.005,
                 "Intervals resample 500 m spatial blocks (500 replicates); they describe spatial uncertainty, not independent validation. "
                 "Daily composites may not use the exact local MSI granule.", ha="center", va="top", fontsize=8.3)
        fig.savefig(OUT / "Figure_CMEMS_S2_NDTI_spatial_association.png", dpi=350, bbox_inches="tight")
        fig.savefig(OUT / "Figure_CMEMS_S2_NDTI_spatial_association.pdf", bbox_inches="tight")
        plt.close(fig)

        # Figure 2: same-grid maps on the selected date with the largest paired support.
        best_date = str(ordered.loc[ordered["paired_spatial_cells"].idxmax(), "date_utc"])
        bundle = eligible_arrays[best_date]
        mask = bundle["pair_mask"]
        ndti_map = np.where(mask, bundle["ndti"], np.nan)
        spm_map = np.where(mask, bundle["SPM"], np.nan)
        ndti_limits = np.nanpercentile(ndti_map, DISPLAY_QUANTILES)
        spm_limits = np.nanpercentile(spm_map, DISPLAY_QUANTILES)
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
        panels = [
            (axes[0], ndti_map, "viridis", "Area-averaged S2 NDTI", "NDTI (dimensionless)", ndti_limits),
            (axes[1], spm_map, "magma", "CMEMS HR-OC SPM", "SPM (g m$^{-3}$)", spm_limits),
        ]
        for ax, array, cmap, title, cbar_label, limits in panels:
            mesh = ax.pcolormesh(grid_info["longitude"], grid_info["latitude"], array, shading="nearest",
                                 cmap=cmap, vmin=float(limits[0]), vmax=float(limits[1]))
            ax.set_title(title, fontsize=11, weight="bold")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_aspect("equal", adjustable="box")
            cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.025)
            cb.set_label(cbar_label)
        fig.suptitle(f"Co-registered product patterns on {best_date} (UTC)", fontsize=13, weight="bold")
        fig.text(0.5, -0.005,
                 "Both panels show the same paired CMEMS grid cells. Display ranges are clipped to the 2nd–98th percentiles; "
                 "the tables retain all valid values. The maps show product consistency, not ground-truth accuracy.",
                 ha="center", va="top", fontsize=8.3)
        fig.savefig(OUT / "Figure_CMEMS_S2_NDTI_paired_maps.png", dpi=350, bbox_inches="tight")
        fig.savefig(OUT / "Figure_CMEMS_S2_NDTI_paired_maps.pdf", bbox_inches="tight")
        plt.close(fig)

    summary_text = [
        "Same-calendar-day Sentinel-2 NDTI / CMEMS HR-OC spatial comparison",
        f"Stable-water area threshold: >= {WATER_AREA_FRACTION_MIN:.2f} fraction of each CMEMS cell.",
        f"NDTI source-pixel support threshold: >= {NDTI_VALID_FRACTION_MIN:.2f} fraction of each CMEMS cell.",
        f"Primary eligible-date threshold: >= {MIN_CMEMS_WATER_CELLS} CMEMS-valid stable-water cells.",
        f"Eligible dates analyzed: {len(eligible_arrays)} of {len(scenes)} selected S2 dates.",
        f"Fixed cells paired on every eligible date: {common_n}.",
        f"Spatial block size for bootstrap: {BLOCK_SIZE_M:.0f} m; replicates: {BOOTSTRAP_REPLICATES}.",
        "Per-date correlations are spatial product-correspondence diagnostics; pixels are spatially dependent.",
        "Confidence intervals resample 500 m blocks, but do not make CMEMS independent of Sentinel-2 or establish causality.",
        "CMEMS HR-OC is satellite-derived; the same day does not prove that its composite used the exact downloaded MSI granule.",
        "No p-values, sediment concentration calibration, transport rate, or hydrodynamic simulation were produced.",
    ]
    (OUT / "cmems_s2_ndti_comparison_notes_FR.txt").write_text("\n".join(summary_text) + "\n", encoding="utf-8")
    print("Paired spatial product comparison complete; source imagery, NetCDF, and bathymetry were not changed.")
    print(f"Eligible selected dates: {len(eligible_arrays)}/{len(scenes)}")
    print(f"Cells paired across every eligible date: {common_n:,}")
    print(f"Tables, paired samples, figures, and notes: {OUT}")
    print("Interpret correlations as spatial product correspondence only; do not describe them as independent validation or sediment transport.")


if __name__ == "__main__":
    main()
