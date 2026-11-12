from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
PAIR_PATH = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit" / "cmems_s2_ndti_paired_cells.csv.gz"
BATHY_PATH = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
NC_PATH = ROOT / "Data" / "Copernicus_Marine" / "Ocean_Colour_HR" / "Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc"
OUT = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit"
TARGET_DATE = "2022-11-10"
MIN_SOUNDINGS_PRIMARY = 5
SOUNDING_THRESHOLDS = (1, 5, 10)
BLOCK_SIZE_M = 500.0
BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20261112


def find_coord(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    names = {name.lower(): name for name in list(ds.coords) + list(ds.dims)}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


def get_grid_centres() -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    with xr.open_dataset(NC_PATH, decode_cf=True, mask_and_scale=True) as ds:
        y_name = find_coord(ds, ("latitude", "lat"))
        x_name = find_coord(ds, ("longitude", "lon"))
        if not x_name or not y_name:
            raise ValueError(f"Could not identify CMEMS coordinates: {list(ds.coords)}")
        lons = np.sort(np.asarray(ds[x_name].values, dtype=float))
        lats = np.sort(np.asarray(ds[y_name].values, dtype=float))[::-1]
    dxs, dys = np.diff(lons), np.diff(lats)
    dx, dy = float(np.median(dxs)), float(np.median(dys))
    if not np.allclose(dxs, dx, rtol=1e-5, atol=1e-10) or not np.allclose(dys, dy, rtol=1e-5, atol=1e-10):
        raise ValueError("CMEMS coordinates are not a regular grid; cannot assign soundings to cells safely.")
    left, top = float(lons[0] - dx / 2), float(lats[0] - dy / 2)
    return lons, lats, dx, dy, left, top


def residual_rank(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    rank_y = rankdata(values, method="average").astype(float)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    ranked_covariates = np.column_stack([rankdata(covariates[:, i], method="average") for i in range(covariates.shape[1])])
    design = np.column_stack([np.ones(rank_y.size), ranked_covariates])
    coefficients, *_ = np.linalg.lstsq(design, rank_y, rcond=None)
    return rank_y - design @ coefficients


def stat_for_method(frame: pd.DataFrame, variable: str, method: str) -> float:
    x = frame["ndti_cell_mean"].to_numpy(dtype=float)
    y = frame[variable].to_numpy(dtype=float)
    if method == "spatial_spearman_unadjusted":
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(spearmanr(x, y).statistic)

    depth = frame["depth_below_local_zero_m"].to_numpy(dtype=float)
    if method == "partial_spearman_depth":
        covariates = depth[:, None]
    elif method == "partial_spearman_depth_and_linear_xy":
        covariates = frame[["depth_below_local_zero_m", "x_utm29_m", "y_utm29_m"]].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown association method: {method}")
    rx = residual_rank(x, covariates)
    ry = residual_rank(y, covariates)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_interval(frame: pd.DataFrame, variable: str, method: str, seed: int) -> tuple[float, float, int]:
    blocks = frame["spatial_block_500m"].to_numpy()
    unique = np.unique(blocks)
    if len(unique) < 10 or len(frame) < 50:
        return np.nan, np.nan, len(unique)
    groups = {block: np.flatnonzero(blocks == block) for block in unique}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([groups[block] for block in sampled])
        value = stat_for_method(frame.iloc[indices], variable, method)
        if np.isfinite(value):
            estimates.append(value)
    if len(estimates) < 50:
        return np.nan, np.nan, len(unique)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi), len(unique)


def main() -> None:
    for path in (PAIR_PATH, BATHY_PATH, NC_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    print("Depth-confounding diagnostic for the 2022-11-10 Anza same-day matchup.")
    print("Soundings retain their source Z relative to the local hydrographic zero of Port of Agadir.")
    print("This is a spatial association sensitivity analysis; it does not convert Z to another datum or infer SSC.")

    paired = pd.read_csv(PAIR_PATH, compression="gzip")
    paired = paired[paired["date_utc"].eq(TARGET_DATE)].copy()
    if paired.empty:
        raise RuntimeError(f"No paired product cells found for {TARGET_DATE}; run Script 23 first.")

    points = pd.read_csv(BATHY_PATH)
    needed = {"z_source_m", "longitude_wgs84_candidate_deg", "latitude_wgs84_candidate_deg"}
    if not needed.issubset(points.columns):
        raise ValueError(f"Bathymetry file lacks required fields: {sorted(needed.difference(points.columns))}")
    lons, lats, dx, dy, left, top = get_grid_centres()
    point_lon = points["longitude_wgs84_candidate_deg"].to_numpy(dtype=float)
    point_lat = points["latitude_wgs84_candidate_deg"].to_numpy(dtype=float)
    col = np.floor((point_lon - left) / dx).astype(int)
    row = np.floor((top - point_lat) / abs(dy)).astype(int)
    in_grid = (row >= 0) & (row < len(lats)) & (col >= 0) & (col < len(lons))
    soundings = points.loc[in_grid, ["z_source_m"]].copy()
    soundings["row"] = row[in_grid]
    soundings["column"] = col[in_grid]
    soundings["z_source_m"] = pd.to_numeric(soundings["z_source_m"], errors="coerce")
    soundings = soundings[np.isfinite(soundings["z_source_m"]) & (soundings["z_source_m"] < 0)].copy()
    soundings["depth_below_local_zero_m"] = -soundings["z_source_m"]

    cell_depth = (
        soundings.groupby(["row", "column"], as_index=False)
        .agg(soundings_per_cell=("depth_below_local_zero_m", "size"), depth_below_local_zero_m=("depth_below_local_zero_m", "median"))
    )
    paired = paired.merge(cell_depth, on=["row", "column"], how="inner", validate="one_to_one")
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
    x_m, y_m = to_metric.transform(paired["longitude"].to_numpy(), paired["latitude"].to_numpy())
    paired["x_utm29_m"] = x_m
    paired["y_utm29_m"] = y_m
    paired["spatial_block_500m"] = (
        np.floor((paired["x_utm29_m"] - paired["x_utm29_m"].min()) / BLOCK_SIZE_M).astype(int)
        + np.floor((paired["y_utm29_m"] - paired["y_utm29_m"].min()) / BLOCK_SIZE_M).astype(int)
        * (int(np.floor((paired["x_utm29_m"] - paired["x_utm29_m"].min()) / BLOCK_SIZE_M).max()) + 1)
    )
    if paired.empty:
        raise RuntimeError("No underwater survey soundings overlap the paired CMEMS/Sentinel-2 cells.")

    print(f"  Same-date paired product cells: {len(pd.read_csv(PAIR_PATH, compression='gzip').query('date_utc == @TARGET_DATE')):,}")
    print(f"  Cells with at least one submerged sounding: {len(paired):,}")
    print(f"  Cells with >= {MIN_SOUNDINGS_PRIMARY} soundings: {(paired['soundings_per_cell'] >= MIN_SOUNDINGS_PRIMARY).sum():,}")

    methods = [
        ("spatial_spearman_unadjusted", "Unadjusted spatial Spearman"),
        ("partial_spearman_depth", "Partial Spearman controlling local depth"),
        ("partial_spearman_depth_and_linear_xy", "Partial Spearman controlling depth and linear XY trend"),
    ]
    comparison_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    primary = paired[paired["soundings_per_cell"] >= MIN_SOUNDINGS_PRIMARY].copy()
    if len(primary) < 50:
        raise RuntimeError("Fewer than 50 paired cells meet the primary sounding-count threshold; review the support rule before interpreting.")

    for min_soundings in SOUNDING_THRESHOLDS:
        subset = paired[paired["soundings_per_cell"] >= min_soundings].copy()
        for variable, label in (("spm_g_m3", "SPM"), ("turbidity_fnu", "TUR")):
            for method, method_label in methods:
                rho = stat_for_method(subset, variable, method)
                threshold_rows.append({
                    "minimum_soundings_per_cell": min_soundings,
                    "variable": label,
                    "method": method,
                    "method_label": method_label,
                    "paired_cells": len(subset),
                    "spatial_blocks_500m": subset["spatial_block_500m"].nunique(),
                    "association_estimate": rho,
                })

    for var_index, (variable, label) in enumerate((("spm_g_m3", "SPM"), ("turbidity_fnu", "TUR"))):
        for method_index, (method, method_label) in enumerate(methods):
            estimate = stat_for_method(primary, variable, method)
            low, high, n_blocks = bootstrap_interval(primary, variable, method, RANDOM_SEED + var_index * 100 + method_index)
            comparison_rows.append({
                "date_utc": TARGET_DATE,
                "variable": label,
                "method": method,
                "method_label": method_label,
                "paired_cells": len(primary),
                "minimum_soundings_per_cell": MIN_SOUNDINGS_PRIMARY,
                "spatial_blocks_500m": n_blocks,
                "association_estimate": estimate,
                "block_bootstrap_95ci_low": low,
                "block_bootstrap_95ci_high": high,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "Anza_20221110_depth_conditioned_associations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_rows).to_csv(OUT / "Anza_20221110_depth_support_sensitivity.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(OUT / "Anza_20221110_depth_conditioned_paired_cells.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")

    depth_bin_width = 2.0
    primary["depth_bin_start_m"] = np.floor(primary["depth_below_local_zero_m"] / depth_bin_width) * depth_bin_width
    bin_rows: list[dict[str, object]] = []
    for bin_start, group in primary.groupby("depth_bin_start_m", sort=True):
        row_out: dict[str, object] = {
            "depth_bin_m": f"{bin_start:.0f}–{bin_start + depth_bin_width:.0f}",
            "paired_cells": len(group),
            "spatial_blocks_500m": group["spatial_block_500m"].nunique(),
            "ndti_median": group["ndti_cell_mean"].median(),
            "depth_median_below_local_zero": group["depth_below_local_zero_m"].median(),
        }
        for variable, label in (("spm_g_m3", "spm"), ("turbidity_fnu", "tur")):
            row_out[f"{label}_median"] = group[variable].median()
            row_out[f"{label}_ndti_spearman_rho"] = float(spearmanr(group["ndti_cell_mean"], group[variable]).statistic) if len(group) >= 30 else np.nan
        bin_rows.append(row_out)
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(OUT / "Anza_20221110_depth_stratified_associations_2m.csv", index=False, encoding="utf-8-sig")

    # Plot SPM relationships, which are not treated as independent from TUR.
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    depth = primary["depth_below_local_zero_m"].to_numpy(dtype=float)
    x = primary["ndti_cell_mean"].to_numpy(dtype=float)
    y = primary["spm_g_m3"].to_numpy(dtype=float)
    raw_rho = stat_for_method(primary, "spm_g_m3", methods[0][0])
    axes[0].scatter(x, y, c=depth, cmap="viridis", s=8, alpha=0.42, linewidths=0, rasterized=True)
    axes[0].set_xlabel("Area-averaged S2 NDTI")
    axes[0].set_ylabel("CMEMS HR-OC SPM (g m$^{-3}$)")
    axes[0].set_title(f"Raw spatial association\nSpearman ρ = {raw_rho:.2f}")
    cb = fig.colorbar(axes[0].collections[0], ax=axes[0], shrink=0.82, pad=0.02)
    cb.set_label("Survey depth below local port zero (m)")

    rank_depth = depth[:, None]
    rx_depth = residual_rank(x, rank_depth)
    ry_depth = residual_rank(y, rank_depth)
    rho_depth = stat_for_method(primary, "spm_g_m3", methods[1][0])
    axes[1].scatter(rx_depth, ry_depth, c=depth, cmap="viridis", s=8, alpha=0.42, linewidths=0, rasterized=True)
    axes[1].axhline(0, color="#666666", lw=0.7, alpha=0.6)
    axes[1].axvline(0, color="#666666", lw=0.7, alpha=0.6)
    axes[1].set_xlabel("NDTI rank residual (depth-adjusted)")
    axes[1].set_ylabel("SPM rank residual (depth-adjusted)")
    axes[1].set_title(f"Controlling for depth\npartial ρ = {rho_depth:.2f}")

    cov = primary[["depth_below_local_zero_m", "x_utm29_m", "y_utm29_m"]].to_numpy(dtype=float)
    rx_space = residual_rank(x, cov)
    ry_space = residual_rank(y, cov)
    rho_space = stat_for_method(primary, "spm_g_m3", methods[2][0])
    axes[2].scatter(rx_space, ry_space, c=depth, cmap="viridis", s=8, alpha=0.42, linewidths=0, rasterized=True)
    axes[2].axhline(0, color="#666666", lw=0.7, alpha=0.6)
    axes[2].axvline(0, color="#666666", lw=0.7, alpha=0.6)
    axes[2].set_xlabel("NDTI rank residual (depth + XY-adjusted)")
    axes[2].set_ylabel("SPM rank residual (depth + XY-adjusted)")
    axes[2].set_title(f"Controlling for depth and XY trend\npartial ρ = {rho_space:.2f}")
    fig.suptitle("Does the same-day NDTI–SPM pattern persist after bathymetric control?", fontsize=13, weight="bold")
    fig.text(0.5, -0.005,
             "One local survey (10 Nov 2022); cell Z is the median of submerged soundings and remains relative to the Agadir port zero. "
             "XY detrending is a sensitivity check, not a hydrodynamic model. Intervals and correlations are spatial diagnostics.",
             ha="center", va="top", fontsize=8.1)
    fig.savefig(OUT / "Figure_Anza_20221110_depth_confounding.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_Anza_20221110_depth_confounding.pdf", bbox_inches="tight")
    plt.close(fig)

    notes = [
        "Anza bathymetric-confounding diagnostic for the 10 November 2022 same-day Sentinel-2 / CMEMS product matchup.",
        f"Primary cell inclusion: at least {MIN_SOUNDINGS_PRIMARY} submerged survey soundings assigned to the CMEMS cell.",
        f"Primary paired cells: {len(primary)}; blocks of {BLOCK_SIZE_M:.0f} m: {primary['spatial_block_500m'].nunique()}.",
        "Survey Z is retained in metres relative to the local hydrographic zero of Port of Agadir; depth below that local zero is -Z for submerged soundings.",
        "No vertical datum conversion was applied. Candidate horizontal coordinates carry the previously reported 7 m transformation accuracy; local control residuals are unavailable.",
        "Partial Spearman estimates correlate rank-transformed residuals after linear regression on ranked covariates.",
        "The depth+XY estimate controls a linear spatial trend only; unresolved nonlinear spatial structure may remain.",
        "The daily HR-OC record may not use the exact local Sentinel-2 granule. The analysis is product correspondence, not independent validation.",
        "Do not interpret the estimates as calibrated suspended sediment concentration, causal effect, or sediment-transport rate.",
    ]
    (OUT / "Anza_20221110_depth_confounding_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("Depth-conditioning sensitivity complete; original bathymetry and satellite products were not changed.")
    print(f"  Paired cells with >= {MIN_SOUNDINGS_PRIMARY} submerged soundings: {len(primary):,}")
    print(f"  500 m spatial blocks: {primary['spatial_block_500m'].nunique()}")
    print(f"  Outputs: {OUT}")
    print("Interpret this as a single-site, same-day confounding diagnostic; it is not a calibrated sediment retrieval.")


if __name__ == "__main__":
    main()
