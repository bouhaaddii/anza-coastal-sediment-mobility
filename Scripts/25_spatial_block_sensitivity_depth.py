from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit" / "Anza_20221110_depth_conditioned_paired_cells.csv.gz"
OUT = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit"
BLOCK_SIZES_M = (250.0, 350.0, 500.0, 750.0, 1000.0)
BOOTSTRAP_REPLICATES = 500
MIN_BLOCKS_FOR_INTERVAL = 10
RANDOM_SEED = 20261113


def residual_rank(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average").astype(float)
    covariates = np.asarray(covariates, dtype=float)
    ranked = np.column_stack([rankdata(covariates[:, i], method="average") for i in range(covariates.shape[1])])
    design = np.column_stack([np.ones(ranks.size), ranked])
    coefficients, *_ = np.linalg.lstsq(design, ranks, rcond=None)
    return ranks - design @ coefficients


def association(frame: pd.DataFrame, x: str, y: str, method: str) -> float:
    xv = frame[x].to_numpy(dtype=float)
    yv = frame[y].to_numpy(dtype=float)
    if method == "unadjusted":
        return float(spearmanr(xv, yv).statistic)
    if method == "depth_adjusted":
        cov = frame[["depth_below_local_zero_m"]].to_numpy(dtype=float)
    elif method == "depth_xy_adjusted":
        cov = frame[["depth_below_local_zero_m", "x_utm29_m", "y_utm29_m"]].to_numpy(dtype=float)
    else:
        raise ValueError(method)
    rx = residual_rank(xv, cov)
    ry = residual_rank(yv, cov)
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(
    frame: pd.DataFrame,
    block_ids: np.ndarray,
    y_col: str,
    method: str,
    seed: int,
) -> tuple[float, float, int]:
    unique = np.unique(block_ids)
    if len(unique) < MIN_BLOCKS_FOR_INTERVAL:
        return np.nan, np.nan, len(unique)
    groups = {block: np.flatnonzero(block_ids == block) for block in unique}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[block] for block in chosen])
        estimate = association(frame.iloc[idx], "ndti_cell_mean", y_col, method)
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < 50:
        return np.nan, np.nan, len(unique)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi), len(unique)


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Run Script 24 first; expected: {INPUT}")
    data = pd.read_csv(INPUT, compression="gzip")
    required = {
        "ndti_cell_mean", "spm_g_m3", "turbidity_fnu", "depth_below_local_zero_m", "x_utm29_m", "y_utm29_m"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"The Script 24 table is missing columns: {sorted(missing)}")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).reset_index(drop=True)
    if len(data) < 50:
        raise ValueError(f"Only {len(data)} complete cells; spatial block comparison requires at least 50.")

    method_map = {
        "unadjusted": "Unadjusted spatial Spearman",
        "depth_adjusted": "Partial Spearman controlling local depth",
        "depth_xy_adjusted": "Partial Spearman controlling depth and linear XY trend",
    }
    rows: list[dict[str, object]] = []
    print("Spatial-block-size sensitivity for the Anza 2022-11-10 depth-conditioned analysis.")
    print(f"Complete matched cells: {len(data)}; bootstrap replicates: {BOOTSTRAP_REPLICATES} per estimate.")

    for block_size in BLOCK_SIZES_M:
        bx = np.floor((data["x_utm29_m"].to_numpy() - data["x_utm29_m"].min()) / block_size).astype(int)
        by = np.floor((data["y_utm29_m"].to_numpy() - data["y_utm29_m"].min()) / block_size).astype(int)
        block_ids = bx + by * (int(bx.max()) + 1)
        for variable, y_col in (("SPM", "spm_g_m3"), ("TUR", "turbidity_fnu")):
            for method_i, (method, label) in enumerate(method_map.items()):
                estimate = association(data, "ndti_cell_mean", y_col, method)
                low, high, nblocks = bootstrap_ci(
                    data,
                    block_ids,
                    y_col,
                    method,
                    RANDOM_SEED + int(block_size) + (0 if variable == "SPM" else 17) + method_i * 101,
                )
                rows.append({
                    "date_utc": "2022-11-10",
                    "variable": variable,
                    "method": method,
                    "method_label": label,
                    "block_size_m": int(block_size),
                    "spatial_blocks": nblocks,
                    "matched_cells": len(data),
                    "association_estimate": estimate,
                    "block_bootstrap_95ci_low": low,
                    "block_bootstrap_95ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES if nblocks >= MIN_BLOCKS_FOR_INTERVAL else 0,
                    "interval_status": "computed" if np.isfinite(low) else f"not computed: fewer than {MIN_BLOCKS_FOR_INTERVAL} blocks",
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Anza_20221110_spatial_block_size_sensitivity.csv", index=False, encoding="utf-8-sig")

    # SPM is plotted once; the CSV retains both SPM and TUR diagnostics.
    plot_data = result[result["variable"].eq("SPM")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharex=True, constrained_layout=True)
    palette = {250: "#007C83", 350: "#2878B5", 500: "#6A5ACD", 750: "#C45A3A", 1000: "#9A6A3A"}
    for ax, (method, label) in zip(axes, method_map.items()):
        subset = plot_data[plot_data["method"].eq(method)].sort_values("block_size_m")
        xval = subset["block_size_m"].to_numpy(dtype=float)
        yval = subset["association_estimate"].to_numpy(dtype=float)
        lo = subset["block_bootstrap_95ci_low"].to_numpy(dtype=float)
        hi = subset["block_bootstrap_95ci_high"].to_numpy(dtype=float)
        nblocks = subset["spatial_blocks"].to_numpy(dtype=int)
        for xx, yy, ll, hh, nb, size in zip(xval, yval, lo, hi, nblocks, subset["block_size_m"].to_numpy(dtype=int)):
            color = palette[size]
            if np.isfinite(ll) and np.isfinite(hh):
                ax.errorbar(xx, yy, yerr=[[max(0.0, yy - ll)], [max(0.0, hh - yy)]], fmt="o", color=color,
                            capsize=3, markersize=5, linewidth=1.3)
            else:
                ax.scatter([xx], [yy], marker="x", color=color, s=36, zorder=3)
            ax.annotate(f"{nb} blocks", (xx, yy), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.0, color=color)
        ax.axhline(0, color="#666666", lw=0.8, linestyle="--")
        ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("Spatial block width (m)")
        ax.set_xticks(list(BLOCK_SIZES_M))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Association estimate / block-bootstrap 95% interval")
    fig.suptitle("Sensitivity of the Anza NDTI–SPM association to block size", fontsize=12, weight="bold")
    fig.text(0.5, -0.005,
             "Points marked × have too few spatial blocks for an interval. Bootstrap blocks are spatial resamples, not independent field replicates. "
             "TUR results are retained in the accompanying table.", ha="center", va="top", fontsize=8.0)
    fig.savefig(OUT / "Figure_Anza_20221110_spatial_block_size_sensitivity.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_Anza_20221110_spatial_block_size_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    notes = [
        "Spatial-block-size sensitivity for the single-day Anza 2022-11-10 depth-confounding analysis.",
        f"Input cells: {len(data)}; block widths tested (m): {', '.join(str(int(x)) for x in BLOCK_SIZES_M)}.",
        f"Intervals use {BOOTSTRAP_REPLICATES} whole-block resamples when at least {MIN_BLOCKS_FOR_INTERVAL} blocks are available.",
        "Point estimates are expected to remain constant across block widths; interval width and block count assess sensitivity to spatial resampling scale.",
        "With few blocks, no interval is reported. The study remains limited to one site, one survey date, and satellite-derived comparison products.",
        "This diagnostic does not establish a calibrated sediment-concentration relation or sediment transport.",
    ]
    (OUT / "Anza_20221110_spatial_block_size_sensitivity_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote block-size sensitivity table and figure to {OUT}")
    print("Review the number of blocks before interpreting interval changes; no new model was fitted.")


if __name__ == "__main__":
    main()
