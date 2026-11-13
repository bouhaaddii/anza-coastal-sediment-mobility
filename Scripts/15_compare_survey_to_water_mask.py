from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
MASK_DIR = ROOT / "Results" / "Marine_Mask"
CLASS_RASTER = MASK_DIR / "S2_water_stability_class_10m.tif"
OCCURRENCE_RASTER = MASK_DIR / "S2_water_occurrence_10m.tif"
COUNT_RASTER = MASK_DIR / "S2_clear_observation_count_10m.tif"

CLASS_LABELS = {
    0: "Insufficient observations",
    1: "Stable non-water",
    2: "Variable water fringe",
    3: "Stable water",
}
DISPLAY_LABELS = {0: "Insufficient", 1: "Non-water", 2: "Fringe", 3: "Water"}
CLASS_COLORS = {0: "#d8dee9", 1: "#475569", 2: "#f59e0b", 3: "#0e7490"}


def read_survey() -> list[dict[str, str]]:
    if not SURVEY.is_file():
        raise FileNotFoundError(f"Candidate survey coordinates not found: {SURVEY}")
    with SURVEY.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("The candidate survey point file contains no rows.")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for path in (CLASS_RASTER, OCCURRENCE_RASTER, COUNT_RASTER):
        if not path.is_file():
            raise FileNotFoundError(f"Run Scripts/14_sentinel_water_occurrence.py first: {path}")

    survey = read_survey()
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(CLASS_RASTER) as cls_ds, rasterio.open(OCCURRENCE_RASTER) as occ_ds, rasterio.open(COUNT_RASTER) as count_ds:
        if cls_ds.crs is None or cls_ds.crs.to_epsg() != 32629:
            raise ValueError(f"Expected water-mask CRS EPSG:32629; found {cls_ds.crs}.")
        signature = (cls_ds.width, cls_ds.height, cls_ds.transform, cls_ds.crs)
        for name, ds in (("occurrence", occ_ds), ("clear-count", count_ds)):
            if (ds.width, ds.height, ds.transform, ds.crs) != signature:
                raise ValueError(f"The {name} raster is not aligned with the stability-class raster.")

        classes = cls_ds.read(1)
        occurrence = occ_ds.read(1)
        clear_count = count_ds.read(1)
        bounds = cls_ds.bounds
        transform = cls_ds.transform

    sampled: list[dict[str, object]] = []
    outside = 0
    for point in survey:
        x = float(point["easting_utm29n_candidate_m"])
        y = float(point["northing_utm29n_candidate_m"])
        z = float(point["z_source_m"])
        row, col = rasterio.transform.rowcol(transform, x, y)
        if row < 0 or col < 0 or row >= classes.shape[0] or col >= classes.shape[1]:
            outside += 1
            continue
        code = int(classes[row, col])
        occ_value = float(occurrence[row, col])
        sampled.append({
            "point_id": point["point_id"],
            "easting_utm29n_candidate_m": x,
            "northing_utm29n_candidate_m": y,
            "z_source_m_local_port_zero": z,
            "z_sign_group": "below_local_zero" if z < 0 else "at_or_above_local_zero",
            "water_class_code": code,
            "water_class_label": CLASS_LABELS.get(code, "Unknown"),
            "water_occurrence_fraction": occ_value if occ_value != -9999 else "",
            "clear_observation_count": int(clear_count[row, col]),
        })

    if not sampled:
        raise RuntimeError("No survey points intersect the Sentinel-2 stability raster.")

    sampled_path = MASK_DIR / "survey_points_sampled_water_mask.csv"
    write_csv(sampled_path, sampled)

    summary: list[dict[str, object]] = []
    for code in (0, 1, 2, 3):
        subset = [row for row in sampled if row["water_class_code"] == code]
        z_values = np.asarray([float(row["z_source_m_local_port_zero"]) for row in subset], dtype=float)
        occ_values = np.asarray(
            [float(row["water_occurrence_fraction"]) for row in subset if row["water_occurrence_fraction"] != ""],
            dtype=float,
        )
        n = len(subset)
        n_negative = sum(float(row["z_source_m_local_port_zero"]) < 0 for row in subset)
        n_nonnegative = n - n_negative
        summary.append({
            "water_class_code": code,
            "water_class_label": CLASS_LABELS[code],
            "survey_points": n,
            "share_of_sampled_points": n / len(sampled),
            "median_local_z_m": float(np.median(z_values)) if n else "",
            "z_p10_m": float(np.percentile(z_values, 10)) if n else "",
            "z_p90_m": float(np.percentile(z_values, 90)) if n else "",
            "points_below_local_zero": n_negative,
            "share_below_local_zero": n_negative / n if n else "",
            "points_at_or_above_local_zero": n_nonnegative,
            "median_water_occurrence": float(np.median(occ_values)) if len(occ_values) else "",
        })
    summary_path = MASK_DIR / "survey_water_mask_class_summary.csv"
    write_csv(summary_path, summary)

    cross_tab: list[dict[str, object]] = []
    for code in (0, 1, 2, 3):
        for sign_group in ("below_local_zero", "at_or_above_local_zero"):
            count = sum(row["water_class_code"] == code and row["z_sign_group"] == sign_group for row in sampled)
            cross_tab.append({
                "water_class_code": code,
                "water_class_label": CLASS_LABELS[code],
                "z_sign_group": sign_group,
                "point_count": count,
            })
    cross_path = MASK_DIR / "survey_water_mask_cross_tab.csv"
    write_csv(cross_path, cross_tab)

    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    plot_extent = (left, right, bottom, top)
    x = np.asarray([float(row["easting_utm29n_candidate_m"]) for row in sampled])
    y = np.asarray([float(row["northing_utm29n_candidate_m"]) for row in sampled])
    z = np.asarray([float(row["z_source_m_local_port_zero"]) for row in sampled])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), constrained_layout=True)

    ax = axes[0]
    occurrence_cmap = plt.get_cmap("viridis").copy()
    occurrence_cmap.set_bad("#e5e7eb")
    occurrence_plot = np.ma.masked_equal(occurrence, -9999.0)
    im = ax.imshow(occurrence_plot, extent=plot_extent, origin="upper", vmin=0, vmax=1, cmap=occurrence_cmap, interpolation="nearest")
    z_norm = TwoSlopeNorm(vmin=float(z.min()), vcenter=0.0, vmax=float(z.max()))
    points = ax.scatter(x, y, c=z, s=1.4, alpha=0.58, cmap="RdBu_r", norm=z_norm, linewidths=0, rasterized=True)
    ax.set_title("A  Local survey Z over water occurrence")
    fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label="Fraction of clear observations classified as water")
    fig.colorbar(points, ax=ax, shrink=0.82, pad=0.11, label="Survey Z (m; local port zero)")

    ax = axes[1]
    active_codes = [code for code in (0, 1, 2, 3) if any(row["water_class_code"] == code for row in sampled)]
    groups = [
        [float(row["z_source_m_local_port_zero"]) for row in sampled if row["water_class_code"] == code]
        for code in active_codes
    ]
    positions = list(range(1, len(active_codes) + 1))
    box = ax.boxplot(groups, positions=positions, widths=0.58, patch_artist=True, showfliers=False, manage_ticks=False)
    for patch, code in zip(box["boxes"], active_codes):
        patch.set_facecolor(CLASS_COLORS[code])
        patch.set_alpha(0.78)
        patch.set_edgecolor("#243247")
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.3)
    ax.axhline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_xticks(
        positions,
        [f"{DISPLAY_LABELS[code]}\n(n={len(groups[i]):,})" for i, code in enumerate(active_codes)],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel("Survey Z (m; local port zero)")
    ax.set_xlabel("Sentinel-2 class at sounding locations")
    ax.set_title("B  Survey-Z distributions by sampled class")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.5, alpha=0.7)

    axes[0].set_xlim(left, right)
    axes[0].set_ylim(bottom, top)
    axes[0].set_aspect("equal")
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[0].set_xlabel("Easting (m; UTM 29N)")
    axes[0].set_ylabel("Northing (m; UTM 29N)")

    figure_path = MASK_DIR / "Figure_survey_water_mask_consistency.png"
    fig.savefig(figure_path, dpi=350, bbox_inches="tight")
    fig.savefig(MASK_DIR / "Figure_survey_water_mask_consistency.pdf", bbox_inches="tight")
    plt.close(fig)
    (MASK_DIR / "Figure_survey_water_mask_consistency_caption_EN.txt").write_text(
        "Figure. Spatial and distributional comparison of the local Anza survey with the multi-scene Sentinel-2 SCL water-occurrence diagnostic. "
        "(A) Survey soundings are plotted over the fraction of clear observations classified as water; point colors show Z relative to the local "
        "Agadir port hydrographic zero. (B) Boxplots summarize survey Z at locations sampled from each SCL stability class; the dashed line marks "
        "the local zero. The image observations span November 2022–March 2023 and are not contemporaneous with the survey. This descriptive overlay "
        "is not a shoreline-accuracy or vertical-datum validation.\n",
        encoding="utf-8",
    )

    notes = (
        "Descriptive spatial consistency diagnostic only. The Sentinel-2 SCL classes are derived from November 2022 to March 2023; "
        "the local bathymetric/topographic survey was acquired at a different time. Survey Z is relative to the local Agadir port "
        "hydrographic zero; no vertical transformation to instantaneous sea level, LAT, or MSL was applied. The projected survey "
        "coordinates use a candidate EPSG:26192 to UTM 29N transformation whose reported operation accuracy is 7 m. Therefore, "
        "class-versus-Z counts are not shoreline validation, datum validation, or a confusion-matrix accuracy assessment. "
        "Use the overlay to identify gross spatial inconsistencies and boundary-sensitive points only.\n"
    )
    (MASK_DIR / "survey_water_mask_interpretation_limits_EN.txt").write_text(notes, encoding="utf-8")

    print("Survey-to-mask overlay complete; source XYZ and water-mask rasters were not changed.")
    print(f"  Survey points in raster: {len(sampled):,}; outside: {outside:,}")
    print(f"  Class summary: {summary_path}")
    print(f"  Point samples: {sampled_path}")
    print(f"  Figure: {figure_path}")
    print("This is not a quantitative shoreline, datum, or SCL accuracy validation.")


if __name__ == "__main__":
    main()
