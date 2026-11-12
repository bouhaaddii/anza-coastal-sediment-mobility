from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
CRS_REPORT = ROOT / "Results" / "Horizontal_CRS" / "candidate_horizontal_crs_diagnostic.json"
OUT_DIR = ROOT / "Results" / "Bathymetry_QC"
SUMMARY = OUT_DIR / "Anza_survey_sampling_summary.csv"
FIGURE_PNG = OUT_DIR / "Figure_Anza_survey_sampling.png"
FIGURE_PDF = OUT_DIR / "Figure_Anza_survey_sampling.pdf"
NOTES = OUT_DIR / "interpretation_notes_EN.txt"


def load_points() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    if not POINTS.is_file():
        raise FileNotFoundError(f"Run Scripts/11_check_survey_horizontal_crs.py first: {POINTS}")
    with POINTS.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No transformed candidate points found in {POINTS}")
    east = np.asarray([float(row["easting_utm29n_candidate_m"]) for row in rows])
    north = np.asarray([float(row["northing_utm29n_candidate_m"]) for row in rows])
    z = np.asarray([float(row["z_source_m"]) for row in rows])
    return rows, east, north, z


def bounded_histogram_bins(values: np.ndarray, max_bins: int = 120) -> int:
    """Use a bounded Freedman–Diaconis bin count for stable, low-memory plots."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 1

    value_range = float(np.max(values) - np.min(values))
    if not np.isfinite(value_range) or value_range <= 0:
        return 1

    q25, q75 = np.percentile(values, [25, 75])
    fd_width = 2.0 * float(q75 - q25) / np.cbrt(values.size)
    if not np.isfinite(fd_width) or fd_width <= 0:
        estimated_bins = int(np.ceil(np.sqrt(values.size)))
    else:
        estimated_bins = int(np.ceil(value_range / fd_width))

    return max(30, min(max_bins, estimated_bins))


def main() -> None:
    rows, east, north, z = load_points()
    if not np.isfinite(np.column_stack((east, north, z))).all():
        raise ValueError("Candidate coordinates or Z contain non-finite values.")

    distances, _ = cKDTree(np.column_stack((east, north))).query(
        np.column_stack((east, north)), k=2
    )
    nearest = distances[:, 1]
    q = np.percentile(nearest, [5, 25, 50, 75, 90, 95, 99, 100])
    crs_report = json.loads(CRS_REPORT.read_text(encoding="utf-8")) if CRS_REPORT.is_file() else {}
    reported_accuracy = crs_report.get("candidate_transformer_reported_accuracy_m", "")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = [
        ("point_count", len(rows), "count"),
        ("nearest_neighbor_min", float(np.min(nearest)), "m"),
        ("nearest_neighbor_p05", float(q[0]), "m"),
        ("nearest_neighbor_p25", float(q[1]), "m"),
        ("nearest_neighbor_median", float(q[2]), "m"),
        ("nearest_neighbor_p75", float(q[3]), "m"),
        ("nearest_neighbor_p90", float(q[4]), "m"),
        ("nearest_neighbor_p95", float(q[5]), "m"),
        ("nearest_neighbor_p99", float(q[6]), "m"),
        ("nearest_neighbor_max", float(q[7]), "m"),
        ("z_min", float(np.min(z)), "m relative to Port of Agadir hydrographic zero"),
        ("z_median", float(np.median(z)), "m relative to Port of Agadir hydrographic zero"),
        ("z_max", float(np.max(z)), "m relative to Port of Agadir hydrographic zero"),
        ("candidate_transformer_reported_accuracy", reported_accuracy, "m"),
    ]
    with SUMMARY.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(stats)

    bathy_cmap = LinearSegmentedColormap.from_list(
        "coastal_depth", ["#123B5D", "#246B83", "#61AFA7", "#D9C995", "#F6F1DF"]
    )
    norm = TwoSlopeNorm(vmin=float(np.min(z)), vcenter=0.0, vmax=float(np.max(z)))
    spacing_cap = float(np.percentile(nearest, 99))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    points = ax.scatter(east / 1000, north / 1000, c=z, s=1.2, cmap=bathy_cmap, norm=norm,
                        linewidths=0, rasterized=True)
    ax.set_title("Survey points coloured by local Z", fontweight="bold")
    ax.set_xlabel("Easting UTM 29N (km)")
    ax.set_ylabel("Northing UTM 29N (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#D9E0E3", linewidth=0.6, alpha=0.65)
    cb = fig.colorbar(points, ax=ax, shrink=0.86, pad=0.02)
    cb.set_label("Z (m; Port of Agadir hydrographic zero)")

    ax = axes[0, 1]
    spacing = ax.scatter(east / 1000, north / 1000, c=np.minimum(nearest, spacing_cap),
                         s=1.2, cmap="magma", vmin=0, vmax=spacing_cap,
                         linewidths=0, rasterized=True)
    ax.set_title("Distance to nearest other sounding", fontweight="bold")
    ax.set_xlabel("Easting UTM 29N (km)")
    ax.set_ylabel("Northing UTM 29N (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#D9E0E3", linewidth=0.6, alpha=0.65)
    cb = fig.colorbar(spacing, ax=ax, shrink=0.86, pad=0.02, extend="max")
    cb.set_label("Nearest XY neighbour distance (m; clipped at P99)")

    ax = axes[1, 0]
    ax.hist(nearest, bins=bounded_histogram_bins(nearest), color="#2A7F83",
            edgecolor="white", linewidth=0.35)
    ax.axvline(np.median(nearest), color="#D97732", linestyle="--", linewidth=1.3,
               label=f"Median = {np.median(nearest):.2f} m")
    ax.axvline(np.percentile(nearest, 95), color="#5F477A", linestyle=":", linewidth=1.3,
               label=f"P95 = {np.percentile(nearest, 95):.2f} m")
    ax.set_title("Nearest-neighbour spacing distribution", fontweight="bold")
    ax.set_xlabel("Distance to nearest other sounding (m)")
    ax.set_ylabel("Number of points")
    ax.grid(axis="y", color="#D9E0E3", linewidth=0.6)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.hist(z, bins=bounded_histogram_bins(z), color="#4C7896",
            edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="#D97732", linestyle="--", linewidth=1.3,
               label="Local hydrographic zero")
    ax.set_title("Survey Z distribution", fontweight="bold")
    ax.set_xlabel("Z (m; Port of Agadir hydrographic zero)")
    ax.set_ylabel("Number of points")
    ax.grid(axis="y", color="#D9E0E3", linewidth=0.6)
    ax.legend(frameon=False)

    fig.suptitle("Anza bathymetric survey coverage and sampling geometry", fontsize=14, fontweight="bold")
    fig.savefig(FIGURE_PNG, dpi=350, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)

    note_text = [
        "Interpretation notes — Anza survey sampling diagnostic",
        "",
        f"The diagnostic uses {len(rows):,} source soundings and plan-confirmed XY coordinates transformed from EPSG:26192 to WGS 84 / UTM zone 29N.",
        f"Nearest-neighbour spacing is a within-survey point-density diagnostic: median {np.median(nearest):.2f} m; P95 {np.percentile(nearest, 95):.2f} m.",
        "Nearest-neighbour spacing often measures along-track spacing and does not by itself estimate cross-track line spacing or justify a raster cell size.",
        "The map is a point display, not an interpolated bathymetric surface. No points were removed, no interpolation was run, and no mesh was created.",
        "Histogram bin counts follow the Freedman–Diaconis estimate and are capped at 120 bins; every valid sounding remains included.",
        "Z values are copied unchanged and remain relative to the hydrographic zero of the Port of Agadir.",
        "The registered horizontal transformation reports 7 m accuracy; this is not a local control-point residual. Check alignment against DWG control points before claiming sub-10 m absolute co-registration with satellite pixels.",
    ]
    NOTES.write_text("\n".join(note_text) + "\n", encoding="utf-8")

    print("Survey sampling diagnostic complete; no interpolation or mesh was created.")
    print(f"  Points: {len(rows):,}")
    print(f"  Nearest-neighbour spacing median: {np.median(nearest):.2f} m; P95: {np.percentile(nearest, 95):.2f} m")
    print(f"  Figure: {FIGURE_PNG}")
    print(f"  Summary: {SUMMARY}")
    print(f"  Notes: {NOTES}")
    print("Nearest-neighbour spacing alone must not be treated as a recommended mesh or raster resolution.")


if __name__ == "__main__":
    main()
