from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
POINTS_FILE = (
    ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir"
    / "Anza_Agadir_bathy_points_candidate_utm29.csv"
)
RASTER_DIR = ROOT / "Data" / "Open_Bathymetry" / "Anza_Agadir"
OUT_DIR = ROOT / "Results" / "Bathymetry_Context"
RASTERS = [
    ("EMODnet 2022", RASTER_DIR / "EMODnet_2022_mean_depth_Anza.tif", "LAT"),
    ("EMODnet current", RASTER_DIR / "EMODnet_current_mean_depth_Anza.tif", "LAT"),
    ("GEBCO 2026", RASTER_DIR / "GEBCO_2026_Anza.tif", "nominal MSL*"),
]
TID_FILE = RASTER_DIR / "GEBCO_2026_TID_Anza.tif"
GEBCO_TID_LABELS = {
    0: "Land",
    16: "Optical depth",
    17: "Combined direct methods",
    40: "Predicted from satellite-derived gravity",
    44: "Bathymetric sounding constrained; gravity-guided grid",
}


def read_survey_points() -> dict[str, np.ndarray]:
    if not POINTS_FILE.is_file():
        raise FileNotFoundError(
            f"Run Scripts/11_check_survey_horizontal_crs.py first: {POINTS_FILE}"
        )
    with POINTS_FILE.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No soundings found in {POINTS_FILE}")
    required = {
        "longitude_wgs84_candidate_deg",
        "latitude_wgs84_candidate_deg",
        "z_source_m",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"Expected CSV columns: {sorted(required)}")
    return {
        "lon": np.asarray([float(row["longitude_wgs84_candidate_deg"]) for row in rows]),
        "lat": np.asarray([float(row["latitude_wgs84_candidate_deg"]) for row in rows]),
        "z": np.asarray([float(row["z_source_m"]) for row in rows]),
    }


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required open-bathymetry raster not found: {path}")
    with rasterio.open(path) as ds:
        values = ds.read(1, masked=True)
        valid = values.compressed()
        if valid.size == 0:
            raise RuntimeError(f"Raster contains no valid cells: {path}")
        lat0 = (ds.bounds.bottom + ds.bounds.top) / 2.0
        res_x_m = abs(ds.res[0]) * 111_320.0 * np.cos(np.deg2rad(lat0))
        res_y_m = abs(ds.res[1]) * 110_574.0
        tags = ds.tags()
        # GeoTIFFs often expose a tuple of band descriptions containing None.
        # Keep only actual strings so metadata inspection does not fail.
        description = " ".join(value for value in ds.descriptions if value)
        summary = {
            "name": path.name,
            "crs": str(ds.crs) if ds.crs else "not recorded",
            "width": int(ds.width),
            "height": int(ds.height),
            "resolution_x_m_approx": float(res_x_m),
            "resolution_y_m_approx": float(res_y_m),
            "valid_cells": int(valid.size),
            "minimum_value_m": float(np.min(valid)),
            "median_value_m": float(np.median(valid)),
            "maximum_value_m": float(np.max(valid)),
            "negative_cells": int(np.count_nonzero(valid < 0)),
            "positive_cells": int(np.count_nonzero(valid > 0)),
            "units": ds.units[0] if ds.units and ds.units[0] else tags.get("elevation#units", "not recorded"),
            "standard_name": tags.get("elevation#standard_name", "not recorded"),
            "bounds_wgs84": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
            "vertical_reference": "LAT (EMODnet DTM documentation; GeoTIFF tags do not identify the vertical reference)"
            if "EMODnet" in path.name
            else "Nominal mean sea level (GEBCO CF metadata); GEBCO cautions some shallow-water source cells may use another reference",
            "description": description or None,
            "metadata_tags": tags,
        }
        return values, summary


def raster_centers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as ds:
        x = ds.bounds.left + (np.arange(ds.width) + 0.5) * ds.res[0]
        y = ds.bounds.top - (np.arange(ds.height) + 0.5) * abs(ds.res[1])
    return x, y


def add_survey_outline(ax, lon: np.ndarray, lat: np.ndarray) -> None:
    x0, x1 = float(np.min(lon)), float(np.max(lon))
    y0, y1 = float(np.min(lat)), float(np.max(lat))
    ax.plot(
        [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
        color="#111827", linewidth=1.05, linestyle="--", label="Survey extent",
    )


def make_figure(points: dict[str, np.ndarray], rasters: dict[str, tuple[np.ndarray, dict]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), constrained_layout=True)

    # Panel A: measured point coverage in local hydrographic-zero coordinates.
    ax = axes[0]
    scatter = ax.scatter(
        points["lon"], points["lat"], c=points["z"], s=1.7,
        cmap="Blues_r", vmin=-25, vmax=0, linewidths=0, rasterized=True,
    )
    ax.set_xlim(float(points["lon"].min()) - 0.0015, float(points["lon"].max()) + 0.0015)
    ax.set_ylim(float(points["lat"].min()) - 0.0015, float(points["lat"].max()) + 0.0015)
    ax.set_title("A  Local survey soundings")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Survey Z (m; Agadir local ZH)")

    # Panels B and C show the regional context grids and survey footprint.
    plot_specs = [
        (axes[1], "EMODnet current", "B  EMODnet DTM (LAT)"),
        (axes[2], "GEBCO 2026", "C  GEBCO grid (nominal MSL*)"),
    ]
    for ax, key, title in plot_specs:
        raster, meta = rasters[key]
        x, y = raster_centers(RASTER_DIR / meta["name"])
        bathy = np.ma.masked_where(np.ma.getdata(raster) >= 0, raster)
        mesh = ax.pcolormesh(x, y, bathy, shading="auto", cmap="Blues_r", vmin=-45, vmax=0)
        ax.scatter(points["lon"], points["lat"], s=0.18, color="#F97316", alpha=0.45,
                   linewidths=0, rasterized=True, label="Local sounding footprint")
        add_survey_outline(ax, points["lon"], points["lat"])
        bounds = meta["bounds_wgs84"]
        pad_x = max((bounds[2] - bounds[0]) * 0.015, 0.001)
        pad_y = max((bounds[3] - bounds[1]) * 0.015, 0.001)
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
        ax.set_title(title)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.legend(loc="lower right", fontsize=7, frameon=True)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label("Grid elevation (m; product-specific datum)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "Figure_bathymetry_context_audit.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_bathymetry_context_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize_gebco_tid(points: dict[str, np.ndarray]) -> dict:
    """Summarize source classes by distinct GEBCO cells touched by survey points."""
    if not TID_FILE.is_file():
        raise FileNotFoundError(f"GEBCO Type Identifier raster not found: {TID_FILE}")
    with rasterio.open(TID_FILE) as ds:
        tid = ds.read(1, masked=True)
        full_values, full_counts = np.unique(tid.compressed(), return_counts=True)
        full_counts_by_code = {int(code): int(count) for code, count in zip(full_values, full_counts)}
        cells_touched: dict[tuple[int, int], int] = {}
        for lon, lat in zip(points["lon"], points["lat"]):
            row, col = ds.index(float(lon), float(lat))
            if 0 <= row < ds.height and 0 <= col < ds.width:
                value = tid[row, col]
                if not np.ma.is_masked(value):
                    cells_touched[(row, col)] = int(value)

    touched_counts: dict[int, int] = {}
    for code in cells_touched.values():
        touched_counts[code] = touched_counts.get(code, 0) + 1
    all_codes = sorted(set(full_counts_by_code) | set(touched_counts))
    records = [
        {
            "class_code": code,
            "source_class": GEBCO_TID_LABELS.get(code, f"Class {code}"),
            "cells_in_downloaded_subset": full_counts_by_code.get(code, 0),
            "distinct_cells_intersecting_survey": touched_counts.get(code, 0),
        }
        for code in all_codes
    ]
    with (OUT_DIR / "gebco_tid_source_classes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return {
        "file": TID_FILE.name,
        "distinct_grid_cells_intersecting_local_soundings": len(cells_touched),
        "class_counts_among_distinct_intersecting_cells": {
            GEBCO_TID_LABELS.get(code, f"Class {code}"): count
            for code, count in sorted(touched_counts.items())
        },
        "class_definitions": {str(k): v for k, v in GEBCO_TID_LABELS.items()},
        "caveat": "Counts are by distinct coarse GEBCO cells (about 400 x 460 m at Anza), not by individual 10 m survey point.",
    }


def compare_emodnet_editions(points: dict[str, np.ndarray]) -> dict:
    """Compare aligned EMODnet releases as a product-revision diagnostic, not seabed change."""
    old_path = RASTER_DIR / "EMODnet_2022_mean_depth_Anza.tif"
    current_path = RASTER_DIR / "EMODnet_current_mean_depth_Anza.tif"
    with rasterio.open(old_path) as old_ds, rasterio.open(current_path) as current_ds:
        if (
            old_ds.crs != current_ds.crs
            or old_ds.transform != current_ds.transform
            or old_ds.shape != current_ds.shape
        ):
            raise ValueError("EMODnet grids are not aligned; edition differencing stopped.")
        old = old_ds.read(1, masked=True)
        current = current_ds.read(1, masked=True)
        bounds = old_ds.bounds
        x = bounds.left + (np.arange(old_ds.width) + 0.5) * old_ds.res[0]
        y = bounds.top - (np.arange(old_ds.height) + 0.5) * abs(old_ds.res[1])

    valid = ~np.ma.getmaskarray(old) & ~np.ma.getmaskarray(current)
    old_values = np.ma.getdata(old)
    current_values = np.ma.getdata(current)
    difference = current_values - old_values
    water_both = valid & (old_values < 0) & (current_values < 0)
    water_mask_mismatch = valid & ((old_values < 0) != (current_values < 0))

    def metrics(values: np.ndarray, label: str) -> dict:
        values = np.asarray(values, dtype=np.float64)
        absolute = np.abs(values)
        return {
            "comparison": label,
            "n_cells": int(values.size),
            "mean_current_minus_2022_m": float(np.mean(values)),
            "median_current_minus_2022_m": float(np.median(values)),
            "mae_m": float(np.mean(absolute)),
            "rmse_m": float(np.sqrt(np.mean(values ** 2))),
            "p95_absolute_difference_m": float(np.quantile(absolute, 0.95)),
            "cells_abs_difference_over_1m": int(np.count_nonzero(absolute > 1)),
            "cells_abs_difference_over_2m": int(np.count_nonzero(absolute > 2)),
            "cells_abs_difference_over_5m": int(np.count_nonzero(absolute > 5)),
            "fraction_abs_difference_over_1m": float(np.mean(absolute > 1)),
            "fraction_abs_difference_over_2m": float(np.mean(absolute > 2)),
            "fraction_abs_difference_over_5m": float(np.mean(absolute > 5)),
        }

    summary_rows = [
        metrics(difference[valid], "All common valid cells"),
        metrics(difference[water_both], "Cells that are underwater in both editions"),
    ]
    for row in summary_rows:
        row["water_land_mask_mismatch_cells"] = int(np.count_nonzero(water_mask_mismatch))
        row["interpretation"] = "EMODnet edition/revision discrepancy, not observed temporal seabed change."
    with (OUT_DIR / "emodnet_edition_difference_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    water_difference = np.ma.masked_where(~water_both, difference)
    fig, ax = plt.subplots(figsize=(8.0, 6.1), constrained_layout=True)
    mesh = ax.pcolormesh(
        x, y, water_difference, shading="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12),
    )
    add_survey_outline(ax, points["lon"], points["lat"])
    ax.scatter(points["lon"], points["lat"], s=0.18, color="#111827", alpha=0.45,
               linewidths=0, rasterized=True, label="Local sounding footprint")
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.82)
    cbar.set_label("Current minus 2022 elevation (m; both products referenced to LAT)")
    fig.savefig(OUT_DIR / "Figure_EMODnet_edition_difference.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_EMODnet_edition_difference.pdf", bbox_inches="tight")
    plt.close(fig)
    (OUT_DIR / "EMODnet_edition_difference_caption_EN.txt").write_text(
        "EMODnet current-layer elevations minus the 2022 release for cells that are underwater in both editions. "
        "Both rasters have identical CRS, grid geometry and nominal LAT reference. The map describes differences "
        "between product editions, which may reflect changed source data or processing; it is not a measurement of "
        "seabed change between 2022 and the current release. The local survey footprint is shown for spatial context.\n",
        encoding="utf-8",
    )
    return {
        "summary": summary_rows,
        "water_land_mask_mismatch_cells": int(np.count_nonzero(water_mask_mismatch)),
        "figure": "Figure_EMODnet_edition_difference.png/.pdf",
        "interpretation_limit": "Edition differences may reflect revised source data, interpolation, or processing; they are not temporal seabed change.",
    }


def main() -> None:
    points = read_survey_points()
    raster_results = {}
    for label, path, vertical_reference in RASTERS:
        values, metadata = read_raster(path)
        metadata["product_label"] = label
        metadata["vertical_reference_short"] = vertical_reference
        raster_results[label] = (values, metadata)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tid_summary = summarize_gebco_tid(points)
    edition_summary = compare_emodnet_editions(points)
    records = []
    for label, _, _ in RASTERS:
        _, meta = raster_results[label]
        records.append({
            "product": label,
            "file": meta["name"],
            "grid_size": f"{meta['width']} × {meta['height']} cells",
            "approx_resolution_m": f"{meta['resolution_x_m_approx']:.0f} × {meta['resolution_y_m_approx']:.0f}",
            "vertical_reference": meta["vertical_reference"],
            "min_grid_value_m": f"{meta['minimum_value_m']:.3f}",
            "median_grid_value_m": f"{meta['median_value_m']:.3f}",
            "max_grid_value_m": f"{meta['maximum_value_m']:.3f}",
            "interpretation": "Regional context/boundary support only; not a replacement for the local survey and not fused to it.",
        })
    with (OUT_DIR / "open_bathymetry_suitability.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    report = {
        "status": "metadata and scale audit only; no resampling, vertical conversion, fusion, mesh, or hydrodynamic simulation",
        "survey_point_count": int(points["lon"].size),
        "survey_z_reference": "Hydrographic zero of Port of Agadir; source Z values retained unchanged",
        "survey_xy_transform_reported_accuracy_m": 7.0,
        "survey_median_nearest_neighbor_spacing_m": 9.99880672875299,
        "open_products": [meta for _, meta in raster_results.values()],
        "gebco_type_identifier_sample": tid_summary,
        "emodnet_edition_comparison": edition_summary,
        "interpretation": [
            "The local soundings are sampled at about 10 m, whereas EMODnet is about 100 x 115 m and GEBCO is about 400 x 460 m at Anza; those grids cannot resolve the local survey detail.",
            "EMODnet DTM elevations are referenced to LAT according to the official product documentation; the downloaded GeoTIFFs do not contain a vertical-reference tag.",
            "The GEBCO GeoTIFF metadata identifies elevations relative to mean sea level; GEBCO cautions that some shallow-water source cells may use another vertical reference.",
            "GEBCO TID source classes intersecting the survey are summarized by distinct raster cell in gebco_tid_source_classes.csv; they describe coarse grid provenance, not each local sounding.",
            "The EMODnet current and 2022 rasters have matching grids. Their differences are edition/revision discrepancies and must not be interpreted as seabed change.",
            "Port Agadir ZH, EMODnet LAT, and GEBCO MSL are distinct references here. Do not merge elevations until their local/date-specific vertical ties are established.",
            "The official 2024 IHO value for Agadir ZH relative to NGM is recorded separately; it is not applied to the 2022 survey.",
        ],
        "figure": "Figure_bathymetry_context_audit.png/.pdf; map extents only, not a datum-normalized comparison",
    }
    (OUT_DIR / "open_bathymetry_suitability.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "figure_caption_EN.txt").write_text(
        "Figure. Local Anza soundings and open bathymetry products shown at their native map scales. "
        "Panel A displays the survey Z values relative to the local hydrographic zero of Port Agadir. "
        "Panels B and C show EMODnet and GEBCO regional grids, respectively; the orange points and dashed outline "
        "locate the local survey. The grids have substantially coarser spatial resolution and use vertical references "
        "different from the local survey; GEBCO cautions that some shallow-water source cells may use another vertical "
        "reference than nominal MSL. The map is a coverage and scale audit only; elevations have not been "
        "converted, compared statistically, resampled, or merged.\n",
        encoding="utf-8",
    )
    make_figure(points, raster_results)
    print("Open-bathymetry context audit complete; source grids and survey XYZ were not changed.")
    print(f"  Survey soundings: {points['lon'].size:,}; median spacing about 10 m.")
    for label, _, _ in RASTERS:
        _, meta = raster_results[label]
        print(
            f"  {label}: {meta['width']} x {meta['height']} cells; "
            f"approximately {meta['resolution_x_m_approx']:.0f} x {meta['resolution_y_m_approx']:.0f} m; "
            f"vertical reference: {meta['vertical_reference_short']}."
        )
    print(f"  Outputs: {OUT_DIR}")
    water_metrics = edition_summary["summary"][1]
    print(
        "  EMODnet edition sensitivity over common underwater cells: "
        f"n={water_metrics['n_cells']:,}; mean current-minus-2022="
        f"{water_metrics['mean_current_minus_2022_m']:.2f} m; "
        f"RMSE={water_metrics['rmse_m']:.2f} m; p95 |difference|="
        f"{water_metrics['p95_absolute_difference_m']:.2f} m."
    )
    print("  This is a product-edition difference, not observed seabed change.")
    print("No vertical shift, raster fusion, interpolation, mesh, or circulation model was produced.")


if __name__ == "__main__":
    main()
