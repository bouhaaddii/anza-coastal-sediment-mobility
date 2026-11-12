from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
PROCESSED = ROOT / "Data" / "Sentinel2" / "Processed"
PROXY_DIR = ROOT / "Results" / "Sediment_Proxy"
MARINE_DIR = ROOT / "Results" / "Marine_Mask"
STABILITY = MARINE_DIR / "S2_water_stability_class_10m.tif"
SUMMARY = PROXY_DIR / "scene_relative_turbidity_summary.csv"
OUT_DIR = PROXY_DIR / "Spatial_Support"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit per-scene valid-pixel coverage and NDTI statistics on fixed water domains."
    )
    parser.add_argument(
        "--minimum-common-scenes", type=int, default=8,
        help="Minimum number of valid NDTI scene observations for the common-support mask (default: 8).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: np.ndarray) -> tuple[int, float | str, float | str, float | str]:
    if values.size == 0:
        return 0, "", "", ""
    q25, q50, q75 = np.percentile(values.astype(np.float64), [25, 50, 75])
    return int(values.size), float(q50), float(q25), float(q75)


def main() -> None:
    args = parse_args()
    if not STABILITY.is_file():
        raise FileNotFoundError(f"Run Scripts/14_sentinel_water_occurrence.py first: {STABILITY}")

    selected = [
        row for row in read_csv(SELECTION)
        if row.get("download_selected", "").strip().upper() == "YES"
        and row.get("product_name", "").startswith(("S2A_MSIL2A_2022", "S2B_MSIL2A_2022", "S2A_MSIL2A_2023", "S2B_MSIL2A_2023"))
    ]
    if not selected:
        raise RuntimeError("No selected 2022–2023 Sentinel-2 products were found.")
    if not (1 <= args.minimum_common_scenes <= len(selected)):
        raise ValueError("--minimum-common-scenes must be between 1 and the number of selected scenes.")

    summaries = {row.get("product_name", ""): row for row in read_csv(SUMMARY)}
    selected = sorted(selected, key=lambda row: row.get("acquisition_utc", ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Stage 6: audit NDTI spatial support across selected scenes.")
    print(f"  Scenes: {len(selected)}; common-support threshold: at least {args.minimum_common_scenes} valid observations.")
    print("  Fixed domains: persistent SCL-water core and pixels valid in the selected scenes at the stated frequency.")
    print("  This is a sampling-support diagnostic; no sediment concentration or causal effect is estimated.")

    with rasterio.open(STABILITY) as stability_ds:
        stability = stability_ds.read(1)
        profile = stability_ds.profile.copy()
        transform = stability_ds.transform
        crs = stability_ds.crs
        bounds = stability_ds.bounds
    if crs is None or crs.to_epsg() != 32629:
        raise ValueError(f"Expected EPSG:32629 analysis grid; found {crs}.")
    stable_water_core = stability == 3
    stable_core_count = int(stable_water_core.sum())
    if stable_core_count == 0:
        raise RuntimeError("The persistent SCL-water core contains no pixels.")

    valid_layers: list[np.ndarray] = []
    ndti_layers: list[np.ndarray] = []
    qa_water_layers: list[np.ndarray] = []
    clear_pixel_counts: list[int] = []
    layer_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        product = row["product_name"]
        qa_path = PROCESSED / f"{product}_qa_10m.tif"
        ndti_path = PROXY_DIR / f"{product}_NDTI_relative_proxy.tif"
        if product not in summaries:
            raise FileNotFoundError(f"Scene summary has no row for selected product: {product}")
        if not qa_path.is_file() or not ndti_path.is_file():
            raise FileNotFoundError(f"Missing QA or NDTI raster for {product}")

        with rasterio.open(qa_path) as qa_ds, rasterio.open(ndti_path) as ndti_ds:
            signature = (qa_ds.width, qa_ds.height, qa_ds.transform, qa_ds.crs)
            expected = (profile["width"], profile["height"], transform, crs)
            if signature != expected or (ndti_ds.width, ndti_ds.height, ndti_ds.transform, ndti_ds.crs) != expected:
                raise ValueError(f"Raster grid mismatch for {product}; stopped before support analysis.")
            clear = qa_ds.read(1) == 1
            water = qa_ds.read(2) == 1
            clear_water = clear & water
            ndti = ndti_ds.read(1).astype(np.float32, copy=False)
            valid = np.isfinite(ndti)
            if np.any(valid & ~clear_water):
                raise ValueError(f"NDTI raster has valid pixels outside the clear SCL-water mask: {product}")
            valid_layers.append(valid)
            ndti_layers.append(ndti)
            qa_water_layers.append(clear_water)
            clear_pixel_counts.append(int(clear.sum()))
            layer_rows.append(row)
        print(f"  Loaded {index}/{len(selected)}: {product}")

    valid_stack = np.stack(valid_layers, axis=0)
    support_count = valid_stack.sum(axis=0).astype(np.uint8)
    common_support = support_count >= args.minimum_common_scenes
    common_count = int(common_support.sum())
    if common_count == 0:
        raise RuntimeError(
            f"No pixel reaches the common-support threshold ({args.minimum_common_scenes} scenes). "
            "Rerun with a lower threshold only as an explicitly labelled sensitivity analysis."
        )

    support_profile = profile.copy()
    support_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(OUT_DIR / "S2_valid_NDTI_observation_count_10m.tif", "w", **support_profile) as dst:
        dst.write(support_count, 1)
        dst.set_band_description(1, "Number of selected scenes with valid clear-water NDTI")
        dst.update_tags(scene_count=len(selected), source="selected Sentinel-2 NDTI rasters")

    common_profile = profile.copy()
    common_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(OUT_DIR / f"S2_common_support_min_{args.minimum_common_scenes}_of_{len(selected)}_10m.tif", "w", **common_profile) as dst:
        dst.write(common_support.astype(np.uint8), 1)
        dst.set_band_description(1, "Fixed common support meeting minimum valid-scene count")
        dst.update_tags(minimum_valid_scenes=args.minimum_common_scenes, total_selected_scenes=len(selected))

    output_rows: list[dict[str, object]] = []
    for i, row in enumerate(layer_rows):
        product = row["product_name"]
        valid = valid_layers[i]
        ndti = ndti_layers[i]
        qa_water = qa_water_layers[i]
        all_values = ndti[valid]
        core_valid = valid & stable_water_core
        common_valid = valid & common_support
        core_values = ndti[core_valid]
        common_values = ndti[common_valid]
        all_n, all_median, all_q25, all_q75 = quantiles(all_values)
        core_n, core_median, core_q25, core_q75 = quantiles(core_values)
        common_n, common_median, common_q25, common_q75 = quantiles(common_values)
        output_rows.append({
            "product_name": product,
            "acquisition_utc": row.get("acquisition_utc", ""),
            "clear_pixel_count": clear_pixel_counts[i],
            "clear_scl_water_pixel_count": int(qa_water.sum()),
            "valid_ndti_pixel_count": all_n,
            "clear_water_fraction_of_clear_pixels": float(qa_water.sum() / clear_pixel_counts[i]) if clear_pixel_counts[i] else "",
            "stable_water_core_pixel_count": stable_core_count,
            "valid_pixels_in_stable_water_core": core_n,
            "stable_water_core_coverage_fraction": core_n / stable_core_count,
            "fixed_common_support_pixel_count": common_count,
            "valid_pixels_in_fixed_common_support": common_n,
            "fixed_common_support_coverage_fraction": common_n / common_count,
            "all_scene_water_median_ndti": all_median,
            "all_scene_water_ndti_p25": all_q25,
            "all_scene_water_ndti_p75": all_q75,
            "stable_core_median_ndti": core_median,
            "stable_core_ndti_p25": core_q25,
            "stable_core_ndti_p75": core_q75,
            "common_support_median_ndti": common_median,
            "common_support_ndti_p25": common_q25,
            "common_support_ndti_p75": common_q75,
        })

    summary_path = OUT_DIR / "ndti_spatial_support_by_scene.csv"
    write_csv(summary_path, output_rows)

    count_colors = ListedColormap(["#d8dee9", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a", "#0f766e", "#14b8a6", "#86efac", "#facc15", "#fb923c", "#dc2626", "#7f1d1d"])
    count_norm = BoundaryNorm(np.arange(-0.5, len(selected) + 1.5, 1), count_colors.N)
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    extent = (left, right, bottom, top)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)
    fig.suptitle("Sentinel-2 NDTI spatial-support audit", fontsize=14, fontweight="bold")

    ax = axes[0]
    image = ax.imshow(support_count, extent=extent, origin="upper", cmap=count_colors, norm=count_norm, interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02, ticks=np.arange(0, len(selected) + 1, 2))
    cbar.set_label("Scenes with valid clear-water NDTI")
    ax.set_title("A  Repeated valid-observation support")
    ax.set_xlabel("Easting (m; WGS 84 / UTM 29N)")
    ax.set_ylabel("Northing (m; WGS 84 / UTM 29N)")
    ax.set_aspect("equal")

    survey_path = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
    with survey_path.open(newline="", encoding="utf-8-sig") as stream:
        survey_points = list(csv.DictReader(stream))
    sx = np.asarray([float(point["easting_utm29n_candidate_m"]) for point in survey_points])
    sy = np.asarray([float(point["northing_utm29n_candidate_m"]) for point in survey_points])
    survey_bounds = (float(sx.min()), float(sy.min()), float(sx.max()), float(sy.max()))
    ax.add_patch(Rectangle((survey_bounds[0], survey_bounds[1]), survey_bounds[2] - survey_bounds[0], survey_bounds[3] - survey_bounds[1],
                           fill=False, edgecolor="white", linewidth=1.0, linestyle="--"))

    ax = axes[1]
    coverage = [float(row["stable_water_core_coverage_fraction"]) for row in output_rows]
    dates = [datetime.fromisoformat(row["acquisition_utc"].replace("Z", "+00:00")) for row in output_rows]
    labels = [date.strftime("%d %b") for date in dates]
    colors = ["#dc2626" if value < 0.5 else "#0e7490" for value in coverage]
    bars = ax.bar(np.arange(len(output_rows)), np.asarray(coverage) * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(80, color="#475569", linestyle="--", linewidth=0.9, label="80% guide")
    ax.set_xticks(np.arange(len(output_rows)), labels, rotation=45, ha="right")
    ax.set_ylabel("Valid NDTI pixels / persistent-water core (%)")
    ax.set_title("B  Per-scene coverage of the fixed water core")
    ax.set_ylim(0, max(100, max(coverage) * 105))
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, loc="lower right")
    for bar, value in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{value:.0%}", ha="center", va="bottom", fontsize=7, rotation=90)

    figure_path = OUT_DIR / "Figure_NDTI_spatial_support_audit.png"
    fig.savefig(figure_path, dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_NDTI_spatial_support_audit.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "selected_scene_count": len(selected),
        "persistent_water_core_definition": "SCL water occurrence stability class 3 from Script 14 (>=80% occurrence and >=5 clear observations)",
        "common_support_definition": f"valid NDTI in at least {args.minimum_common_scenes} of {len(selected)} selected scenes",
        "stable_water_core_pixel_count": stable_core_count,
        "common_support_pixel_count": common_count,
        "common_support_fraction_of_persistent_water_core": common_count / stable_core_count,
        "interpretation": [
            "Different per-scene NDTI medians can reflect changing valid-pixel geography as well as optical conditions.",
            "The persistent-water core is a fixed spatial domain, but individual dates still have missing pixels within it.",
            "The common-support mask restricts summaries to locations observed in at least the configured number of scenes; it does not correct remaining missingness.",
            "NDTI remains a unitless, uncalibrated optical proxy; no suspended-sediment concentration, transport rate, causality, or statistical significance is inferred.",
        ],
        "outputs": [
            "ndti_spatial_support_by_scene.csv",
            "S2_valid_NDTI_observation_count_10m.tif",
            f"S2_common_support_min_{args.minimum_common_scenes}_of_{len(selected)}_10m.tif",
            "Figure_NDTI_spatial_support_audit.png/.pdf",
        ],
    }
    (OUT_DIR / "ndti_spatial_support_settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "Figure_NDTI_spatial_support_caption_EN.txt").write_text(
        "Figure. Spatial availability of valid Sentinel-2 NDTI observations across the selected acquisition dates. "
        "(A) Number of scenes with valid clear-water NDTI per 10 m pixel; the dashed outline marks the local bathymetric survey extent. "
        "(B) Per-scene percentage of the fixed persistent-water core with valid NDTI. Red bars flag coverage below 50% as a visual screening threshold, "
        "not a universal quality criterion. The common-support outputs restrict summaries to pixels valid in at least the stated number of scenes; "
        "they do not eliminate residual missing-data or optical-proxy limitations.\n",
        encoding="utf-8",
    )

    print("Spatial-support audit complete; reflectance, QA masks, and bathymetry were not changed.")
    print(f"  Persistent-water core: {stable_core_count:,} pixels")
    print(f"  Common-support pixels (>= {args.minimum_common_scenes}/{len(selected)} scenes): {common_count:,} ({common_count / stable_core_count:.1%} of the core)")
    print(f"  Per-scene summary: {summary_path}")
    print(f"  Figure: {figure_path}")
    print("Inspect coverage before comparing date-to-date NDTI medians; no SSC retrieval or sediment-transport rate was computed.")


if __name__ == "__main__":
    main()
