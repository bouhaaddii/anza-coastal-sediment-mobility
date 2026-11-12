from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Sentinel2" / "Processed"
SELECTION = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
SURVEY = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
OUT_DIR = ROOT / "Results" / "Marine_Mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an exploratory multi-scene Sentinel-2 water-occurrence diagnostic."
    )
    parser.add_argument(
        "--minimum-clear-observations", type=int, default=5,
        help="Minimum number of clear observations required for a stability class (default: 5).",
    )
    parser.add_argument(
        "--stable-land-threshold", type=float, default=0.20,
        help="Water-occurrence fraction at or below which a pixel is labelled stable non-water (default: 0.20).",
    )
    parser.add_argument(
        "--stable-water-threshold", type=float, default=0.80,
        help="Water-occurrence fraction at or above which a pixel is labelled stable water (default: 0.80).",
    )
    return parser.parse_args()


def selected_products() -> list[dict[str, str]]:
    if not SELECTION.is_file():
        raise FileNotFoundError(f"Sentinel-2 selection table not found: {SELECTION}")
    with SELECTION.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("download_selected", "").strip().upper() == "YES"]
    rows = [row for row in rows if row.get("product_name", "").startswith(("S2A_MSIL2A_2022", "S2B_MSIL2A_2022", "S2A_MSIL2A_2023", "S2B_MSIL2A_2023"))]
    if not rows:
        raise RuntimeError("No selected 2022–2023 Sentinel-2 scenes were found in download_selection.csv.")
    return sorted(rows, key=lambda row: row.get("acquisition_utc", ""))


def survey_extent_utm() -> tuple[float, float, float, float]:
    if not SURVEY.is_file():
        raise FileNotFoundError(f"Run Scripts/11_check_survey_horizontal_crs.py first: {SURVEY}")
    with SURVEY.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("No candidate survey points found.")
    x = np.asarray([float(row["easting_utm29n_candidate_m"]) for row in rows])
    y = np.asarray([float(row["northing_utm29n_candidate_m"]) for row in rows])
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def main() -> None:
    args = parse_args()
    if args.minimum_clear_observations < 1:
        raise ValueError("--minimum-clear-observations must be at least 1.")
    if not (0 <= args.stable_land_threshold < args.stable_water_threshold <= 1):
        raise ValueError("Thresholds must satisfy 0 <= stable-land < stable-water <= 1.")

    products = selected_products()
    clear_count = None
    water_count = None
    profile = None
    used_scenes = []
    missing_scenes = []
    grid_signature = None

    print("Stage 5: multi-scene Sentinel-2 water-occurrence diagnostic.")
    print("  Water occurrence = clear observations labelled as SCL water / all clear observations.")
    print("  This is a candidate marine mask, not a surveyed coastline or a hydrodynamic boundary.")
    for row in products:
        name = row["product_name"]
        qa_path = PROCESSED / f"{name}_qa_10m.tif"
        if not qa_path.is_file():
            missing_scenes.append(name)
            print(f"  Missing QA mask; skipped: {name}")
            continue
        with rasterio.open(qa_path) as ds:
            if ds.count < 2:
                raise ValueError(f"Expected clear and SCL-water mask bands in {qa_path.name}; found {ds.count}.")
            signature = (ds.width, ds.height, ds.crs, ds.transform)
            if grid_signature is None:
                grid_signature = signature
                profile = ds.profile.copy()
                clear_count = np.zeros((ds.height, ds.width), dtype=np.uint16)
                water_count = np.zeros((ds.height, ds.width), dtype=np.uint16)
            elif signature != grid_signature:
                raise ValueError(f"QA grids do not match; stopped before compositing: {qa_path.name}")
            clear = ds.read(1) == 1
            water = (ds.read(2) == 1) & clear
            clear_count += clear.astype(np.uint16)
            water_count += water.astype(np.uint16)
            used_scenes.append({
                "product_name": name,
                "acquisition_utc": row.get("acquisition_utc", ""),
                "clear_pixel_count": int(clear.sum()),
                "clear_water_pixel_count": int(water.sum()),
                "qa_file": qa_path.name,
            })
            print(f"  Included {len(used_scenes)}/{len(products)}: {name} ({int(water.sum()):,} clear-water pixels)")

    if not used_scenes or clear_count is None or water_count is None or profile is None:
        raise RuntimeError("No selected scene had a usable two-band clear/SCL-water raster.")

    has_observation = clear_count > 0
    occurrence = np.full(clear_count.shape, np.nan, dtype=np.float32)
    occurrence[has_observation] = water_count[has_observation] / clear_count[has_observation]
    sufficiently_observed = clear_count >= args.minimum_clear_observations

    # 0: insufficient clear observations; 1: stable non-water; 2: variable fringe;
    # 3: stable water. These labels are exploratory and depend on the stated thresholds.
    stability = np.zeros(clear_count.shape, dtype=np.uint8)
    stability[sufficiently_observed & (occurrence <= args.stable_land_threshold)] = 1
    stability[
        sufficiently_observed
        & (occurrence > args.stable_land_threshold)
        & (occurrence < args.stable_water_threshold)
    ] = 2
    stability[sufficiently_observed & (occurrence >= args.stable_water_threshold)] = 3

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    occurrence_profile = profile.copy()
    occurrence_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)
    occurrence_saved = np.where(np.isfinite(occurrence), occurrence, -9999.0).astype(np.float32)
    with rasterio.open(OUT_DIR / "S2_water_occurrence_10m.tif", "w", **occurrence_profile) as dst:
        dst.write(occurrence_saved, 1)
        dst.set_band_description(1, "Fraction of clear observations labelled as SCL water")
        dst.update_tags(
            formula="clear SCL-water observations / all clear observations",
            source_scenes=str(len(used_scenes)),
            note="Exploratory occurrence surface; not a charted coastline or tide-normalized shoreline.",
        )

    count_profile = profile.copy()
    count_profile.update(count=1, dtype="uint16", nodata=0, compress="deflate", predictor=2)
    with rasterio.open(OUT_DIR / "S2_clear_observation_count_10m.tif", "w", **count_profile) as dst:
        dst.write(clear_count.astype(np.uint16), 1)
        dst.set_band_description(1, "Number of clear Sentinel-2 observations")

    class_profile = profile.copy()
    class_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(OUT_DIR / "S2_water_stability_class_10m.tif", "w", **class_profile) as dst:
        dst.write(stability, 1)
        dst.set_band_description(1, "Exploratory water-stability class")
        dst.update_tags(
            class_0="insufficient_clear_observations",
            class_1=f"stable_non_water_water_fraction_le_{args.stable_land_threshold}",
            class_2=f"variable_fringe_{args.stable_land_threshold}_to_{args.stable_water_threshold}",
            class_3=f"stable_water_water_fraction_ge_{args.stable_water_threshold}",
            minimum_clear_observations=args.minimum_clear_observations,
        )

    rows = []
    total_obs = int(sufficiently_observed.sum())
    class_labels = {0: "Insufficient clear observations", 1: "Stable non-water", 2: "Variable water fringe", 3: "Stable water"}
    pixel_area_m2 = abs(profile["transform"].a * profile["transform"].e)
    for code, label in class_labels.items():
        count = int(np.count_nonzero(stability == code))
        rows.append({
            "class_code": code,
            "class_label": label,
            "pixel_count": count,
            "area_km2": count * pixel_area_m2 / 1_000_000.0,
            "fraction_of_raster": count / stability.size,
        })
    with (OUT_DIR / "S2_water_stability_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "S2_water_scene_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(used_scenes[0]))
        writer.writeheader()
        writer.writerows(used_scenes)

    # rasterio.array_bounds returns (left, bottom, right, top), whereas
    # matplotlib.imshow expects (left, right, bottom, top).
    left, bottom, right, top = rasterio.transform.array_bounds(
        stability.shape[0], stability.shape[1], profile["transform"]
    )
    plot_extent = (left, right, bottom, top)
    survey_bounds = survey_extent_utm()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
    fig.suptitle("Sentinel-2 multi-scene water occurrence at Anza", fontsize=14, fontweight="bold")

    ax = axes[0]
    image = ax.imshow(occurrence, extent=plot_extent, origin="upper", vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    ax.set_title("A  Water occurrence among clear observations")
    cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Fraction of clear observations classified as water")

    ax = axes[1]
    class_cmap = ListedColormap(["#d8dee9", "#475569", "#f59e0b", "#0e7490"])
    class_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], class_cmap.N)
    ax.imshow(stability, extent=plot_extent, origin="upper", cmap=class_cmap, norm=class_norm, interpolation="nearest")
    ax.set_title("B  Exploratory stability classes")
    handles = [
        Patch(color="#d8dee9", label=f"Insufficient (<{args.minimum_clear_observations} clear scenes)"),
        Patch(color="#475569", label=f"Stable non-water (≤{args.stable_land_threshold:.0%} water)"),
        Patch(color="#f59e0b", label="Variable / intertidal fringe"),
        Patch(color="#0e7490", label=f"Stable water (≥{args.stable_water_threshold:.0%} water)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=True)

    for ax in axes:
        ax.plot(
            [survey_bounds[0], survey_bounds[2], survey_bounds[2], survey_bounds[0], survey_bounds[0]],
            [survey_bounds[1], survey_bounds[1], survey_bounds[3], survey_bounds[3], survey_bounds[1]],
            color="white", linestyle="--", linewidth=1.0, label="Survey extent",
        )
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_xlabel("Easting (m; WGS 84 / UTM 29N)")
        ax.set_ylabel("Northing (m; WGS 84 / UTM 29N)")
        ax.set_aspect("equal")

    fig.savefig(OUT_DIR / "Figure_S2_water_occurrence.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_S2_water_occurrence.pdf", bbox_inches="tight")
    plt.close(fig)
    (OUT_DIR / "figure_caption_EN.txt").write_text(
        "Figure. Sentinel-2 SCL water-class occurrence across selected clear observations from November 2022 to March 2023. "
        "Panel A shows the proportion of clear observations classified as water. Panel B groups pixels into exploratory "
        "stability classes using a minimum-observation requirement and stated occurrence thresholds; the dashed outline "
        "shows the local bathymetric survey extent. The variable fringe may include shoreline migration, intertidal water, "
        "surf, mixed pixels, and classification error. This is not a surveyed coastline and must not be used as a final "
        "hydrodynamic boundary without comparison to a suitable reference shoreline.\n",
        encoding="utf-8",
    )

    report = {
        "status": "exploratory multi-scene SCL-water occurrence; not a surveyed coastline",
        "scene_count": len(used_scenes),
        "scene_names": [row["product_name"] for row in used_scenes],
        "missing_scene_products": missing_scenes,
        "formula": "clear observations with SCL water / all clear observations",
        "minimum_clear_observations_for_stability_class": args.minimum_clear_observations,
        "stable_non_water_threshold": args.stable_land_threshold,
        "stable_water_threshold": args.stable_water_threshold,
        "classes": class_labels,
        "limitations": [
            "Sentinel-2 SCL water class is not a surveyed shoreline and can omit foam, turbid water, wet sand, or mixed coastal pixels.",
            "Scene-to-scene variation mixes tide, wave run-up, seasonal waterline changes, and classification error.",
            "No tidal normalization, local gauge validation, or charted coastline comparison was performed.",
            "Use the stable-water core and variable fringe as a screening diagnostic only; do not use them as an authoritative mesh boundary.",
            "The bathymetric survey horizontal transformation has a reported 7 m operation accuracy; the drawn extent is approximate.",
        ],
        "outputs": [
            "S2_water_occurrence_10m.tif",
            "S2_clear_observation_count_10m.tif",
            "S2_water_stability_class_10m.tif",
            "S2_water_stability_summary.csv",
            "S2_water_scene_summary.csv",
            "Figure_S2_water_occurrence.png/.pdf",
        ],
    }
    (OUT_DIR / "S2_water_occurrence_metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Water-occurrence diagnostic complete; source scenes and bathymetry were not changed.")
    print(f"  Selected scenes found: {len(products)}; used: {len(used_scenes)}; missing processed masks: {len(missing_scenes)}")
    print(f"  Stable-water threshold: ≥{args.stable_water_threshold:.0%}; minimum clear observations: {args.minimum_clear_observations}")
    for row in rows:
        if row["class_code"] != 0:
            print(f"  {row['class_label']}: {row['pixel_count']:,} pixels ({row['area_km2']:.2f} km²)")
    print(f"  Outputs: {OUT_DIR}")
    print("No tide correction, shoreline survey, bathymetric fusion, mesh, or model boundary was produced.")


if __name__ == "__main__":
    main()
