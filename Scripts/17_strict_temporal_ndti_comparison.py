from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
PROCESSED = ROOT / "Data" / "Sentinel2" / "Processed"
PROXY_DIR = ROOT / "Results" / "Sediment_Proxy"
SUPPORT_DIR = PROXY_DIR / "Spatial_Support"
STABILITY = ROOT / "Results" / "Marine_Mask" / "S2_water_stability_class_10m.tif"
OUT_DIR = SUPPORT_DIR


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No output rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values: np.ndarray) -> tuple[int, float | str, float | str, float | str]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0, "", "", ""
    q25, q50, q75 = np.percentile(values.astype(np.float64), [25, 50, 75])
    return int(values.size), float(q50), float(q25), float(q75)


def main() -> None:
    selected = [
        row for row in read_csv(SELECTION)
        if row.get("download_selected", "").strip().upper() == "YES"
        and row.get("product_name", "").startswith(("S2A_MSIL2A_2022", "S2B_MSIL2A_2022", "S2A_MSIL2A_2023", "S2B_MSIL2A_2023"))
    ]
    selected.sort(key=lambda row: row.get("acquisition_utc", ""))
    if not selected:
        raise RuntimeError("No selected 2022–2023 Sentinel-2 products were found.")

    support_csv = SUPPORT_DIR / "ndti_spatial_support_by_scene.csv"
    support_by_product = {row["product_name"]: row for row in read_csv(support_csv)}
    support_count_path = SUPPORT_DIR / "S2_valid_NDTI_observation_count_10m.tif"
    if not STABILITY.is_file() or not support_count_path.is_file():
        raise FileNotFoundError("Run Scripts/14_sentinel_water_occurrence.py and Scripts/16_audit_ndti_spatial_support.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(STABILITY) as stability_ds:
        stable_water_core = stability_ds.read(1) == 3
        profile = stability_ds.profile.copy()
        transform = stability_ds.transform
        crs = stability_ds.crs
    with rasterio.open(support_count_path) as count_ds:
        if (count_ds.width, count_ds.height, count_ds.transform, count_ds.crs) != (
            profile["width"], profile["height"], transform, crs
        ):
            raise ValueError("Valid-observation-count raster is not aligned with the stable-water core.")
        valid_observation_count = count_ds.read(1)

    valid_masks: list[np.ndarray] = []
    ndti_arrays: list[np.ndarray] = []
    scene_rows: list[dict[str, str]] = []
    print("Stage 7: strict temporal comparison of NDTI on identical pixel locations.")
    print(f"  Selected scenes: {len(selected)}; fixed domain: stable-water core from Script 14.")
    for index, row in enumerate(selected, start=1):
        product = row["product_name"]
        ndti_path = PROXY_DIR / f"{product}_NDTI_relative_proxy.tif"
        if product not in support_by_product:
            raise FileNotFoundError(f"Spatial-support summary lacks product: {product}")
        if not ndti_path.is_file():
            raise FileNotFoundError(ndti_path)
        with rasterio.open(ndti_path) as ds:
            if (ds.width, ds.height, ds.transform, ds.crs) != (profile["width"], profile["height"], transform, crs):
                raise ValueError(f"NDTI raster grid mismatch for {product}.")
            arr = ds.read(1).astype(np.float32, copy=False)
        valid = np.isfinite(arr)
        valid_masks.append(valid)
        ndti_arrays.append(arr)
        scene_rows.append(row)
        print(f"  Loaded {index}/{len(selected)}: {product}")

    valid_stack = np.stack(valid_masks, axis=0)
    strict_mask = stable_water_core & valid_stack.all(axis=0)
    strict_n = int(strict_mask.sum())
    core_n = int(stable_water_core.sum())
    at_least_8 = stable_water_core & (valid_observation_count >= min(8, len(selected)))
    if strict_n == 0:
        print("  WARNING: no stable-water pixel is valid in every scene; strict temporal comparison cannot be computed.")

    rows: list[dict[str, object]] = []
    for i, row in enumerate(scene_rows):
        product = row["product_name"]
        arr = ndti_arrays[i]
        support = support_by_product[product]
        all_valid_values = arr[valid_masks[i]]
        core_values = arr[valid_masks[i] & stable_water_core]
        at_least_8_values = arr[valid_masks[i] & at_least_8]
        strict_values = arr[strict_mask]
        all_n, all_med, all_q25, all_q75 = stats(all_valid_values)
        core_count, core_med, core_q25, core_q75 = stats(core_values)
        eight_count, eight_med, eight_q25, eight_q75 = stats(at_least_8_values)
        strict_count, strict_med, strict_q25, strict_q75 = stats(strict_values)
        rows.append({
            "product_name": product,
            "acquisition_utc": row.get("acquisition_utc", ""),
            "valid_pixels_all_water": all_n,
            "median_ndti_all_water": all_med,
            "median_ndti_p25_all_water": all_q25,
            "median_ndti_p75_all_water": all_q75,
            "stable_water_core_pixels": core_n,
            "valid_pixels_stable_water_core": core_count,
            "coverage_stable_water_core": core_count / core_n if core_n else "",
            "median_ndti_valid_part_of_core": core_med,
            "median_ndti_p25_valid_part_of_core": core_q25,
            "median_ndti_p75_valid_part_of_core": core_q75,
            "valid_pixels_ge_8_scene_support": eight_count,
            "median_ndti_ge_8_scene_support": eight_med,
            "median_ndti_p25_ge_8_scene_support": eight_q25,
            "median_ndti_p75_ge_8_scene_support": eight_q75,
            "strict_all_scene_common_pixels": strict_n,
            "median_ndti_strict_common_pixels": strict_med,
            "median_ndti_p25_strict_common_pixels": strict_q25,
            "median_ndti_p75_strict_common_pixels": strict_q75,
        })

    output = OUT_DIR / "ndti_strict_temporal_comparison.csv"
    write_csv(output, rows)

    strict_profile = profile.copy()
    strict_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    strict_path = OUT_DIR / "S2_strict_common_support_all_scenes_10m.tif"
    with rasterio.open(strict_path, "w", **strict_profile) as dst:
        dst.write(strict_mask.astype(np.uint8), 1)
        dst.set_band_description(1, "Stable-water pixels with valid NDTI in every selected scene")
        dst.update_tags(scene_count=len(selected), selection_rule="valid in all selected NDTI scenes and stable-water core")

    dates = [datetime.fromisoformat(row["acquisition_utc"].replace("Z", "+00:00")) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.0), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.0, 1.45]})
    fig.suptitle("NDTI temporal comparison under spatial-support controls", fontsize=14, fontweight="bold")

    ax = axes[0]
    coverage_core = np.asarray([float(row["coverage_stable_water_core"]) for row in rows]) * 100
    coverage_eight = np.asarray([float(support_by_product[row["product_name"]]["fixed_common_support_coverage_fraction"]) for row in rows]) * 100
    ax.plot(dates, coverage_core, marker="o", color="#0e7490", linewidth=1.6, label="Persistent-water core")
    ax.plot(dates, coverage_eight, marker="s", color="#f59e0b", linewidth=1.4, linestyle="--", label="≥8-scene-support domain")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Valid pixels (%)")
    ax.set_title("A  Per-scene observation coverage")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="lower left")

    ax = axes[1]
    series = [
        ("median_ndti_all_water", "median_ndti_p25_all_water", "median_ndti_p75_all_water", "All valid SCL-water pixels", "#94a3b8", "o"),
        ("median_ndti_valid_part_of_core", "median_ndti_p25_valid_part_of_core", "median_ndti_p75_valid_part_of_core", "Valid part of persistent-water core", "#0e7490", "s"),
        ("median_ndti_ge_8_scene_support", "median_ndti_p25_ge_8_scene_support", "median_ndti_p75_ge_8_scene_support", "Valid pixels in ≥8-scene domain", "#f59e0b", "^"),
        ("median_ndti_strict_common_pixels", "median_ndti_p25_strict_common_pixels", "median_ndti_p75_strict_common_pixels", "Same pixels in all scenes", "#7c3aed", "D"),
    ]
    for median_key, low_key, high_key, label, color, marker in series:
        med = np.asarray([float(row[median_key]) if row[median_key] != "" else np.nan for row in rows])
        low = np.asarray([float(row[low_key]) if row[low_key] != "" else np.nan for row in rows])
        high = np.asarray([float(row[high_key]) if row[high_key] != "" else np.nan for row in rows])
        if np.isfinite(med).any():
            ax.plot(dates, med, marker=marker, markersize=4.5, linewidth=1.4, color=color, label=label)
            ax.fill_between(dates, low, high, color=color, alpha=0.10, linewidth=0)
    ax.set_ylabel("NDTI median (unitless)")
    ax.set_xlabel("Sentinel-2 acquisition date (UTC)")
    ax.set_title(f"B  Median NDTI by spatial domain (strict overlap: {strict_n:,} pixels)")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    figure_path = OUT_DIR / "Figure_NDTI_strict_temporal_comparison.png"
    fig.savefig(figure_path, dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_NDTI_strict_temporal_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "selected_scenes": len(selected),
        "fixed_core_pixels": core_n,
        "strict_all_scene_common_pixels": strict_n,
        "strict_common_share_of_core": strict_n / core_n if core_n else None,
        "strict_domain_definition": "class-3 stable-water core from Script 14 intersected with valid NDTI pixels from every selected scene",
        "median_definition": "unweighted per-pixel median within each spatial domain and acquisition date",
        "uncertainty_bands": "25th–75th percentile across pixels; spatially autocorrelated and not confidence intervals",
        "limitations": [
            "NDTI is an uncalibrated unitless optical proxy, not suspended-sediment concentration.",
            "The strict all-scene domain may be small and is spatially selected by scene availability; report its size and location.",
            "Pixel-level quantiles describe spatial spread and are not inferential confidence intervals because pixels are spatially dependent.",
            "No causal wave response, sediment flux, or statistical significance is estimated.",
        ],
        "outputs": ["ndti_strict_temporal_comparison.csv", "S2_strict_common_support_all_scenes_10m.tif", "Figure_NDTI_strict_temporal_comparison.png/.pdf"],
    }
    (OUT_DIR / "ndti_strict_temporal_comparison_settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "Figure_NDTI_strict_temporal_comparison_caption_EN.txt").write_text(
        "Figure. Sentinel-2 NDTI time series with observation-support diagnostics. (A) Valid-pixel coverage in the persistent-water core and the domain "
        "whose pixels are observed in at least eight scenes. (B) Per-date median and interquartile spatial range for all valid SCL-water pixels, the "
        "persistent-water core, the ≥8-scene-support domain, and the strict intersection valid in every scene. The shaded ranges are pixelwise 25th–75th "
        "percentiles, not confidence intervals. NDTI is an uncalibrated optical proxy; the figure does not establish suspended-sediment concentration or "
        "a causal hydrodynamic response.\n",
        encoding="utf-8",
    )

    print("Strict temporal-support comparison complete; source imagery and survey were not changed.")
    print(f"  Stable-water core: {core_n:,} pixels")
    print(f"  Exact all-scene overlap: {strict_n:,} pixels ({strict_n / core_n:.1%} of core)")
    print(f"  Summary: {output}")
    print(f"  Figure: {figure_path}")
    print("IQR bands show spatial dispersion, not statistical confidence intervals; NDTI remains uncalibrated.")


if __name__ == "__main__":
    main()
