from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
SELECTION = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
PROCESSED = ROOT / "Data" / "Sentinel2" / "Processed"
FORCING = ROOT / "Results" / "scene_forcing_and_quality.csv"
WAVE_CONTEXT = ROOT / "Results" / "Additional_Analyses" / "sentinel2_antecedent_wave_context.csv"
OUT_DIR = ROOT / "Results" / "Sediment_Proxy"
SUMMARY = OUT_DIR / "scene_relative_turbidity_summary.csv"
NOTES = OUT_DIR / "interpretation_notes.txt"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def percentile(values: np.ndarray, q: float) -> float | str:
    return float(np.percentile(values, q)) if values.size else ""


def grids_match(ref: rasterio.io.DatasetReader, qa: rasterio.io.DatasetReader) -> bool:
    return (
        ref.width == qa.width
        and ref.height == qa.height
        and ref.crs == qa.crs
        and ref.transform == qa.transform
    )


def main() -> None:
    selections = [
        row for row in read_rows(SELECTION)
        if row.get("download_selected", "").strip().upper() == "YES"
    ]
    if not selections:
        raise RuntimeError(f"No products marked YES in {SELECTION}")

    forcing_by_product = {row.get("product_name", ""): row for row in read_rows(FORCING)}
    wave_by_product = {row.get("product_name", ""): row for row in read_rows(WAVE_CONTEXT)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Stage 4: descriptive Sentinel-2 relative turbidity proxy; no MES calibration.")
    print("  Index: NDTI = (B04 red - B03 green) / (B04 red + B03 green).")
    print("  Pixels: SCL class-6 water intersected with the existing clear-pixel mask.")
    print("  Only the 11 currently selected study scenes are considered; the 2015 test scene is excluded.")

    output_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for selection in selections:
        product = selection.get("product_name", "").strip()
        if not product:
            continue
        reflectance_path = PROCESSED / f"{product}_reflectance_10m.tif"
        qa_path = PROCESSED / f"{product}_qa_10m.tif"
        if not reflectance_path.is_file() or not qa_path.is_file():
            skipped.append(product)
            print(f"  Missing processed reflectance/QA; skipped: {product}")
            continue

        with rasterio.open(reflectance_path) as ref, rasterio.open(qa_path) as qa:
            if ref.count < 4 or qa.count < 2:
                raise ValueError(f"Unexpected band count for {product}: reflectance={ref.count}, QA={qa.count}")
            if not grids_match(ref, qa):
                raise ValueError(f"Reflectance and QA grids do not match for {product}")

            green = ref.read(2).astype(np.float32, copy=False)
            red = ref.read(3).astype(np.float32, copy=False)
            clear = qa.read(1) == 1
            water = qa.read(2) == 1
            clear_water = clear & water
            denominator = red + green
            valid = (
                clear_water
                & np.isfinite(red)
                & np.isfinite(green)
                & (red >= 0.0)
                & (green >= 0.0)
                & (denominator > 1e-6)
            )

            ndti = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
            ndti[valid] = (red[valid] - green[valid]) / denominator[valid]
            values = ndti[valid].astype(np.float64)

            raster_path = OUT_DIR / f"{product}_NDTI_relative_proxy.tif"
            profile = ref.profile.copy()
            profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
            with rasterio.open(raster_path, "w", **profile) as dst:
                dst.write(ndti, 1)
                dst.set_band_description(1, "NDTI_relative_turbidity_proxy_unitless")
                dst.update_tags(
                    formula="(B04_red - B03_green) / (B04_red + B03_green)",
                    mask="clear SCL class-6 pixels within approximate working box",
                    interpretation="relative optical proxy only; not calibrated to suspended matter concentration",
                    glint_correction="not applied",
                    water_mask_note="SCL class 6 is not a checked marine polygon; coastline and surf-zone pixels require visual review",
                )

        forcing = forcing_by_product.get(product, {})
        wave = wave_by_product.get(product, {})
        uo = as_float(forcing.get("ibi_uo_m_s"))
        vo = as_float(forcing.get("ibi_vo_m_s"))
        current_speed = float(np.hypot(uo, vo)) if uo is not None and vo is not None else ""
        output_rows.append({
            "product_name": product,
            "acquisition_utc": forcing.get("acquisition_utc", selection.get("acquisition_utc", "")),
            "product_cloud_cover_pct": forcing.get("cloud_cover_product_pct", selection.get("cloud_cover_pct", "")),
            "clear_scl_water_pixels_in_approx_box": int(clear_water.sum()),
            "valid_ndti_pixels": int(values.size),
            "valid_fraction_of_clear_scl_water": float(values.size / clear_water.sum()) if clear_water.any() else "",
            "ndti_median": percentile(values, 50),
            "ndti_p25": percentile(values, 25),
            "ndti_p75": percentile(values, 75),
            "ndti_p90": percentile(values, 90),
            "antecedent_24h_max_wave_height_m": wave.get("preceding_24h_max_hs_m", ""),
            "ibi_wave_height_at_acquisition_m": forcing.get("ibi_vhm0_m", ""),
            "ibi_peak_wave_period_s": forcing.get("ibi_vtpk_s", ""),
            "ibi_mean_wave_direction_deg": forcing.get("ibi_vmdr_deg", ""),
            "ibi_surface_current_speed_m_s": current_speed,
            "got56_anomaly_from_utc_day_mean_m": forcing.get("got56_anomaly_from_utc_day_mean_m", ""),
            "ndti_raster": str(raster_path.relative_to(ROOT)),
            "interpretation_limit": "Unitless relative optical proxy; not calibrated to MES; SCL water mask and glint require review.",
        })
        print(f"  {product}: {values.size:,} valid clear-water pixels; wrote {raster_path.name}")

    if not output_rows:
        raise RuntimeError("No selected product had both processed reflectance and QA rasters.")
    fields = list(output_rows[0].keys())
    with SUMMARY.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    notes = [
        "Sentinel-2 relative turbidity proxy — interpretation notes",
        "",
        "NDTI = (B04 red surface reflectance - B03 green surface reflectance) / (B04 + B03).",
        "The calculation is limited to clear pixels labelled as water by SCL class 6 and with finite, non-negative red/green reflectance.",
        "NDTI is reported as a unitless relative optical proxy. No in-situ suspended matter concentration (MES/TSS) calibration is available, so do not report NDTI as mg/L or as a validated sediment concentration.",
        "The SCL water mask is not a verified marine polygon. The approximate bounding box includes land, and no sunglint, foam, adjacency-effect, or atmospheric-residual correction is applied here.",
        "The IBI wave/current fields provide regional context and do not resolve the surf zone. Scene-level associations are descriptive and are not evidence of sediment transport or causality.",
        f"Scenes selected: {len(selections)}; scenes summarized: {len(output_rows)}; scenes skipped for missing products: {len(skipped)}.",
    ]
    if skipped:
        notes.append("Skipped products: " + "; ".join(skipped))
    NOTES.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"Wrote scene summary: {SUMMARY}")
    print(f"Wrote interpretation limits: {NOTES}")
    print("No turbidity calibration, concentration retrieval, mesh, or sediment-transport simulation was performed.")


if __name__ == "__main__":
    main()
