from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from matplotlib.lines import Line2D
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "Results" / "Sediment_Proxy" / "scene_relative_turbidity_summary.csv"
SETTINGS = ROOT / "Results" / "Additional_Analyses" / "wave_event_analysis_settings.json"
OUT_DIR = ROOT / "Results" / "Sediment_Proxy"
MAP_PNG = OUT_DIR / "Figure_relative_turbidity_maps.png"
MAP_PDF = OUT_DIR / "Figure_relative_turbidity_maps.pdf"
TIME_PNG = OUT_DIR / "Figure_turbidity_wave_context.png"
TIME_PDF = OUT_DIR / "Figure_turbidity_wave_context.pdf"
CAPTIONS = OUT_DIR / "figure_captions_EN.txt"

# Fixed color limits keep the NDTI color meaning consistent across all scene panels.
NDTI_VMIN = -0.60
NDTI_VMAX = 0.10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def as_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_on_template(path: Path, template: rasterio.io.DatasetReader) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.crs is None or template.crs is None:
            raise ValueError(f"Raster CRS is missing: {path}")
        if (
            src.width == template.width
            and src.height == template.height
            and src.crs == template.crs
            and src.transform == template.transform
        ):
            return src.read(1).astype(np.float32)
        output = np.full((template.height, template.width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=output,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=template.transform,
            dst_crs=template.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return output


def make_map_panels(rows: list[dict[str, str]]) -> None:
    first = next((row for row in rows if (ROOT / row["ndti_raster"]).is_file()), None)
    if first is None:
        raise FileNotFoundError("No NDTI raster listed in the summary was found.")

    template = rasterio.open(ROOT / first["ndti_raster"])
    try:
        fig, axes = plt.subplots(4, 3, figsize=(14, 12), constrained_layout=True)
        flat_axes = axes.ravel()
        image = None
        crs = template.crs
        if crs and crs.is_projected:
            axis_info = PyprojCRS.from_user_input(crs).axis_info
            unit = axis_info[0].unit_name if axis_info else "map units"
            use_kilometres = unit.lower() in {"metre", "meter", "metres", "meters"}
            coordinate_scale = 1000.0 if use_kilometres else 1.0
            display_unit = "km" if use_kilometres else unit
            x_label, y_label = f"Easting ({display_unit})", f"Northing ({display_unit})"
        else:
            coordinate_scale = 1.0
            x_label, y_label = "X coordinate", "Y coordinate"

        median_support = np.median([int(float(row["valid_ndti_pixels"])) for row in rows])
        for panel_index, (ax, row) in enumerate(zip(flat_axes, rows)):
            raster_path = ROOT / row["ndti_raster"]
            if not raster_path.is_file():
                ax.set_visible(False)
                continue
            data = read_on_template(raster_path, template)
            left, bottom, right, top = array_bounds(template.height, template.width, template.transform)
            image = ax.imshow(
                data,
                extent=(left / coordinate_scale, right / coordinate_scale,
                        bottom / coordinate_scale, top / coordinate_scale),
                origin="upper",
                cmap="cividis",
                vmin=NDTI_VMIN,
                vmax=NDTI_VMAX,
                interpolation="nearest",
            )
            date = parse_time(row["acquisition_utc"]).strftime("%Y-%m-%d")
            n_pixels = int(float(row["valid_ndti_pixels"]))
            title = f"{date}  |  n = {n_pixels / 1_000_000:.2f} M"
            if n_pixels < 0.5 * median_support:
                title += "  |  low coverage"
            ax.set_title(title, fontsize=9, pad=5)
            panel_row, panel_col = divmod(panel_index, 3)
            ax.set_xlabel(x_label if panel_row == 3 else "", fontsize=8)
            ax.set_ylabel(y_label if panel_col == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(color="white", alpha=0.18, linewidth=0.35)

        for ax in flat_axes[len(rows):]:
            ax.set_visible(False)
        if image is None:
            raise RuntimeError("No NDTI raster was plotted.")
        colorbar = fig.colorbar(image, ax=list(flat_axes), shrink=0.78, pad=0.02, extend="both")
        colorbar.set_label("NDTI (unitless relative optical proxy; display clipped)", fontsize=9)
        fig.suptitle(
            "Clear-water Sentinel-2 relative turbidity proxy across selected acquisitions",
            fontsize=14,
            fontweight="bold",
        )
        fig.savefig(MAP_PNG, dpi=350, bbox_inches="tight")
        fig.savefig(MAP_PDF, bbox_inches="tight")
        plt.close(fig)
    finally:
        template.close()


def make_time_series(rows: list[dict[str, str]], seasonal_q95: float | None) -> None:
    rows = sorted(rows, key=lambda row: parse_time(row["acquisition_utc"]))
    dates = [parse_time(row["acquisition_utc"]) for row in rows]
    medians = np.array([as_float(row["ndti_median"]) for row in rows])
    p25 = np.array([as_float(row["ndti_p25"]) for row in rows])
    p75 = np.array([as_float(row["ndti_p75"]) for row in rows])
    wave = np.array([as_float(row["antecedent_24h_max_wave_height_m"]) for row in rows])
    n_pixels = np.array([int(float(row["valid_ndti_pixels"])) for row in rows])
    low_support = n_pixels < 0.5 * np.median(n_pixels)

    fig, (ax_ndti, ax_wave) = plt.subplots(
        2, 1, figsize=(11.5, 7.2), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1]},
    )
    ax_ndti.errorbar(
        dates,
        medians,
        yerr=np.vstack((medians - p25, p75 - medians)),
        fmt="o",
        color="#176B73",
        ecolor="#70A9A8",
        elinewidth=1.4,
        capsize=3,
        markersize=6,
        zorder=3,
    )
    for date, value, low in zip(dates, medians, low_support):
        if low:
            ax_ndti.annotate("low pixel support", (date, value), xytext=(5, 8),
                             textcoords="offset points", fontsize=8, color="#A44A3F")
    ax_ndti.axvline(datetime(2022, 11, 10, tzinfo=dates[0].tzinfo), color="#D97732",
                    linestyle="--", linewidth=1.1, label="Bathymetric survey date")
    ax_ndti.set_ylabel("NDTI median and IQR\n(unitless)")
    ax_ndti.grid(axis="y", color="#D9E0E3", linewidth=0.7)
    ax_ndti.legend(frameon=False, loc="best", fontsize=8)

    threshold = seasonal_q95 if seasonal_q95 is not None else np.nanmedian(wave)
    colors = np.where(wave >= threshold, "#D97732", "#4D8291")
    ax_wave.scatter(dates, wave, c=colors, s=64, alpha=0.92, edgecolor="white", linewidth=0.7, zorder=3)
    ax_wave.axvline(datetime(2022, 11, 10, tzinfo=dates[0].tzinfo), color="#D97732",
                    linestyle="--", linewidth=1.1)
    if seasonal_q95 is not None:
        ax_wave.axhline(seasonal_q95, color="#5F477A", linestyle=":", linewidth=1.2,
                        label=f"Seasonal hourly Q95 = {seasonal_q95:.2f} m")
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4D8291",
               markeredgecolor="white", markersize=7, label="24-h maximum below seasonal Q95"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D97732",
               markeredgecolor="white", markersize=7, label="24-h maximum at/above seasonal Q95"),
    ]
    if seasonal_q95 is not None:
        legend_items.append(Line2D([0], [0], color="#5F477A", linestyle=":", linewidth=1.2,
                                   label=f"Seasonal hourly Q95 = {seasonal_q95:.2f} m"))
    ax_wave.legend(handles=legend_items, frameon=False, loc="upper left", fontsize=8)
    ax_wave.set_ylabel("Maximum IBI Hs in\npreceding 24 h (m)")
    ax_wave.set_ylim(bottom=0)
    ax_wave.set_xlabel("Sentinel-2 acquisition date (UTC)")
    ax_wave.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax_wave.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax_wave.grid(axis="y", color="#D9E0E3", linewidth=0.7)
    fig.suptitle("Scene-level optical proxy and antecedent regional wave context",
                 fontsize=14, fontweight="bold")
    fig.savefig(TIME_PNG, dpi=350, bbox_inches="tight")
    fig.savefig(TIME_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"Run Scripts/09_analyze_relative_turbidity.py first: {SUMMARY}")
    rows = read_csv(SUMMARY)
    if not rows:
        raise RuntimeError(f"No scene rows found in {SUMMARY}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seasonal_q95 = None
    if SETTINGS.is_file():
        import json
        with SETTINGS.open(encoding="utf-8") as stream:
            seasonal_q95 = as_float(json.load(stream).get("season_q95_hs_m", ""))

    make_map_panels(rows)
    make_time_series(rows, seasonal_q95)
    captions = [
        "Figure A. Sentinel-2 NDTI maps for the selected acquisitions. NDTI is a unitless relative optical proxy calculated from B04 and B03 over clear SCL class-6 pixels. All panels use the same color limits (−0.60 to 0.10); values outside the displayed interval are clipped. The low-coverage 10 March 2023 scene is flagged. The index is not calibrated to suspended sediment concentration, and the SCL water mask is not a verified marine polygon.",
        "Figure B. Scene-level median and interquartile range of NDTI (upper panel) alongside the maximum regional IBI significant wave height during the preceding 24 hours (lower panel). The dashed vertical line marks the 10 November 2022 bathymetric survey date. Orange markers denote a preceding-24-h maximum at or above the six-month seasonal hourly Q95; the dotted horizontal line shows that hourly threshold for context. It is not a climatological extreme threshold. The comparison is descriptive, based on 11 selected scenes, and does not demonstrate causality or sediment transport.",
    ]
    CAPTIONS.write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    print("Created two descriptive figures as 350-dpi PNG and vector PDF:")
    print(f"  {MAP_PNG}")
    print(f"  {TIME_PNG}")
    print(f"Captions: {CAPTIONS}")
    print("No regression, statistical significance test, SSC calibration, or causal claim was made.")


if __name__ == "__main__":
    main()
