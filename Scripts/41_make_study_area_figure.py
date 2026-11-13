"""Build the manuscript locator map with local survey points over an open bathymetry layer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RASTER = ROOT / "Data" / "Open_Bathymetry" / "Anza_Agadir" / "EMODnet_2022_mean_depth_Anza.tif"
SURVEY = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
COUNTRIES = ROOT / "Data" / "Cartographic" / "ne_110m_admin_0_countries.geojson"
OUT = ROOT / "Results" / "Manuscript_Figures"


def survey_arrays():
    rows = []
    with SURVEY.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["longitude_wgs84_candidate_deg"]),
                         float(row["latitude_wgs84_candidate_deg"]),
                         float(row["z_source_m"])))
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Expected survey longitude, latitude, and local-Z columns.")
    return a[:, 0], a[:, 1], a[:, 2]


def polygon_parts(geometry):
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        yield from geometry["coordinates"]


def draw_country(ax, coordinates, face, edge, linewidth=0.35):
    for polygon in polygon_parts(coordinates):
        exterior = np.asarray(polygon[0], dtype=float)
        ax.add_patch(Polygon(exterior, closed=True, facecolor=face,
                             edgecolor=edge, linewidth=linewidth, zorder=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lon, lat, z_local = survey_arrays()
    submerged = z_local <= 0
    lon, lat, z_local = lon[submerged], lat[submerged], z_local[submerged]

    with rasterio.open(RASTER) as src:
        bathy = src.read(1, masked=True)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

    with COUNTRIES.open(encoding="utf-8") as f:
        world = json.load(f)
    morocco = next(feat for feat in world["features"]
                   if feat.get("properties", {}).get("ADMIN") == "Morocco")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelcolor": "#243746", "xtick.color": "#425466",
        "ytick.color": "#425466", "axes.edgecolor": "#6B7C88",
        "axes.linewidth": 0.75,
    })
    fig = plt.figure(figsize=(11.4, 7.2), facecolor="white")
    ax = fig.add_axes([0.085, 0.105, 0.655, 0.82])
    ax.set_facecolor("#F5F8FA")
    image = ax.imshow(bathy, extent=extent, origin="upper", cmap="Blues_r",
                      vmin=-45, vmax=0, interpolation="nearest", zorder=0,
                      rasterized=True)
    xx = np.linspace(extent[0], extent[1], bathy.shape[1])
    yy = np.linspace(extent[3], extent[2], bathy.shape[0])
    X, Y = np.meshgrid(xx, yy)
    layer = np.ma.masked_where(np.ma.getdata(bathy) >= 0, bathy)
    ax.contour(X, Y, layer, levels=[-40, -30, -20, -10, -5],
               colors="#23485D", linewidths=0.42, alpha=0.62, zorder=2)

    # Local depths are plotted as a distinct point layer: they are not vertically
    # shifted to EMODnet's LAT datum and are not fused into its raster.
    scatter = ax.scatter(lon, lat, c=-z_local, s=1.4, cmap="magma_r",
                         vmin=0, vmax=25, alpha=0.78, linewidths=0,
                         rasterized=True, zorder=4)

    ax.set_xlim(-9.6835, -9.6600)
    ax.set_ylim(30.4430, 30.4628)
    ax.set_aspect(1 / np.cos(np.deg2rad(30.45)))
    ax.set_xlabel("Longitude (°W)", labelpad=5)
    ax.set_ylabel("Latitude (°N)", labelpad=5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.3f}°W"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}°N"))
    ax.grid(color="white", linewidth=0.6, alpha=0.75, zorder=3)
    ax.text(0.018, 0.974, "A", transform=ax.transAxes, va="top", ha="left",
            fontsize=12, fontweight="bold", color="#122B3A",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#CBD5DC", "alpha": 0.95}, zorder=10)

    cax1 = fig.add_axes([0.785, 0.405, 0.025, 0.27])
    cax2 = fig.add_axes([0.885, 0.405, 0.025, 0.27])
    cbar1 = fig.colorbar(image, cax=cax1)
    cbar1.set_label("EMODnet 2022 elevation (m; LAT)", fontsize=8.2, labelpad=7)
    cbar1.ax.tick_params(labelsize=7.5)
    cbar2 = fig.colorbar(scatter, cax=cax2)
    cbar2.set_label("Survey depth (m below Port of Agadir ZH)", fontsize=8.2, labelpad=7)
    cbar2.ax.tick_params(labelsize=7.5)

    # Scale bar in approximate local metres at 30.45°N.
    center_lat = 30.452
    lon_per_km = 1 / (111.32 * np.cos(np.deg2rad(center_lat)))
    x0, y0, w = -9.6818, 30.4440, 1.0 * lon_per_km
    ax.plot([x0, x0 + w], [y0, y0], color="#152B38", linewidth=3.2, solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0], [y0 - 0.00018, y0 + 0.00018], color="#152B38", linewidth=1.0, zorder=8)
    ax.plot([x0 + w, x0 + w], [y0 - 0.00018, y0 + 0.00018], color="#152B38", linewidth=1.0, zorder=8)
    ax.text(x0 + w / 2, y0 + 0.00034, "1 km", ha="center", va="bottom", fontsize=7.5, color="#152B38",
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.2}, zorder=8)

    # A conventional compass rose, not a bare north arrow.
    cx, cy = 0.94, 0.12
    ax.annotate("", xy=(cx, cy + 0.09), xytext=(cx, cy - 0.015), xycoords="axes fraction",
                arrowprops={"arrowstyle": "simple", "facecolor": "#183B50", "edgecolor": "#183B50", "lw": 0.5}, zorder=9)
    ax.annotate("N", (cx, cy + 0.101), xycoords="axes fraction", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#183B50", zorder=9)

    # Locator inset integrated into the main map layout.
    inset = fig.add_axes([0.765, 0.735, 0.225, 0.205], facecolor="white", zorder=12)
    for feat in world["features"]:
        props = feat.get("properties", {})
        if props.get("ADMIN") == "Morocco":
            draw_country(inset, feat["geometry"], "#2A7F83", "#244B5A", 0.48)
        elif props.get("ADMIN") in {"Algeria", "Western Sahara", "Mauritania", "Spain", "Portugal"}:
            draw_country(inset, feat["geometry"], "#E6ECEF", "#98A8B1", 0.28)
    inset.scatter([-9.67], [30.45], s=18, c="#F28E2B", edgecolors="white", linewidths=0.7, zorder=5)
    inset.annotate("Anza / Agadir", xy=(-9.67, 30.45), xytext=(-6.5, 32.5),
                   fontsize=6.2, color="#183B50", ha="center",
                   arrowprops={"arrowstyle": "-", "color": "#E87524", "lw": 0.8}, zorder=6)
    inset.set_xlim(-14.5, -0.5)
    inset.set_ylim(20.5, 36.5)
    inset.set_aspect(1 / np.cos(np.deg2rad(29)))
    inset.set_xticks([]); inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#617987"); spine.set_linewidth(0.8)
    inset.text(0.025, 0.97, "B  MOROCCO", transform=inset.transAxes, ha="left", va="top",
               fontsize=7.2, fontweight="bold", color="#163C50",
               bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#D5DEE3", "alpha": 0.95})

    png = OUT / "Figure_1_study_area_overlay.png"
    pdf = OUT / "Figure_1_study_area_overlay.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    caption = (
        "Figure 1. Study-area location and overlaid bathymetric inputs. (A) Anza survey soundings, colored by "
        "depth below the local hydrographic zero of the Port of Agadir, plotted over the EMODnet DTM 2022 "
        "elevation layer (LAT) with native-grid contours. (B) Natural Earth locator showing Anza on the Atlantic "
        "coast of Morocco. The local sounding elevations and EMODnet elevations retain their distinct vertical "
        "references; the layers are overlaid for geographic context only and were not datum-transformed or fused. "
        "Natural Earth country outlines are shown for cartographic location."
    )
    (OUT / "Figure_1_study_area_overlay_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
