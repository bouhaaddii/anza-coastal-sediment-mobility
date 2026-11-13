"""Create a layered 3D figure and nested Morocco-to-Anza locators."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, Window


ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / "Data" / "Sentinel2" / "Processed" / "S2B_MSIL2A_20221110T112209_N0510_R037_T29RMP_20240801T233757_reflectance_10m.tif"
QA = ROOT / "Data" / "Sentinel2" / "Processed" / "S2B_MSIL2A_20221110T112209_N0510_R037_T29RMP_20240801T233757_qa_10m.tif"
SURVEY = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
COUNTRIES = ROOT / "Data" / "Cartographic" / "ne_110m_admin_0_countries.geojson"
OUT = ROOT / "Results" / "Manuscript_Figures"


def read_survey():
    rows = []
    with SURVEY.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            z = float(row["z_source_m"])
            if z < 0:
                depth = -z
                depth_bin = round(depth * 2.0) / 2.0
                if 0.5 <= depth_bin <= 29.0:
                    rows.append((float(row["easting_utm29n_candidate_m"]),
                                 float(row["northing_utm29n_candidate_m"]), depth))
    a = np.asarray(rows, dtype=float)
    if len(a) == 0:
        raise ValueError("The survey file contains no submerged points.")
    return a


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


def read_rgb_crop(bounds, output_shape=(150, 150)):
    xmin, ymin, xmax, ymax = bounds
    with rasterio.open(RGB) as src, rasterio.open(QA) as qa:
        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 2 or window.height < 2:
            raise ValueError("The survey footprint lies outside the Sentinel-2 image.")
        h, w = output_shape
        raw = src.read((3, 2, 1), window=window, out_shape=(3, h, w),
                       resampling=Resampling.bilinear).astype(np.float32)
        qa_arr = qa.read(1, window=window, out_shape=(h, w),
                         resampling=Resampling.nearest)
        clear = (qa_arr == 1) & np.isfinite(raw).all(axis=0)
        if clear.sum() < 20:
            clear = np.isfinite(raw).all(axis=0)
        rgb = np.zeros((h, w, 4), dtype=np.float32)
        for band in range(3):
            vals = raw[band][clear]
            lo, hi = np.nanpercentile(vals, (2, 98))
            if hi <= lo:
                hi = lo + 1e-6
            # A mild display gamma reveals the dark-water texture without
            # changing reflectance inputs or implying bathymetric information.
            stretched = np.clip((raw[band] - lo) / (hi - lo), 0, 1)
            rgb[:, :, band] = np.power(stretched, 0.72)
        rgb[:, :, 3] = clear.astype(np.float32)
        crop_transform = src.window_transform(window)
        extent = (crop_transform.c,
                  crop_transform.c + crop_transform.a * window.width,
                  crop_transform.f + crop_transform.e * window.height,
                  crop_transform.f)
    return rgb, extent


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    points = read_survey()
    e, n, depth = points.T
    center_e, center_n = float(np.median(e)), float(np.median(n))
    pad = 220.0
    bounds = (float(e.min() - pad), float(n.min() - pad),
              float(e.max() + pad), float(n.max() + pad))
    rgb, rgb_extent = read_rgb_crop(bounds)

    with COUNTRIES.open(encoding="utf-8") as f:
        world = json.load(f)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.labelcolor": "#243746", "xtick.color": "#425466",
        "ytick.color": "#425466", "axes.edgecolor": "#6B7C88",
        "axes.linewidth": 0.7,
    })
    fig = plt.figure(figsize=(12.4, 7.0), facecolor="white")
    ax3d = fig.add_axes([0.025, 0.12, 0.625, 0.82], projection="3d")
    ax3d.set_facecolor("white")
    height, width = rgb.shape[:2]
    x = np.linspace(rgb_extent[0], rgb_extent[1], width)
    y = np.linspace(rgb_extent[3], rgb_extent[2], height)
    X, Y = np.meshgrid((x - center_e) / 1000.0, (y - center_n) / 1000.0)
    z_image = np.full_like(X, 26.0)
    ax3d.plot_surface(X, Y, z_image, facecolors=rgb, shade=False,
                      linewidth=0, antialiased=False, rstride=1, cstride=1)
    scatter = ax3d.scatter((e - center_e) / 1000.0,
                           (n - center_n) / 1000.0,
                           depth, c=depth, cmap="viridis",
                           vmin=0, vmax=23, s=1.35, alpha=0.82,
                           linewidths=0, depthshade=False, rasterized=True)
    ax3d.set_xlim((bounds[0] - center_e) / 1000, (bounds[2] - center_e) / 1000)
    ax3d.set_ylim((bounds[1] - center_n) / 1000, (bounds[3] - center_n) / 1000)
    ax3d.set_zlim(26.2, 0)
    ax3d.set_xlabel("Easting X offset (km)", labelpad=7)
    ax3d.set_ylabel("Northing Y offset (km)", labelpad=7)
    ax3d.set_zlabel("Depth below local Port of Agadir ZH (m)", labelpad=6)
    ax3d.set_zticks([0, 5, 10, 15, 20, 25])
    ax3d.view_init(elev=28, azim=-57)
    ax3d.set_box_aspect((1.35, 1.05, 0.68))
    ax3d.grid(True, alpha=0.24)
    ax3d.text2D(0.015, 0.98, "A", transform=ax3d.transAxes,
                fontsize=12, fontweight="bold", color="#142F43",
                va="top", bbox={"boxstyle": "round,pad=0.18", "fc": "white",
                                "ec": "#D2DCE2", "alpha": 0.95})
    cax = fig.add_axes([0.653, 0.38, 0.016, 0.36])
    cb = fig.colorbar(scatter, cax=cax)
    cb.set_label("Measured sounding depth (m; positive downward)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_morocco = fig.add_axes([0.715, 0.555, 0.265, 0.37], facecolor="white")
    for feat in world["features"]:
        admin = feat.get("properties", {}).get("ADMIN", "")
        if admin == "Morocco":
            draw_country(ax_morocco, feat["geometry"], "#4E9BA0", "#315C68", 0.55)
        elif admin in {"Algeria", "Western Sahara", "Mauritania", "Spain", "Portugal"}:
            draw_country(ax_morocco, feat["geometry"], "#E8EDF0", "#AAB7BE", 0.35)
    ax_morocco.scatter([-9.67], [30.45], s=35, marker="*", c="#D55E00",
                       edgecolors="white", linewidths=0.65, zorder=5)
    ax_morocco.annotate("Anza / Agadir", xy=(-9.67, 30.45), xytext=(-5.1, 32.3),
                        fontsize=7.1, color="#193B4D", ha="center",
                        arrowprops={"arrowstyle": "-", "color": "#D55E00", "lw": 0.9},
                        zorder=6)
    ax_morocco.set_xlim(-13.5, -0.5)
    ax_morocco.set_ylim(20.5, 36.5)
    ax_morocco.set_aspect(1 / np.cos(np.deg2rad(29)))
    ax_morocco.set_xticks([]); ax_morocco.set_yticks([])
    for spine in ax_morocco.spines.values():
        spine.set_edgecolor("#758995"); spine.set_linewidth(0.8)
    ax_morocco.text(0.03, 0.97, "B  MOROCCO", transform=ax_morocco.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold",
                    color="#183B50",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white",
                          "ec": "#D5DEE3", "alpha": 0.95})

    ax_local = fig.add_axes([0.715, 0.105, 0.265, 0.36], facecolor="white")
    local_extent = tuple(v / 1000.0 for v in (rgb_extent[0] - center_e,
                     rgb_extent[1] - center_e, rgb_extent[2] - center_n,
                     rgb_extent[3] - center_n))
    ax_local.imshow(rgb, extent=local_extent, origin="upper", interpolation="nearest")
    xmin, ymin, xmax, ymax = bounds
    footprint = Rectangle(((xmin - center_e) / 1000, (ymin - center_n) / 1000),
                          (xmax - xmin) / 1000, (ymax - ymin) / 1000,
                          fill=False, edgecolor="#D55E00", linewidth=1.1)
    ax_local.add_patch(footprint)
    ax_local.set_xlim((bounds[0] - center_e) / 1000, (bounds[2] - center_e) / 1000)
    ax_local.set_ylim((bounds[1] - center_n) / 1000, (bounds[3] - center_n) / 1000)
    ax_local.set_xlabel("Easting X offset (km)", labelpad=2, fontsize=7)
    ax_local.set_ylabel("Northing Y offset (km)", labelpad=2, fontsize=7)
    ax_local.tick_params(labelsize=6.5)
    ax_local.set_aspect("equal", adjustable="box")
    ax_local.grid(color="white", linewidth=0.55, alpha=0.65)
    ax_local.text(0.025, 0.97, "C  ANZA / AGADIR", transform=ax_local.transAxes,
                  ha="left", va="top", fontsize=7.8, fontweight="bold",
                  color="#183B50",
                  bbox={"boxstyle": "round,pad=0.18", "fc": "white",
                        "ec": "#D5DEE3", "alpha": 0.95})

    proxies = [
        Line2D([0], [0], color="#647E89", lw=7,
               label="Sentinel-2 L2A RGB layer (display offset only)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4B6B8A",
               markeredgecolor="white", markersize=5,
               label="Measured survey soundings; colour encodes local-ZH depth"),
    ]
    fig.legend(handles=proxies, loc="lower left", bbox_to_anchor=(0.055, 0.025),
               frameon=False, ncol=1, fontsize=7.5, handlelength=1.8, labelspacing=0.35)
    png = OUT / "Figure_1_study_area_overlay.png"
    pdf = OUT / "Figure_1_study_area_overlay.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    caption = (
        "Figure 1. Nested location and layered observation support for Anza, Morocco. (A) Three-dimensional "
        "view of the Sentinel-2 L2A RGB composite (bands B04/B03/B02; acquisition 10 November 2022) and the "
        "22,986 submerged soundings from the local survey. Sounding depth is plotted positive downward from "
        "the hydrographic zero (ZH) of the Port of Agadir and encoded by colour. The RGB image plane is placed "
        "at an artificial depth of 26 m solely to separate the layers visually; it is not a bathymetric surface "
        "or a vertical-datum transformation. (B) Morocco locator with Anza/Agadir highlighted. (C) Local RGB "
        "view with the survey footprint outlined. The satellite and survey layers are co-located horizontally "
        "for context but were not vertically fused."
    )
    (OUT / "Figure_1_study_area_overlay_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Submerged soundings plotted: {len(points):,}")


if __name__ == "__main__":
    main()
