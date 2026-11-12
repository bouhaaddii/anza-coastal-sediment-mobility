from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
PRODUCT = "S2B_MSIL2A_20221110T112209_N0510_R037_T29RMP_20240801T233757"
SURVEY = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
SAFE = ROOT / "Data" / "Sentinel2" / "SAFE" / f"{PRODUCT}.SAFE"
NDTI = ROOT / "Results" / "Sediment_Proxy" / f"{PRODUCT}_NDTI_relative_proxy.tif"
QA = ROOT / "Data" / "Sentinel2" / "Processed" / f"{PRODUCT}_qa_10m.tif"
OUT_DIR = ROOT / "Results" / "Survey_Sentinel2_Concurrent"
SURVEY_START_LOCAL = "10:33"
SURVEY_END_LOCAL = "15:16"
LOCAL_UTC_OFFSET_HOURS = 1  # Africa/Casablanca on 10 Nov 2022, per IANA tz database.
BLOCK_SIZE_M = 100.0


def read_survey() -> list[dict[str, str]]:
    if not SURVEY.is_file():
        raise FileNotFoundError(SURVEY)
    with SURVEY.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("The candidate bathymetry file contains no points.")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No output rows to write: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def granule_sensing_time() -> datetime:
    if not SAFE.is_dir():
        raise FileNotFoundError(f"Extracted Sentinel-2 SAFE product not found: {SAFE}")
    candidates = list(SAFE.glob("GRANULE/*/MTD_TL.xml"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one granule MTD_TL.xml for {PRODUCT}; found {len(candidates)}.")
    root = ElementTree.parse(candidates[0]).getroot()
    for element in root.iter():
        if element.tag.split("}")[-1] == "SENSING_TIME" and element.text:
            return datetime.fromisoformat(element.text.strip().replace("Z", "+00:00"))
    raise RuntimeError(f"SENSING_TIME was not found in {candidates[0]}.")


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def depth_group(z: float) -> tuple[str, str]:
    if z >= 0:
        return "At or above local zero", "Z ≥ 0 m"
    depth = -z
    if depth <= 2:
        return "0–2 m below local zero", "0 < depth ≤ 2 m"
    if depth <= 5:
        return "2–5 m below local zero", "2 < depth ≤ 5 m"
    if depth <= 10:
        return "5–10 m below local zero", "5 < depth ≤ 10 m"
    return ">10 m below local zero", "depth > 10 m"


def main() -> None:
    if CONFIG.get("survey_date") != "2022-11-10":
        raise ValueError(f"Project survey_date is not 2022-11-10: {CONFIG.get('survey_date')}")
    for path in (NDTI, QA):
        if not path.is_file():
            raise FileNotFoundError(f"Required raster not found: {path}")

    sensing_utc = granule_sensing_time()
    if sensing_utc.tzinfo is None:
        raise ValueError("Sentinel granule sensing time must include a UTC offset.")
    local_tz = timezone(timedelta(hours=LOCAL_UTC_OFFSET_HOURS), name="Africa/Casablanca")
    sensing_local = sensing_utc.astimezone(local_tz)
    start_local = datetime.fromisoformat(f"2022-11-10T{SURVEY_START_LOCAL}:00+01:00")
    end_local = datetime.fromisoformat(f"2022-11-10T{SURVEY_END_LOCAL}:00+01:00")
    time_inside_window = start_local <= sensing_local <= end_local

    points = read_survey()
    with rasterio.open(NDTI) as ndti_ds, rasterio.open(QA) as qa_ds:
        if (ndti_ds.width, ndti_ds.height, ndti_ds.transform, ndti_ds.crs) != (
            qa_ds.width, qa_ds.height, qa_ds.transform, qa_ds.crs
        ):
            raise ValueError("NDTI and QA raster grids are not aligned.")
        if ndti_ds.crs is None or ndti_ds.crs.to_epsg() != 32629:
            raise ValueError(f"Expected raster CRS EPSG:32629; found {ndti_ds.crs}.")
        ndti = ndti_ds.read(1).astype(np.float32, copy=False)
        qa_clear = qa_ds.read(1) == 1
        qa_water = qa_ds.read(2) == 1
        clear_water = qa_clear & qa_water & np.isfinite(ndti)
        transform = ndti_ds.transform
        bounds = ndti_ds.bounds
        raster_profile = ndti_ds.profile.copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matched: list[dict[str, object]] = []
    outside = 0
    for point in points:
        x = float(point["easting_utm29n_candidate_m"])
        y = float(point["northing_utm29n_candidate_m"])
        z = float(point["z_source_m"])
        row, col = rasterio.transform.rowcol(transform, x, y)
        if row < 0 or col < 0 or row >= ndti.shape[0] or col >= ndti.shape[1]:
            outside += 1
            continue

        exact_valid = bool(clear_water[row, col])
        exact_ndti = float(ndti[row, col]) if exact_valid else ""
        r0, r1 = max(0, row - 1), min(ndti.shape[0], row + 2)
        c0, c1 = max(0, col - 1), min(ndti.shape[1], col + 2)
        neighborhood_values = ndti[r0:r1, c0:c1][clear_water[r0:r1, c0:c1]]
        neighborhood_median = float(np.median(neighborhood_values)) if neighborhood_values.size else ""
        group, group_label = depth_group(z)
        matched.append({
            "point_id": point["point_id"],
            "easting_utm29n_candidate_m": x,
            "northing_utm29n_candidate_m": y,
            "z_local_agadir_port_zero_m": z,
            "depth_below_local_zero_m_positive_down": -z if z < 0 else "",
            "survey_z_group": group,
            "survey_z_group_label": group_label,
            "sentinel_pixel_row": int(row),
            "sentinel_pixel_col": int(col),
            "clear_scl_water_at_containing_pixel": exact_valid,
            "ndti_exact_clear_water_pixel": exact_ndti,
            "valid_clear_water_pixels_in_3x3": int(neighborhood_values.size),
            "ndti_3x3_clear_water_median_sensitivity": neighborhood_median,
        })

    if not matched:
        raise RuntimeError("No survey points overlap the Sentinel-2 analysis raster.")
    matched_path = OUT_DIR / "Anza_20221110_survey_NDTI_point_matchups.csv"
    write_csv(matched_path, matched)

    group_order = [
        "At or above local zero",
        "0–2 m below local zero",
        "2–5 m below local zero",
        "5–10 m below local zero",
        ">10 m below local zero",
    ]
    bin_rows: list[dict[str, object]] = []
    for group in group_order:
        subset = [row for row in matched if row["survey_z_group"] == group]
        valid_subset = [row for row in subset if row["clear_scl_water_at_containing_pixel"]]
        vals = np.asarray([float(row["ndti_exact_clear_water_pixel"]) for row in valid_subset], dtype=float)
        neighbor_vals = np.asarray(
            [float(row["ndti_3x3_clear_water_median_sensitivity"]) for row in subset if row["ndti_3x3_clear_water_median_sensitivity"] != ""],
            dtype=float,
        )
        if vals.size:
            q25, median, q75 = np.percentile(vals, [25, 50, 75])
        else:
            q25 = median = q75 = ""
        if neighbor_vals.size:
            neighbor_q25, neighbor_median, neighbor_q75 = np.percentile(neighbor_vals, [25, 50, 75])
        else:
            neighbor_q25 = neighbor_median = neighbor_q75 = ""
        bin_rows.append({
            "survey_z_group": group,
            "survey_z_group_label": next((row["survey_z_group_label"] for row in subset), ""),
            "survey_point_count": len(subset),
            "clear_scl_water_exact_pixel_count": len(valid_subset),
            "clear_scl_water_exact_pixel_fraction": len(valid_subset) / len(subset) if subset else "",
            "ndti_median_exact_pixel": median,
            "ndti_p25_exact_pixel": q25,
            "ndti_p75_exact_pixel": q75,
            "points_with_valid_3x3_water_neighborhood": int(neighbor_vals.size),
            "ndti_median_3x3_sensitivity": neighbor_median,
            "ndti_p25_3x3_sensitivity": neighbor_q25,
            "ndti_p75_3x3_sensitivity": neighbor_q75,
            "interpretation": "Spatial coincidence only; local-zero sign does not by itself establish wet/dry status at overpass.",
        })
    bins_path = OUT_DIR / "Anza_20221110_NDTI_by_local_depth_bin.csv"
    write_csv(bins_path, bin_rows)

    valid_submerged = [
        row for row in matched
        if float(row["z_local_agadir_port_zero_m"]) < 0 and row["clear_scl_water_at_containing_pixel"]
    ]
    block_map: dict[tuple[int, int], list[dict[str, object]]] = {}
    for point in valid_submerged:
        ix = math.floor((float(point["easting_utm29n_candidate_m"]) - bounds.left) / BLOCK_SIZE_M)
        iy = math.floor((float(point["northing_utm29n_candidate_m"]) - bounds.bottom) / BLOCK_SIZE_M)
        block_map.setdefault((ix, iy), []).append(point)
    block_rows: list[dict[str, object]] = []
    for (ix, iy), subset in sorted(block_map.items()):
        depths = np.asarray([float(row["depth_below_local_zero_m_positive_down"]) for row in subset])
        ndti_values = np.asarray([float(row["ndti_exact_clear_water_pixel"]) for row in subset])
        block_rows.append({
            "block_id": f"{ix}_{iy}",
            "block_size_m": BLOCK_SIZE_M,
            "block_center_easting_utm29n_m": bounds.left + (ix + 0.5) * BLOCK_SIZE_M,
            "block_center_northing_utm29n_m": bounds.bottom + (iy + 0.5) * BLOCK_SIZE_M,
            "matched_sounding_count": len(subset),
            "median_depth_below_local_zero_m": float(np.median(depths)),
            "median_ndti": float(np.median(ndti_values)),
            "ndti_p25": float(np.percentile(ndti_values, 25)),
            "ndti_p75": float(np.percentile(ndti_values, 75)),
        })
    block_path = OUT_DIR / "Anza_20221110_NDTI_depth_100m_blocks.csv"
    if block_rows:
        write_csv(block_path, block_rows)
    else:
        block_path.write_text("block_id,block_size_m,matched_sounding_count,median_depth_below_local_zero_m,median_ndti\n", encoding="utf-8")

    depth_bin_rows: list[dict[str, object]] = []
    if block_rows:
        max_depth = max(float(row["median_depth_below_local_zero_m"]) for row in block_rows)
        upper_limit = max(2.0, math.ceil(max_depth / 2.0) * 2.0)
        for lower in np.arange(0.0, upper_limit, 2.0):
            upper = lower + 2.0
            subset = [
                row for row in block_rows
                if lower <= float(row["median_depth_below_local_zero_m"]) < upper
            ]
            if not subset:
                continue
            values = np.asarray([float(row["median_ndti"]) for row in subset])
            depths = np.asarray([float(row["median_depth_below_local_zero_m"]) for row in subset])
            depth_bin_rows.append({
                "depth_bin_lower_m_local_zero": lower,
                "depth_bin_upper_m_local_zero": upper,
                "block_count": len(subset),
                "sounding_count_in_blocks": sum(int(row["matched_sounding_count"]) for row in subset),
                "median_block_depth_m": float(np.median(depths)),
                "median_of_block_median_ndti": float(np.median(values)),
                "block_ndti_p25": float(np.percentile(values, 25)),
                "block_ndti_p75": float(np.percentile(values, 75)),
                "note": "Unweighted summary across occupied 100 m blocks; IQR is spatial dispersion, not a confidence interval.",
            })
    depth_bins_path = OUT_DIR / "Anza_20221110_NDTI_100m_block_depth_bins_2m.csv"
    if depth_bin_rows:
        write_csv(depth_bins_path, depth_bin_rows)
    else:
        depth_bins_path.write_text("depth_bin_lower_m_local_zero,depth_bin_upper_m_local_zero,block_count,median_of_block_median_ndti\n", encoding="utf-8")

    if len(block_rows) >= 3:
        block_depth = np.asarray([float(row["median_depth_below_local_zero_m"]) for row in block_rows])
        block_ndti = np.asarray([float(row["median_ndti"]) for row in block_rows])
        rho = float(np.corrcoef(rankdata_average(block_depth), rankdata_average(block_ndti))[0, 1])
    else:
        rho = None

    labels = [row["survey_z_group"] for row in bin_rows]
    hit_fraction = [float(row["clear_scl_water_exact_pixel_fraction"]) if row["clear_scl_water_exact_pixel_fraction"] != "" else np.nan for row in bin_rows]
    group_medians = [float(row["ndti_median_exact_pixel"]) if row["ndti_median_exact_pixel"] != "" else np.nan for row in bin_rows]
    group_q25 = [float(row["ndti_p25_exact_pixel"]) if row["ndti_p25_exact_pixel"] != "" else np.nan for row in bin_rows]
    group_q75 = [float(row["ndti_p75_exact_pixel"]) if row["ndti_p75_exact_pixel"] != "" else np.nan for row in bin_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1.0, 0.9]})
    fig.suptitle("Same-day Sentinel-2 NDTI and Anza bathymetric survey", fontsize=14, fontweight="bold")

    ax = axes[0]
    valid_points = [row for row in matched if row["clear_scl_water_at_containing_pixel"] and float(row["z_local_agadir_port_zero_m"]) < 0]
    if valid_points:
        sc = ax.scatter(
            [float(row["easting_utm29n_candidate_m"]) for row in valid_points],
            [float(row["northing_utm29n_candidate_m"]) for row in valid_points],
            c=[float(row["ndti_exact_clear_water_pixel"]) for row in valid_points],
            s=4.0, cmap="viridis", vmin=float(np.nanpercentile([float(row["ndti_exact_clear_water_pixel"]) for row in valid_points], 2)),
            vmax=float(np.nanpercentile([float(row["ndti_exact_clear_water_pixel"]) for row in valid_points], 98)),
            linewidths=0, rasterized=True,
        )
        fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02, label="NDTI (unitless)")
    ax.set_title("A  Survey points with clear SCL-water pixels")
    ax.set_xlabel("Easting (m; UTM 29N)")
    ax.set_ylabel("Northing (m; UTM 29N)")
    if valid_points:
        px = [float(row["easting_utm29n_candidate_m"]) for row in matched]
        py = [float(row["northing_utm29n_candidate_m"]) for row in matched]
        margin_x = max((max(px) - min(px)) * 0.05, 25)
        margin_y = max((max(py) - min(py)) * 0.05, 25)
        ax.set_xlim(min(px) - margin_x, max(px) + margin_x)
        ax.set_ylim(min(py) - margin_y, max(py) + margin_y)
    ax.set_aspect("equal")
    ax.grid(color="#CBD5E1", linewidth=0.35, alpha=0.5)

    ax = axes[1]
    if block_rows:
        block_depth = np.asarray([float(row["median_depth_below_local_zero_m"]) for row in block_rows])
        block_ndti = np.asarray([float(row["median_ndti"]) for row in block_rows])
        block_n = np.asarray([int(row["matched_sounding_count"]) for row in block_rows])
        scatter = ax.scatter(block_depth, block_ndti, c=block_depth, s=16 + 1.8 * np.sqrt(block_n), cmap="cividis", edgecolor="white", linewidth=0.4, alpha=0.85)
        fig.colorbar(scatter, ax=ax, shrink=0.82, pad=0.02, label="Median depth (m; local zero)")
        rho_text = f"ρs={rho:.2f}" if rho is not None else "ρs not defined"
        if depth_bin_rows:
            bx = np.asarray([float(row["median_block_depth_m"]) for row in depth_bin_rows])
            by = np.asarray([float(row["median_of_block_median_ndti"]) for row in depth_bin_rows])
            blo = np.asarray([float(row["block_ndti_p25"]) for row in depth_bin_rows])
            bhi = np.asarray([float(row["block_ndti_p75"]) for row in depth_bin_rows])
            ax.errorbar(bx, by, yerr=np.vstack([by - blo, bhi - by]), fmt="D-", color="#ef4444", ecolor="#ef4444",
                        markersize=4, linewidth=1.2, capsize=2, label="2 m depth-bin medians ± block IQR", zorder=4)
            ax.legend(frameon=True, fontsize=7, loc="best")
        ax.set_title(f"B  100 m block medians; {rho_text} (descriptive)\nn={len(block_rows)} blocks; red line summarizes 2 m bins")
    else:
        ax.text(0.5, 0.5, "No valid underwater blocks", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B  Block medians")
    ax.set_xlabel("Median depth below local zero (m)")
    ax.set_ylabel("Median NDTI (unitless)")
    ax.grid(color="#CBD5E1", linewidth=0.5, alpha=0.7)

    ax = axes[2]
    positions = np.arange(len(labels))
    bars = ax.bar(positions, np.nan_to_num(hit_fraction, nan=0) * 100, color=["#64748b", "#14b8a6", "#0e7490", "#2563eb", "#7c3aed"], edgecolor="white", linewidth=0.6)
    tick_labels = [
        f"{label}\nn={int(row['survey_point_count']):,}"
        for label, row in zip(("Z ≥ 0 m", "0–2 m", "2–5 m", "5–10 m", ">10 m"), bin_rows)
    ]
    ax.set_xticks(positions, tick_labels, rotation=0, ha="center")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Points with clear SCL-water pixel (%)")
    ax.set_title("C  SCL-water coincidence by survey Z")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.5, alpha=0.7)
    for bar, row in zip(bars, bin_rows):
        fraction = row["clear_scl_water_exact_pixel_fraction"]
        y = float(fraction) * 100 if fraction != "" else 0
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1.2, f"{y:.1f}%", ha="center", va="bottom", fontsize=7, color="#334155")

    figure_path = OUT_DIR / "Figure_Anza_same_day_survey_NDTI.png"
    fig.savefig(figure_path, dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_Anza_same_day_survey_NDTI.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "survey_date": CONFIG["survey_date"],
        "survey_local_time_window_reported_by_user": f"{SURVEY_START_LOCAL}–{SURVEY_END_LOCAL}",
        "survey_time_window_timezone_assumption": "Africa/Casablanca local civil time, UTC+01:00 on 2022-11-10; individual sounding times were not supplied",
        "sentinel_product": PRODUCT,
        "granule_sensing_time_utc_from_MTD_TL_xml": sensing_utc.isoformat(),
        "granule_sensing_time_local": sensing_local.isoformat(),
        "granule_sensing_time_within_reported_window": time_inside_window,
        "temporal_interpretation": "The granule acquisition falls within the reported survey window if the supplied survey times are local. No point-specific observation timestamps are available, so individual soundings are same-day/window context, not proven simultaneous measurements.",
        "horizontal_reference": "EPSG:32629 candidate coordinates from EPSG:26192; registered operation accuracy 7 m, not local control residual",
        "vertical_reference": "Local Agadir port hydrographic zero; Z retained as supplied; no transformation to tide, LAT, or MSL",
        "satellite_index": "NDTI=(B04-B03)/(B04+B03), clear SCL class-6 water pixels only; uncalibrated",
        "point_sampling": "Containing 10 m Sentinel-2 pixel; 3x3 valid-water neighborhood median is supplied only as co-registration sensitivity",
        "block_aggregation": f"{BLOCK_SIZE_M:.0f} m squares, descriptive aggregation to reduce point-level spatial pseudoreplication; not a mesh resolution recommendation",
        "block_spearman_rho": rho if rho is not None else "NOT_COMPUTED",
        "block_depth_summary": "Median-of-block NDTI and interquartile spatial dispersion in occupied 2 m depth bins; descriptive only.",
        "limits": [
            "A valid SCL-water coincidence is not a formal shoreline or wet/dry accuracy score because the survey zero is local and the overpass water level is not tied to it.",
            "The NDTI-depth relation can reflect bottom influence, water-column optics, glint, foam, and spatial co-registration; it is not a suspended-sediment calibration.",
            "The 7 m reported transformation accuracy is comparable to a 10 m pixel; interpret exact-pixel sampling with the 3x3 sensitivity output.",
            "No sediment concentration, transport flux, causal relationship, or inferential p-value is computed.",
        ],
        "outputs": [
            "Anza_20221110_survey_NDTI_point_matchups.csv",
            "Anza_20221110_NDTI_by_local_depth_bin.csv",
            "Anza_20221110_NDTI_depth_100m_blocks.csv",
            "Anza_20221110_NDTI_100m_block_depth_bins_2m.csv",
            "Figure_Anza_same_day_survey_NDTI.png/.pdf",
            "Anza_20221110_interpretation_FR.txt",
        ],
    }
    (OUT_DIR / "Anza_20221110_concurrent_scene_diagnostic.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    def fmt_fr(value: object) -> str:
        if value == "" or value is None:
            return "non calculé"
        return f"{float(value):.3f}".replace(".", ",")

    bin_lines = []
    for row in bin_rows:
        if row["ndti_median_exact_pixel"] == "":
            ndti_text = "NDTI non disponible"
        else:
            ndti_text = (
                f"médiane NDTI = {fmt_fr(row['ndti_median_exact_pixel'])} "
                f"(IQR {fmt_fr(row['ndti_p25_exact_pixel'])} à {fmt_fr(row['ndti_p75_exact_pixel'])})"
            )
        fraction = "non calculé" if row["clear_scl_water_exact_pixel_fraction"] == "" else f"{100 * float(row['clear_scl_water_exact_pixel_fraction']):.1f}%".replace(".", ",")
        bin_lines.append(
            f"- {row['survey_z_group_label']}: {row['survey_point_count']} points; "
            f"{row['clear_scl_water_exact_pixel_count']} pixels clairs classés eau ({fraction}); {ndti_text}."
        )
    if depth_bin_rows:
        min_bin = min(depth_bin_rows, key=lambda row: float(row["median_of_block_median_ndti"]))
        deepest_bin = max(depth_bin_rows, key=lambda row: float(row["median_block_depth_m"]))
        min_bin_label = f"{float(min_bin['depth_bin_lower_m_local_zero']):.0f}–{float(min_bin['depth_bin_upper_m_local_zero']):.0f} m"
        deepest_label = f"{float(deepest_bin['depth_bin_lower_m_local_zero']):.0f}–{float(deepest_bin['depth_bin_upper_m_local_zero']):.0f} m"
        depth_shape_note = (
            f"Parmi les blocs de 100 m résumés par tranches de 2 m, la médiane NDTI la plus faible apparaît dans la tranche {min_bin_label} "
            f"({fmt_fr(min_bin['median_of_block_median_ndti'])}); elle est moins faible dans la tranche la plus profonde occupée {deepest_label} "
            f"({fmt_fr(deepest_bin['median_of_block_median_ndti'])}). Cette forme non monotone rend le seul rho de Spearman ({fmt_fr(rho)}) insuffisant "
            "pour décrire la relation profondeur-signal."
        )
    else:
        depth_shape_note = "Aucune tranche de profondeur par blocs n'a pu être résumée."
    interpretation = (
        "Diagnostic exploratoire de la scène Sentinel-2 du 10 novembre 2022\n\n"
        f"Le granule a été détecté à {sensing_local.strftime('%H:%M:%S')} heure locale, dans la fenêtre de levé rapportée de {SURVEY_START_LOCAL} à {SURVEY_END_LOCAL}, "
        "si celle-ci correspond à l'heure locale. Les heures par point ne sont pas disponibles; la simultanéité de chaque sondage n'est donc pas démontrée.\n\n"
        f"{sum(bool(row['clear_scl_water_at_containing_pixel']) for row in matched):,} des {len(matched):,} points ont un pixel contenant clair classé eau par SCL. "
        "Cette fraction est une coïncidence spatiale, pas une validation de rivage ni une matrice d'exactitude : le Z local, le niveau d'eau instantané, la marée, "
        "le déferlement et le zéro du port ne sont pas raccordés.\n\n"
        "Médianes du NDTI exact par classes de Z local :\n" + "\n".join(bin_lines) + "\n\n"
        f"Sur {len(block_rows)} blocs de 100 m, le coefficient de Spearman entre profondeur locale et NDTI médian est {fmt_fr(rho)} (descriptif, sans valeur p). "
        f"{depth_shape_note}\n\n"
        "Le NDTI est un indice optique non calibré en MES. La structure avec la profondeur peut refléter la contribution du fond en eau peu profonde, "
        "les propriétés optiques de la colonne d'eau, le glint, l'écume et la co-inscription horizontale (précision de transformation rapportée : 7 m pour un pixel de 10 m). "
        "Ne pas interpréter cette relation comme une concentration de sédiments, un flux ou une preuve de transport. Les médianes interquartiles décrivent la dispersion spatiale; elles ne sont pas des intervalles de confiance.\n"
    )
    interpretation_path = OUT_DIR / "Anza_20221110_interpretation_FR.txt"
    interpretation_path.write_text(interpretation, encoding="utf-8")
    (OUT_DIR / "Figure_Anza_same_day_survey_NDTI_caption_EN.txt").write_text(
        "Figure. Same-day spatial comparison of the Anza bathymetric survey and Sentinel-2 NDTI on 10 November 2022. "
        "(A) Survey points whose containing 10 m pixel is clear and classified as water by SCL, colored by NDTI. "
        "(B) Median NDTI and local-zero depth aggregated to 100 m spatial blocks; the red line and error bars summarize medians and interquartile spatial dispersion across occupied 2 m depth bins. "
        "The block Spearman coefficient is a descriptive monotonic-rank summary and may not represent a non-monotonic depth response. "
        "(C) Fraction of survey points with a clear SCL-water pixel in each local-Z group; this is a coincidence diagnostic, not shoreline accuracy. "
        "The granule sensing time is compared with the reported 10:33–15:16 survey window assuming Africa/Casablanca local time; individual point times are unavailable. "
        "Z is not transformed from the Agadir port hydrographic zero, and NDTI is uncalibrated to suspended-sediment concentration.\n",
        encoding="utf-8",
    )

    print("Same-day survey/Sentinel-2 diagnostic complete; original survey XYZ and source rasters were not changed.")
    print(f"  Survey points: {len(points):,}; in raster: {len(matched):,}; outside: {outside:,}")
    print(f"  Granule sensing time: {sensing_utc.isoformat()} UTC / {sensing_local.strftime('%H:%M:%S')} local")
    print(f"  Inside reported local survey window ({SURVEY_START_LOCAL}–{SURVEY_END_LOCAL}): {time_inside_window}")
    print(f"  Exact clear-water pixel matches: {sum(bool(row['clear_scl_water_at_containing_pixel']) for row in matched):,}")
    print(f"  100 m blocks with underwater matches: {len(block_rows)}; descriptive block Spearman rho: {rho if rho is not None else 'not computed'}")
    print(f"  Depth-bin summary: {depth_bins_path}")
    print(f"  Interpretation: {interpretation_path}")
    print(f"  Outputs: {OUT_DIR}")
    print("The same-day window does not prove each sounding is simultaneous; NDTI-depth is not an SSC calibration.")


if __name__ == "__main__":
    main()
