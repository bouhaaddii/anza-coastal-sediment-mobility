from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
SOURCE = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_normalized.csv"
OUT_DIR = ROOT / "Results" / "Horizontal_CRS"
CANDIDATE_POINTS = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
REPORT = OUT_DIR / "candidate_horizontal_crs_diagnostic.json"

# The user confirmed from the Anza DWG: Lambert Zone 2, Clarke 1880, X/easting
# then Y/northing, and the title-block translation direction. EPSG:26192 is the
# registered Sud Maroc realization used for the XY transformation. PROJ applies
# the registered local-to-WGS84 operation (+31, +146, +47 m in geocentric space).
# Its reported 7 m accuracy is not a local control-point residual; survey Z is
# kept unchanged and remains relative to the Port of Agadir hydrographic zero.
SOURCE_CRS = CRS.from_epsg(26192)
GEOGRAPHIC_CRS = CRS.from_epsg(4326)
TARGET_CRS = CRS.from_epsg(32629)


def read_points() -> list[dict[str, str]]:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Run Scripts/05_audit_bathymetry_xyz.py first: {SOURCE}")
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    points = read_points()
    if not points:
        raise RuntimeError(f"No bathymetric points found in {SOURCE}")

    x = np.asarray([float(row["x_source_m"]) for row in points], dtype=np.float64)
    y = np.asarray([float(row["y_source_m"]) for row in points], dtype=np.float64)
    z = np.asarray([float(row["z_source_m"]) for row in points], dtype=np.float64)

    to_geographic = Transformer.from_crs(SOURCE_CRS, GEOGRAPHIC_CRS, always_xy=True)
    lon, lat = to_geographic.transform(x, y)
    to_target = Transformer.from_crs(GEOGRAPHIC_CRS, TARGET_CRS, always_xy=True)
    east_utm, north_utm = to_target.transform(lon, lat)

    bbox = CONFIG["site_bbox_wgs84"]
    inside_bbox = (
        (lon >= bbox[0]) & (lon <= bbox[2]) &
        (lat >= bbox[1]) & (lat <= bbox[3])
    )
    lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    center_distance_km = np.hypot(
        (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0)),
        (lat - lat0) * 111.32,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATE_POINTS.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "point_id", "x_source_m", "y_source_m", "z_source_m",
            "longitude_wgs84_candidate_deg", "latitude_wgs84_candidate_deg",
            "easting_utm29n_candidate_m", "northing_utm29n_candidate_m",
        ])
        for row, xx, yy, zz, lo, la, east, north in zip(
            points, x, y, z, lon, lat, east_utm, north_utm
        ):
            writer.writerow([
                row["point_id"], f"{xx:.3f}", f"{yy:.3f}", f"{zz:.3f}",
                f"{lo:.9f}", f"{la:.9f}", f"{east:.3f}", f"{north:.3f}",
            ])

    report = {
        "status": "plan-confirmed horizontal CRS; registered transformation applied; local control residual not assessed",
        "source_file": str(SOURCE),
        "candidate_points_file": str(CANDIDATE_POINTS),
        "source_crs": "EPSG:26192 — Merchich / Sud Maroc; confirmed from the Anza survey plan",
        "target_crs": "EPSG:32629 — WGS 84 / UTM zone 29N",
        "point_count": int(len(points)),
        "source_x_range_m": [float(np.min(x)), float(np.max(x))],
        "source_y_range_m": [float(np.min(y)), float(np.max(y))],
        "candidate_longitude_range_deg": [float(np.nanmin(lon)), float(np.nanmax(lon))],
        "candidate_latitude_range_deg": [float(np.nanmin(lat)), float(np.nanmax(lat))],
        "points_inside_approximate_project_bbox": int(inside_bbox.sum()),
        "bbox_point_fraction": float(inside_bbox.mean()),
        "median_distance_to_bbox_center_km": float(np.median(center_distance_km)),
        "candidate_transformer_description": to_geographic.description,
        "candidate_transformer_reported_accuracy_m": float(to_geographic.accuracy),
        "vertical_values": "Z copied unchanged; retained in metres relative to the local hydrographic zero of Port of Agadir",
        "vertical_translation_applied": False,
        "plan_translation_note": "The user confirmed the Anza plan's Lambert Zone 2 / Clarke 1880 reference, X/easting then Y/northing order, and translation direction. The registered EPSG:26192-to-WGS84 operation applies +31,+146,+47 m in geocentric space, inverse to the plan's listed negative direction. These datum translations are not applied to the survey's local hydrographic Z values.",
        "horizontal_operation_confirmation": "Confirmed by the user from the Anza DWG; EPSG:26192 to WGS 84 operation used with always_xy=True.",
        "limitations": [
            "The registered horizontal operation reports 7 m accuracy; this is not a local survey control-point residual.",
            "Check alignment against DWG control points before claiming sub-10 m absolute co-registration with satellite pixels.",
            "The approximate working box is not a surveyed marine polygon.",
            "No vertical datum conversion, gridding, open-bathymetry fusion, mesh, or hydrodynamic simulation is performed.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Plan-confirmed horizontal CRS diagnostic complete; source XYZ was not changed.")
    print("  Source CRS: EPSG:26192 (Merchich / Sud Maroc)")
    print(f"  Points: {len(points):,}; inside approximate working box: {int(inside_bbox.sum()):,} ({inside_bbox.mean():.1%})")
    print(f"  Candidate WGS84 longitude: {np.min(lon):.6f} to {np.max(lon):.6f}")
    print(f"  Candidate WGS84 latitude:  {np.min(lat):.6f} to {np.max(lat):.6f}")
    print(f"  Transformer-reported accuracy: {to_geographic.accuracy:g} m")
    print("  Z retained unchanged; no vertical translation was applied.")
    print(f"  Candidate points: {CANDIDATE_POINTS}")
    print(f"  Diagnostic report: {REPORT}")
    print("  Plan translation direction confirmed; registered transformation accuracy remains 7 m.")
    print("Check DWG control-point residuals before claiming sub-10 m satellite co-registration.")


if __name__ == "__main__":
    main()
