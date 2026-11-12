from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
CATALOG = ROOT / "Data" / "Sentinel2" / "scene_catalog.csv"
PROCESSED = ROOT / "Data" / "Sentinel2" / "Processed"
MARINE = ROOT / "Data" / "Copernicus_Marine"
WAVE_FILE = MARINE / "IBI_wave_Anza_subset.nc"
CURRENT_FILE = MARINE / "IBI_currents_surface_Anza_subset.nc"
TIDE_FILE = ROOT / "Results" / "sentinel2_GOT5.6_tide_estimates.csv"
OUT = ROOT / "Results" / "scene_forcing_and_quality.csv"


def nearest_value(ds: xr.Dataset, variable: str, timestamp: str, lon: float, lat: float, depth: float | None = None):
    if variable not in ds:
        return float("nan")
    da = ds[variable]
    selection = {}
    time_coord = next((name for name in ("time", "TIME") if name in da.coords), None)
    lon_coord = next((name for name in ("longitude", "lon", "LONGITUDE") if name in da.coords), None)
    lat_coord = next((name for name in ("latitude", "lat", "LATITUDE") if name in da.coords), None)
    if time_coord:
        t = np.datetime64(timestamp.replace("Z", ""))
        da = da.sel({time_coord: t}, method="nearest", tolerance=np.timedelta64(90, "m") )
    if lon_coord:
        da = da.sel({lon_coord: lon}, method="nearest")
    if lat_coord:
        da = da.sel({lat_coord: lat}, method="nearest")
    depth_coord = next((name for name in ("depth", "deptht", "DEPTH") if name in da.coords), None)
    if depth is not None and depth_coord:
        da = da.sel({depth_coord: depth}, method="nearest")
    values = np.asarray(da.values).astype(float).ravel()
    values = values[np.isfinite(values)]
    return float(values[0]) if values.size else float("nan")


def nearest_valid_current_pair(
    ds: xr.Dataset, timestamp: str, lon: float, lat: float
) -> tuple[float, float, float, float, float]:
    """Return currents from the nearest wet IBI cell and its distance from the target."""
    if not all(name in ds for name in ("uo", "vo")):
        return (float("nan"),) * 5

    lon_coord = next((name for name in ("longitude", "lon", "LONGITUDE") if name in ds.coords), None)
    lat_coord = next((name for name in ("latitude", "lat", "LATITUDE") if name in ds.coords), None)
    time_coord = next((name for name in ("time", "TIME") if name in ds.coords), None)
    if not lon_coord or not lat_coord:
        return (float("nan"),) * 5

    try:
        t = np.datetime64(timestamp.replace("Z", ""))
        fields = []
        for variable in ("uo", "vo"):
            da = ds[variable]
            if time_coord:
                da = da.sel({time_coord: t}, method="nearest", tolerance=np.timedelta64(90, "m"))
            depth_coord = next((name for name in ("depth", "deptht", "DEPTH") if name in da.coords), None)
            if depth_coord:
                da = da.sel({depth_coord: 0.5}, method="nearest")
            fields.append(np.asarray(da.transpose(lat_coord, lon_coord).values, dtype=float))

        u_grid, v_grid = fields
        if u_grid.ndim != 2 or v_grid.shape != u_grid.shape:
            return (float("nan"),) * 5

        lons = np.asarray(ds[lon_coord].values, dtype=float)
        lats = np.asarray(ds[lat_coord].values, dtype=float)
        if lons.ndim != 1 or lats.ndim != 1 or u_grid.shape != (lats.size, lons.size):
            return (float("nan"),) * 5

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        valid = np.isfinite(u_grid) & np.isfinite(v_grid)
        if not valid.any():
            return (float("nan"),) * 5

        # Equirectangular distance is sufficient for choosing within this small regional subset.
        dx_km = (lon_grid - lon) * 111.32 * np.cos(np.deg2rad(lat))
        dy_km = (lat_grid - lat) * 111.32
        distance_km = np.hypot(dx_km, dy_km)
        distance_km[~valid] = np.inf
        iy, ix = np.unravel_index(np.argmin(distance_km), distance_km.shape)
        return (
            float(u_grid[iy, ix]), float(v_grid[iy, ix]),
            float(lon_grid[iy, ix]), float(lat_grid[iy, ix]),
            float(distance_km[iy, ix]),
        )
    except (KeyError, ValueError, IndexError):
        return (float("nan"),) * 5


def quality_counts(product_name: str) -> tuple[int | str, int | str, float | str]:
    path = PROCESSED / f"{product_name}_qa_10m.tif"
    if not path.exists():
        return "", "", ""
    with rasterio.open(path) as ds:
        clear = ds.read(1) == 1
        water = ds.read(2) == 1
    total = int(clear.size)
    nclear, nwater = int(clear.sum()), int(water.sum())
    # The bounding box includes land; this is an image-QA fraction, not a water-only availability estimate.
    return nclear, nwater, (nclear / total if total else float("nan"))


def main() -> None:
    if not CATALOG.exists():
        raise FileNotFoundError("Run Scripts/01_catalog_sentinel2.py first.")
    if not WAVE_FILE.exists():
        raise FileNotFoundError(f"IBI wave subset missing: {WAVE_FILE}. Run Scripts/03_download_copernicus_marine.ps1 first.")
    bbox = CONFIG["site_bbox_wgs84"]
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    wave_ds = xr.open_dataset(WAVE_FILE)
    current_ds = xr.open_dataset(CURRENT_FILE) if CURRENT_FILE.exists() else None
    tide_by_product = {}
    if TIDE_FILE.exists():
        with TIDE_FILE.open(newline="", encoding="utf-8-sig") as stream:
            tide_by_product = {r.get("product_name", ""): r for r in csv.DictReader(stream)}
    else:
        print(f"  Tide estimates not found at {TIDE_FILE}; run Scripts/07_estimate_tides_sentinel2.py first.")
    print("Stage 3: matching each Sentinel-2 acquisition to the nearest IBI grid cell and hour.")
    print(f"  Nearest-grid target: {lon:.5f}°E, {lat:.5f}°N; IBI grid is regional and does not resolve surf-zone currents.")
    rows = []
    with CATALOG.open(newline="", encoding="utf-8-sig") as stream:
        for index, scene in enumerate(csv.DictReader(stream), start=1):
            timestamp = scene.get("acquisition_utc", "")
            if not timestamp:
                continue
            name = scene.get("product_name", "")
            clear_n, water_n, clear_fraction = quality_counts(name)
            row = {
                "scene_id": scene.get("scene_id", ""), "product_name": name,
                "acquisition_utc": timestamp, "cloud_cover_product_pct": scene.get("cloud_cover_pct", ""),
                "ibi_vhm0_m": nearest_value(wave_ds, "VHM0", timestamp, lon, lat),
                "ibi_vtpk_s": nearest_value(wave_ds, "VTPK", timestamp, lon, lat),
                "ibi_vmdr_deg": nearest_value(wave_ds, "VMDR", timestamp, lon, lat),
                "scl_clear_pixel_count_in_approx_box": clear_n,
                "scl_class6_water_pixel_count_in_approx_box": water_n,
                "scl_clear_fraction_of_entire_box_not_water_only": clear_fraction,
                "currents_file_present": bool(current_ds is not None)
            }
            tide = tide_by_product.get(name, {})
            row["got56_model_reference_tide_m"] = tide.get("model_reference_tidal_elevation_m", "")
            row["got56_anomaly_from_utc_day_mean_m"] = tide.get("anomaly_relative_to_modeled_utc_day_mean_m", "")
            row["tide_estimate_status"] = tide.get("status", "not calculated")
            if current_ds is not None:
                uo, vo, current_lon, current_lat, current_distance = nearest_valid_current_pair(
                    current_ds, timestamp, lon, lat
                )
                row["ibi_uo_m_s"] = uo
                row["ibi_vo_m_s"] = vo
                row["ibi_current_grid_lon"] = current_lon
                row["ibi_current_grid_lat"] = current_lat
                row["ibi_current_grid_distance_km"] = current_distance
            rows.append(row)
            if index % 50 == 0:
                print(f"  Scenes matched: {index}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with OUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    wave_ds.close()
    if current_ds is not None:
        current_ds.close()
    print(f"Wrote {len(rows)} scene matchups: {OUT}")
    print("Caution: clear fraction is computed over an approximate rectangle that includes land; do not interpret it as the fraction of water observed.")
    print("A checked marine polygon and a defensible nearshore mask are required before estimating observation-probability weights.")


if __name__ == "__main__":
    main()
