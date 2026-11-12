from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
WAVE_FILE = ROOT / "Data" / "Copernicus_Marine" / "IBI_wave_Anza_subset.nc"
CATALOG = ROOT / "Data" / "Sentinel2" / "scene_catalog.csv"
OUT_DIR = ROOT / "Results" / "Additional_Analyses"
EVENTS_OUT = OUT_DIR / "ibi_winter_2022_2023_q95_wave_events.csv"
SCENES_OUT = OUT_DIR / "sentinel2_antecedent_wave_context.csv"
SETTINGS_OUT = OUT_DIR / "wave_event_analysis_settings.json"


def find_coord(ds: xr.Dataset, choices: tuple[str, ...]) -> str | None:
    return next((name for name in choices if name in ds.coords), None)


def nearest_wave_cell(ds: xr.Dataset, lon: float, lat: float):
    time_coord = find_coord(ds, ("time", "TIME"))
    lon_coord = find_coord(ds, ("longitude", "lon", "LONGITUDE"))
    lat_coord = find_coord(ds, ("latitude", "lat", "LATITUDE"))
    needed = ("VHM0", "VTPK", "VMDR")
    if not time_coord or not lon_coord or not lat_coord or not all(v in ds for v in needed):
        raise ValueError("Wave file is missing time/latitude/longitude coordinates or VHM0/VTPK/VMDR.")

    lons = np.asarray(ds[lon_coord].values, dtype=float)
    lats = np.asarray(ds[lat_coord].values, dtype=float)
    if lons.ndim != 1 or lats.ndim != 1:
        raise ValueError("Expected one-dimensional IBI latitude and longitude axes.")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    distance = np.hypot(
        (lon_grid - lon) * 111.32 * np.cos(np.deg2rad(lat)),
        (lat_grid - lat) * 111.32,
    )

    arrays = {}
    wet_fraction = np.ones((lats.size, lons.size), dtype=bool)
    for variable in needed:
        da = ds[variable].transpose(time_coord, lat_coord, lon_coord)
        values = np.asarray(da.values, dtype=float)
        arrays[variable] = values
        wet_fraction &= np.isfinite(values).mean(axis=0) >= 0.95
    if not wet_fraction.any():
        raise ValueError("No IBI grid cell has at least 95% valid wave records for all requested variables.")

    distance[~wet_fraction] = np.inf
    iy, ix = np.unravel_index(np.argmin(distance), distance.shape)
    series = {name: values[:, iy, ix] for name, values in arrays.items()}
    return series, float(lon_grid[iy, ix]), float(lat_grid[iy, ix]), float(distance[iy, ix])


def contiguous_events(indices: np.ndarray, max_below_hours: int = 6) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    # Exceedance hours separated by at most max_below_hours are grouped as one episode.
    split_at = np.flatnonzero(np.diff(indices) > max_below_hours + 1) + 1
    groups = np.split(indices, split_at)
    return [(int(group[0]), int(group[-1])) for group in groups if group.size]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not WAVE_FILE.exists():
        raise FileNotFoundError(f"Missing {WAVE_FILE}; run Scripts/03_download_copernicus_marine.ps1 first.")
    if not CATALOG.exists():
        raise FileNotFoundError(f"Missing {CATALOG}; run Scripts/01_catalog_sentinel2.py first.")

    bbox = CONFIG["site_bbox_wgs84"]
    target_lon, target_lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    ds = xr.open_dataset(WAVE_FILE)
    try:
        series, cell_lon, cell_lat, cell_distance = nearest_wave_cell(ds, target_lon, target_lat)
        times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values, utc=True))
    finally:
        ds.close()

    hs = np.asarray(series["VHM0"], dtype=float)
    tp = np.asarray(series["VTPK"], dtype=float)
    direction = np.asarray(series["VMDR"], dtype=float)
    valid_hs = hs[np.isfinite(hs)]
    q90, q95 = (float(np.quantile(valid_hs, q)) for q in (0.90, 0.95))
    above = np.flatnonzero(np.isfinite(hs) & (hs >= q95))

    event_rows = []
    for event_number, (first, last) in enumerate(contiguous_events(above), start=1):
        peak = first + int(np.nanargmax(hs[first:last + 1]))
        event_rows.append({
            "event_id": f"Q95_{event_number:02d}",
            "threshold_definition": "seasonal 95th percentile of hourly IBI VHM0; episodes merged across gaps <=6 h",
            "season_q90_hs_m": q90,
            "season_q95_hs_m": q95,
            "episode_start_utc": times[first].isoformat(),
            "episode_end_utc": times[last].isoformat(),
            "peak_time_utc": times[peak].isoformat(),
            "peak_hs_m": float(hs[peak]),
            "peak_period_s": float(tp[peak]) if np.isfinite(tp[peak]) else float("nan"),
            "peak_direction_deg": float(direction[peak]) if np.isfinite(direction[peak]) else float("nan"),
            "episode_span_hours": int(round((times[last] - times[first]).total_seconds() / 3600)) + 1,
            "q95_exceedance_hours": int(np.count_nonzero(hs[first:last + 1] >= q95)),
            "sum_hs_squared_m2_hours_diagnostic": float(np.nansum(hs[first:last + 1] ** 2)),
        })

    scene_rows = []
    with CATALOG.open(newline="", encoding="utf-8-sig") as stream:
        for scene in csv.DictReader(stream):
            stamp_text = scene.get("acquisition_utc", "")
            if not stamp_text:
                continue
            stamp = pd.Timestamp(stamp_text)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            stamp = stamp.tz_convert("UTC")
            t_index = int(np.argmin(np.abs(times.asi8 - stamp.value)))
            nearest_offset = (times[t_index] - stamp).total_seconds() / 60
            left = int(np.searchsorted(times.asi8, (stamp - pd.Timedelta(hours=24)).value, side="left"))
            right = int(np.searchsorted(times.asi8, stamp.value, side="right"))
            prior_hs = hs[left:right]
            prior_hs = prior_hs[np.isfinite(prior_hs)]
            scene_rows.append({
                "product_name": scene.get("product_name", ""),
                "acquisition_utc": stamp.isoformat(),
                "product_cloud_cover_pct": scene.get("cloud_cover_pct", ""),
                "ibi_hs_at_nearest_hour_m": float(hs[t_index]) if np.isfinite(hs[t_index]) else float("nan"),
                "ibi_peak_period_at_nearest_hour_s": float(tp[t_index]) if np.isfinite(tp[t_index]) else float("nan"),
                "ibi_wave_direction_at_nearest_hour_deg": float(direction[t_index]) if np.isfinite(direction[t_index]) else float("nan"),
                "nearest_wave_time_offset_min": float(nearest_offset),
                "preceding_24h_max_hs_m": float(np.max(prior_hs)) if prior_hs.size else float("nan"),
                "preceding_24h_mean_hs_m": float(np.mean(prior_hs)) if prior_hs.size else float("nan"),
                "preceding_24h_max_exceeds_season_q95": bool(prior_hs.size and np.max(prior_hs) >= q95),
                "wave_grid_lon": cell_lon,
                "wave_grid_lat": cell_lat,
                "wave_grid_distance_from_target_km": cell_distance,
            })

    event_rows.sort(key=lambda row: row["peak_hs_m"], reverse=True)
    write_csv(EVENTS_OUT, event_rows)
    write_csv(SCENES_OUT, scene_rows)
    settings = {
        "period_start_utc": times.min().isoformat(),
        "period_end_utc": times.max().isoformat(),
        "target_box_center_lon": target_lon,
        "target_box_center_lat": target_lat,
        "selected_wave_grid_lon": cell_lon,
        "selected_wave_grid_lat": cell_lat,
        "selected_wave_grid_distance_km": cell_distance,
        "valid_hour_count": int(valid_hs.size),
        "season_q90_hs_m": q90,
        "season_q95_hs_m": q95,
        "event_definition": "hourly VHM0 >= this six-month period's Q95; exceedance segments separated by no more than 6 below-threshold hours are grouped",
        "interpretation_limit": "These are seasonal upper-tail wave episodes, not climatological extreme storms and not nearshore wave simulations.",
    }
    SETTINGS_OUT.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    print(f"Wrote {len(event_rows)} seasonal Q95 wave episodes to {EVENTS_OUT}")
    print(f"Wrote {len(scene_rows)} Sentinel-2 acquisition/antecedent-wave rows to {SCENES_OUT}")
    print(f"Wave cell: {cell_lon:.5f}, {cell_lat:.5f}; distance from approximate site center: {cell_distance:.2f} km")
    print(f"Seasonal thresholds: Q90={q90:.2f} m; Q95={q95:.2f} m")
    print("Interpret these as regional seasonal wave context, not local surf-zone or sediment-transport observations.")


if __name__ == "__main__":
    main()
