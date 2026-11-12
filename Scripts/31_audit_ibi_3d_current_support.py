"""Read-only audit of IBI daily 3-D current support for a later mobility calculation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "Data" / "Copernicus_Marine" / "Currents_3D" / "IBI_currents_3D_daily_Anza_0_30m_20221001_20230331.nc"
SURVEY = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
OUT = ROOT / "Results" / "Currents_3D_QC"


def find_coord(ds: xr.Dataset, names: tuple[str, ...], standard: str | None = None) -> str:
    for name in names:
        if name in ds.variables:
            return name
    if standard:
        for name, var in ds.variables.items():
            if var.attrs.get("standard_name") == standard:
                return name
    raise ValueError(f"Could not identify coordinate among {names}; available={list(ds.variables)}")


def haversine_m(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6_371_000.0
    a1, a2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dp = np.radians(lat2 - lat1)
    a = np.sin(dp / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main() -> None:
    if not NC.is_file():
        raise SystemExit(f"3-D current subset not found: {NC}\nRun Scripts/30_download_ibi_3d_currents_anza.ps1 first.")
    if not SURVEY.is_file():
        raise SystemExit(f"Survey candidate coordinate file not found: {SURVEY}")
    survey = pd.read_csv(SURVEY)
    required = {"longitude_wgs84_candidate_deg", "latitude_wgs84_candidate_deg", "z_source_m"}
    if not required.issubset(survey.columns):
        raise SystemExit(f"Survey CSV lacks required columns: {sorted(required - set(survey.columns))}")
    wet_survey = survey[np.isfinite(survey["z_source_m"]) & (survey["z_source_m"] < 0)]
    site_lon = float(wet_survey["longitude_wgs84_candidate_deg"].median())
    site_lat = float(wet_survey["latitude_wgs84_candidate_deg"].median())
    survey_depth = -wet_survey["z_source_m"].to_numpy(dtype=float)

    with xr.open_dataset(NC) as ds:
        for needed in ("uo", "vo"):
            if needed not in ds:
                raise SystemExit(f"Variable {needed} absent. Found: {list(ds.data_vars)}")
        lat_name = find_coord(ds, ("latitude", "lat", "nav_lat"), "latitude")
        lon_name = find_coord(ds, ("longitude", "lon", "nav_lon"), "longitude")
        time_name = find_coord(ds, ("time", "time_counter"), "time")
        depth_name = find_coord(ds, ("depth", "deptht", "depthu", "lev", "z"), "depth")
        lat = np.asarray(ds[lat_name].values, dtype=float)
        lon = np.asarray(ds[lon_name].values, dtype=float)
        if lat.ndim == lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            spatial_dims = (ds[lat_name].dims[0], ds[lon_name].dims[0])
        elif lat.ndim == lon.ndim == 2 and ds[lat_name].dims == ds[lon_name].dims:
            lat_grid, lon_grid = lat, lon
            spatial_dims = ds[lat_name].dims
        else:
            raise SystemExit(f"Unsupported lat/lon coordinate shapes: {lat.shape}, {lon.shape}")

        u = ds["uo"]
        nonspatial_dims = [d for d in u.dims if d not in spatial_dims]
        wet = u.notnull().any(dim=nonspatial_dims).transpose(*spatial_dims).values
        wet = np.asarray(wet, dtype=bool).squeeze()
        if wet.shape != lat_grid.shape:
            raise SystemExit(f"Wet mask shape {wet.shape} differs from grid {lat_grid.shape}")
        distance = haversine_m(site_lon, site_lat, lon_grid, lat_grid)
        distance[~wet] = np.inf
        if not np.isfinite(distance).any():
            raise SystemExit("No wet current grid cell was found in this subset.")
        iy, ix = np.unravel_index(np.argmin(distance), distance.shape)
        nearest_lon = float(lon_grid[iy, ix])
        nearest_lat = float(lat_grid[iy, ix])
        nearest_km = float(distance[iy, ix] / 1000)
        idx = {spatial_dims[0]: int(iy), spatial_dims[1]: int(ix)}

        depth_vals = np.asarray(ds[depth_name].values, dtype=float).squeeze()
        depth_attrs = dict(ds[depth_name].attrs)
        if depth_vals.ndim != 1:
            raise SystemExit(f"Depth coordinate is not one-dimensional: {depth_vals.shape}")
        positive = str(depth_attrs.get("positive", "down")).lower()
        depth_down = depth_vals if positive == "down" or np.nanmedian(depth_vals) >= 0 else -depth_vals
        depth_dim = ds[depth_name].dims[0]
        tdim = ds[time_name].dims[0]
        profile_u = ds["uo"].isel({**idx, tdim: slice(None)}).notnull().any(dim=tdim).values
        profile_v = ds["vo"].isel({**idx, tdim: slice(None)}).notnull().any(dim=tdim).values
        profile_valid = np.asarray(profile_u & profile_v, dtype=bool).squeeze()
        available = depth_down[profile_valid]
        if available.size == 0:
            raise SystemExit("Nearest wet node has no depth level with paired uo/vo values.")

        deptho_name = next((n for n in ("deptho", "bathymetry", "sea_floor_depth_below_geoid") if n in ds), None)
        model_bottom = None
        if deptho_name:
            model_bottom = float(ds[deptho_name].isel(idx).values)

        level_rows = []
        for h in (2, 5, 10, 15, 20):
            k = int(np.argmin(np.abs(available - h)))
            level_rows.append({"local_survey_depth_m": h, "closest_available_current_level_m": float(available[k]),
                               "vertical_offset_m": float(available[k] - h)})
        time_values = np.asarray(ds[time_name].values)
        time_range = [str(time_values.min()), str(time_values.max())]
        result = {
            "input_file": str(NC), "product": "IBI multi-year physics reanalysis; daily mean currents",
            "dataset_id": "cmems_mod_ibi_phy-cur_my_0.027deg_P1D-m", "doi": "10.48670/moi-00029",
            "variables": {"uo": dict(ds["uo"].attrs), "vo": dict(ds["vo"].attrs)},
            "dimensions": {k: int(v) for k, v in ds.sizes.items()}, "time_range": time_range,
            "depth_coordinate": depth_name, "depth_units": depth_attrs.get("units"),
            "depth_positive_attribute": depth_attrs.get("positive"),
            "available_depth_levels_m_positive_down": [float(v) for v in available],
            "survey_wet_point_count": int(len(wet_survey)),
            "survey_median_lon_lat": [site_lon, site_lat],
            "nearest_wet_model_node_lon_lat": [nearest_lon, nearest_lat],
            "nearest_wet_node_distance_km": nearest_km,
            "model_bottom_depth_variable": deptho_name,
            "model_bottom_depth_value": model_bottom,
            "closest_levels_for_target_local_depths": level_rows,
            "interpretation_warning": "A wet grid point and depth levels do not prove near-bed current support for the surf zone. Daily regional currents (~3 km grid) must not be treated as observed local bed currents. Compare model bathymetry, local depths, and vertical offsets before using combined shear equations."
        }

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "IBI_3D_current_support_audit.json"
    csv_path = OUT / "IBI_3D_current_depth_level_match.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(level_rows).to_csv(csv_path, index=False)
    print("Read-only IBI 3-D current support audit complete; no current/bathymetry data were changed.")
    print(f"Grid dimensions: {result['dimensions']}")
    print(f"Daily time range: {time_range[0]} to {time_range[1]}")
    print(f"Nearest wet node: {nearest_lon:.5f}, {nearest_lat:.5f}; distance={nearest_km:.2f} km")
    print(f"Available wet profile levels (m, positive down): {', '.join(f'{v:g}' for v in available)}")
    print(f"Audit: {json_path}")
    print(f"Depth comparison: {csv_path}")
    print("Do not run a combined bed-stress calculation until these vertical and horizontal offsets are reviewed.")


if __name__ == "__main__":
    main()
