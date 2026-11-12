from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "Config" / "project_config.json"
NC_PATH = ROOT / "Data" / "Copernicus_Marine" / "Ocean_Colour_HR" / "Anza_CMEMS_HR_OC_SPM_TUR_L3_20221101_20230331.nc"
WAVE_PATH = ROOT / "Data" / "Copernicus_Marine" / "IBI_wave_Anza_subset.nc"
SUPPORT_PATH = ROOT / "Results" / "CMEMS_Ocean_Colour_Audit" / "cmems_hr_oc_water_support_by_date.csv"
SURVEY_POINTS_PATH = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
FORCING_PATH = ROOT / "Results" / "scene_forcing_and_quality.csv"
OUT = ROOT / "Results" / "Additional_Analyses" / "Within_Cell_Wave_Response"

WATER_FRACTION_THRESHOLD = 0.50
MIN_DAILY_WATER_CELLS = 1000
REPEATED_SUPPORT_FRACTION = 0.70
MIN_DATES = 20
MIN_SURVEY_CELLS = 50
MIN_HOURLY_WAVE_SAMPLES = 18
TEMPORAL_BLOCK_LENGTH = 3  # consecutive eligible acquisitions, not calendar days
BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20261112


def find_name(names: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {str(name).lower(): str(name) for name in names}
    return next((by_lower[c.lower()] for c in candidates if c.lower() in by_lower), None)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=None, keep_default_na=True)


def nearest_grid_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or not values.size:
        raise ValueError("IBI wave grid coordinates are empty or invalid.")
    return int(np.argmin(np.abs(values - target)))


def load_wave_series(path: Path, lon: float, lat: float) -> tuple[pd.DataFrame, float]:
    with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as source:
        t_name = find_name(list(source.coords) + list(source.dims), ("time", "datetime"))
        x_name = find_name(list(source.coords) + list(source.dims), ("longitude", "lon"))
        y_name = find_name(list(source.coords) + list(source.dims), ("latitude", "lat"))
        if not all((t_name, x_name, y_name)):
            raise ValueError(f"Cannot identify wave coordinates in {path.name}.")
        for variable in ("VHM0", "VTPK"):
            if variable not in source:
                raise ValueError(f"Required wave variable {variable} is absent from {path.name}.")
        lons = np.asarray(source[x_name].values, dtype=float)
        lats = np.asarray(source[y_name].values, dtype=float)
        ix, iy = nearest_grid_index(lons, lon), nearest_grid_index(lats, lat)
        grid_lon, grid_lat = float(lons[ix]), float(lats[iy])
        distance_km = float(np.hypot((grid_lon - lon) * 111.32 * np.cos(np.deg2rad(lat)), (grid_lat - lat) * 111.32))
        times = pd.to_datetime(np.asarray(source[t_name].values).reshape(-1), utc=True, errors="coerce")
        hs = source["VHM0"].isel({y_name: iy, x_name: ix}).transpose(t_name).values.astype(float)
        tp = source["VTPK"].isel({y_name: iy, x_name: ix}).transpose(t_name).values.astype(float)
    frame = pd.DataFrame({"time_utc": times, "hs_m": hs, "tp_s": tp}).dropna(subset=["time_utc"])
    return frame.sort_values("time_utc").reset_index(drop=True), distance_km


def make_forcing_by_date(forcing: pd.DataFrame) -> dict[str, dict[str, object]]:
    if "acquisition_utc" not in forcing or "product_name" not in forcing:
        raise ValueError("The scene-forcing table lacks acquisition_utc/product_name.")
    forcing = forcing.copy()
    forcing["acquisition_time"] = pd.to_datetime(forcing["acquisition_utc"], utc=True, errors="coerce")
    forcing = forcing.dropna(subset=["acquisition_time"])
    forcing["date_utc"] = forcing["acquisition_time"].dt.strftime("%Y-%m-%d")
    result: dict[str, dict[str, object]] = {}
    for date, group in forcing.groupby("date_utc", sort=True):
        if "cloud_cover_product_pct" in group:
            group = group.assign(_cloud=pd.to_numeric(group["cloud_cover_product_pct"], errors="coerce"))
            group = group.sort_values(["_cloud", "acquisition_time"], na_position="last")
        row = group.iloc[0].to_dict()
        row["same_day_scene_count"] = int(len(group))
        result[str(date)] = row
    return result


def antecedent_wave(window: pd.DataFrame, overpass: pd.Timestamp) -> dict[str, float | int]:
    start = overpass - pd.Timedelta(hours=24)
    selected = window[(window["time_utc"] > start) & (window["time_utc"] <= overpass)]
    hs = pd.to_numeric(selected["hs_m"], errors="coerce").to_numpy(dtype=float)
    tp = pd.to_numeric(selected["tp_s"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(hs) & (hs >= 0)
    hs = hs[valid]
    if hs.size < MIN_HOURLY_WAVE_SAMPLES:
        return {"wave_samples_24h_n": int(hs.size), "hs_24h_mean_m": np.nan, "hs_24h_max_m": np.nan}
    return {
        "wave_samples_24h_n": int(hs.size),
        "hs_24h_mean_m": float(np.mean(hs)),
        "hs_24h_max_m": float(np.max(hs)),
    }


def fixed_effect_fit(frame: pd.DataFrame, predictors: list[str], group_col: str = "cell_code") -> dict[str, object]:
    columns = ["log_spm", *predictors, group_col]
    if "date_index" in frame and "date_index" not in columns:
        columns.append("date_index")
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError("No complete observations remain for the fixed-cell model.")
    groups, inverse = np.unique(data[group_col].to_numpy(), return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    y = data["log_spm"].to_numpy(dtype=float)
    y_mean = np.bincount(inverse, weights=y) / counts
    yd = y - y_mean[inverse]
    x_cols = []
    for predictor in predictors:
        x = data[predictor].to_numpy(dtype=float)
        x_mean = np.bincount(inverse, weights=x) / counts
        x_cols.append(x - x_mean[inverse])
    xd = np.column_stack(x_cols)
    beta, _, rank, _ = np.linalg.lstsq(xd, yd, rcond=None)
    if rank < len(predictors):
        raise ValueError("The fixed-cell design matrix is rank deficient; reduce covariates or dates.")
    residual = yd - xd @ beta
    denom = float(np.sum(yd**2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else np.nan
    return {
        "beta": beta,
        "n_obs": int(len(data)),
        "n_cells": int(len(groups)),
        "n_dates": int(data["date_index"].nunique()) if "date_index" in data else 0,
        "within_r2": r2,
    }


def moving_date_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    length = max(1, min(int(length), n))
    starts = np.arange(max(1, n - length + 1))
    sampled: list[int] = []
    while len(sampled) < n:
        start = int(rng.choice(starts))
        sampled.extend(range(start, start + length))
    return np.asarray(sampled[:n], dtype=int)


def two_axis_block_bootstrap(
    frame: pd.DataFrame,
    predictors: list[str],
    *,
    reps: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block_values = np.sort(frame["spatial_block_500m"].dropna().unique())
    dates = np.sort(frame["date_index"].unique())
    if len(block_values) < 10 or len(dates) < 10:
        return np.empty((0, len(predictors)), dtype=float)
    frame = frame.copy().reset_index(drop=True)
    lookup: dict[tuple[object, int], np.ndarray] = {}
    for (block, date_index), group in frame.groupby(["spatial_block_500m", "date_index"], sort=False):
        lookup[(block, int(date_index))] = group.index.to_numpy(dtype=int)
    cell_codes = frame["cell_code"].to_numpy(dtype=int)
    estimates: list[np.ndarray] = []
    for _ in range(reps):
        selected_blocks = rng.choice(block_values, size=len(block_values), replace=True)
        selected_dates = moving_date_indices(len(dates), block_length, rng)
        sampled_rows: list[int] = []
        sampled_groups: list[int] = []
        for block_draw, block in enumerate(selected_blocks):
            group_offset = (block_draw + 1) * (int(frame["cell_code"].max()) + 1)
            for date_pos in selected_dates:
                date_index = int(dates[date_pos])
                row_ids = lookup.get((block, date_index))
                if row_ids is None or not len(row_ids):
                    continue
                sampled_rows.extend(row_ids.tolist())
                sampled_groups.extend((cell_codes[row_ids] + group_offset).tolist())
        if len(sampled_rows) < 50:
            continue
        sample = frame.iloc[np.asarray(sampled_rows, dtype=int)].copy()
        sample["bootstrap_cell"] = np.asarray(sampled_groups, dtype=int)
        try:
            result = fixed_effect_fit(sample, predictors, group_col="bootstrap_cell")
            estimates.append(np.asarray(result["beta"], dtype=float))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return np.vstack(estimates) if estimates else np.empty((0, len(predictors)), dtype=float)


def main() -> None:
    for path in (CONFIG_PATH, NC_PATH, WAVE_PATH, SUPPORT_PATH, SURVEY_POINTS_PATH, FORCING_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bbox = config["site_bbox_wgs84"]
    target_lon, target_lat = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    support = read_csv(SUPPORT_PATH)
    threshold = pd.to_numeric(support["stable_water_fraction_threshold"], errors="coerce")
    valid_n = pd.to_numeric(support["valid_water_cmems_cells"], errors="coerce")
    support = support[
        support["variable"].eq("SPM")
        & np.isclose(threshold, WATER_FRACTION_THRESHOLD)
        & valid_n.ge(MIN_DAILY_WATER_CELLS)
    ].copy()
    support["date_utc"] = pd.to_datetime(support["date_utc"], errors="coerce").dt.strftime("%Y-%m-%d")
    support = support.dropna(subset=["date_utc"]).drop_duplicates("date_utc").sort_values("date_utc")
    if support.empty:
        raise RuntimeError("No daily HR-OC SPM dates meet the water-support threshold from Script 22.")

    forcing_by_date = make_forcing_by_date(read_csv(FORCING_PATH))
    survey_points = read_csv(SURVEY_POINTS_PATH)
    required_points = {"z_source_m", "longitude_wgs84_candidate_deg", "latitude_wgs84_candidate_deg"}
    if not required_points.issubset(survey_points.columns):
        raise ValueError(f"Survey point file must contain: {sorted(required_points)}")

    wave_series, wave_grid_distance_km = load_wave_series(WAVE_PATH, target_lon, target_lat)
    wave_series["time_utc"] = pd.to_datetime(wave_series["time_utc"], utc=True)

    panel_rows: list[dict[str, object]] = []
    date_rows: list[dict[str, object]] = []
    with xr.open_dataset(NC_PATH, decode_cf=True, mask_and_scale=True) as source:
        t_name = find_name(list(source.coords) + list(source.dims), ("time", "datetime"))
        x_name = find_name(list(source.coords) + list(source.dims), ("longitude", "lon"))
        y_name = find_name(list(source.coords) + list(source.dims), ("latitude", "lat"))
        if not all((t_name, x_name, y_name)) or "SPM" not in source.data_vars:
            raise ValueError("The HR-OC file lacks expected time/latitude/longitude/SPM variables.")
        ds = source.sortby(y_name, ascending=False).sortby(x_name, ascending=True)
        time_values = pd.to_datetime(np.asarray(ds[t_name].values).reshape(-1), utc=True, errors="coerce")
        if time_values.isna().any():
            raise ValueError("HR-OC has invalid time coordinates.")
        date_to_index = {timestamp.strftime("%Y-%m-%d"): idx for idx, timestamp in enumerate(time_values)}
        grid_lons = np.asarray(ds[x_name].values, dtype=float)
        grid_lats = np.asarray(ds[y_name].values, dtype=float)
        dxs, dys = np.diff(grid_lons), np.diff(grid_lats)
        dx, dy = float(np.median(dxs)), float(np.median(dys))
        if not np.allclose(dxs, dx, rtol=1e-5, atol=1e-10) or not np.allclose(dys, dy, rtol=1e-5, atol=1e-10):
            raise ValueError("HR-OC grid is not regular; refusing approximate bathymetry-cell assignment.")
        left, top = float(grid_lons[0] - dx / 2), float(grid_lats[0] - dy / 2)
        point_lon = pd.to_numeric(survey_points["longitude_wgs84_candidate_deg"], errors="coerce").to_numpy(dtype=float)
        point_lat = pd.to_numeric(survey_points["latitude_wgs84_candidate_deg"], errors="coerce").to_numpy(dtype=float)
        point_z = pd.to_numeric(survey_points["z_source_m"], errors="coerce").to_numpy(dtype=float)
        finite_xy = np.isfinite(point_lon) & np.isfinite(point_lat)
        safe_lon = np.where(finite_xy, point_lon, left - 10 * abs(dx))
        safe_lat = np.where(finite_xy, point_lat, top + 10 * abs(dy))
        col = np.floor((safe_lon - left) / dx).astype(int)
        row = np.floor((top - safe_lat) / abs(dy)).astype(int)
        in_grid = finite_xy & (row >= 0) & (row < len(grid_lats)) & (col >= 0) & (col < len(grid_lons))
        submerged = in_grid & np.isfinite(point_z) & (point_z < 0)
        sounding_rows = pd.DataFrame({
            "row": row[submerged],
            "column": col[submerged],
            "depth_below_local_zero_m": -point_z[submerged],
        })
        survey_cells = (
            sounding_rows.groupby(["row", "column"], as_index=False)
            .agg(soundings_per_cell=("depth_below_local_zero_m", "size"),
                 depth_below_local_zero_m=("depth_below_local_zero_m", "median"))
        )
        survey_cells = survey_cells[survey_cells["soundings_per_cell"] >= 5].copy().reset_index(drop=True)
        survey_cells["longitude"] = grid_lons[survey_cells["column"].to_numpy(dtype=int)]
        survey_cells["latitude"] = grid_lats[survey_cells["row"].to_numpy(dtype=int)]
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
        easting, northing = to_utm.transform(survey_cells["longitude"].to_numpy(), survey_cells["latitude"].to_numpy())
        block_x = np.floor(np.asarray(easting) / 500.0).astype(int)
        block_y = np.floor(np.asarray(northing) / 500.0).astype(int)
        survey_cells["spatial_block_500m"] = [f"{bx}:{by}" for bx, by in zip(block_x, block_y)]
        survey_cells["cell_code"] = np.arange(len(survey_cells), dtype=int)
        if survey_cells.empty:
            raise RuntimeError("No HR-OC cells contain at least five submerged survey soundings.")
        spm_attrs = dict(ds["SPM"].attrs)
        valid_min = spm_attrs.get("valid_min")
        valid_max = spm_attrs.get("valid_max")

        for _, support_row in support.iterrows():
            date = str(support_row["date_utc"])
            scene = forcing_by_date.get(date)
            if date not in date_to_index or scene is None:
                date_rows.append({"date_utc": date, "included_in_candidate_dates": False,
                                  "exclusion_reason": "no matching HR-OC date or Sentinel-2 scene forcing record"})
                continue
            overpass = pd.Timestamp(scene["acquisition_time"])
            if overpass.tzinfo is None:
                overpass = overpass.tz_localize("UTC")
            wave_metrics = antecedent_wave(wave_series, overpass)
            tide = pd.to_numeric(pd.Series([scene.get("got56_anomaly_from_utc_day_mean_m")]), errors="coerce").iloc[0]
            hs24 = float(wave_metrics["hs_24h_max_m"])
            spm_array = np.array(ds["SPM"].isel({t_name: date_to_index[date]}).transpose(y_name, x_name).values, dtype=float, copy=True)
            if valid_min is not None:
                spm_array[spm_array < float(valid_min)] = np.nan
            if valid_max is not None:
                spm_array[spm_array > float(valid_max)] = np.nan
            if survey_cells["row"].max() >= spm_array.shape[0] or survey_cells["column"].max() >= spm_array.shape[1]:
                raise ValueError("Survey-cell indices fall outside the sorted HR-OC grid; refusing approximate sampling.")
            sampled = spm_array[survey_cells["row"].to_numpy(), survey_cells["column"].to_numpy()]
            usable = np.isfinite(sampled) & (sampled > 0)
            local_n = int(usable.sum())
            date_row = {
                "date_utc": date,
                "acquisition_utc_used_for_forcing": overpass.isoformat(),
                "same_day_scene_count": int(scene.get("same_day_scene_count", 1)),
                "cloud_cover_scene_metadata_pct": pd.to_numeric(pd.Series([scene.get("cloud_cover_product_pct")]), errors="coerce").iloc[0],
                "hr_oc_water_cells_ge_1000": int(pd.to_numeric(pd.Series([support_row["valid_water_cmems_cells"]]), errors="coerce").iloc[0]),
                "survey_cells_total": int(len(survey_cells)),
                "valid_survey_cells_n": local_n,
                "valid_survey_cell_fraction": local_n / len(survey_cells),
                "survey_cell_spm_median_g_m3": float(np.median(sampled[usable])) if local_n else np.nan,
                "survey_cell_spm_p90_g_m3": float(np.percentile(sampled[usable], 90)) if local_n else np.nan,
                "hs_24h_mean_m": wave_metrics["hs_24h_mean_m"],
                "hs_24h_max_m": wave_metrics["hs_24h_max_m"],
                "wave_samples_24h_n": wave_metrics["wave_samples_24h_n"],
                "got56_anomaly_from_utc_day_mean_m": tide,
                "wave_grid_distance_to_bbox_center_km": wave_grid_distance_km,
                "calendar_day_match_only_hr_oc_source_scene_unverified": True,
            }
            date_rows.append(date_row)
            if wave_metrics["wave_samples_24h_n"] < MIN_HOURLY_WAVE_SAMPLES or not np.isfinite(hs24):
                continue
            for idx, cell in survey_cells.iterrows():
                value = sampled[idx]
                if not np.isfinite(value) or value <= 0:
                    continue
                panel_rows.append({
                    "date_utc": date,
                    "date_index": 0,
                    "cell_code": int(cell["cell_code"]),
                    "spatial_block_500m": str(cell["spatial_block_500m"]),
                    "depth_below_local_zero_m": float(cell["depth_below_local_zero_m"]),
                    "spm_g_m3": float(value),
                    "log_spm": float(np.log(value)),
                    "hs_24h_max_m": hs24,
                    "hs_24h_mean_m": float(wave_metrics["hs_24h_mean_m"]),
                    "tide_anomaly_m": float(tide) if pd.notna(tide) else np.nan,
                })

    date_audit = pd.DataFrame(date_rows).sort_values("date_utc")
    date_audit_path = OUT / "within_cell_wave_response_date_support.csv"
    date_audit.to_csv(date_audit_path, index=False, encoding="utf-8-sig")
    if not panel_rows:
        raise RuntimeError(f"No local bathymetry-cell matchups were built. Inspect {date_audit_path}.")
    raw_panel = pd.DataFrame(panel_rows)
    candidate_dates = sorted(raw_panel["date_utc"].unique())
    candidate_cells = sorted(raw_panel["cell_code"].unique())

    # Iteratively retain a repeated high-coverage core: >=70% of survey cells per date,
    # and >=70% of dates per cell. This avoids comparing changing shoreline footprints.
    current_dates = list(candidate_dates)
    current_cells = list(candidate_cells)
    for _ in range(8):
        cell_presence = raw_panel[raw_panel["date_utc"].isin(current_dates)].groupby("cell_code")["date_utc"].nunique()
        min_cell_dates = int(np.ceil(REPEATED_SUPPORT_FRACTION * max(1, len(current_dates))))
        new_cells = sorted(cell_presence[cell_presence >= min_cell_dates].index.astype(int).tolist())
        if not new_cells:
            current_dates, current_cells = [], []
            break
        date_presence = raw_panel[raw_panel["cell_code"].isin(new_cells)].groupby("date_utc")["cell_code"].nunique()
        min_date_cells = int(np.ceil(REPEATED_SUPPORT_FRACTION * len(new_cells)))
        new_dates = sorted(date_presence[date_presence >= min_date_cells].index.astype(str).tolist())
        if new_dates == current_dates and new_cells == current_cells:
            current_dates, current_cells = new_dates, new_cells
            break
        current_dates, current_cells = new_dates, new_cells

    panel = raw_panel[raw_panel["date_utc"].isin(current_dates) & raw_panel["cell_code"].isin(current_cells)].copy()
    date_order = {date: index for index, date in enumerate(current_dates)}
    panel["date_index"] = panel["date_utc"].map(date_order).astype(int)
    if current_dates:
        date_numeric = pd.to_datetime(current_dates, utc=True)
        elapsed_days = (date_numeric - date_numeric.min()).days.to_numpy(dtype=float)
        day_sd = float(np.std(elapsed_days, ddof=0))
        if day_sd == 0:
            day_sd = 1.0
        time_z_map = {date: float((elapsed_days[i] - np.mean(elapsed_days)) / day_sd) for i, date in enumerate(current_dates)}
        wave_by_date = raw_panel.drop_duplicates("date_utc").set_index("date_utc")["hs_24h_max_m"]
        log_wave = np.log1p(wave_by_date.reindex(current_dates).to_numpy(dtype=float))
        wave_sd = float(np.std(log_wave, ddof=0))
        if wave_sd == 0:
            wave_sd = 1.0
        log_wave_z_map = {date: float((log_wave[i] - np.mean(log_wave)) / wave_sd) for i, date in enumerate(current_dates)}
        tide_by_date = raw_panel.drop_duplicates("date_utc").set_index("date_utc")["tide_anomaly_m"].reindex(current_dates)
        tide_values = tide_by_date.to_numpy(dtype=float)
        tide_mean, tide_sd = float(np.nanmean(tide_values)), float(np.nanstd(tide_values, ddof=0))
        if not np.isfinite(tide_sd) or tide_sd == 0:
            tide_sd = 1.0
        tide_z_map = {date: float((tide_by_date.loc[date] - tide_mean) / tide_sd) if pd.notna(tide_by_date.loc[date]) else np.nan for date in current_dates}
        panel["log_hs24_max_z"] = panel["date_utc"].map(log_wave_z_map)
        panel["time_z"] = panel["date_utc"].map(time_z_map)
        panel["tide_anomaly_z"] = panel["date_utc"].map(tide_z_map)

    panel_path = OUT / "within_cell_wave_response_panel.csv.gz"
    panel.to_csv(panel_path, index=False, compression="gzip", encoding="utf-8-sig")
    date_audit["included_in_repeated_cell_panel"] = date_audit["date_utc"].isin(current_dates)
    date_audit.to_csv(date_audit_path, index=False, encoding="utf-8-sig")

    blocks = panel["spatial_block_500m"].nunique() if not panel.empty else 0
    summary_lines = [
        "Within-cell, bathymetry-conditioned temporal analysis at Anza.",
        f"HR-OC dates meeting water support: {len(support)}; repeated-core dates retained: {len(current_dates)}.",
        f"Survey cells retained: {panel['cell_code'].nunique() if not panel.empty else 0} of {len(survey_cells)}; 500 m spatial blocks: {blocks}.",
        f"Repeated-support rule: at least {REPEATED_SUPPORT_FRACTION:.0%} of retained dates per cell and cells per date.",
        f"IBI wave grid distance from the Anza working-box centre: {wave_grid_distance_km:.2f} km.",
        "The HR-OC date is paired by calendar day with a local Sentinel-2 scene; exact source-granule identity is not established.",
        "Outcome is the satellite-derived HR-OC SPM estimate, not an in-situ concentration measurement.",
        "Cell fixed effects absorb time-invariant depth/location/bottom-optics differences; a linear time trend controls gradual seasonal drift.",
        "GOT5.6 is a modeled anomaly relative to the UTC-day mean, not a local tide-gauge observation or port-datum correction.",
        f"Bootstrap: {BOOTSTRAP_REPLICATES} replicates, resampling 500 m spatial blocks and moving blocks of {TEMPORAL_BLOCK_LENGTH} consecutive eligible acquisitions.",
        "Intervals are exploratory block-bootstrap intervals; they do not establish causality or sediment transport.",
    ]
    if len(current_dates) < MIN_DATES or panel["cell_code"].nunique() < MIN_SURVEY_CELLS or blocks < 10:
        summary_lines.append(
            f"NO MODEL FIT: minimums not met (need >= {MIN_DATES} dates, >= {MIN_SURVEY_CELLS} cells, and >= 10 spatial blocks)."
        )
        (OUT / "within_cell_wave_response_notes_FR.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print("Within-cell wave-response support audit complete; model not fitted because minimum repeated support was not met.")
        print(f"  Repeated-core dates/cells/blocks: {len(current_dates)}/{panel['cell_code'].nunique()}/{blocks}")
        print(f"  Outputs: {OUT}")
        return

    models = [
        ("Wave + temporal trend", ["log_hs24_max_z", "time_z"]),
        ("Wave + tide anomaly + temporal trend", ["log_hs24_max_z", "tide_anomaly_z", "time_z"]),
    ]
    model_rows: list[dict[str, object]] = []
    for model_label, predictors in models:
        complete = panel.dropna(subset=["log_spm", *predictors]).copy()
        try:
            fit = fixed_effect_fit(complete, predictors)
        except ValueError as exc:
            print(f"  Skipped {model_label}: {exc}")
            continue
        boot = two_axis_block_bootstrap(
            complete, predictors, reps=BOOTSTRAP_REPLICATES,
            block_length=TEMPORAL_BLOCK_LENGTH,
            seed=RANDOM_SEED + len(model_rows),
        )
        for idx, predictor in enumerate(predictors):
            beta = float(fit["beta"][idx])
            if boot.shape[0] >= 100:
                low, high = np.percentile(boot[:, idx], [2.5, 97.5])
                low, high = float(low), float(high)
            else:
                low, high = np.nan, np.nan
            model_rows.append({
                "outcome": "HR-OC SPM",
                "model": model_label,
                "predictor": predictor,
                "standardized_beta_log_spm_per_1sd_predictor": beta,
                "effect_pct_per_1sd_predictor": float(100.0 * np.expm1(beta)) if predictor == "log_hs24_max_z" else "",
                "block_bootstrap_95ci_low": low,
                "block_bootstrap_95ci_high": high,
                "bootstrap_replicates_valid": int(boot.shape[0]),
                "observations_n": fit["n_obs"],
                "cells_n": fit["n_cells"],
                "dates_n": fit["n_dates"],
                "spatial_blocks_n": int(blocks),
                "within_r2": fit["within_r2"],
            })
    if not model_rows:
        summary_lines.append("NO MODEL FIT: the within-cell design was not estimable after predictor-support checks.")
        (OUT / "within_cell_wave_response_notes_FR.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print("Within-cell wave-response support audit complete; no fixed-effect model was estimable.")
        print(f"  Support and panel outputs: {OUT}")
        return
    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(OUT / "within_cell_wave_response_model.csv", index=False, encoding="utf-8-sig")

    main_rows = model_table[model_table["predictor"].eq("log_hs24_max_z")].dropna(
        subset=["standardized_beta_log_spm_per_1sd_predictor"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), gridspec_kw={"height_ratios": [1.1, 0.9]}, constrained_layout=True)
    fig.suptitle("Bathymetry-conditioned HR-OC response to antecedent wave forcing at Anza", fontsize=14, fontweight="bold")
    audit_by_date = date_audit.set_index("date_utc")
    plotted = audit_by_date.reindex(current_dates)
    dates = pd.to_datetime(current_dates)
    med = pd.to_numeric(plotted["survey_cell_spm_median_g_m3"], errors="coerce").to_numpy(dtype=float)
    wave_max = pd.to_numeric(plotted["hs_24h_max_m"], errors="coerce").to_numpy(dtype=float)
    ax = axes[0]
    ax.plot(dates, med, color="#176B87", marker="o", lw=1.7, ms=4, label="HR-OC SPM median on repeated survey cells")
    ax.set_ylabel("HR-OC SPM estimate (g m$^{-3}$)", color="#176B87")
    ax.tick_params(axis="y", labelcolor="#176B87")
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.6)
    axr = ax.twinx()
    axr.plot(dates, wave_max, color="#D97732", marker="s", lw=1.2, ms=3.5, alpha=0.9, label="Maximum IBI Hs in prior 24 h")
    axr.set_ylabel("IBI significant wave height (m)", color="#D97732")
    axr.tick_params(axis="y", labelcolor="#D97732")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False, fontsize=8)
    ax.set_title("A  Scene-level temporal context (calendar-day HR-OC matchup)", loc="left", fontweight="bold")

    ax = axes[1]
    plot_rows = main_rows.reset_index(drop=True)
    y_positions = np.arange(len(plot_rows))[::-1]
    for i, row in plot_rows.iterrows():
        beta = float(row["standardized_beta_log_spm_per_1sd_predictor"])
        low = float(row["block_bootstrap_95ci_low"]) if pd.notna(row["block_bootstrap_95ci_low"]) else np.nan
        high = float(row["block_bootstrap_95ci_high"]) if pd.notna(row["block_bootstrap_95ci_high"]) else np.nan
        color = "#176B87" if "tide anomaly" not in str(row["model"]).lower() else "#6B63A8"
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [y_positions[i], y_positions[i]], color=color, lw=1.5, solid_capstyle="round")
            ax.plot([low, low], [y_positions[i] - 0.08, y_positions[i] + 0.08], color=color, lw=1.2)
            ax.plot([high, high], [y_positions[i] - 0.08, y_positions[i] + 0.08], color=color, lw=1.2)
            ax.plot(beta, y_positions[i], "o", color=color, ms=6)
        else:
            ax.plot(beta, y_positions[i], "x", color=color, ms=7)
    ax.axvline(0, color="#475569", linestyle="--", linewidth=0.9)
    ax.set_yticks(y_positions, plot_rows["model"].tolist())
    ax.set_xlabel("Change in log(SPM) per 1-SD increase in antecedent wave proxy")
    ax.set_title("B  Within-cell fixed-effect wave coefficient and two-axis block-bootstrap interval", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#CBD5E1", alpha=0.55, linewidth=0.6)
    fig_path = OUT / "Figure_within_cell_wave_response.png"
    fig.savefig(fig_path, dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_within_cell_wave_response.pdf", bbox_inches="tight")
    plt.close(fig)

    primary = main_rows[main_rows["model"].eq("Wave + temporal trend")]
    adjusted = main_rows[main_rows["model"].eq("Wave + tide anomaly + temporal trend")]
    beta_text = ""
    if not primary.empty:
        r = primary.iloc[0]
        beta_text += f"Primary Hs coefficient: beta={float(r['standardized_beta_log_spm_per_1sd_predictor']):.3f}, 95% block-bootstrap CI [{float(r['block_bootstrap_95ci_low']):.3f}, {float(r['block_bootstrap_95ci_high']):.3f}] when estimable.\n"
    if not adjusted.empty:
        r = adjusted.iloc[0]
        beta_text += f"Tide-adjusted Hs coefficient: beta={float(r['standardized_beta_log_spm_per_1sd_predictor']):.3f}, 95% block-bootstrap CI [{float(r['block_bootstrap_95ci_low']):.3f}, {float(r['block_bootstrap_95ci_high']):.3f}] when estimable.\n"
    summary_lines.extend([
        f"Repeated-core inclusion: {len(current_dates)} dates, {panel['cell_code'].nunique()} cells, {blocks} spatial blocks.",
        "Primary model: log(HR-OC SPM) with cell fixed effects, standardized log(1 + 24 h maximum Hs), and a linear time trend.",
        "Sensitivity model adds the standardized GOT5.6 tide anomaly; this is not a locally validated tide-gauge value.",
        beta_text.rstrip(),
        "Interpret coefficient intervals as exploratory associations. HR-OC is satellite-derived and same-calendar-day identity with the local Sentinel-2 scene is unverified.",
        "The model does not estimate a sediment flux, transport direction, or causal wave effect; IBI forcing is regional and does not resolve the surf zone.",
    ])
    (OUT / "within_cell_wave_response_notes_FR.txt").write_text("\n".join(line for line in summary_lines if line) + "\n", encoding="utf-8")
    (OUT / "figure_caption_EN.txt").write_text(
        "Figure. Bathymetry-conditioned temporal response of the Copernicus Marine HR-OC SPM estimate to antecedent wave forcing at Anza. "
        "(A) Daily median SPM estimate over repeated cells containing local survey soundings, paired by calendar day with the maximum IBI significant wave height in the preceding 24 h. "
        "(B) Standardized within-cell fixed-effect coefficient for antecedent wave height, with exploratory intervals from resampling 500 m spatial blocks and moving blocks of three eligible acquisitions. "
        "The HR-OC source granule is not confirmed to be identical to the local Sentinel-2 product; modeled regional forcing and satellite-derived SPM are not in-situ validation.",
        encoding="utf-8",
    )

    print("Additional analysis 26 complete: repeated-cell, depth-conditioned wave-response panel.")
    print(f"  HR-OC dates meeting water support: {len(support)}")
    print(f"  Repeated-core dates/cells/500 m blocks: {len(current_dates)}/{panel['cell_code'].nunique()}/{blocks}")
    print(f"  Date support: {date_audit_path}")
    print(f"  Cell-date panel: {panel_path}")
    print(f"  Coefficients: {OUT / 'within_cell_wave_response_model.csv'}")
    print(f"  Figure: {fig_path}")
    print(f"  Notes: {OUT / 'within_cell_wave_response_notes_FR.txt'}")
    print("This is a within-cell association analysis, not an independent SPM validation or a sediment-transport model.")


if __name__ == "__main__":
    main()
