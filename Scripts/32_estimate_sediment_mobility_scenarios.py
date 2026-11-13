"""Scenario screening of wave/current bed-mobility at Anza survey depths.

This is not a sediment-flux or morphodynamic model. Read the metadata and
limitations written by this script before interpreting the scenario outputs.
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyTMD
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
SURVEY = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_candidate_utm29.csv"
WAVE_NC = ROOT / "Data" / "Copernicus_Marine" / "IBI_wave_Anza_subset.nc"
CURRENT_NC = ROOT / "Data" / "Copernicus_Marine" / "Currents_3D" / "IBI_currents_3D_daily_Anza_0_30m_20221001_20230331.nc"
CURRENT_AUDIT = ROOT / "Results" / "Currents_3D_QC" / "IBI_3D_current_support_audit.json"
CURRENT_PRODUCT = CONFIG["copernicus_marine"]["current_3d"]
TIDE_DIR = ROOT / "Data" / "Tide_Model"
OUT = ROOT / "Results" / "Sediment_Mobility_Scenarios"

G = 9.80665
RHO_W = 1025.0
RHO_S = 2650.0
NU = 1.05e-6
KAPPA = 0.4
D50_MM = (0.063, 0.100, 0.125, 0.250, 0.500, 1.000)
D50_PROVENANCE = [
    {
        "D50_scenario_mm": 0.063,
        "scenario_role": "generic sensitivity bound",
        "literature_context": "No direct local measurement identified; included only as a fine-sediment sensitivity.",
        "source": "None; analyst-defined sensitivity",
        "limitation": "Not measured in the 2022 Anza survey and not a fitted or calibrated grain size.",
    },
    {
        "D50_scenario_mm": 0.100,
        "scenario_role": "literature-informed near-bottom benchmark",
        "literature_context": "Aouiche et al. (2016) report a historical LCHF (1972) D50 of 0.10 mm 'au fond' in Agadir Bay.",
        "source": "Aouiche et al. (2016), J. Mater. Environ. Sci., 7(10), 3711-3723; https://www.jmaterenvironsci.com/Document/vol7/vol7_N10/402-JMES-Aouiche.pdf",
        "limitation": "Historical regional context, not a measured D50 for the 2022 Anza soundings or a verified match to each seabed location.",
    },
    {
        "D50_scenario_mm": 0.125,
        "scenario_role": "generic sensitivity",
        "literature_context": "No direct local measurement identified; included as an intermediate fine-sand sensitivity.",
        "source": "None; analyst-defined sensitivity",
        "limitation": "Not measured in the 2022 Anza survey and not a fitted or calibrated grain size.",
    },
    {
        "D50_scenario_mm": 0.250,
        "scenario_role": "historical beach comparison",
        "literature_context": "Aouiche et al. (2016) report a historical LCHF (1972) D50 of 0.25 mm 'sur la plage' in Agadir Bay.",
        "source": "Aouiche et al. (2016), J. Mater. Environ. Sci., 7(10), 3711-3723; https://www.jmaterenvironsci.com/Document/vol7/vol7_N10/402-JMES-Aouiche.pdf",
        "limitation": "Beach value is not treated as a subtidal measurement; retained only for comparison with the previously reported scenario.",
    },
    {
        "D50_scenario_mm": 0.500,
        "scenario_role": "generic sensitivity bound",
        "literature_context": "No direct local measurement identified; included only as a coarser-grain sensitivity.",
        "source": "None; analyst-defined sensitivity",
        "limitation": "Not measured in the 2022 Anza survey and not a fitted or calibrated grain size.",
    },
    {
        "D50_scenario_mm": 1.000,
        "scenario_role": "generic sensitivity bound",
        "literature_context": "No direct local measurement identified; included only as a coarse-grain sensitivity.",
        "source": "None; analyst-defined sensitivity",
        "limitation": "Not measured in the 2022 Anza survey and not a fitted or calibrated grain size.",
    },
]
DATUM_CASES = (("ZH_unadjusted", 0.0), ("IHO_2024_candidate_offset_sensitivity", 2.159))
BED_CLEARANCE_M = 0.5
BREAKING_SCREEN_HS_OVER_H = 0.78


def coord_name(ds: xr.Dataset, aliases: tuple[str, ...], standard: str) -> str:
    for name in aliases:
        if name in ds.variables:
            return name
    for name, var in ds.variables.items():
        if var.attrs.get("standard_name") == standard:
            return name
    raise ValueError(f"Cannot identify {standard} coordinate. Available: {list(ds.variables)}")


def haversine_m(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    radius = 6_371_000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def solve_wavenumber(period_s: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Vectorized bisection solution of omega^2 = g k tanh(k h)."""
    period, depth = np.broadcast_arrays(np.asarray(period_s, float), np.asarray(depth_m, float))
    result = np.full(period.shape, np.nan, dtype=float)
    valid = np.isfinite(period) & np.isfinite(depth) & (period > 0) & (depth > 0)
    if not valid.any():
        return result
    omega = 2.0 * np.pi / period[valid]
    h = depth[valid]
    k_deep = omega**2 / G
    lo = np.zeros_like(omega)
    hi = np.maximum(2.0 * k_deep, 2.0 * omega / np.sqrt(G * h))
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        positive = G * mid * np.tanh(mid * h) > omega**2
        hi = np.where(positive, mid, hi)
        lo = np.where(positive, lo, mid)
    result[valid] = 0.5 * (lo + hi)
    return result


def predict_got_tide(times_utc: pd.DatetimeIndex) -> np.ndarray:
    model = CONFIG["tide_model"]
    lon, lat = map(float, model["point_wgs84"])
    if not (TIDE_DIR / "GOT5.6").is_dir():
        raise FileNotFoundError(f"GOT5.6 files not found under {TIDE_DIR}")
    times_naive = times_utc.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    result = pyTMD.compute.tide_elevations(
        np.full(len(times_naive), lon, dtype=float),
        np.full(len(times_naive), lat, dtype=float),
        times_naive,
        directory=TIDE_DIR,
        model=model["name"],
        crs="4326",
        standard="datetime",
        type="trajectory",
        method="linear",
        crop=True,
        bounds=[lon - 0.5, lon + 0.5, lat - 0.5, lat + 0.5],
        buffer=0.25,
        extrapolate=True,
        cutoff=float(model["maximum_wet_node_distance_km"]),
    )
    values = result.values if hasattr(result, "values") else result
    return np.asarray(values, dtype=float).reshape(-1)


def current_profile_at_site() -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray, dict]:
    audit = json.loads(CURRENT_AUDIT.read_text(encoding="utf-8"))
    lon, lat = map(float, audit["nearest_wet_model_node_lon_lat"])
    with xr.open_dataset(CURRENT_NC) as ds:
        lat_name = coord_name(ds, ("latitude", "lat", "nav_lat"), "latitude")
        lon_name = coord_name(ds, ("longitude", "lon", "nav_lon"), "longitude")
        time_name = coord_name(ds, ("time", "time_counter"), "time")
        depth_name = coord_name(ds, ("depth", "deptht", "depthu", "lev", "z"), "depth")
        point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        time_dim = ds[time_name].dims[0]
        depth_dim = ds[depth_name].dims[0]
        u = point["uo"].transpose(time_dim, depth_dim).values.astype(float)
        v = point["vo"].transpose(time_dim, depth_dim).values.astype(float)
        time = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values, utc=True))
        z = np.asarray(ds[depth_name].values, dtype=float).squeeze()
        if str(ds[depth_name].attrs.get("positive", "down")).lower() != "down" and np.nanmedian(z) < 0:
            z = -z
        order = np.argsort(z)
        z, u, v = z[order], u[:, order], v[:, order]
        meta = {
            "nearest_wet_node_lon_lat": [lon, lat],
            "nearest_wet_node_distance_km": float(audit["nearest_wet_node_distance_km"]),
            "vertical_levels_m_positive_down": [float(q) for q in z],
            "current_product_temporal_resolution": "daily mean",
        }
    return z, time, u, v, meta


def main() -> None:
    missing = [p for p in (SURVEY, WAVE_NC, CURRENT_NC, CURRENT_AUDIT) if not p.is_file()]
    if missing:
        raise SystemExit("Required input missing:\n" + "\n".join(str(p) for p in missing))

    survey = pd.read_csv(SURVEY)
    needed = {"point_id", "longitude_wgs84_candidate_deg", "latitude_wgs84_candidate_deg", "easting_utm29n_candidate_m", "northing_utm29n_candidate_m", "z_source_m"}
    if not needed.issubset(survey.columns):
        raise SystemExit(f"Survey file missing columns: {sorted(needed - set(survey.columns))}")
    survey = survey[np.isfinite(survey["z_source_m"]) & (survey["z_source_m"] < 0)].copy()
    survey["depth_below_local_ZH_m"] = -survey["z_source_m"].astype(float)
    survey["depth_bin_m"] = (survey["depth_below_local_ZH_m"] * 2.0).round() / 2.0
    survey = survey[(survey["depth_bin_m"] >= 0.5) & (survey["depth_bin_m"] <= 29.0)].copy()
    if survey.empty:
        raise SystemExit("No underwater survey points fall in the supported 0.5–29.0 m depth range.")
    depth_bins = np.sort(survey["depth_bin_m"].unique().astype(float))
    site_lon = float(survey["longitude_wgs84_candidate_deg"].median())
    site_lat = float(survey["latitude_wgs84_candidate_deg"].median())

    with xr.open_dataset(WAVE_NC) as ds:
        lat_name = coord_name(ds, ("latitude", "lat", "nav_lat"), "latitude")
        lon_name = coord_name(ds, ("longitude", "lon", "nav_lon"), "longitude")
        time_name = coord_name(ds, ("time", "time_counter"), "time")
        lat = np.asarray(ds[lat_name].values, dtype=float)
        lon = np.asarray(ds[lon_name].values, dtype=float)
        if lat.ndim != 1 or lon.ndim != 1:
            raise SystemExit("Wave grid coordinates are not 1-D; stop and inspect the NetCDF before proceeding.")
        iy = int(np.argmin(np.abs(lat - site_lat)))
        ix = int(np.argmin(np.abs(lon - site_lon)))
        wave_node_lon, wave_node_lat = float(lon[ix]), float(lat[iy])
        wave_node_distance_km = float(haversine_m(site_lon, site_lat, np.asarray(wave_node_lon), np.asarray(wave_node_lat)) / 1000)
        point = ds.isel({ds[lat_name].dims[0]: iy, ds[lon_name].dims[0]: ix})
        wave_time = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values, utc=True))
        hs = np.asarray(point["VHM0"].values, dtype=float).reshape(-1)
        tp = np.asarray(point["VTPK"].values, dtype=float).reshape(-1)
        wave_dir = np.asarray(point["VMDR"].values, dtype=float).reshape(-1)
    if not (len(wave_time) == len(hs) == len(tp) == len(wave_dir)):
        raise SystemExit("Wave variable lengths differ; inspect the source file.")

    current_depth, current_time, current_u, current_v, current_meta = current_profile_at_site()
    current_date_to_row = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(current_time)}
    wave_dates = np.asarray(wave_time.strftime("%Y-%m-%d"))
    current_rows_for_wave = np.array([current_date_to_row.get(day, -1) for day in wave_dates], dtype=int)
    if (current_rows_for_wave < 0).any():
        warnings.warn(f"{int((current_rows_for_wave < 0).sum())} wave hours have no same-date daily current profile and will be excluded from combined stress.")

    # GOT5.6 supplies a modelled astronomical tide signal. Remove the period mean
    # and retain it only as a relative water-depth sensitivity, not a port datum tie.
    tide_prediction = predict_got_tide(wave_time)
    tide_anomaly = tide_prediction - np.nanmean(tide_prediction)
    if not np.isfinite(tide_anomaly).any():
        raise SystemExit("GOT5.6 did not return valid tide predictions at the configured point.")

    n_depth, n_time = len(depth_bins), len(wave_time)
    rows_summary: list[dict] = []
    rows_daily: list[dict] = []
    rows_curves: list[dict] = []
    point_lookup: dict[tuple[str, float], dict] = {}
    all_case_arrays: dict[tuple[str, float, float], dict] = {}

    for case_name, datum_offset in DATUM_CASES:
        h_eff = depth_bins[:, None] + datum_offset + tide_anomaly[None, :]
        k = solve_wavenumber(np.broadcast_to(tp[None, :], h_eff.shape), h_eff)
        kh = k * h_eff
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            uw = np.pi * hs[None, :] / (tp[None, :] * np.sinh(np.clip(kh, 0, 700)))
        valid_wave = (np.isfinite(h_eff) & (h_eff > 0) & np.isfinite(hs)[None, :] & np.isfinite(tp)[None, :]
                      & np.isfinite(wave_dir)[None, :] & (hs[None, :] / h_eff <= BREAKING_SCREEN_HS_OVER_H)
                      & np.isfinite(uw))

        # Interpolate daily regional current profiles to 0.5 m above each
        # scenario seabed. The horizontal node is 2.33 km from the survey median.
        z_eval = h_eff - BED_CLEARANCE_M
        u_bed = np.full((n_depth, n_time), np.nan)
        v_bed = np.full((n_depth, n_time), np.nan)
        for it, current_row in enumerate(current_rows_for_wave):
            if current_row < 0:
                continue
            valid_levels = np.isfinite(current_u[current_row]) & np.isfinite(current_v[current_row])
            if valid_levels.sum() < 2:
                continue
            zz = current_depth[valid_levels]
            uu = current_u[current_row, valid_levels]
            vv = current_v[current_row, valid_levels]
            inside = (z_eval[:, it] >= zz.min()) & (z_eval[:, it] <= zz.max()) & np.isfinite(z_eval[:, it])
            if inside.any():
                u_bed[inside, it] = np.interp(z_eval[inside, it], zz, uu)
                v_bed[inside, it] = np.interp(z_eval[inside, it], zz, vv)

        wave_bearing = np.deg2rad(wave_dir)[None, :]
        current_bearing = np.arctan2(u_bed, v_bed)
        # VMDR may encode the direction waves come from. Stress magnitude is
        # invariant to reversing the wave axis, so use the acute axis angle.
        cos_phi = np.abs(np.cos(wave_bearing - current_bearing))
        sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi**2))
        valid_combined_base = valid_wave & np.isfinite(u_bed) & np.isfinite(v_bed)

        for d50_mm in D50_MM:
            d50 = d50_mm / 1000.0
            ks = 2.5 * d50
            z0 = ks / 30.0
            orbital_excursion = uw * tp[None, :] / (2.0 * np.pi)
            relative_excursion = orbital_excursion / ks
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                fw = np.where(relative_excursion <= 1.0, 0.30,
                              np.exp(5.213 * np.maximum(relative_excursion, 1e-12) ** (-0.194) - 5.977))
            fw = np.minimum(fw, 0.30)
            tau_wave = 0.5 * RHO_W * fw * uw**2

            dstar = d50 * (G * (RHO_S / RHO_W - 1.0) / NU**2) ** (1.0 / 3.0)
            theta_cr = 0.3 / (1.0 + 1.2 * dstar) + 0.055 * (1.0 - np.exp(-0.02 * dstar))
            tau_cr = theta_cr * (RHO_S - RHO_W) * G * d50
            theta_wave = tau_wave / ((RHO_S - RHO_W) * G * d50)
            mobility_wave = theta_wave / theta_cr

            speed_bed = np.hypot(u_bed, v_bed)
            log_argument = BED_CLEARANCE_M / z0
            if log_argument <= 1.0:
                raise SystemExit(f"Invalid log-law roughness ratio for D50={d50_mm} mm")
            tau_current = RHO_W * (KAPPA * speed_bed / np.log(log_argument)) ** 2
            tau_mean = tau_current * (1.0 + 1.2 * (tau_wave / np.maximum(tau_current + tau_wave, 1e-30)) ** 3.2)
            tau_max = np.sqrt((tau_mean + tau_wave * cos_phi) ** 2 + (tau_wave * sin_phi) ** 2)
            theta_combined = tau_max / ((RHO_S - RHO_W) * G * d50)
            mobility_combined = theta_combined / theta_cr
            valid_combined = valid_combined_base & np.isfinite(tau_current) & np.isfinite(mobility_combined)

            for j, hbin in enumerate(depth_bins):
                vw = valid_wave[j]
                vc = valid_combined[j]
                wave_n = int(vw.sum())
                combined_n = int(vc.sum())
                wave_fraction = float(np.mean(mobility_wave[j, vw] >= 1.0)) if wave_n else float("nan")
                combined_fraction = float(np.mean(mobility_combined[j, vc] >= 1.0)) if combined_n else float("nan")
                rows_summary.append({
                    "datum_case": case_name, "datum_offset_sensitivity_m": datum_offset,
                    "survey_depth_bin_below_local_ZH_m": float(hbin), "D50_scenario_mm_not_measured": d50_mm,
                    "valid_wave_hours": wave_n, "wave_only_exceedance_fraction": wave_fraction,
                    "wave_only_M_median": float(np.nanmedian(mobility_wave[j, vw])) if wave_n else np.nan,
                    "wave_only_M_p95": float(np.nanpercentile(mobility_wave[j, vw], 95)) if wave_n else np.nan,
                    "valid_combined_hours": combined_n, "combined_exceedance_fraction": combined_fraction,
                    "combined_M_median": float(np.nanmedian(mobility_combined[j, vc])) if combined_n else np.nan,
                    "combined_M_p95": float(np.nanpercentile(mobility_combined[j, vc], 95)) if combined_n else np.nan,
                    "valid_fraction_of_full_wave_record": wave_n / n_time,
                })
                rows_curves.append({
                    "datum_case": case_name, "survey_depth_bin_below_local_ZH_m": float(hbin),
                    "D50_scenario_mm_not_measured": d50_mm,
                    "wave_only_exceedance_fraction": wave_fraction,
                    "combined_exceedance_fraction": combined_fraction,
                })

            for day in np.unique(wave_dates):
                ix_day = np.flatnonzero(wave_dates == day)
                for j, hbin in enumerate(depth_bins):
                    vw = valid_wave[j, ix_day]
                    vc = valid_combined[j, ix_day]
                    wave_m = mobility_wave[j, ix_day][vw]
                    combined_m = mobility_combined[j, ix_day][vc]
                    rows_daily.append({
                        "date_utc": str(day), "datum_case": case_name,
                        "survey_depth_bin_below_local_ZH_m": float(hbin),
                        "D50_scenario_mm_not_measured": d50_mm,
                        "wave_valid_hours": int(vw.sum()),
                        "wave_only_exceedance_fraction": float(np.mean(wave_m >= 1.0)) if wave_m.size else np.nan,
                        "wave_M_median": float(np.nanmedian(wave_m)) if wave_m.size else np.nan,
                        "wave_M_p95": float(np.nanpercentile(wave_m, 95)) if wave_m.size else np.nan,
                        "combined_valid_hours": int(vc.sum()),
                        "combined_exceedance_fraction": float(np.mean(combined_m >= 1.0)) if combined_m.size else np.nan,
                        "combined_M_median": float(np.nanmedian(combined_m)) if combined_m.size else np.nan,
                        "combined_M_p95": float(np.nanpercentile(combined_m, 95)) if combined_m.size else np.nan,
                    })

            all_case_arrays[(case_name, datum_offset, d50_mm)] = {
                "mobility_wave": mobility_wave, "mobility_combined": mobility_combined,
                "valid_wave": valid_wave, "valid_combined": valid_combined,
            }

    summary = pd.DataFrame(rows_summary)
    daily = pd.DataFrame(rows_daily)
    curves = pd.DataFrame(rows_curves)
    OUT.mkdir(parents=True, exist_ok=True)
    provenance_path = OUT / "D50_scenario_provenance.csv"
    pd.DataFrame(D50_PROVENANCE).to_csv(provenance_path, index=False, encoding="utf-8-sig")
    summary_path = OUT / "mobility_by_depth_grain_datum_scenario.csv"
    daily_path = OUT / "mobility_daily_by_depth_grain_datum_scenario.csv.gz"
    summary.to_csv(summary_path, index=False, float_format="%.7g", encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, compression="gzip", float_format="%.7g", encoding="utf-8-sig")

    # A display table for every surveyed point, using the 0.10-mm literature-informed scenario.
    # This is a scenario map, not an observed transport map.
    selected = summary[(summary["datum_case"] == "ZH_unadjusted") & (summary["D50_scenario_mm_not_measured"] == 0.10)]
    lookup = selected.set_index("survey_depth_bin_below_local_ZH_m").to_dict(orient="index")
    point_rows = []
    for row in survey.itertuples(index=False):
        key = float(row.depth_bin_m)
        val = lookup.get(key, {})
        point_rows.append({
            "point_id": row.point_id,
            "longitude_wgs84_candidate_deg": row.longitude_wgs84_candidate_deg,
            "latitude_wgs84_candidate_deg": row.latitude_wgs84_candidate_deg,
            "easting_utm29n_candidate_m": row.easting_utm29n_candidate_m,
            "northing_utm29n_candidate_m": row.northing_utm29n_candidate_m,
            "z_source_local_port_ZH_m": row.z_source_m,
            "depth_bin_m": key,
            "D50_scenario_mm_not_measured": 0.10,
            "wave_only_exceedance_fraction": val.get("wave_only_exceedance_fraction", np.nan),
            "combined_exceedance_fraction": val.get("combined_exceedance_fraction", np.nan),
            "combined_valid_hours": val.get("valid_combined_hours", 0),
        })
    point_table = pd.DataFrame(point_rows)
    point_path = OUT / "mobility_point_map_D50_0p10mm_ZH_unadjusted.csv.gz"
    point_table.to_csv(point_path, index=False, compression="gzip", float_format="%.7g", encoding="utf-8-sig")

    # Figure A: local sounding geometry colored by the 0.10-mm scenario frequency.
    fig, ax = plt.subplots(figsize=(7.1, 6.0), constrained_layout=True)
    sc = ax.scatter(point_table["easting_utm29n_candidate_m"], point_table["northing_utm29n_candidate_m"],
                    c=point_table["combined_exceedance_fraction"], s=2.2, cmap="viridis", vmin=0.82, vmax=1.0,
                    linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82)
    cb.set_label("Fraction of valid hours with modeled M ≥ 1 (display range: 0.82–1.00)")
    ax.set_xlabel("Easting (UTM 29N; m)")
    ax.set_ylabel("Northing (UTM 29N; m)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.02, 0.02, "Regional forcing held spatially constant; pattern mainly follows depth. No local wave transformation.",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 3})
    map_png, map_pdf = OUT / "Figure_mobility_scenario_map.png", OUT / "Figure_mobility_scenario_map.pdf"
    fig.savefig(map_png, dpi=350, bbox_inches="tight")
    fig.savefig(map_pdf, bbox_inches="tight")
    plt.close(fig)

    # Figure B: frequency by depth and assumed grain size; no confidence interval implied.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.89, bottom=0.19, hspace=0.28, wspace=0.13)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(D50_MM)))
    panels = [
        (axes[0, 0], "ZH_unadjusted", "wave_only_exceedance_fraction", "Wave-only; local ZH depth"),
        (axes[0, 1], "ZH_unadjusted", "combined_exceedance_fraction", "Wave + daily regional current; local ZH depth"),
        (axes[1, 0], "IHO_2024_candidate_offset_sensitivity", "wave_only_exceedance_fraction", "Wave-only; candidate +2.159 m offset"),
        (axes[1, 1], "IHO_2024_candidate_offset_sensitivity", "combined_exceedance_fraction", "Wave + daily regional current; candidate offset"),
    ]
    for ax, case, metric, title in panels:
        subset = summary[summary["datum_case"] == case]
        for color, d50 in zip(colors, D50_MM):
            line = subset[subset["D50_scenario_mm_not_measured"] == d50].sort_values("survey_depth_bin_below_local_ZH_m")
            ax.plot(line[metric], line["survey_depth_bin_below_local_ZH_m"], marker="o", ms=2.5,
                    lw=1.4, color=color, label=f"{d50:g} mm")
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#D9E2E8", linewidth=0.55)
        ax.set_xlim(0.45, 1.0)
        ax.set_xticks(np.arange(0.5, 1.01, 0.1))
        ax.invert_yaxis()
    fig.text(0.025, 0.54, "Survey depth below local ZH (m; increasing downward)",
             rotation=90, ha="center", va="center", fontsize=9)
    fig.text(0.56, 0.115, "Fraction of valid hourly records with M ≥ 1 (axis shown from 0.45)",
             ha="center", va="center", fontsize=9)
    axes[0, 1].legend(title="Assumed D50", frameon=True, fontsize=8, title_fontsize=8, loc="lower right")
    fig.subplots_adjust(top=0.95)
    fig.text(0.55, 0.035,
             "Valid-hour support varies by depth; see Figure C. Scenario frequencies are not probabilities or transport rates.",
             ha="center", fontsize=8)
    curve_png, curve_pdf = OUT / "Figure_mobility_depth_grain_sensitivity.png", OUT / "Figure_mobility_depth_grain_sensitivity.pdf"
    fig.savefig(curve_png, dpi=350, bbox_inches="tight")
    fig.savefig(curve_pdf, bbox_inches="tight")
    plt.close(fig)

    # Figure C: show the denominator/support behind the frequency curves.
    support = summary[summary["D50_scenario_mm_not_measured"] == 0.10].copy()
    support["wave_valid_fraction"] = support["valid_wave_hours"] / n_time
    support["combined_valid_fraction"] = support["valid_combined_hours"] / n_time
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    styles = [
        ("ZH_unadjusted", "wave_valid_fraction", "Local ZH; wave-only", "#2A788E", "-"),
        ("ZH_unadjusted", "combined_valid_fraction", "Local ZH; wave + current", "#D55E00", "-"),
        ("IHO_2024_candidate_offset_sensitivity", "wave_valid_fraction", "Candidate offset; wave-only", "#2A788E", "--"),
        ("IHO_2024_candidate_offset_sensitivity", "combined_valid_fraction", "Candidate offset; wave + current", "#D55E00", "--"),
    ]
    for case, col, label, color, style in styles:
        line = support[support["datum_case"] == case].sort_values("survey_depth_bin_below_local_ZH_m")
        ax.plot(line[col], line["survey_depth_bin_below_local_ZH_m"], marker="o", ms=2.3,
                lw=1.5, color=color, linestyle=style, label=label)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(float(depth_bins.max()), float(depth_bins.min()))
    ax.set_xlabel("Fraction of the 4,368-hour record with valid forcing")
    ax.set_ylabel("Survey depth below local ZH (m; increasing downward)")
    ax.grid(True, color="#D9E2E8", linewidth=0.55)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()
    support_png, support_pdf = OUT / "Figure_mobility_forcing_support.png", OUT / "Figure_mobility_forcing_support.pdf"
    fig.savefig(support_png, dpi=350, bbox_inches="tight")
    fig.savefig(support_pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "study": "Anza / Agadir, Morocco",
        "survey_depth_reference": "local hydrographic zero of Port of Agadir; source Z unchanged",
        "current_product": "Copernicus Marine IBI multi-year physics reanalysis, daily mean uo/vo, 0.027-degree dataset",
        "current_product_id": CURRENT_PRODUCT["product_id"],
        "current_dataset_id": CURRENT_PRODUCT["dataset_id"],
        "current_doi": CURRENT_PRODUCT["doi"],
        "current_variables": CURRENT_PRODUCT["variables"],
        "current_download_depth_range_m": [CURRENT_PRODUCT["minimum_depth_m"], CURRENT_PRODUCT["maximum_depth_m"]],
        "current_node": current_meta,
        "wave_product": "existing IBI hourly VHM0/VTPK/VMDR fields sampled at the nearest grid node",
        "wave_node_lon_lat": [wave_node_lon, wave_node_lat],
        "wave_node_distance_from_survey_median_km": wave_node_distance_km,
        "wave_records": int(n_time),
        "wave_time_range_utc": [str(wave_time.min()), str(wave_time.max())],
        "tide_model": CONFIG["tide_model"]["name"],
        "tide_signal": "GOT5.6 prediction demeaned over this forcing period; relative water-depth sensitivity only, not gauge validated and not a port datum tie",
        "datum_cases": [{"name": n, "offset_m": v, "meaning": "sensitivity only; +2.159 m is a candidate IHO 2024 relation and is not applied as verified survey-datum correction"} for n, v in DATUM_CASES],
        "D50_scenarios_mm": list(D50_MM),
        "D50_note": "All D50 values remain unmeasured scenarios. The 0.10 mm case is a literature-informed historical near-bottom benchmark reported for Agadir Bay by Aouiche et al. (2016), based on LCHF (1972); it is not a measurement from the 2022 Anza survey. The 0.25 mm case is a historical beach value and is not treated as subtidal grain size. See D50_scenario_provenance.csv.",
        "D50_provenance_table": str(provenance_path),
        "equations": ["linear dispersion and near-bed orbital velocity", "Swart wave friction factor with ks=2.5 D50", "Soulsby combined wave-current maximum stress", "Soulsby-Whitehouse critical Shields threshold"],
        "current_stress_method": "log-law extrapolation from interpolated regional-model velocity at 0.5 m above scenario seabed; daily mean currents and horizontal grid node 2.33 km from survey median",
        "screening": "Records with Hs/h > 0.78 are excluded from linear-wave stress calculation as breaker-likely; no local wave transformation, refraction, setup, or surf-zone current model is included.",
        "interpretation": "M>=1 is modeled incipient-motion threshold exceedance for noncohesive grains under the stated scenarios. Not a probability, measured mobility, sediment concentration, transport direction/rate, bed change, or causal validation.",
        "source_inputs_modified": False,
    }
    metadata_path = OUT / "mobility_scenario_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    captions = OUT / "figure_captions_EN.txt"
    captions.write_text(
        "Figure A. Bathymetry-conditioned scenario map of the modeled fraction of valid hourly records exceeding the incipient-motion threshold (M >= 1), for the 0.10 mm literature-informed but unmeasured near-bottom scenario and unadjusted local hydrographic zero. Aouiche et al. (2016) report this historical LCHF (1972) grain-size value for the bottom of Agadir Bay; it is not a measurement from the 2022 Anza survey. The same regional wave and current forcing is applied across the survey area; the spatial pattern mainly reflects depth bins, not resolved spatial variability in hydrodynamic forcing. No nearshore wave transformation is represented. This is not a sediment-flux map.\n\n"
        "Figure B. Modeled fraction of valid hourly records with M >= 1 by local survey depth and assumed grain-size scenario for wave-only and combined wave-current stress, under two vertical-reference sensitivities. The 0.10 mm scenario is a historical near-bottom literature benchmark; 0.25 mm is a historical beach value; other values are analyst-defined sensitivity bounds. None is a measured D50 for the 2022 survey. The x-axis is displayed from 0.45 to resolve the high-exceedance range; valid-hour support varies with depth and is shown in Figure C. The +2.159 m case is an unverified candidate datum offset. Lines are conditional scenario frequencies, not confidence intervals, probabilities, or transport rates.\n\n"
        "Figure C. Fraction of the 4,368-hour forcing record with valid wave-only or combined wave-current inputs by depth for the 0.10 mm literature-informed but unmeasured near-bottom scenario. Missing support is concentrated in shallow depth bins because of the breaker screen and the vertical range of the regional current profile.\n",
        encoding="utf-8",
    )
    print("Scenario-based sediment-mobility screening complete; source bathymetry, NetCDF, and tide model were not changed.")
    print(f"Survey points: {len(survey):,}; depth bins: {n_depth}; hourly wave records: {n_time:,}")
    print(f"Wave node: {wave_node_lon:.5f}, {wave_node_lat:.5f}; distance from survey median={wave_node_distance_km:.2f} km")
    print(f"Current node: {current_meta['nearest_wet_node_lon_lat']}; distance={current_meta['nearest_wet_node_distance_km']:.2f} km; daily mean profile")
    print(f"Summary: {summary_path}")
    print(f"Daily scenarios: {daily_path}")
    print(f"Point scenario map table: {point_path}")
    print(f"Figures: {map_png}; {curve_png}; {support_png}")
    print(f"Metadata and caveats: {metadata_path}; {captions}; {provenance_path}")
    print("Interpret as a screening study only: grain size is assumed, current forcing is regional/daily, wave transformation is unresolved, and absolute vertical tie is unverified.")


if __name__ == "__main__":
    main()


