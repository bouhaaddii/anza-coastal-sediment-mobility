from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "Scripts" / "26_within_cell_wave_response.py"
PANEL_PATH = ROOT / "Results" / "Additional_Analyses" / "Within_Cell_Wave_Response" / "within_cell_wave_response_panel.csv.gz"
DATE_AUDIT_PATH = ROOT / "Results" / "Additional_Analyses" / "Within_Cell_Wave_Response" / "within_cell_wave_response_date_support.csv"
WAVE_PATH = ROOT / "Data" / "Copernicus_Marine" / "IBI_wave_Anza_subset.nc"
CONFIG_PATH = ROOT / "Config" / "project_config.json"
OUT = ROOT / "Results" / "Additional_Analyses" / "Within_Cell_Wave_Response"
MIN_HOURLY_SAMPLES_PER_WINDOW = 18
TEMPORAL_BLOCK_LENGTH = 3
BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20261114


def load_base_module():
    spec = importlib.util.spec_from_file_location("within_cell_wave_response_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analysis functions from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_wave_window(wave: pd.DataFrame, overpass: pd.Timestamp, lag_start_h: int) -> tuple[float, int]:
    # Non-overlapping 24-h windows ending 0–24, 24–48, or 48–72 h before overpass.
    window_end = overpass - pd.Timedelta(hours=lag_start_h)
    window_start = window_end - pd.Timedelta(hours=24)
    selected = wave[(wave["time_utc"] > window_start) & (wave["time_utc"] <= window_end)]
    values = pd.to_numeric(selected["hs_m"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) < MIN_HOURLY_SAMPLES_PER_WINDOW:
        return np.nan, int(len(values))
    return float(np.mean(values)), int(len(values))


def main() -> None:
    for path in (BASE_SCRIPT, PANEL_PATH, DATE_AUDIT_PATH, WAVE_PATH, CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}. Run Script 26 first.")
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    panel = pd.read_csv(PANEL_PATH, compression="gzip")
    date_audit = pd.read_csv(DATE_AUDIT_PATH)
    config = pd.read_json(CONFIG_PATH, typ="series").to_dict()
    bbox = config["site_bbox_wgs84"]
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    wave, grid_distance_km = base.load_wave_series(WAVE_PATH, float(lon), float(lat))
    wave["time_utc"] = pd.to_datetime(wave["time_utc"], utc=True)

    required_panel = {"date_utc", "date_index", "cell_code", "spatial_block_500m", "log_spm", "time_z", "tide_anomaly_z"}
    if not required_panel.issubset(panel.columns):
        raise ValueError(f"Saved Script 26 panel is missing columns: {sorted(required_panel.difference(panel.columns))}")
    if not {"date_utc", "acquisition_utc_used_for_forcing"}.issubset(date_audit.columns):
        raise ValueError("Script 26 date audit lacks the overpass timestamp required for lag matching.")

    panel_dates = sorted(panel["date_utc"].astype(str).unique())
    audit = date_audit.copy()
    audit["date_utc"] = audit["date_utc"].astype(str)
    audit = audit[audit["date_utc"].isin(panel_dates)].copy()
    overpass_by_date = dict(zip(audit["date_utc"], pd.to_datetime(audit["acquisition_utc_used_for_forcing"], utc=True, errors="coerce")))

    feature_rows = []
    for date in panel_dates:
        overpass = overpass_by_date.get(date, pd.NaT)
        row: dict[str, object] = {"date_utc": date, "overpass_utc": overpass.isoformat() if pd.notna(overpass) else ""}
        if pd.isna(overpass):
            for name in ("lag_0_24", "lag_24_48", "lag_48_72"):
                row[f"{name}_mean_hs_m"] = np.nan
                row[f"{name}_valid_hourly_n"] = 0
        else:
            for name, lag_start in (("lag_0_24", 0), ("lag_24_48", 24), ("lag_48_72", 48)):
                mean_hs, n = summarize_wave_window(wave, overpass, lag_start)
                row[f"{name}_mean_hs_m"] = mean_hs
                row[f"{name}_valid_hourly_n"] = n
        feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    lag_names = ("lag_0_24", "lag_24_48", "lag_48_72")
    for name in lag_names:
        raw = pd.to_numeric(features[f"{name}_mean_hs_m"], errors="coerce")
        log_value = np.log1p(raw)
        sd = float(log_value.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            features[f"{name}_logmean_hs_z"] = np.nan
        else:
            features[f"{name}_logmean_hs_z"] = (log_value - float(log_value.mean())) / sd
    feature_cols = [f"{name}_logmean_hs_z" for name in lag_names]
    features["all_three_windows_available"] = features[feature_cols].notna().all(axis=1)
    features.to_csv(OUT / "wave_lag_window_support.csv", index=False, encoding="utf-8-sig")

    common_dates = set(features.loc[features["all_three_windows_available"], "date_utc"].astype(str))
    analysis = panel[panel["date_utc"].astype(str).isin(common_dates)].merge(
        features[["date_utc", *feature_cols]], on="date_utc", how="inner", validate="many_to_one"
    )
    if analysis["date_utc"].nunique() < 20 or analysis["spatial_block_500m"].nunique() < 10:
        note = (
            "Lag-response support audit only; models not fitted because common complete support was below the pre-specified minimum.\n"
            f"Common dates={analysis['date_utc'].nunique()}, cells={analysis['cell_code'].nunique()}, "
            f"500 m blocks={analysis['spatial_block_500m'].nunique()}.\n"
            "No thresholds were relaxed after seeing the data."
        )
        (OUT / "wave_lag_response_notes_FR.txt").write_text(note, encoding="utf-8")
        print("Wave-lag support audit complete; no model fitted because minimum support was not met.")
        print(f"  Common dates/cells/500 m blocks: {analysis['date_utc'].nunique()}/{analysis['cell_code'].nunique()}/{analysis['spatial_block_500m'].nunique()}")
        print(f"  Outputs: {OUT}")
        return

    # Individual lag models estimate each window separately; the joint model estimates the
    # conditional contribution of each window and is retained only if the design is estimable.
    specifications = []
    for name, label in zip(lag_names, ("0–24 h", "24–48 h", "48–72 h")):
        specifications.append((f"Separate lag {label} + trend", [f"{name}_logmean_hs_z", "time_z"]))
    all_lags = [f"{name}_logmean_hs_z" for name in lag_names]
    specifications.append(("Joint three-window distributed lag + trend", [*all_lags, "time_z"]))
    specifications.append(("Joint three-window distributed lag + tide + trend", [*all_lags, "tide_anomaly_z", "time_z"]))

    result_rows = []
    for spec_idx, (label, predictors) in enumerate(specifications):
        complete = analysis.dropna(subset=["log_spm", *predictors]).copy()
        try:
            fit = base.fixed_effect_fit(complete, predictors)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"  Skipped {label}: {exc}")
            continue
        bootstrap = base.two_axis_block_bootstrap(
            complete, predictors, reps=BOOTSTRAP_REPLICATES,
            block_length=TEMPORAL_BLOCK_LENGTH,
            seed=RANDOM_SEED + spec_idx,
        )
        for predictor_idx, predictor in enumerate(predictors):
            if predictor in ("time_z", "tide_anomaly_z"):
                continue
            beta = float(fit["beta"][predictor_idx])
            if len(bootstrap) >= 100:
                low, high = np.percentile(bootstrap[:, predictor_idx], [2.5, 97.5])
                low, high = float(low), float(high)
            else:
                low, high = np.nan, np.nan
            result_rows.append({
                "model": label,
                "lag_window": predictor.replace("_logmean_hs_z", "").replace("lag_", "").replace("_", "–") + " h",
                "predictor": predictor,
                "standardized_beta_log_spm_per_1sd_wave": beta,
                "effect_percent_per_1sd_wave": float(100 * np.expm1(beta)),
                "block_bootstrap_95ci_low": low,
                "block_bootstrap_95ci_high": high,
                "valid_bootstrap_replicates": int(len(bootstrap)),
                "observations_n": fit["n_obs"],
                "cells_n": fit["n_cells"],
                "dates_n": fit["n_dates"],
                "spatial_blocks_500m_n": int(complete["spatial_block_500m"].nunique()),
                "within_r2": fit["within_r2"],
                "ibi_wave_grid_distance_km": grid_distance_km,
            })
        print(f"  Fitted {label}; valid bootstrap={len(bootstrap)}/{BOOTSTRAP_REPLICATES}")

    if not result_rows:
        raise RuntimeError("No lag model was estimable; inspect the wave-window support audit.")
    result = pd.DataFrame(result_rows)
    table_path = OUT / "wave_lag_response_models.csv"
    result.to_csv(table_path, index=False, encoding="utf-8-sig")

    figure, ax = plt.subplots(figsize=(11.5, max(5.2, .52 * len(result))), constrained_layout=True)
    plot = result.reset_index(drop=True)
    positions = np.arange(len(plot))[::-1]
    palette = {"0–24 h": "#176B87", "24–48 h": "#D97732", "48–72 h": "#7567A8"}
    for idx, row in plot.iterrows():
        beta = float(row["standardized_beta_log_spm_per_1sd_wave"])
        low, high = row["block_bootstrap_95ci_low"], row["block_bootstrap_95ci_high"]
        color = palette.get(str(row["lag_window"]), "#334155")
        if pd.notna(low) and pd.notna(high):
            ax.plot([float(low), float(high)], [positions[idx], positions[idx]], color=color, lw=1.7, solid_capstyle="round")
            ax.plot([float(low), float(low)], [positions[idx] - .08, positions[idx] + .08], color=color, lw=1.2)
            ax.plot([float(high), float(high)], [positions[idx] - .08, positions[idx] + .08], color=color, lw=1.2)
        ax.plot(beta, positions[idx], "o", color=color, ms=6)
    ax.axvline(0, color="#475569", linestyle="--", linewidth=1)
    labels = [f"{row.model} | {row.lag_window}" for row in plot.itertuples()]
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Within-cell change in log(HR-OC SPM) per 1-SD increase in mean Hs")
    ax.grid(axis="x", color="#CBD5E1", alpha=.6, linewidth=.7)
    ax.text(.01, -.13, "Intervals: two-axis block bootstrap (500 m spatial blocks; 3 consecutive eligible acquisitions).", transform=ax.transAxes, fontsize=8)
    ax.margins(y=.08)
    fig_path = OUT / "Figure_wave_lag_response.png"
    figure.savefig(fig_path, dpi=350, bbox_inches="tight")
    figure.savefig(OUT / "Figure_wave_lag_response.pdf", bbox_inches="tight")
    plt.close(figure)

    included_dates = sorted(analysis["date_utc"].astype(str).unique())
    notes = [
        "Exploratory, bathymetry-conditioned wave-lag analysis at Anza.",
        f"Common complete support: {len(included_dates)} dates, {analysis['cell_code'].nunique()} cells, {analysis['spatial_block_500m'].nunique()} spatial blocks of 500 m.",
        "Three non-overlapping 24-h forcing windows were defined relative to the local Sentinel-2 overpass time as a proxy: 0–24 h, 24–48 h, and 48–72 h.",
        "Critical timing limit: HR-OC L3 is a daily aggregation that may merge multiple Sentinel-2 tile mosaics/overpasses; the source acquisition time for each SPM pixel is not identified in this subset. These are nominal calendar-day forcing windows, not verified physical SPM response lags.",
        "Wave exposure is the standardized log(1 + mean IBI Hs) in each window. Separate models estimate each lag individually; joint distributed-lag models estimate conditional lag contributions when their design matrices are estimable.",
        "The joint lag terms may be collinear because wave conditions persist across days. Report rank/fit warnings and intervals; do not interpret separate-window associations as independent effects.",
        "The two-axis block bootstrap uses 500 m spatial blocks and moving blocks of three consecutive eligible acquisitions (500 replicates). It is exploratory, not an exact coverage guarantee for every dependence structure.",
        "Calendar-day HR-OC / local Sentinel-2 source-granule identity is unverified. The outcome is satellite-derived SPM, not in-situ concentration. IBI is regional and does not resolve the surf zone; GOT5.6 is modeled, not gauge-validated.",
        "This analysis tests response timing in this Anza data set. Lagged remote-sensing analyses already exist in the literature; do not claim the general method is unprecedented or infer causal sediment transport.",
    ]
    (OUT / "wave_lag_response_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (OUT / "wave_lag_response_figure_caption_EN.txt").write_text(
        "Figure. Exploratory bathymetry-conditioned association between daily HR-OC L3 SPM and mean IBI significant wave height in nominal 24-h windows referenced to the local Sentinel-2 overpass (0–24, 24–48, and 48–72 h). HR-OC L3 may merge multiple same-day tile mosaics, and the source overpass time for each SPM pixel is not identified; these are not verified physical response lags. Points are standardized within-cell fixed-effect coefficients and bars are exploratory two-axis block-bootstrap 95% intervals. The analysis is not in-situ validation or a causal sediment-transport estimate.",
        encoding="utf-8",
    )
    print("Exploratory wave-lag analysis complete; no source imagery, bathymetry, or CMEMS files were changed.")
    print(f"  Common support dates/cells/blocks: {len(included_dates)}/{analysis['cell_code'].nunique()}/{analysis['spatial_block_500m'].nunique()}")
    print(f"  Window support audit: {OUT / 'wave_lag_window_support.csv'}")
    print(f"  Model table: {table_path}")
    print(f"  Figure: {fig_path}")
    print(f"  Interpretation: {OUT / 'wave_lag_response_notes_FR.txt'}")
    print("Treat this as a pre-specified timing sensitivity, not as a search for a favorable lag or evidence of causation.")


if __name__ == "__main__":
    main()
