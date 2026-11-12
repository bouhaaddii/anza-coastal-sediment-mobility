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
OUT = ROOT / "Results" / "Additional_Analyses" / "Within_Cell_Wave_Response"
REPLICATES = 500
SEED = 20261113


def load_base_module():
    spec = importlib.util.spec_from_file_location("within_cell_wave_response_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base analysis functions from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not BASE_SCRIPT.is_file() or not PANEL_PATH.is_file():
        raise FileNotFoundError("Run Script 26 first; its saved panel and source script are required.")
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    panel = pd.read_csv(PANEL_PATH, compression="gzip")
    required = {
        "date_utc", "date_index", "cell_code", "spatial_block_500m", "log_spm",
        "log_hs24_max_z", "hs_24h_mean_m", "time_z", "tide_anomaly_z",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Saved Script 26 panel is missing columns: {sorted(missing)}")

    daily = panel.drop_duplicates("date_index").sort_values("date_index").copy()
    mean_hs = pd.to_numeric(daily["hs_24h_mean_m"], errors="coerce").to_numpy(dtype=float)
    mean_log = np.log1p(mean_hs)
    mean_sd = float(np.nanstd(mean_log, ddof=0))
    if not np.isfinite(mean_sd) or mean_sd == 0:
        raise ValueError("The 24-hour mean wave-height proxy has no usable variation.")
    mean_z = (mean_log - float(np.nanmean(mean_log))) / mean_sd
    daily_mean_map = dict(zip(daily["date_index"].astype(int), mean_z))
    panel["log_hs24_mean_z"] = panel["date_index"].astype(int).map(daily_mean_map)

    specifications = [
        ("Maximum Hs + trend; temporal block 2", ["log_hs24_max_z", "time_z"], 2),
        ("Maximum Hs + trend; temporal block 3", ["log_hs24_max_z", "time_z"], 3),
        ("Maximum Hs + trend; temporal block 5", ["log_hs24_max_z", "time_z"], 5),
        ("Mean Hs + trend; temporal block 3", ["log_hs24_mean_z", "time_z"], 3),
        ("Maximum Hs + tide + trend; temporal block 3", ["log_hs24_max_z", "tide_anomaly_z", "time_z"], 3),
    ]
    result_rows = []
    for spec_idx, (label, predictors, block_length) in enumerate(specifications):
        complete = panel.dropna(subset=["log_spm", *predictors]).copy()
        fit = base.fixed_effect_fit(complete, predictors)
        boot = base.two_axis_block_bootstrap(
            complete, predictors, reps=REPLICATES,
            block_length=block_length, seed=SEED + spec_idx,
        )
        wave_predictor = predictors[0]
        wave_idx = predictors.index(wave_predictor)
        beta = float(fit["beta"][wave_idx])
        if len(boot) >= 100:
            low, high = np.percentile(boot[:, wave_idx], [2.5, 97.5])
            low, high = float(low), float(high)
        else:
            low, high = np.nan, np.nan
        result_rows.append({
            "specification": label,
            "wave_metric": "24-h maximum Hs" if wave_predictor == "log_hs24_max_z" else "24-h mean Hs",
            "wave_beta_log_spm_per_1sd": beta,
            "wave_effect_percent_per_1sd": float(100 * np.expm1(beta)),
            "block_bootstrap_95ci_low": low,
            "block_bootstrap_95ci_high": high,
            "valid_bootstrap_replicates": int(len(boot)),
            "temporal_block_length_acquisitions": block_length,
            "observations_n": fit["n_obs"],
            "cells_n": fit["n_cells"],
            "dates_n": fit["n_dates"],
            "spatial_blocks_500m_n": int(complete["spatial_block_500m"].nunique()),
            "within_r2": fit["within_r2"],
        })
        print(f"  {label}: beta={beta:.3f}; valid bootstrap={len(boot)}/{REPLICATES}")

    result = pd.DataFrame(result_rows)
    result_path = OUT / "wave_response_robustness_sensitivity.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")

    # Leave-one-acquisition-date-out influence diagnostic for the primary and tide-adjusted specifications.
    loo_rows = []
    loo_specs = [
        ("Maximum Hs + trend", ["log_hs24_max_z", "time_z"]),
        ("Maximum Hs + tide + trend", ["log_hs24_max_z", "tide_anomaly_z", "time_z"]),
    ]
    date_values = sorted(panel["date_index"].dropna().astype(int).unique())
    for label, predictors in loo_specs:
        for date_index in date_values:
            subset = panel[panel["date_index"].astype(int) != date_index].dropna(subset=["log_spm", *predictors])
            try:
                fit = base.fixed_effect_fit(subset, predictors)
                beta = float(fit["beta"][0])
            except (ValueError, np.linalg.LinAlgError):
                beta = np.nan
            date_text = panel.loc[panel["date_index"].astype(int).eq(date_index), "date_utc"].iloc[0]
            loo_rows.append({"specification": label, "omitted_date_utc": date_text, "wave_beta_log_spm_per_1sd": beta})
    loo = pd.DataFrame(loo_rows)
    loo_path = OUT / "wave_response_leave_one_date_out.csv"
    loo.to_csv(loo_path, index=False, encoding="utf-8-sig")

    wave_loo = loo[loo["specification"].eq("Maximum Hs + trend")]["wave_beta_log_spm_per_1sd"].dropna()
    sensitivity = result[result["wave_beta_log_spm_per_1sd"].notna()]
    notes = [
        "Robustness checks for the within-cell wave association at Anza.",
        "All fits use the saved repeated-cell panel from Script 26; no new satellite or bathymetric data are generated.",
        "Checks compare 24-h maximum and mean Hs proxies, temporal moving-block lengths of 2, 3, and 5 acquisitions, a tide-adjusted specification, and leave-one-date-out influence.",
        "Spatial blocks remain 500 m; the moving-block resampling is exploratory and does not create independent observations.",
        f"Leave-one-date-out primary wave coefficient range: {wave_loo.min():.3f} to {wave_loo.max():.3f}; positive in {(wave_loo > 0).sum()} of {len(wave_loo)} omissions." if len(wave_loo) else "Leave-one-date-out primary coefficient range could not be estimated.",
        "Do not select a specification because its interval excludes zero. Interpret the complete pre-specified sensitivity set and retain the primary model as reported in Script 26.",
        "HR-OC remains a satellite-derived product; the exact source granule is unverified, IBI does not resolve the surf zone, and these associations are not causal or a sediment-transport estimate.",
    ]
    (OUT / "wave_response_robustness_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    plot = sensitivity.reset_index(drop=True)
    y = np.arange(len(plot))[::-1]
    colors = ["#176B87" if "tide" not in text.lower() else "#7567A8" for text in plot["specification"]]
    for i, row in plot.iterrows():
        beta = float(row["wave_beta_log_spm_per_1sd"])
        low, high = row["block_bootstrap_95ci_low"], row["block_bootstrap_95ci_high"]
        color = colors[i]
        if pd.notna(low) and pd.notna(high):
            ax.plot([float(low), float(high)], [y[i], y[i]], color=color, lw=1.7, solid_capstyle="round")
            ax.plot([float(low), float(low)], [y[i] - .08, y[i] + .08], color=color, lw=1.2)
            ax.plot([float(high), float(high)], [y[i] - .08, y[i] + .08], color=color, lw=1.2)
        ax.plot(beta, y[i], "o", color=color, ms=6)
    ax.axvline(0, color="#475569", linestyle="--", linewidth=1)
    ax.set_yticks(y, plot["specification"])
    ax.set_xlabel("Within-cell change in log(HR-OC SPM) per 1-SD wave proxy")
    ax.set_title("Sensitivity of the Anza wave coefficient to wave summary and time-block length", loc="left", weight="bold")
    ax.grid(axis="x", color="#CBD5E1", alpha=.6, linewidth=.7)
    fig.savefig(OUT / "Figure_wave_response_robustness.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_wave_response_robustness.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Wave-response robustness analysis complete; fixed-effect regressions and bootstrap sensitivity fits were performed. No source observations were changed.")
    print(f"  Sensitivity table: {result_path}")
    print(f"  Leave-one-date-out table: {loo_path}")
    print(f"  Figure: {OUT / 'Figure_wave_response_robustness.png'}")
    print(f"  Interpretation notes: {OUT / 'wave_response_robustness_notes_FR.txt'}")
    print("These diagnostics assess sensitivity; they are not new independent validation or proof of causality.")


if __name__ == "__main__":
    main()

