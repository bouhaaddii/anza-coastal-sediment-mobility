"""Exploratory fixed-cell association of modeled sediment mobility with HR-OC SPM.

HR-OC is satellite-derived and the source granule/day match is not verified.
This is a product-association analysis, not independent validation or causality.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "Results" / "Additional_Analyses" / "Within_Cell_Wave_Response" / "within_cell_wave_response_panel.csv.gz"
MOBILITY = ROOT / "Results" / "Sediment_Mobility_Scenarios" / "mobility_daily_by_depth_grain_datum_scenario.csv.gz"
OUT = ROOT / "Results" / "Sediment_Mobility_Scenarios" / "HR_OC_Association"
D50S = (0.063, 0.100, 0.125, 0.250, 0.500, 1.000)
DATUM_CASES = ("ZH_unadjusted", "IHO_2024_candidate_offset_sensitivity")
MIN_VALID_HOURS = 12
BOOTSTRAP_REPLICATES = 500
TEMPORAL_BLOCK_LENGTH = 3
RANDOM_SEED = 33041


def fixed_effect_fit(frame: pd.DataFrame, predictor: str) -> dict[str, object]:
    data = frame[["log_spm", predictor, "cell_code", "date_index"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if data.empty:
        raise ValueError("No complete observations remain after support filtering.")
    _, inv = np.unique(data["cell_code"].to_numpy(), return_inverse=True)
    counts = np.bincount(inv).astype(float)
    y = data["log_spm"].to_numpy(float)
    ymean = np.bincount(inv, weights=y) / counts
    yd = y - ymean[inv]
    x = data[predictor].to_numpy(float)
    xmean = np.bincount(inv, weights=x) / counts
    xd = x - xmean[inv]
    t = data["date_index"].to_numpy(float)
    tmean = np.bincount(inv, weights=t) / counts
    td = t - tmean[inv]
    X = np.column_stack([xd, td])
    beta, _, rank, _ = np.linalg.lstsq(X, yd, rcond=None)
    if rank < 2:
        raise ValueError("The fixed-cell mobility/time design is rank deficient.")
    resid = yd - X @ beta
    denom = float(np.sum(yd**2))
    return {"beta": beta, "n_obs": int(len(data)), "n_cells": int(data.cell_code.nunique()),
            "n_dates": int(data.date_index.nunique()), "within_r2": 1 - float(np.sum(resid**2)) / denom if denom > 0 else np.nan}


def moving_date_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    chosen: list[int] = []
    if n == 0:
        return np.array([], dtype=int)
    max_start = max(1, n - length + 1)
    while len(chosen) < n:
        start = int(rng.integers(0, max_start))
        chosen.extend(range(start, min(start + length, n)))
    return np.asarray(chosen[:n], dtype=int)


def two_axis_block_bootstrap(frame: pd.DataFrame, predictor: str, *, reps: int, block_length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = np.sort(frame["spatial_block_500m"].dropna().unique())
    dates = np.sort(frame["date_index"].unique())
    if len(blocks) < 10 or len(dates) < 10:
        return np.empty((0, 2), dtype=float)
    frame = frame.copy().reset_index(drop=True)
    lookup = {(block, int(day)): g.index.to_numpy(int)
              for (block, day), g in frame.groupby(["spatial_block_500m", "date_index"], sort=False)}
    cell_codes = frame["cell_code"].to_numpy(int)
    estimates: list[np.ndarray] = []
    max_cell = int(frame["cell_code"].max()) + 1
    for _ in range(reps):
        selected_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        selected_dates = moving_date_indices(len(dates), block_length, rng)
        row_ids: list[int] = []
        group_ids: list[int] = []
        for block_draw, block in enumerate(selected_blocks):
            offset = (block_draw + 1) * max_cell
            for date_pos in selected_dates:
                rows = lookup.get((block, int(dates[date_pos])))
                if rows is not None and rows.size:
                    row_ids.extend(rows.tolist())
                    group_ids.extend((cell_codes[rows] + offset).tolist())
        if len(row_ids) < 50:
            continue
        sample = frame.iloc[np.asarray(row_ids, dtype=int)].copy()
        sample["cell_code"] = np.asarray(group_ids, dtype=int)
        try:
            estimates.append(np.asarray(fixed_effect_fit(sample, predictor)["beta"], dtype=float))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return np.vstack(estimates) if estimates else np.empty((0, 2), dtype=float)


def main() -> None:
    for path in (PANEL, MOBILITY):
        if not path.is_file():
            raise SystemExit(f"Input missing: {path}. Run script 26 and the updated script 32 first.")
    panel = pd.read_csv(PANEL)
    mob = pd.read_csv(MOBILITY)
    panel["date_utc"] = pd.to_datetime(panel["date_utc"]).dt.strftime("%Y-%m-%d")
    panel["depth_bin_m"] = (panel["depth_below_local_zero_m"] * 2).round() / 2
    mob["date_utc"] = pd.to_datetime(mob["date_utc"]).dt.strftime("%Y-%m-%d")
    join_keys = ["date_utc", "survey_depth_bin_below_local_ZH_m"]
    panel = panel.rename(columns={"depth_bin_m": "survey_depth_bin_below_local_ZH_m"})
    merged_all: list[pd.DataFrame] = []
    result_rows: list[dict] = []
    support_rows: list[dict] = []
    scenario_id = 0

    for datum in DATUM_CASES:
        for d50 in D50S:
            right = mob[(mob["datum_case"] == datum) & np.isclose(mob["D50_scenario_mm_not_measured"], d50)].copy()
            base = panel.merge(right, on=join_keys, how="left", validate="many_to_one")
            if base["combined_valid_hours"].isna().any():
                warnings.warn(f"Unmatched mobility rows for {datum}, D50={d50:g} mm.")
            support_rows.append({
                "datum_case": datum, "D50_scenario_mm_not_measured": d50,
                "panel_rows": len(base), "matched_rows": int(base["combined_valid_hours"].notna().sum()),
                "dates_n": int(base.loc[base["combined_valid_hours"].notna(), "date_utc"].nunique()),
                "cells_n": int(base.loc[base["combined_valid_hours"].notna(), "cell_code"].nunique()),
                "min_wave_valid_hours": int(base["wave_valid_hours"].min()) if base.wave_valid_hours.notna().any() else 0,
                "min_combined_valid_hours": int(base["combined_valid_hours"].min()) if base.combined_valid_hours.notna().any() else 0,
                "rows_retained_at_12h_common_support": int((base[["wave_valid_hours", "combined_valid_hours"]].min(axis=1) >= MIN_VALID_HOURS).sum()),
            })
            common = base[base[["wave_valid_hours", "combined_valid_hours"]].min(axis=1) >= MIN_VALID_HOURS].copy()
            for method, metric, valid_col in [
                ("Wave_only", "wave_M_p95", "wave_valid_hours"),
                ("Wave_plus_daily_current", "combined_M_p95", "combined_valid_hours"),
            ]:
                frame = common.dropna(subset=["log_spm", metric, "date_index", "cell_code", "spatial_block_500m"]).copy()
                if frame.empty:
                    continue
                raw = np.log1p(np.maximum(frame[metric].to_numpy(float), 0.0))
                sd = float(np.std(raw, ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    print(f"Skipped {datum}, D50={d50:g}, {method}: mobility predictor has no variation.")
                    continue
                predictor = "mobility_log1p_p95_z"
                frame[predictor] = (raw - float(np.mean(raw))) / sd
                frame["date_index"] = pd.to_numeric(frame["date_index"], errors="coerce")
                if frame.date_index.nunique() < 10 or frame.spatial_block_500m.nunique() < 10:
                    print(f"Skipped {datum}, D50={d50:g}, {method}: insufficient independent date/block support.")
                    continue
                try:
                    fit = fixed_effect_fit(frame, predictor)
                except ValueError as exc:
                    print(f"Skipped {datum}, D50={d50:g}, {method}: {exc}")
                    continue
                boot = two_axis_block_bootstrap(frame, predictor, reps=BOOTSTRAP_REPLICATES,
                                                block_length=TEMPORAL_BLOCK_LENGTH,
                                                seed=RANDOM_SEED + scenario_id)
                scenario_id += 1
                low, high = (np.nan, np.nan)
                if boot.shape[0] >= 100:
                    low, high = np.nanpercentile(boot[:, 0], [2.5, 97.5])
                result_rows.append({
                    "outcome": "log(HR-OC SPM; satellite-derived)", "model": method,
                    "predictor": "standardized log1p(daily p95 mobility ratio M)",
                    "datum_case": datum, "D50_scenario_mm_not_measured": d50,
                    "beta_log_spm_per_1sd_mobility": float(fit["beta"][0]),
                    "effect_pct_per_1sd_mobility": float(100.0 * np.expm1(fit["beta"][0])),
                    "block_bootstrap_95ci_low": float(low),
                    "block_bootstrap_95ci_high": float(high), "bootstrap_replicates_valid": int(boot.shape[0]),
                    "effect_pct_95ci_low": float(100.0 * np.expm1(low)) if np.isfinite(low) else np.nan,
                    "effect_pct_95ci_high": float(100.0 * np.expm1(high)) if np.isfinite(high) else np.nan,
                    "observations_n": fit["n_obs"], "cells_n": fit["n_cells"], "dates_n": fit["n_dates"],
                    "spatial_blocks_n": int(frame["spatial_block_500m"].nunique()), "within_r2": float(fit["within_r2"]),
                    "minimum_common_valid_hours_per_day": MIN_VALID_HOURS,
                })
                frame["method"] = method
                frame["datum_case"] = datum
                frame["D50_scenario_mm_not_measured"] = d50
                frame["mobility_predictor_z"] = frame[predictor]
                frame["scenario_metric_p95_M"] = frame[metric]
                frame["scenario_valid_hours"] = frame[valid_col]
                merged_all.append(frame[["date_utc", "cell_code", "spatial_block_500m", "depth_below_local_zero_m",
                                         "spm_g_m3", "log_spm", "method", "datum_case", "D50_scenario_mm_not_measured",
                                         "scenario_metric_p95_M", "scenario_valid_hours", "mobility_predictor_z"]])

    OUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(result_rows)
    support = pd.DataFrame(support_rows)
    result_path = OUT / "mobility_HR_OC_fixed_cell_models.csv"
    support_path = OUT / "mobility_HR_OC_support_audit.csv"
    panel_path = OUT / "mobility_HR_OC_matched_panel.csv.gz"
    result.to_csv(result_path, index=False, encoding="utf-8-sig", float_format="%.7g")
    support.to_csv(support_path, index=False, encoding="utf-8-sig", float_format="%.7g")
    if merged_all:
        pd.concat(merged_all, ignore_index=True).to_csv(panel_path, index=False, compression="gzip", encoding="utf-8-sig", float_format="%.7g")

    if not result.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 8.0), sharex=True)
        for ax, datum in zip(axes, DATUM_CASES):
            sub = result[result.datum_case == datum].copy()
            labels, ys = [], []
            for _, row in sub.sort_values(["D50_scenario_mm_not_measured", "model"]).iterrows():
                label = f"{row.D50_scenario_mm_not_measured:g} mm · {row.model.replace('_', ' ')}"
                labels.append(label)
                ys.append(len(ys))
                y = ys[-1]
                color = "#2A788E" if row.model == "Wave_only" else "#D55E00"
                lo, hi, beta = (row.block_bootstrap_95ci_low, row.block_bootstrap_95ci_high,
                                row.beta_log_spm_per_1sd_mobility)
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round")
                    ax.plot([lo, lo], [y - 0.12, y + 0.12], color=color, lw=1.0)
                    ax.plot([hi, hi], [y - 0.12, y + 0.12], color=color, lw=1.0)
                ax.plot(beta, y, "o", color=color, ms=4.8)
            ax.axvline(0, color="#475569", lw=1, linestyle="--")
            ax.set_yticks(ys, labels)
            ax.invert_yaxis()
            ax.set_title("Local ZH, unadjusted" if datum == "ZH_unadjusted" else "Candidate +2.159 m sensitivity")
            ax.grid(axis="x", color="#D9E2E8", linewidth=0.55)
        axes[0].set_ylabel("Grain-size scenario and forcing case")
        axes[0].set_xlabel("Change in log(SPM) per 1-SD mobility increase (95% block-bootstrap interval)")
        axes[1].set_xlabel("Change in log(SPM) per 1-SD mobility increase (95% block-bootstrap interval)")
        fig.suptitle("Association of modeled mobility intensity with satellite-derived HR-OC SPM", fontsize=12)
        fig.text(0.5, 0.015,
                 "Exploratory association only; same-day HR-OC source granule is unverified, and this is not in-situ validation or causality.",
                 ha="center", fontsize=8)
        fig.subplots_adjust(left=0.31, right=0.98, top=0.91, bottom=0.12, wspace=0.12)
        fig.savefig(OUT / "Figure_mobility_HR_OC_association.png", dpi=350, bbox_inches="tight")
        fig.savefig(OUT / "Figure_mobility_HR_OC_association.pdf", bbox_inches="tight")
        plt.close(fig)

    notes = [
        "Association analysis: model-derived daily sediment mobility versus satellite-derived HR-OC SPM at Anza.",
        "Fixed-cell model: natural log(SPM) with cell fixed effects, log1p(daily 95th percentile of M) standardized as the predictor, and a linear date trend.",
        "Coefficient interpretation: change in natural log(SPM) per 1-standard-deviation increase in the mobility predictor; the response itself is not standardized. exp(beta)-1 is also provided as a model-implied relative percentage change, not as a calibrated concentration change.",
        "The same at-least-12-valid-hour daily support filter is used for wave-only and combined cases before within-cell fitting.",
        "Uncertainty intervals resample 500 m spatial blocks and moving temporal blocks of three dates; they are exploratory with only 15 spatial blocks and 33 dates.",
        "HR-OC is satellite-derived; its source granule is not confirmed to match the local Sentinel-2 scene. This is product association, not independent validation, an SSC calibration, sediment flux, or causal inference.",
        "All D50 values are unmeasured scenarios. The 0.10 mm case is a literature-informed historical near-bottom benchmark reported for Agadir Bay by Aouiche et al. (2016), based on LCHF (1972), not a measurement from the 2022 Anza survey. The 0.25 mm value is a historical beach value and is not treated as subtidal grain size; remaining D50 values are analyst-defined sensitivities. See Results/Sediment_Mobility_Scenarios/D50_scenario_provenance.csv. The +2.159 m offset is also a sensitivity, not a verified vertical datum correction.",
        f"Models written: {len(result)}; support audit rows: {len(support)}.",
    ]
    if not result.empty:
        crosses_zero = (result["block_bootstrap_95ci_low"] <= 0) & (result["block_bootstrap_95ci_high"] >= 0)
        beta_min = float(result["beta_log_spm_per_1sd_mobility"].min())
        beta_max = float(result["beta_log_spm_per_1sd_mobility"].max())
        r2_min = float(result["within_r2"].min())
        r2_max = float(result["within_r2"].max())
        notes.append(
            f"Across the {len(result)} scenario fits, beta ranged from {beta_min:.3f} to {beta_max:.3f} log units per 1-SD predictor increase; "
            f"{int(crosses_zero.sum())}/{len(result)} block-bootstrap 95% intervals included zero. Within-cell R2 ranged from {r2_min:.3f} to {r2_max:.3f}."
        )
    (OUT / "mobility_HR_OC_interpretation_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (OUT / "figure_caption_EN.txt").write_text(
        "Figure. Within-cell fixed-effect association between daily modeled bed-mobility intensity (the 95th percentile of the non-dimensional mobility ratio M) and the Copernicus Marine HR-OC satellite-derived SPM estimate. Points show changes in natural log(SPM) per one-standard-deviation increase in the mobility predictor; only the predictor is standardized. D50 values are unmeasured scenarios: 0.10 mm is a historical near-bottom literature benchmark for Agadir Bay reported by Aouiche et al. (2016), whereas 0.25 mm is a historical beach value; neither is a measurement from the 2022 Anza survey. Intervals are exploratory 95% two-axis bootstrap ranges based on resampling 500 m spatial blocks and moving blocks of three dates. Panels show the unadjusted local hydrographic zero and the unverified +2.159 m sensitivity; colors separate wave-only from wave-plus-daily-current forcing. This is not in-situ validation, an SSC calibration, or evidence of causation; the HR-OC source granule is not confirmed to match the local Sentinel-2 acquisition.",
        encoding="utf-8",
    )
    print("Mobility–HR-OC association analysis complete; input panels and modeled source data were read only.")
    print(f"Model rows: {len(result)}; support rows: {len(support)}")
    print(f"Models: {result_path}")
    print(f"Support audit: {support_path}")
    if not result.empty:
        print(f"Figure: {OUT / 'Figure_mobility_HR_OC_association.png'}")
    print(f"Interpretation limits: {OUT / 'mobility_HR_OC_interpretation_FR.txt'}")
    print("Do not call HR-OC an independent field validation; interpret results as within-cell product association only.")


if __name__ == "__main__":
    main()

