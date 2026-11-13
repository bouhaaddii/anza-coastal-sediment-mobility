"""Describe seasonal persistence and temporal uncertainty in mobility scenarios.

Uses the daily summaries written by Script 32. The moving-block intervals
describe temporal sampling variability conditional on the existing forcing,
survey depths, grain-size scenarios, and mobility equations; they do not
represent model-structure or input-data uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Results" / "Sediment_Mobility_Scenarios" / "mobility_daily_by_depth_grain_datum_scenario.csv.gz"
OUT = ROOT / "Results" / "Sediment_Mobility_Scenarios" / "Temporal_Persistence"

ANCHOR_D50_MM = (0.10, 0.25, 1.00)
DATUM_CASES = ("ZH_unadjusted", "IHO_2024_candidate_offset_sensitivity")
FORCING_CASES = {
    "wave_only": ("wave_valid_hours", "wave_only_exceedance_fraction"),
    "waves_plus_daily_currents": ("combined_valid_hours", "combined_exceedance_fraction"),
}
MIN_VALID_HOURS_PER_DAY = 12
MIN_VALID_DAYS_PER_MONTH = 10
BLOCK_LENGTHS_DAYS = (3, 7)
BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 361904


def moving_block_indices(n_days: int, block_length: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Return non-circular moving-block bootstrap indices, one row per replicate."""
    if n_days < 1:
        return np.empty((0, 0), dtype=int)
    block = min(int(block_length), int(n_days))
    blocks_per_replicate = int(np.ceil(n_days / block))
    max_start = n_days - block + 1
    starts = rng.integers(0, max_start, size=(reps, blocks_per_replicate))
    indices = np.empty((reps, n_days), dtype=int)
    for block_i in range(blocks_per_replicate):
        left = block_i * block
        width = min(block, n_days - left)
        indices[:, left:left + width] = starts[:, block_i, None] + np.arange(width)[None, :]
    return indices


def weighted_frequency(fraction: np.ndarray, hours: np.ndarray, minimum_hours: int) -> tuple[float, int, int]:
    keep = np.isfinite(fraction) & np.isfinite(hours) & (hours >= minimum_hours) & (hours > 0)
    if not keep.any():
        return np.nan, 0, 0
    total_hours = int(np.sum(hours[keep]))
    estimate = float(np.sum(fraction[keep] * hours[keep]) / total_hours)
    return estimate, int(keep.sum()), total_hours


def bootstrap_interval(
    fraction: np.ndarray,
    hours: np.ndarray,
    block_length: int,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    valid = np.isfinite(fraction) & np.isfinite(hours) & (hours >= MIN_VALID_HOURS_PER_DAY) & (hours > 0)
    weights = np.where(valid, hours, 0.0)
    values = np.where(valid, fraction, 0.0)
    index = moving_block_indices(len(fraction), block_length, reps, rng)
    if index.size == 0:
        return np.nan, np.nan, 0
    sample_w = weights[index]
    denominator = sample_w.sum(axis=1)
    numerator = (values[index] * sample_w).sum(axis=1)
    estimate = np.divide(numerator, denominator, out=np.full(reps, np.nan), where=denominator > 0)
    estimate = estimate[np.isfinite(estimate)]
    if estimate.size < max(50, reps // 2):
        return np.nan, np.nan, int(estimate.size)
    lo, hi = np.quantile(estimate, [0.025, 0.975])
    return float(lo), float(hi), int(estimate.size)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Script 32 daily output not found: {INPUT}")
    data = pd.read_csv(INPUT, parse_dates=["date_utc"])
    required = {
        "date_utc", "datum_case", "survey_depth_bin_below_local_ZH_m",
        "D50_scenario_mm_not_measured", *[c for pair in FORCING_CASES.values() for c in pair],
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Daily mobility table is missing required columns: {missing}")

    data = data[data["D50_scenario_mm_not_measured"].apply(
        lambda value: any(np.isclose(float(value), anchor) for anchor in ANCHOR_D50_MM)
    )].copy()
    data["month"] = data["date_utc"].dt.strftime("%Y-%m")
    all_dates = pd.DatetimeIndex(sorted(data["date_utc"].dropna().unique()))
    if len(all_dates) < 30:
        raise ValueError(f"Only {len(all_dates)} distinct daily records; temporal block analysis is not supported.")

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []

    group_keys = [
        "datum_case", "survey_depth_bin_below_local_ZH_m", "D50_scenario_mm_not_measured"
    ]
    selected = data[data["datum_case"].isin(DATUM_CASES)].copy()
    for keys, group in selected.groupby(group_keys, sort=True):
        datum, depth, d50 = keys
        group = group.set_index("date_utc").reindex(all_dates).rename_axis("date_utc").reset_index()
        group["datum_case"] = datum
        group["survey_depth_bin_below_local_ZH_m"] = depth
        group["D50_scenario_mm_not_measured"] = d50
        group["month"] = group["date_utc"].dt.strftime("%Y-%m")

        for forcing_name, (hours_col, fraction_col) in FORCING_CASES.items():
            hours = pd.to_numeric(group[hours_col], errors="coerce").fillna(0).to_numpy(float)
            fraction = pd.to_numeric(group[fraction_col], errors="coerce").to_numpy(float)
            estimate, valid_days, total_hours = weighted_frequency(
                fraction, hours, MIN_VALID_HOURS_PER_DAY
            )
            ci: dict[int, tuple[float, float, int]] = {}
            for block_length in BLOCK_LENGTHS_DAYS:
                ci[block_length] = bootstrap_interval(
                    fraction, hours, block_length, BOOTSTRAP_REPLICATES, rng
                )

            row: dict[str, object] = {
                "datum_case": datum,
                "survey_depth_bin_below_local_ZH_m": float(depth),
                "D50_scenario_mm_not_measured": float(d50),
                "forcing_case": forcing_name,
                "period_start_utc": str(all_dates.min().date()),
                "period_end_utc": str(all_dates.max().date()),
                "dates_in_record_n": int(len(all_dates)),
                "valid_dates_ge_12h_n": valid_days,
                "valid_hours_total": total_hours,
                "hour_weighted_exceedance_fraction_M_ge_1": estimate,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "interval_interpretation": "temporal sampling variability conditional on this model and forcing record; not total/model uncertainty",
            }
            for block_length, (lo, hi, n_valid) in ci.items():
                row[f"moving_block_{block_length}d_ci95_low"] = lo
                row[f"moving_block_{block_length}d_ci95_high"] = hi
                row[f"moving_block_{block_length}d_valid_replicates"] = n_valid
            summary_rows.append(row)

            for month, month_group in group.groupby("month", sort=True):
                month_hours = pd.to_numeric(month_group[hours_col], errors="coerce").fillna(0).to_numpy(float)
                month_fraction = pd.to_numeric(month_group[fraction_col], errors="coerce").to_numpy(float)
                month_estimate, month_days, month_total_hours = weighted_frequency(
                    month_fraction, month_hours, MIN_VALID_HOURS_PER_DAY
                )
                monthly_rows.append({
                    "month_utc": month,
                    "datum_case": datum,
                    "survey_depth_bin_below_local_ZH_m": float(depth),
                    "D50_scenario_mm_not_measured": float(d50),
                    "forcing_case": forcing_name,
                    "valid_dates_ge_12h_n": month_days,
                    "valid_hours_total": month_total_hours,
                    "hour_weighted_exceedance_fraction_M_ge_1": month_estimate,
                    "display_support_flag": "supported" if month_days >= MIN_VALID_DAYS_PER_MONTH else "low support",
                })

    summary_frame = pd.DataFrame(summary_rows)
    no_support = summary_frame[summary_frame["valid_dates_ge_12h_n"].eq(0)]
    summary_path = OUT / "mobility_temporal_block_uncertainty.csv"
    monthly_path = OUT / "mobility_monthly_depth_frequency.csv"
    summary_frame.to_csv(summary_path, index=False, float_format="%.7g", encoding="utf-8-sig")
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(monthly_path, index=False, float_format="%.7g", encoding="utf-8-sig")

    # A compact seasonal view of a literature-informed fine-sand case and a coarse sensitivity.
    months = sorted(monthly["month_utc"].dropna().unique())
    labels = [pd.Timestamp(f"{month}-01").strftime("%b %Y") for month in months]
    selected_monthly = monthly[
        monthly["datum_case"].eq("ZH_unadjusted")
        & monthly["D50_scenario_mm_not_measured"].isin([0.10, 1.00])
    ].copy()
    depths = np.sort(selected_monthly["survey_depth_bin_below_local_ZH_m"].unique())
    depth_edges = np.r_[depths[0] - 0.25, (depths[:-1] + depths[1:]) / 2, depths[-1] + 0.25]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E7ECF1")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), sharex=True, sharey=True)
    mesh = None
    panels = [
        (0.10, "wave_only", "D50 = 0.10 mm scenario · waves"),
        (0.10, "waves_plus_daily_currents", "D50 = 0.10 mm scenario · waves + daily currents"),
        (1.00, "wave_only", "D50 = 1.00 mm sensitivity · waves"),
        (1.00, "waves_plus_daily_currents", "D50 = 1.00 mm sensitivity · waves + daily currents"),
    ]
    for ax, (d50, forcing, title) in zip(axes.flat, panels):
        subset = selected_monthly[
            np.isclose(selected_monthly["D50_scenario_mm_not_measured"], d50)
            & selected_monthly["forcing_case"].eq(forcing)
        ]
        pivot = subset.pivot(
            index="survey_depth_bin_below_local_ZH_m", columns="month_utc",
            values="hour_weighted_exceedance_fraction_M_ge_1",
        ).reindex(index=depths, columns=months)
        support = subset.pivot(
            index="survey_depth_bin_below_local_ZH_m", columns="month_utc",
            values="valid_dates_ge_12h_n",
        ).reindex(index=depths, columns=months)
        # A writable copy is needed before masking low-support cells below.
        values = pivot.to_numpy(dtype=float, copy=True)
        low_support = support.to_numpy(float) < MIN_VALID_DAYS_PER_MONTH
        values[low_support] = np.nan
        mesh = ax.pcolormesh(
            np.arange(len(months) + 1), depth_edges, np.ma.masked_invalid(values),
            shading="flat", cmap=cmap, vmin=0, vmax=1,
            edgecolors="white", linewidth=0.18,
        )
        ax.set_title(title, fontsize=10.1, pad=8)
        ax.set_yticks([0, 5, 10, 15, 20, 23])
        ax.set_ylim(depth_edges[-1], depth_edges[0])
        ax.grid(axis="y", color="white", lw=0.45, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xticks(np.arange(len(months)) + 0.5, labels=labels, rotation=25, ha="right")
    fig.supylabel("Survey depth below local port zero (m; positive down)", x=0.035)
    fig.supxlabel("Forcing month (UTC)", y=0.075)
    fig.subplots_adjust(left=0.12, right=0.87, top=0.90, bottom=0.18, hspace=0.28, wspace=0.15)
    cax = fig.add_axes([0.90, 0.21, 0.018, 0.64])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Hour-weighted fraction with modeled M ≥ 1")
    fig.text(
        0.5, 0.018,
        f"Gray cells: fewer than {MIN_VALID_DAYS_PER_MONTH} supported days in that month. "
        "Scenario frequencies are conditional model outputs, not probabilities or transport rates.",
        ha="center", va="bottom", fontsize=8.0, color="#475569",
    )
    figure_png = OUT / "Figure_mobility_temporal_persistence.png"
    figure_pdf = OUT / "Figure_mobility_temporal_persistence.pdf"
    fig.savefig(figure_png, dpi=350, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    caption_path = OUT / "figure_caption_EN.txt"
    caption_path.write_text(
        "Figure. Month-by-depth frequency of modeled incipient-motion threshold exceedance (M ≥ 1) for the local hydrographic zero without datum adjustment. Panels compare the 0.10 mm literature-informed but unmeasured near-bottom scenario and the 1.00 mm coarse sensitivity under wave-only and combined wave plus daily regional-current forcing. Each monthly value is weighted by the number of valid hourly records in supported days (at least 12 valid hours per day); gray cells have fewer than 10 supported days. The 0.10 mm value is historical regional context reported for the bottom of Agadir Bay, not a measurement from the 2022 Anza survey. The map does not resolve nearshore wave transformation or spatial variation in regional forcing. Exceedance frequency is not a probability, a sediment-transport rate, or a field observation.\n",
        encoding="utf-8",
    )
    notes_path = OUT / "interpretation_notes_FR.txt"
    note_lines = [
        "INTERPRÉTATION DE LA PERSISTANCE TEMPORELLE DE LA MOBILITÉ MODÉLISÉE",
        f"Période : {all_dates.min().date()} au {all_dates.max().date()} ({len(all_dates)} dates UTC).",
        f"Support journalier : au moins {MIN_VALID_HOURS_PER_DAY} heures valides. Les cellules mensuelles avec moins de {MIN_VALID_DAYS_PER_MONTH} jours supportés sont masquées dans la figure.",
        "Les moyennes globales ci-dessous sont des moyennes non pondérées entre classes de profondeur; elles ne sont pas pondérées par la superficie du fond.",
        "",
        "1. Résumé global par scénario, pour le zéro hydrographique local non ajusté",
    ]
    forcing_labels = {
        "wave_only": "vagues seules",
        "waves_plus_daily_currents": "vagues + courants journaliers",
    }
    for d50 in ANCHOR_D50_MM:
        for forcing, label in forcing_labels.items():
            part = summary_frame[
                summary_frame["datum_case"].eq("ZH_unadjusted")
                & np.isclose(summary_frame["D50_scenario_mm_not_measured"], d50)
                & summary_frame["forcing_case"].eq(forcing)
            ]
            mean_fraction = float(part["hour_weighted_exceedance_fraction_M_ge_1"].mean())
            median_fraction = float(part["hour_weighted_exceedance_fraction_M_ge_1"].median())
            low = float(part["hour_weighted_exceedance_fraction_M_ge_1"].min())
            high = float(part["hour_weighted_exceedance_fraction_M_ge_1"].max())
            width3 = (part["moving_block_3d_ci95_high"] - part["moving_block_3d_ci95_low"]).median()
            width7 = (part["moving_block_7d_ci95_high"] - part["moving_block_7d_ci95_low"]).median()
            note_lines.append(
                f"D50 scénario {d50:g} mm, {label} : moyenne entre classes={mean_fraction:.3f}; "
                f"médiane={median_fraction:.3f}; étendue={low:.3f}–{high:.3f}; "
                f"largeur médiane des IC95 temporels, blocs 3 j={width3:.3f} et 7 j={width7:.3f}."
            )

    note_lines.extend(["", "2. Moyenne mensuelle entre classes de profondeur (zéro local non ajusté)"])
    month_names = {
        "2022-10": "oct. 2022", "2022-11": "nov. 2022", "2022-12": "déc. 2022",
        "2023-01": "janv. 2023", "2023-02": "févr. 2023", "2023-03": "mars 2023",
    }
    for d50 in (0.10, 1.00):
        for forcing, label in forcing_labels.items():
            part = monthly[
                monthly["datum_case"].eq("ZH_unadjusted")
                & np.isclose(monthly["D50_scenario_mm_not_measured"], d50)
                & monthly["forcing_case"].eq(forcing)
            ]
            month_means = part.groupby("month_utc")["hour_weighted_exceedance_fraction_M_ge_1"].mean()
            values = "; ".join(
                f"{month_names.get(month, month)} : {value:.3f}"
                for month, value in month_means.items()
            )
            note_lines.append(f"D50 {d50:g} mm, {label} — fraction moyenne : {values}.")

    note_lines.extend([
        "",
        "3. Lecture et limites",
        "Le scénario D50 = 0,10 mm est informé par une valeur historique rapportée pour le fond de la baie d’Agadir; il ne s’agit pas d’une mesure du levé d’Anza de 2022. Le scénario 0,25 mm est également un contexte historique de plage, et 1,00 mm est une sensibilité grossière.",
        "Les intervalles par blocs mobiles de 3 et 7 jours quantifient seulement la variabilité temporelle conditionnelle à cette série de forçage et aux hypothèses du modèle. Ils ne couvrent ni l’incertitude structurelle des équations, ni la granulométrie non mesurée, ni l’incertitude de datum, ni la résolution insuffisante des champs près du rivage.",
        "Les fréquences de M ≥ 1 sont des sorties de scénario. Elles ne sont ni des probabilités d’érosion, ni des taux de transport, ni des observations de remise en suspension. La carte de mobilité et les différences profondeur–mois sont fortement conditionnées par le profil bathymétrique, car le forçage régional n’est pas résolu spatialement sur le levé.",
        f"Combinaisons sans jour respectant le seuil de support (≥ {MIN_VALID_HOURS_PER_DAY} heures valides par jour) : {len(no_support)}. Elles ont une fraction et des intervalles non estimables; ne pas les interpréter comme une mobilité nulle.",
        f"Cellules mensuelles de faible support masquées dans la figure (< {MIN_VALID_DAYS_PER_MONTH} jours supportés) : {(monthly['display_support_flag'] == 'low support').sum()} sur {len(monthly)}.",
    ])
    notes_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    metadata_path = OUT / "temporal_mobility_run_metadata.json"
    metadata_path.write_text(json.dumps({
        "input": str(INPUT),
        "output_period_utc": [str(all_dates.min().date()), str(all_dates.max().date())],
        "daily_support_threshold_hours": MIN_VALID_HOURS_PER_DAY,
        "monthly_support_threshold_days": MIN_VALID_DAYS_PER_MONTH,
        "grain_size_scenarios_mm": list(ANCHOR_D50_MM),
        "datum_cases": list(DATUM_CASES),
        "bootstrap_block_lengths_days": list(BLOCK_LENGTHS_DAYS),
        "bootstrap_replicates_per_block_length": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": RANDOM_SEED,
        "uncertainty_scope": "conditional temporal sampling variability only; excludes model structure and input-data uncertainty",
        "spatial_forcing_note": "Regional wave and daily current forcing is not resolved across the survey; modeled spatial patterns are mainly depth-conditioned.",
        "source_inputs_modified": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Temporal persistence and moving-block uncertainty analysis complete; Script 32 outputs were read only.")
    print(f"Daily records: {len(all_dates)}; grain-size scenarios: {', '.join(f'{d:g} mm' for d in ANCHOR_D50_MM)}")
    print(f"Summary intervals: {summary_path}")
    print(f"Monthly depth-frequency table: {monthly_path}")
    print(f"Combinations with no day meeting the {MIN_VALID_HOURS_PER_DAY}-hour support threshold: {len(no_support)} (not zero mobility)")
    print(f"Figures: {figure_png}; {figure_pdf}")
    print(f"Caption and interpretation: {caption_path}; {notes_path}")
    print(f"Run metadata: {metadata_path}")
    print("Intervals quantify temporal variability conditional on this modeled forcing record, not total model uncertainty.")


if __name__ == "__main__":
    main()
