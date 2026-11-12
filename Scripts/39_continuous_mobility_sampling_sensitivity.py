"""Continuous-mobility sampling sensitivity for the Anza Sentinel-2 date audit.

Reads the saved Script 32 daily scenario table and Script 6 scene catalog.
The continuous metric reduces saturation of binary M >= 1 fractions, but remains
a model-conditional scenario indicator, not sediment flux or observation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORCING_FILE = ROOT / "Results" / "scene_forcing_and_quality.csv"
PROCESSED_SUPPORT_FILE = (
    ROOT / "Results" / "Sediment_Proxy" / "Spatial_Support"
    / "ndti_spatial_support_by_scene.csv"
)
MOBILITY_FILE = (
    ROOT / "Results" / "Sediment_Mobility_Scenarios"
    / "mobility_daily_by_depth_grain_datum_scenario.csv.gz"
)
OUT = ROOT / "Results" / "Additional_Analyses" / "Scene_Sampling_Representativeness"
DATUM_CASE = "ZH_unadjusted"
D50_CASES = (0.10, 0.25, 1.00)
MIN_VALID_HOURS_PER_DAY = 12
GROUPS = (
    "All Sentinel-2 catalog dates",
    "Cloud cover <= 50%",
    "Cloud cover <= 20%",
    "Cloud cover <= 10%",
    "Processed ROI-QA scenes",
)
SHORT_LABELS = {
    "All Sentinel-2 catalog dates": "Catalog",
    "Cloud cover <= 50%": "Cloud ≤50%",
    "Cloud cover <= 20%": "Cloud ≤20%",
    "Cloud cover <= 10%": "Cloud ≤10%",
    "Processed ROI-QA scenes": "ROI-QA 11",
}
COLORS = {
    "All Sentinel-2 catalog dates": "#247BA0",
    "Cloud cover <= 50%": "#70A288",
    "Cloud cover <= 20%": "#E3A018",
    "Cloud cover <= 10%": "#E76F51",
    "Processed ROI-QA scenes": "#7B61A8",
}


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def scene_groups(scenes: pd.DataFrame, processed_names: set[str]) -> dict[str, pd.DataFrame]:
    scenes = scenes.copy()
    scenes["processed_roi_qa"] = scenes["product_name"].astype(str).isin(processed_names)
    return {
        "All Sentinel-2 catalog dates": scenes,
        "Cloud cover <= 50%": scenes[
            scenes["cloud_cover_product_pct"].notna()
            & scenes["cloud_cover_product_pct"].le(50)
        ],
        "Cloud cover <= 20%": scenes[
            scenes["cloud_cover_product_pct"].notna()
            & scenes["cloud_cover_product_pct"].le(20)
        ],
        "Cloud cover <= 10%": scenes[
            scenes["cloud_cover_product_pct"].notna()
            & scenes["cloud_cover_product_pct"].le(10)
        ],
        "Processed ROI-QA scenes": scenes[scenes["processed_roi_qa"]],
    }


def summarize_scenario(
    daily: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    d50: float,
    forcing_case: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if forcing_case == "wave_only":
        hours_col, metric_col = "wave_valid_hours", "wave_M_p95"
    else:
        hours_col, metric_col = "combined_valid_hours", "combined_M_p95"

    frame = daily[
        np.isclose(pd.to_numeric(daily["D50_scenario_mm_not_measured"], errors="coerce"), d50)
        & daily["datum_case"].eq(DATUM_CASE)
    ].copy()
    frame[hours_col] = pd.to_numeric(frame[hours_col], errors="coerce")
    frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
    dates = pd.DatetimeIndex(sorted(frame["date_utc"].dropna().unique()))
    hours = frame.pivot_table(
        index="date_utc", columns="survey_depth_bin_below_local_ZH_m",
        values=hours_col, aggfunc="first",
    ).reindex(dates)
    metric = frame.pivot_table(
        index="date_utc", columns="survey_depth_bin_below_local_ZH_m",
        values=metric_col, aggfunc="first",
    ).reindex(dates)

    common_depths = [
        depth for depth in hours.columns
        if hours[depth].notna().all() and hours[depth].ge(MIN_VALID_HOURS_PER_DAY).all()
    ]
    if not common_depths:
        raise ValueError(
            f"No fixed depth support for D50={d50:g} mm and {forcing_case}."
        )

    # The p95 M values are reduced to log10 scale before averaging across the
    # identical depth classes, limiting dominance by a few very large ratios.
    fixed_metric = metric[common_depths].where(hours[common_depths].ge(MIN_VALID_HOURS_PER_DAY))
    daily_score = np.log10(fixed_metric.where(fixed_metric.gt(0))).mean(axis=1, skipna=True)
    all_weather = pd.DataFrame({
        "date_utc": dates,
        "date_score_mean_log10_M_p95": daily_score.to_numpy(),
    })
    all_weather["month"] = pd.to_datetime(all_weather["date_utc"]).dt.month
    baseline = float(all_weather["date_score_mean_log10_M_p95"].mean())
    monthly_means = all_weather.groupby("month")["date_score_mean_log10_M_p95"].mean()
    records: list[dict[str, object]] = []

    for label in GROUPS:
        group = groups[label]
        group_dates = pd.to_datetime(group["acquisition_utc"], utc=True, errors="coerce").dt.normalize().dropna()
        unique_dates = pd.DatetimeIndex(group_dates.unique()).tz_localize(None)
        date_set = set(unique_dates)
        sampled = all_weather[pd.to_datetime(all_weather["date_utc"]).isin(date_set)].copy()
        sampled = sampled.dropna(subset=["date_score_mean_log10_M_p95"])
        sample_mean = float(sampled["date_score_mean_log10_M_p95"].mean()) if not sampled.empty else np.nan
        month_counts = pd.Series(pd.to_datetime(sorted(date_set)).month).value_counts().sort_index() if date_set else pd.Series(dtype=float)
        month_weights = month_counts / month_counts.sum() if not month_counts.empty else month_counts
        matched_months = month_weights.index.intersection(monthly_means.index)
        matched_baseline = (
            float((monthly_means.loc[matched_months] * month_weights.loc[matched_months]).sum()
                  / month_weights.loc[matched_months].sum())
            if len(matched_months) and month_weights.loc[matched_months].sum() > 0 else np.nan
        )
        total_delta = sample_mean - baseline if np.isfinite(sample_mean) else np.nan
        season_delta = matched_baseline - baseline if np.isfinite(matched_baseline) else np.nan
        within_month_delta = sample_mean - matched_baseline if np.isfinite(sample_mean) and np.isfinite(matched_baseline) else np.nan
        records.append({
            "group": label,
            "D50_scenario_mm_not_measured": d50,
            "forcing_case": forcing_case,
            "unique_scene_dates_n": int(len(date_set)),
            "dates_with_valid_fixed_depth_metric_n": int(len(sampled)),
            "fixed_depth_bins_n": int(len(common_depths)),
            "fixed_depth_min_m": float(min(common_depths)),
            "fixed_depth_max_m": float(max(common_depths)),
            "all_weather_mean_log10_p95_M": baseline,
            "sample_mean_log10_p95_M": sample_mean,
            "total_difference_log10_units": total_delta,
            "month_matched_all_weather_mean_log10_p95_M": matched_baseline,
            "season_composition_difference_log10_units": season_delta,
            "within_month_difference_log10_units": within_month_delta,
            "sampled_months": ";".join(str(int(m)) for m in month_counts.index),
            "metric_definition": "mean across fixed depth bins of daily log10(p95 of modeled mobility ratio M)",
            "interpretation": "descriptive model-conditional sensitivity; not sediment flux, observed mobility, probability, or causal effect",
        })
    return records, all_weather


def main() -> None:
    for path in (FORCING_FILE, PROCESSED_SUPPORT_FILE, MOBILITY_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    scenes = pd.read_csv(FORCING_FILE)
    require_columns(
        scenes,
        {"product_name", "acquisition_utc", "cloud_cover_product_pct"},
        str(FORCING_FILE),
    )
    scenes["acquisition_utc"] = pd.to_datetime(scenes["acquisition_utc"], utc=True, errors="coerce")
    scenes["cloud_cover_product_pct"] = pd.to_numeric(scenes["cloud_cover_product_pct"], errors="coerce")
    scenes = scenes.dropna(subset=["acquisition_utc"]).drop_duplicates("product_name").copy()

    processed = pd.read_csv(PROCESSED_SUPPORT_FILE)
    require_columns(processed, {"product_name"}, str(PROCESSED_SUPPORT_FILE))
    groups = scene_groups(scenes, set(processed["product_name"].dropna().astype(str)))

    daily = pd.read_csv(MOBILITY_FILE, parse_dates=["date_utc"])
    required = {
        "date_utc", "datum_case", "survey_depth_bin_below_local_ZH_m",
        "D50_scenario_mm_not_measured", "wave_valid_hours", "wave_M_p95",
        "combined_valid_hours", "combined_M_p95",
    }
    require_columns(daily, required, str(MOBILITY_FILE))
    daily = daily[daily["datum_case"].eq(DATUM_CASE)].copy()
    OUT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for d50 in D50_CASES:
        for forcing_case in ("wave_only", "waves_plus_daily_currents"):
            rows, _ = summarize_scenario(daily, groups, d50, forcing_case)
            records.extend(rows)
    summary = pd.DataFrame(records)
    table_path = OUT / "continuous_mobility_sampling_sensitivity.csv"
    summary.to_csv(table_path, index=False, float_format="%.7g", encoding="utf-8-sig")
    figure_path = make_figure(summary, OUT)
    notes_path = write_notes(summary, OUT)
    print("Continuous mobility-ratio sampling sensitivity complete; no model was refitted.")
    print("Metric: mean across fixed depth bins of daily log10(p95 modeled mobility ratio M).")
    print("This reduces binary threshold saturation; it remains conditional on the Script 32 assumptions.")
    print(f"Catalog dates: {scenes.product_name.nunique()}; processed ROI-QA scenes: {len(groups['Processed ROI-QA scenes'])}")
    print(f"Table: {table_path}")
    print(f"Figure: {figure_path}")
    print(f"Interpretation: {notes_path}")


def make_figure(summary: pd.DataFrame, out: Path) -> Path:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titleweight": "bold", "axes.labelcolor": "#243447",
        "text.color": "#243447", "axes.edgecolor": "#9AA8B5",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.5), sharex=True, sharey=True)
    cases = [
        (0.10, "wave_only", "A  D50 scenario 0.10 mm · waves"),
        (0.10, "waves_plus_daily_currents", "B  D50 scenario 0.10 mm · waves + daily currents"),
        (0.25, "wave_only", "C  D50 scenario 0.25 mm · waves"),
        (0.25, "waves_plus_daily_currents", "D  D50 scenario 0.25 mm · waves + daily currents"),
        (1.00, "wave_only", "E  D50 scenario 1.00 mm · waves"),
        (1.00, "waves_plus_daily_currents", "F  D50 scenario 1.00 mm · waves + daily currents"),
    ]
    y = np.arange(len(GROUPS))
    for ax, (d50, forcing, title) in zip(axes.flat, cases):
        part = summary[
            np.isclose(summary["D50_scenario_mm_not_measured"], d50)
            & summary["forcing_case"].eq(forcing)
        ].set_index("group").reindex(GROUPS)
        ax.axvline(0, color="#4C5964", lw=0.85, ls=(0, (3, 2)), zorder=0)
        total = pd.to_numeric(part["total_difference_log10_units"], errors="coerce").to_numpy()
        within = pd.to_numeric(part["within_month_difference_log10_units"], errors="coerce").to_numpy()
        ax.scatter(total, y - 0.11, s=34, color="#247BA0", edgecolor="white", linewidth=0.55,
                   label="Total difference from 182-day mean", zorder=3)
        ax.scatter(within, y + 0.11, s=34, marker="D", color="#E76F51", edgecolor="white", linewidth=0.55,
                   label="Within-month difference", zorder=3)
        ax.set_title(title, loc="left", fontsize=9, pad=7)
        ax.grid(axis="x", color="#E8EDF1", lw=0.65)
        ax.set_yticks(y, [SHORT_LABELS[g] for g in GROUPS])
        ax.tick_params(axis="y", labelsize=8)
        if ax in axes[1]:
            ax.set_xlabel("Difference in mean log10(p95 M) from all-weather reference")
    axes[0, 0].set_ylabel("Sentinel-2 sampling set")
    axes[1, 0].set_ylabel("Sentinel-2 sampling set")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.055))
    fig.suptitle("Continuous model-based mobility intensity across optical sampling sets", fontsize=13, weight="bold", y=0.98)
    fig.text(
        0.5, 0.018,
        "Daily p95 M is a scenario metric; grain size is unmeasured, and regional forcing does not resolve the surf zone.",
        ha="center", fontsize=8.2, color="#5F6B76",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.92, bottom=0.16, wspace=0.22, hspace=0.28)
    png = out / "Figure_continuous_mobility_sampling_sensitivity.png"
    pdf = out / "Figure_continuous_mobility_sampling_sensitivity.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (out / "figure_continuous_mobility_sampling_caption_EN.txt").write_text(
        "Figure. Differences in a continuous, model-based mobility-intensity indicator between Sentinel-2 sampling sets and the complete 182-day forcing record. The plotted score is the mean across fixed bathymetric depth classes of daily log10 of the 95th percentile of the modeled mobility ratio M. Blue circles show the unstandardized difference from the full-period mean; orange diamonds show the within-month difference after matching the full-period reference to the months represented in each image set. The six panels compare three unmeasured D50 scenarios and wave-only versus wave-plus-regional-daily-current forcing. Cloud cover is product-tile metadata. Results are descriptive and conditional on the regional forcing, local hydrographic zero, and Script 32 equations; they are not measurements of sediment flux, observed mobility, SSC, or causal effects.\n",
        encoding="utf-8",
    )
    return png


def write_notes(summary: pd.DataFrame, out: Path) -> Path:
    rows = summary[summary["group"].eq("Processed ROI-QA scenes")].copy()
    lines = [
        "INTERPRÉTATION — SENSIBILITÉ CONTINUE DE LA MOBILITÉ AUX DATES SENTINEL-2",
        "",
        "Le score est la moyenne, sur les classes fixes de profondeur, de log10 du p95 journalier du rapport de mobilité M.",
        "Il réduit la saturation des fréquences binaires M >= 1, mais demeure entièrement conditionnel aux équations,",
        "aux forçages régionaux et aux scénarios de D50 du Script 32. Il ne mesure pas un transport sédimentaire.",
        "",
        "Résultats pour les 11 scènes avec contrôle image ROI :",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"- D50 scénario {row.D50_scenario_mm_not_measured:g} mm; {row.forcing_case}: "
            f"différence totale={row.total_difference_log10_units:+.4f} log10; "
            f"composition mensuelle={row.season_composition_difference_log10_units:+.4f}; "
            f"écart intra-mensuel={row.within_month_difference_log10_units:+.4f}; "
            f"profondeurs fixes={row.fixed_depth_bins_n} ({row.fixed_depth_min_m:g}–{row.fixed_depth_max_m:g} m)."
        )
    lines.extend([
        "",
        "La décomposition conserve les mois représentés dans le sous-échantillon et compare leur moyenne pondérée à la",
        "référence toutes conditions. Elle ne transforme pas les acquisitions en échantillon aléatoire et ne corrige pas",
        "la couverture nuageuse. N’utiliser ni les intervalles ni le langage d’un biais de SSC; aucune concentration",
        "in situ, granulométrie locale ou mesure de flux n’est disponible.",
    ])
    path = out / "continuous_mobility_sampling_interpretation_FR.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()

