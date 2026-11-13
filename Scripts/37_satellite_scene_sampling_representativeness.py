"""Compare Sentinel-2 acquisition sampling with modeled mobility conditions.

This is a descriptive representativeness audit, not an SSC-bias estimate.
Product cloud cover is tile-level metadata, and the mobility values are
conditional on the existing wave/current forcing and sediment scenarios.
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

CLOUD_THRESHOLDS = (10, 20, 50)
D50_CASES_MM = (0.10, 0.25, 1.00)
MIN_VALID_HOURS_PER_DEPTH_DAY = 12
DATUM_CASE = "ZH_unadjusted"
FORCING_CASES = {
    "wave_only": ("wave_valid_hours", "wave_only_exceedance_fraction"),
    "waves_plus_daily_currents": (
        "combined_valid_hours", "combined_exceedance_fraction"
    ),
}
GROUP_LABELS = {
    "All Sentinel-2 catalog dates": "All_S2_catalog",
    "Cloud cover <= 50%": "Cloud_le_50pct",
    "Cloud cover <= 20%": "Cloud_le_20pct",
    "Cloud cover <= 10%": "Cloud_le_10pct",
    "Processed ROI-QA scenes": "Processed_ROI_QA",
}
GROUP_COLORS = {
    "All Sentinel-2 catalog dates": "#247BA0",
    "Cloud cover <= 50%": "#70A288",
    "Cloud cover <= 20%": "#E3A018",
    "Cloud cover <= 10%": "#E76F51",
    "Processed ROI-QA scenes": "#7B61A8",
}


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_mobility() -> tuple[pd.DataFrame, pd.DataFrame]:
    mobility = pd.read_csv(MOBILITY_FILE, parse_dates=["date_utc"])
    required = {
        "date_utc", "datum_case", "survey_depth_bin_below_local_ZH_m",
        "D50_scenario_mm_not_measured",
        *[name for pair in FORCING_CASES.values() for name in pair],
    }
    require_columns(mobility, required, str(MOBILITY_FILE))
    mobility = mobility[mobility["datum_case"].eq(DATUM_CASE)].copy()
    mobility["date_utc"] = mobility["date_utc"].dt.normalize()

    date_rows: list[dict[str, object]] = []
    common_support_rows: list[dict[str, object]] = []
    for d50 in D50_CASES_MM:
        scenario = mobility[np.isclose(
            pd.to_numeric(mobility["D50_scenario_mm_not_measured"], errors="coerce"), d50
        )].copy()
        if scenario.empty:
            raise ValueError(f"No mobility rows found for D50 scenario {d50:g} mm.")
        dates = pd.DatetimeIndex(sorted(scenario["date_utc"].dropna().unique()))
        for forcing_name, (hours_column, fraction_column) in FORCING_CASES.items():
            hours = scenario.pivot_table(
                index="date_utc", columns="survey_depth_bin_below_local_ZH_m",
                values=hours_column, aggfunc="first",
            ).reindex(dates)
            fractions = scenario.pivot_table(
                index="date_utc", columns="survey_depth_bin_below_local_ZH_m",
                values=fraction_column, aggfunc="first",
            ).reindex(dates)
            valid_hours = hours.apply(pd.to_numeric, errors="coerce")
            # Primary comparison uses depth bins supported for every modeled day,
            # keeping the depth composition identical across sampling groups.
            common_depths = [
                depth for depth in valid_hours.columns
                if valid_hours[depth].notna().all()
                and (valid_hours[depth] >= MIN_VALID_HOURS_PER_DEPTH_DAY).all()
            ]
            if not common_depths:
                raise ValueError(
                    f"No fixed depth support for D50={d50:g}, forcing={forcing_name}."
                )
            common_support_rows.append({
                "D50_scenario_mm_not_measured": d50,
                "forcing_case": forcing_name,
                "all_weather_dates_n": len(dates),
                "common_depth_bins_n": len(common_depths),
                "common_depth_min_m": float(min(common_depths)),
                "common_depth_max_m": float(max(common_depths)),
                "depth_support_rule": ">=12 valid forcing hours on every modeled date",
            })
            fixed_values = fractions[common_depths].apply(
                pd.to_numeric, errors="coerce"
            )
            fixed_values = fixed_values.where(valid_hours[common_depths] >= MIN_VALID_HOURS_PER_DEPTH_DAY)
            daily_fixed = fixed_values.mean(axis=1, skipna=True)
            # Secondary diagnostic retains every depth bin supported that day;
            # its depth composition can vary and is not used for the main estimate.
            all_supported = fractions.apply(pd.to_numeric, errors="coerce").where(
                valid_hours >= MIN_VALID_HOURS_PER_DEPTH_DAY
            )
            daily_variable_support = all_supported.mean(axis=1, skipna=True)
            for date in dates:
                date_rows.append({
                    "date_utc": pd.Timestamp(date),
                    "D50_scenario_mm_not_measured": d50,
                    "forcing_case": forcing_name,
                    "mobility_fraction_fixed_common_depth_support": daily_fixed.loc[date],
                    "mobility_fraction_variable_supported_depths_sensitivity": daily_variable_support.loc[date],
                    "fixed_depth_bins_n": len(common_depths),
                    "variable_supported_depth_bins_n": int(
                        (valid_hours.loc[date] >= MIN_VALID_HOURS_PER_DEPTH_DAY).sum()
                    ),
                })
    return pd.DataFrame(date_rows), pd.DataFrame(common_support_rows)


def main() -> None:
    for path in (FORCING_FILE, PROCESSED_SUPPORT_FILE, MOBILITY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    scenes = pd.read_csv(FORCING_FILE)
    require_columns(
        scenes,
        {"product_name", "acquisition_utc", "cloud_cover_product_pct", "ibi_vhm0_m"},
        str(FORCING_FILE),
    )
    scenes["date_utc"] = pd.to_datetime(scenes["acquisition_utc"], utc=True, errors="coerce").dt.date
    scenes["cloud_cover_product_pct"] = pd.to_numeric(scenes["cloud_cover_product_pct"], errors="coerce")
    scenes["ibi_vhm0_m"] = pd.to_numeric(scenes["ibi_vhm0_m"], errors="coerce")
    scenes = scenes.dropna(subset=["date_utc"]).drop_duplicates("product_name").copy()

    processed = pd.read_csv(PROCESSED_SUPPORT_FILE)
    require_columns(processed, {"product_name", "acquisition_utc"}, str(PROCESSED_SUPPORT_FILE))
    processed_names = set(processed["product_name"].dropna().astype(str))
    daily, support = load_mobility()
    daily["date_key"] = pd.to_datetime(daily["date_utc"]).dt.date
    catalog_products_total = int(scenes["product_name"].nunique())
    mobility_dates = set(daily["date_key"])
    out_of_period_products = int((~scenes["date_utc"].isin(mobility_dates)).sum())
    scenes = scenes[scenes["date_utc"].isin(mobility_dates)].copy()
    if scenes.empty:
        raise ValueError("No Sentinel-2 catalog acquisition dates overlap the mobility forcing period.")
    scenes["processed_roi_qa"] = scenes["product_name"].astype(str).isin(processed_names)
    OUT.mkdir(parents=True, exist_ok=True)
    support.to_csv(OUT / "fixed_depth_support.csv", index=False, float_format="%.6g", encoding="utf-8-sig")

    # Product/date-level audit keeps both archive products and the 11 image-QA scenes.
    date_scene = scenes.groupby("date_utc", as_index=False).agg(
        product_count=("product_name", "nunique"),
        cloud_cover_product_pct=("cloud_cover_product_pct", "mean"),
        ibi_vhm0_m=("ibi_vhm0_m", "mean"),
        processed_roi_qa=("processed_roi_qa", "any"),
    )
    date_scene.to_csv(OUT / "sentinel2_date_sampling_audit.csv", index=False, encoding="utf-8-sig")

    # Each group is represented by unique acquisition dates, so multiple products
    # on one day cannot inflate the effective temporal sample size.
    groups: dict[str, pd.DataFrame] = {
        "All Sentinel-2 catalog dates": scenes.copy(),
        "Processed ROI-QA scenes": scenes[scenes["processed_roi_qa"]].copy(),
    }
    for threshold in CLOUD_THRESHOLDS:
        groups[f"Cloud cover <= {threshold}%"] = scenes[
            scenes["cloud_cover_product_pct"].notna()
            & (scenes["cloud_cover_product_pct"] <= threshold)
        ].copy()

    all_mobility = daily.copy()
    all_mobility["group"] = "All-weather modeled period"
    scene_groups = []
    for label, group in groups.items():
        day = group.groupby("date_utc", as_index=False).agg(
            cloud_cover_product_pct=("cloud_cover_product_pct", "mean"),
            ibi_vhm0_m=("ibi_vhm0_m", "mean"),
            product_count=("product_name", "nunique"),
        )
        day["group"] = label
        day["scene_count_in_group"] = int(group["product_name"].nunique())
        scene_groups.append(day)
    sampled_dates = pd.concat(scene_groups, ignore_index=True) if scene_groups else pd.DataFrame()
    sampled_dates["date_key"] = pd.to_datetime(sampled_dates["date_utc"]).dt.date

    # Summaries are based on unique dates and the exact same depth bins within each
    # D50/forcing scenario. No significance test or causal interpretation is made.
    records: list[dict[str, object]] = []
    baseline = all_mobility.groupby(
        ["D50_scenario_mm_not_measured", "forcing_case"], as_index=False
    )["mobility_fraction_fixed_common_depth_support"].mean().rename(columns={
        "mobility_fraction_fixed_common_depth_support": "all_weather_baseline"
    })
    group_order = [
        "All Sentinel-2 catalog dates", "Cloud cover <= 50%",
        "Cloud cover <= 20%", "Cloud cover <= 10%", "Processed ROI-QA scenes",
    ]
    for label in group_order:
        subset_scenes = groups[label]
        date_set = set(pd.to_datetime(subset_scenes["date_utc"]).dt.date)
        sampled_month_counts = pd.Series(
            pd.to_datetime(sorted(date_set)).month
        ).value_counts().sort_index()
        for (d50, forcing), scenario in daily.groupby(
            ["D50_scenario_mm_not_measured", "forcing_case"], sort=True
        ):
            values = scenario[scenario["date_key"].isin(date_set)]
            eligible = values["mobility_fraction_fixed_common_depth_support"].dropna()
            scene_cloud = pd.to_numeric(subset_scenes["cloud_cover_product_pct"], errors="coerce")
            scene_hs = pd.to_numeric(subset_scenes["ibi_vhm0_m"], errors="coerce")
            base_value = float(baseline.loc[
                np.isclose(baseline["D50_scenario_mm_not_measured"], d50)
                & baseline["forcing_case"].eq(forcing), "all_weather_baseline"
            ].iloc[0])
            mean_value = float(eligible.mean()) if len(eligible) else np.nan
            monthly_reference = scenario.copy()
            monthly_reference["month_number"] = pd.to_datetime(monthly_reference["date_utc"]).dt.month
            month_means = monthly_reference.groupby("month_number")[
                "mobility_fraction_fixed_common_depth_support"
            ].mean()
            month_weights = sampled_month_counts / sampled_month_counts.sum()
            matched_months = month_weights.index.intersection(month_means.dropna().index)
            matched_weight_sum = float(month_weights.loc[matched_months].sum())
            month_matched_baseline = (
                float((month_means.loc[matched_months] * month_weights.loc[matched_months]).sum() / matched_weight_sum)
                if matched_weight_sum > 0 else np.nan
            )
            records.append({
                "group": label,
                "group_code": GROUP_LABELS[label],
                "D50_scenario_mm_not_measured": float(d50),
                "forcing_case": forcing,
                "scene_count_n": int(subset_scenes["product_name"].nunique()),
                "unique_acquisition_dates_n": int(len(date_set)),
                "dates_with_mobility_support_n": int(len(eligible)),
                "mean_product_tile_cloud_pct": float(scene_cloud.mean()) if scene_cloud.notna().any() else np.nan,
                "median_product_tile_cloud_pct": float(scene_cloud.median()) if scene_cloud.notna().any() else np.nan,
                "mean_overpass_Hs_m": float(scene_hs.mean()) if scene_hs.notna().any() else np.nan,
                "median_overpass_Hs_m": float(scene_hs.median()) if scene_hs.notna().any() else np.nan,
                "mean_fixed_depth_mobility_exceedance_fraction": mean_value,
                "all_weather_182d_baseline_fraction": base_value,
                "difference_from_all_weather_percentage_points": (mean_value - base_value) * 100 if np.isfinite(mean_value) else np.nan,
                "month_matched_all_weather_baseline_fraction": month_matched_baseline,
                "season_composition_difference_percentage_points": (month_matched_baseline - base_value) * 100 if np.isfinite(month_matched_baseline) else np.nan,
                "within_month_sampling_difference_percentage_points": (mean_value - month_matched_baseline) * 100 if np.isfinite(mean_value) and np.isfinite(month_matched_baseline) else np.nan,
                "sampled_months": ";".join(str(int(m)) for m in sampled_month_counts.index),
                "depth_support": "fixed depth bins with >=12 valid hours on every day in the modeled period",
                "interpretation": "descriptive, model-conditional contrast; month-standardization separates month composition from within-month sampling; not SSC bias, probability, or causal effect",
            })
    summary = pd.DataFrame(records)
    summary_path = OUT / "scene_sampling_representativeness_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.7g", encoding="utf-8-sig")

    # Long date-level file for independent inspection and reuse.
    daily_merge = daily.merge(
        sampled_dates[["date_key", "group", "cloud_cover_product_pct", "ibi_vhm0_m"]],
        on="date_key", how="left", validate="many_to_many",
    )
    daily_merge.to_csv(OUT / "mobility_by_sampling_group_date.csv.gz", index=False, compression="gzip", float_format="%.7g")

    make_figure(scenes, summary, OUT)
    make_seasonality_figure(summary, OUT)
    write_notes(scenes, summary, support, OUT, catalog_products_total, out_of_period_products)
    print("Scene-sampling representativeness audit complete; source observations and model outputs were read only.")
    print(f"Catalog products outside the modeled period excluded: {out_of_period_products} of {catalog_products_total}")
    print(f"Catalog products/dates: {scenes.product_name.nunique()}/{scenes.date_utc.nunique()}")
    print(f"Processed ROI-QA products/dates: {scenes.loc[scenes.processed_roi_qa, 'product_name'].nunique()}/{scenes.loc[scenes.processed_roi_qa, 'date_utc'].nunique()}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {OUT / 'Figure_scene_sampling_representativeness.png'}")
    print(f"Seasonality decomposition: {OUT / 'Figure_processed_scene_month_decomposition.png'}")
    print(f"Interpretation: {OUT / 'interpretation_notes_FR.txt'}")
    print("Product cloud cover is tile-level; results are conditional mobility-scenario diagnostics, not SSC or sediment-flux estimates.")


def make_figure(scenes: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2,
        "axes.titleweight": "bold", "axes.labelcolor": "#243447",
        "text.color": "#243447", "axes.edgecolor": "#9AA8B5",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(7.0, 5.8), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.82, 1.0, 1.0], hspace=0.62, wspace=0.23)
    ax_scatter = fig.add_subplot(grid[0, :])
    valid = scenes.dropna(subset=["cloud_cover_product_pct", "ibi_vhm0_m"])
    regular = valid[~valid["processed_roi_qa"]]
    qa = valid[valid["processed_roi_qa"]]
    ax_scatter.scatter(
        regular["cloud_cover_product_pct"], regular["ibi_vhm0_m"],
        s=43, alpha=0.72, color="#247BA0", edgecolor="white", linewidth=0.5,
        label="Catalog scenes outside the processed subset", zorder=2,
    )
    ax_scatter.scatter(
        qa["cloud_cover_product_pct"], qa["ibi_vhm0_m"],
        s=76, alpha=0.96, color="#E76F51", marker="D", edgecolor="white", linewidth=0.7,
        label="11 scenes with image-based ROI QA", zorder=3,
    )
    for threshold in CLOUD_THRESHOLDS:
        ax_scatter.axvline(threshold, color="#9AA8B5", lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_xlabel("Sentinel-2 product tile cloud cover (%)")
    ax_scatter.set_ylabel("Hs at overpass (m)")
    ax_scatter.set_title("A  Optical acquisition sample and modeled wave conditions", loc="left", pad=9)
    ax_scatter.grid(color="#E8EDF1", lw=0.65)
    ax_scatter.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.6)
    ax_scatter.text(
        0.01, 0.02, "Tile-level cloud metadata; the 11-scene subset is the set with processed image QA.",
        transform=ax_scatter.transAxes, color="#5F6B76", fontsize=7.1,
    )

    panels = [
        (0.10, "wave_only", "B  0.10 mm · waves"),
        (0.10, "waves_plus_daily_currents", "C  0.10 mm · waves + currents"),
        (1.00, "wave_only", "D  1.00 mm · waves"),
        (1.00, "waves_plus_daily_currents", "E  1.00 mm · waves + currents"),
    ]
    display_groups = [
        "All Sentinel-2 catalog dates", "Cloud cover <= 50%",
        "Cloud cover <= 20%", "Cloud cover <= 10%", "Processed ROI-QA scenes",
    ]
    for i, (d50, forcing, title) in enumerate(panels):
        row, col = 1 + i // 2, i % 2
        ax = fig.add_subplot(grid[row, col])
        subset = summary[
            np.isclose(summary["D50_scenario_mm_not_measured"], d50)
            & summary["forcing_case"].eq(forcing)
        ].set_index("group").reindex(display_groups)
        x = subset["difference_from_all_weather_percentage_points"].to_numpy(float)
        y = np.arange(len(display_groups))
        colors = [GROUP_COLORS[group] for group in display_groups]
        ax.barh(y, x, color=colors, height=0.66, edgecolor="white", linewidth=0.7)
        ax.axvline(0, color="#273746", lw=0.9)
        for yi, value in zip(y, x):
            if np.isfinite(value):
                ax.text(value + (0.55 if value >= 0 else -0.55), yi, f"{value:+.1f}",
                        va="center", ha="left" if value >= 0 else "right", fontsize=8.0, color="#263442")
        ax.set_yticks(y, labels=(
            ["All scenes", "Cloud ≤50%", "Cloud ≤20%", "Cloud ≤10%", "Processed 11"]
            if col == 0 else ["", "", "", "", ""]
        ))
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=8.7, pad=5)
        ax.set_xlabel("Δ vs baseline (pp)", fontsize=8.0, labelpad=3)
        ax.tick_params(axis="x", labelsize=7.7)
        ax.tick_params(axis="y", labelsize=7.7, length=0)
        ax.grid(axis="x", color="#E8EDF1", lw=0.6)
        ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.975, bottom=0.12, left=0.15, right=0.985)
    fig.savefig(out / "Figure_scene_sampling_representativeness.png", dpi=350, bbox_inches="tight")
    fig.savefig(out / "Figure_scene_sampling_representativeness.pdf", bbox_inches="tight")
    plt.close(fig)
    (out / "figure_caption_EN.txt").write_text(
        "Figure. (A) Sentinel-2 product tile cloud cover and regional significant wave height at acquisition; diamonds mark the 11 scenes with image-based ROI quality assessment. Vertical guides show the cloud thresholds used to define subsets. (B–E) Difference in the mean depth-equal modeled incipient-motion exceedance fraction between each satellite-date subset and the full 182-day model record, using only depth bins with at least 12 valid forcing hours on every day. Panels compare 0.10 mm literature-informed but unmeasured and 1.00 mm sensitivity scenarios under wave-only and wave plus daily-current forcing. Product cloud cover is tile-level; mobility is model-conditional and is not observed suspended-sediment concentration, sediment flux, or a causal effect.\n",
        encoding="utf-8",
    )


def make_seasonality_figure(summary: pd.DataFrame, out: Path) -> None:
    """Separate month composition from scene selection within each month."""
    from matplotlib.patches import Patch

    subset = summary[summary["group"].eq("Processed ROI-QA scenes")]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharex=True, facecolor="white")
    season_color, within_color = "#A8B5C2", "#D66B4D"
    for ax, d50, title in zip(
        axes, (0.10, 1.00),
        ("A  D50 = 0.10 mm scenario", "B  D50 = 1.00 mm sensitivity"),
    ):
        cases = subset[np.isclose(subset["D50_scenario_mm_not_measured"], d50)].set_index("forcing_case")
        y = np.arange(2)
        seasonal = np.array([
            cases.loc["wave_only", "season_composition_difference_percentage_points"],
            cases.loc["waves_plus_daily_currents", "season_composition_difference_percentage_points"],
        ], dtype=float)
        within = np.array([
            cases.loc["wave_only", "within_month_sampling_difference_percentage_points"],
            cases.loc["waves_plus_daily_currents", "within_month_sampling_difference_percentage_points"],
        ], dtype=float)
        total = seasonal + within
        ax.barh(y, seasonal, color=season_color, height=0.56, edgecolor="white", linewidth=0.6)
        ax.barh(y, within, left=seasonal, color=within_color, height=0.56, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="#243447", lw=0.9, ls="--")
        for yi, value in zip(y, total):
            if np.isfinite(value):
                ax.scatter(value, yi, s=34, color="#243447", zorder=4)
                offset = 0.4 if value >= 0 else -0.4
                ax.text(value + offset, yi, f"total {value:+.1f} pp", va="center",
                        ha="left" if value >= 0 else "right", fontsize=8.0, color="#243447")
        ax.set_yticks(y, ["Wave only", "Waves + daily currents"])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=9, fontsize=10, fontweight="bold")
        ax.set_xlabel("Difference from 182-day baseline (percentage points)")
        ax.grid(axis="x", color="#E8EDF1", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[Patch(facecolor=season_color, label="Month composition"),
                 Patch(facecolor=within_color, label="Within-month sampling")],
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.98),
    )
    fig.subplots_adjust(top=0.95)
    fig.text(
        0.5, 0.015,
        "Month composition weights each month's all-weather mean by the share of selected dates in that month. "
        "Within-month sampling compares selected dates with all modeled days in those same months.",
        ha="center", va="bottom", fontsize=8.0, color="#596775", wrap=True,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.12, right=0.98, wspace=0.30)
    fig.savefig(out / "Figure_processed_scene_month_decomposition.png", dpi=350, bbox_inches="tight")
    fig.savefig(out / "Figure_processed_scene_month_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    (out / "figure_month_decomposition_caption_EN.txt").write_text(
        "Figure. Decomposition of the processed 11-scene subset's modeled mobility-exceedance difference from the full 182-day reference. The gray component represents the effect of the subset's month composition, calculated by weighting each month's all-weather mean by the number of selected acquisition dates in that month. The orange component is the remaining within-month difference between selected scene dates and all-weather days in the same months. Points mark the total difference. Panels show the 0.10 mm literature-informed but unmeasured scenario and the 1.00 mm sensitivity under wave-only and wave plus daily regional-current forcing. This decomposition controls descriptively for month composition but does not establish a cloud-caused bias, observed SSC bias, or sediment flux.\n",
        encoding="utf-8",
    )


def write_notes(
    scenes: pd.DataFrame, summary: pd.DataFrame, support: pd.DataFrame, out: Path,
    catalog_products_total: int, out_of_period_products: int,
) -> None:
    processed = scenes[scenes["processed_roi_qa"]]
    all_n = scenes["product_name"].nunique()
    processed_n = processed["product_name"].nunique()
    mean_cloud = scenes["cloud_cover_product_pct"].mean()
    mean_hs = scenes["ibi_vhm0_m"].mean()
    processed_hs = processed["ibi_vhm0_m"].mean()
    lines = [
        "AUDIT DE REPRÉSENTATIVITÉ TEMPORELLE DES SCÈNES SENTINEL-2",
        "",
        f"Le catalogue contient {all_n} produits sur {scenes['date_utc'].nunique()} dates uniques. "
        f"Le sous-ensemble traité avec contrôle image dans la zone d'étude contient {processed_n} produits sur "
        f"{processed['date_utc'].nunique()} dates.",
        f"{out_of_period_products} produit(s) hors de la période de forçage ont été exclus du test de représentativité "
        f"({catalog_products_total} produit(s) dans le fichier d'entrée avant filtrage).",
        f"La couverture nuageuse moyenne indiquée au niveau du produit/tuile est {mean_cloud:.1f} %. "
        f"La hauteur significative moyenne de houle modélisée aux passages est {mean_hs:.2f} m pour le catalogue "
        f"et {processed_hs:.2f} m pour le sous-ensemble traité.",
        "",
        "Les résultats de mobilité sont calculés à profondeur fixe : seules les classes ayant au moins 12 heures "
        "de forçage valide pour chaque jour des 182 jours entrent dans la comparaison principale. Cette règle "
        "évite que les groupes de dates reçoivent des poids différents parce que leur composition en profondeur varie.",
        "",
        "DÉCOMPOSITION DE L'ÉCART POUR LES 11 SCÈNES TRAITÉES (points de pourcentage)",
        "Écart total = composition des mois échantillonnés + différence de sélection à l'intérieur de ces mois. "
        "La référence mensuelle est la moyenne de tous les jours modélisés du mois, pondérée par le nombre de dates "
        "satellitaires retenues dans ce mois.",
    ]
    for d50 in (0.10, 1.00):
        for forcing in FORCING_CASES:
            row = summary[
                np.isclose(summary["D50_scenario_mm_not_measured"], d50)
                & summary["forcing_case"].eq(forcing)
                & summary["group"].eq("Processed ROI-QA scenes")
            ]
            if not row.empty:
                item = row.iloc[0]
                lines.append(
                    f"D50 {d50:.2f} mm, {forcing} : total {item['difference_from_all_weather_percentage_points']:+.2f}; "
                    f"composition mensuelle {item['season_composition_difference_percentage_points']:+.2f}; "
                    f"différence intra-mensuelle {item['within_month_sampling_difference_percentage_points']:+.2f}; "
                    f"{int(item['dates_with_mobility_support_n'])} dates appariées sur "
                    f"{int(item['unique_acquisition_dates_n'])}."
                )
    lines.extend([
        "",
        "INTERPRÉTATION SCIENTIFIQUE",
        "Les différences en points de pourcentage indiquent si les dates d'acquisition disponibles échantillonnent "
        "un niveau moyen différent de mobilité modélisée par rapport à l'ensemble de la période de forçage. "
        "Une différence positive suggère un sur-échantillonnage de conditions où le modèle dépasse plus souvent "
        "le seuil de mise en mouvement; une différence négative indique l'inverse. La composante intra-mensuelle "
        "isole descriptivement l'écart après contrôle de la distribution des dates par mois. Aucune de ces composantes "
        "ne mesure un biais réel de concentration en sédiments en suspension.",
        "",
        "LIMITES À GARDER DANS LE MANUSCRIT",
        "1. La couverture nuageuse du catalogue est une métadonnée de produit/tuile, pas une fraction nuageuse validée sur l'AOI marine.",
        "2. Le sous-ensemble traité est petit et sélectionné pour l'analyse; cette comparaison décrit sa représentativité, sans inférence causale.",
        "3. Aucun capteur continu local de SSC/turbidité n'est disponible ici pour calculer un biais d'observation réel.",
        "4. La fréquence de dépassement dépend des hypothèses du modèle de mobilité, de la taille de grain non mesurée, du forçage régional et du support bathymétrique.",
        "5. Les classes granulométriques sont des scénarios; la valeur historique de 0.10 mm pour le fond d'Agadir est un contexte régional, pas une mesure du levé Anza 2022.",
        "6. Les résultats ne représentent ni un flux sédimentaire ni une prédiction universelle pour d'autres côtes.",
        "",
        "SOURCES ET POSITIONNEMENT DE LA LACUNE",
        "Lin et al. (2026) montrent que la disponibilité optique non aléatoire peut biaiser les estimations côtières de SSC, "
        "mais leur quantification du biais s'appuie sur des séries in situ fréquentes de turbidité et de houle. "
        "Notre jeu de données ne permet pas cette validation directe; il permet un test plus limité de la représentativité "
        "des dates optiques vis-à-vis d'un indicateur de mobilité conditionnel au modèle.",
        "Nmiss et al. (2025) ont mesuré des flux par pièges à sable sur cinq plages au sud d'Agadir. Leur étude rappelle "
        "que des mesures locales de flux existent dans la région et qu'elles ne doivent pas être confondues avec notre "
        "diagnostic satellite/modèle, réalisé à Anza et sans mesure directe de flux.",
        "",
        f"Classes de profondeur retenues (table fixed_depth_support.csv) : {int(support['common_depth_bins_n'].min())} à "
        f"{int(support['common_depth_bins_n'].max())} selon scénario; limites verticales communes de "
        f"{support['common_depth_min_m'].min():.1f}–{support['common_depth_max_m'].max():.1f} m.",
        "",
        "Références cliquables :",
        "- Lin, H., Fichot, C. G., Wang, Y., Yu, Q., Wang, A., & Gao, S. (2026). Global Fair-Weather Bias in Remotely Sensed Coastal Suspended Sediment Concentration. Geophysical Research Letters, 53, e2026GL122332. https://doi.org/10.1029/2026GL122332",
        "- Nmiss, M., Amyay, M., Atiki, N., Benbih, M., & Nait-Si, H. (2025). Quantifying sediment flux under moderate wave energy on beaches south of Agadir, Morocco. Ecological Engineering & Environmental Technology, 26(1), 78–91. https://doi.org/10.12912/27197050/195526",
    ])
    (out / "interpretation_notes_FR.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
