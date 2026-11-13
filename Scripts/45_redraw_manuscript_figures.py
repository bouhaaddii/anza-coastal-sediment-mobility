"""Redraw main manuscript figures 3, 4 and 6 from saved analysis tables.

This presentation-only script does not refit models or alter source data.
Legends are placed in dedicated whitespace so they do not cover plotted data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
OUT = RESULTS / "Manuscript_Figures"
MOBILITY = RESULTS / "Sediment_Mobility_Scenarios" / "mobility_by_depth_grain_datum_scenario.csv"
SCENES = RESULTS / "scene_forcing_and_quality.csv"
ROI_QA = RESULTS / "Sediment_Proxy" / "Spatial_Support" / "ndti_spatial_support_by_scene.csv"
SAMPLING = RESULTS / "Additional_Analyses" / "Scene_Sampling_Representativeness" / "scene_sampling_representativeness_summary.csv"
CONTINUOUS = RESULTS / "Additional_Analyses" / "Scene_Sampling_Representativeness" / "continuous_mobility_sampling_sensitivity.csv"

NAVY = "#17324D"
BLUE = "#247BA0"
ORANGE = "#E76F51"
PALETTE = ["#482878", "#3E4989", "#26828E", "#1F9E89", "#6DCD59", "#B4DE2C"]


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure3() -> None:
    data = pd.read_csv(MOBILITY)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True, sharey=True, facecolor="white")
    panels = [
        (axes[0, 0], "ZH_unadjusted", "wave_only_exceedance_fraction", "A  Waves only · local port zero"),
        (axes[0, 1], "ZH_unadjusted", "combined_exceedance_fraction", "B  Waves + daily regional currents · local zero"),
        (axes[1, 0], "IHO_2024_candidate_offset_sensitivity", "wave_only_exceedance_fraction", "C  Waves only · unverified offset scenario"),
        (axes[1, 1], "IHO_2024_candidate_offset_sensitivity", "combined_exceedance_fraction", "D  Waves + currents · unverified offset scenario"),
    ]
    for ax, datum, metric, title in panels:
        subset = data[data["datum_case"].eq(datum)]
        for color, d50 in zip(PALETTE, sorted(subset["D50_scenario_mm_not_measured"].unique())):
            part = subset[np.isclose(subset["D50_scenario_mm_not_measured"], d50)].sort_values(
                "survey_depth_bin_below_local_ZH_m"
            )
            ax.plot(part[metric], part["survey_depth_bin_below_local_ZH_m"],
                    marker="o", ms=2.1, lw=1.35, color=color, label=f"{d50:g}")
        ax.set_title(title, loc="left", fontsize=9.1, color=NAVY, pad=8, fontweight="bold")
        ax.grid(True, color="#D9E2E8", linewidth=0.55)
        ax.set_xlim(0.45, 1.0)
        ax.set_xticks(np.arange(0.5, 1.01, 0.1))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    fig.supylabel("Survey depth below local hydrographic zero (m; positive downward)", x=0.035, fontsize=9)
    fig.supxlabel("Fraction of valid hourly records with modeled M ≥ 1", y=0.08, fontsize=9)
    handles = [Line2D([0], [0], color=color, marker="o", lw=1.4, ms=4, label=f"{d:g} mm")
               for color, d in zip(PALETTE, sorted(data["D50_scenario_mm_not_measured"].unique()))]
    fig.legend(handles=handles, title="Assumed D₅₀ scenarios (not measured locally)",
               loc="upper center", bbox_to_anchor=(0.58, 0.995), ncol=3,
               frameon=False, fontsize=8, title_fontsize=8)
    fig.text(0.56, 0.025,
             "Valid-hour support varies with depth. The +2.159 m case is unverified sensitivity only; curves are not transport rates.",
             ha="center", fontsize=7.7, color="#536471")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.83, bottom=0.15, hspace=0.30, wspace=0.13)
    save(fig, "Figure_3_depth_grain_sensitivity")


def figure4() -> None:
    scenes = pd.read_csv(SCENES)
    qa_names = set(pd.read_csv(ROI_QA)["product_name"].astype(str))
    scenes["processed_roi_qa"] = scenes["product_name"].astype(str).isin(qa_names)
    scenes["cloud_cover_product_pct"] = pd.to_numeric(scenes["cloud_cover_product_pct"], errors="coerce")
    scenes["ibi_vhm0_m"] = pd.to_numeric(scenes["ibi_vhm0_m"], errors="coerce")
    summary = pd.read_csv(SAMPLING)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.7,
                         "axes.titleweight": "bold", "axes.labelcolor": "#243447",
                         "text.color": "#243447", "axes.edgecolor": "#9AA8B5",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.6, 6.35), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.0, 1.0], hspace=0.62, wspace=0.25)
    ax = fig.add_subplot(grid[0, :])
    valid = scenes.dropna(subset=["cloud_cover_product_pct", "ibi_vhm0_m"])
    for selected, label, color, marker, size in [
        (False, "Other catalog dates (n=62)", BLUE, "o", 36),
        (True, "Image-QA scenes (n=11)", ORANGE, "D", 58),
    ]:
        rows = valid[valid["processed_roi_qa"].eq(selected)]
        ax.scatter(rows["cloud_cover_product_pct"], rows["ibi_vhm0_m"], s=size, alpha=0.86,
                   color=color, marker=marker, edgecolor="white", linewidth=0.55, label=label, zorder=3)
    for threshold in (10, 20, 50):
        ax.axvline(threshold, color="#9AA8B5", lw=0.75, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(left=0)
    ax.set_xlabel("Sentinel-2 product-tile cloud cover (%)")
    ax.set_ylabel("Modeled Hₛ at overpass (m)")
    ax.set_title("A  Optical acquisition sample and regional wave conditions", loc="left", pad=7, color=NAVY)
    ax.grid(color="#E8EDF1", lw=0.6)
    ax.set_axisbelow(True)
    panels = [
        (0.10, "wave_only", "B  D₅₀ 0.10 mm · waves"),
        (0.10, "waves_plus_daily_currents", "C  D₅₀ 0.10 mm · waves + currents"),
        (1.00, "wave_only", "D  D₅₀ 1.00 mm · waves"),
        (1.00, "waves_plus_daily_currents", "E  D₅₀ 1.00 mm · waves + currents"),
    ]
    groups = ["All Sentinel-2 catalog dates", "Cloud cover <= 50%", "Cloud cover <= 20%",
              "Cloud cover <= 10%", "Processed ROI-QA scenes"]
    ylabels = ["All dates", "Cloud ≤50%", "Cloud ≤20%", "Cloud ≤10%", "Image-QA 11"]
    for i, (d50, forcing, title) in enumerate(panels):
        row, col = 1 + i // 2, i % 2
        panel_ax = fig.add_subplot(grid[row, col])
        part = summary[np.isclose(summary["D50_scenario_mm_not_measured"], d50)
                       & summary["forcing_case"].eq(forcing)].set_index("group").reindex(groups)
        values = pd.to_numeric(part["difference_from_all_weather_percentage_points"], errors="coerce").to_numpy()
        colors = ["#247BA0", "#66A182", "#E3A008", "#E76F51", "#7A63A8"]
        panel_ax.barh(np.arange(len(groups)), values, color=colors, height=0.62,
                      edgecolor="white", linewidth=0.6)
        panel_ax.axvline(0, color="#273746", lw=0.85)
        for yi, value in enumerate(values):
            if np.isfinite(value):
                panel_ax.annotate(f"{value:+.1f}", (value, yi), xytext=(4 if value >= 0 else -4, 0),
                                  textcoords="offset points", va="center",
                                  ha="left" if value >= 0 else "right", fontsize=7.3, color="#263442")
        panel_ax.set_yticks(np.arange(len(groups)), ylabels if col == 0 else ["" for _ in groups])
        panel_ax.invert_yaxis()
        panel_ax.set_title(title, loc="left", fontsize=8.0, pad=5, color=NAVY)
        panel_ax.set_xlabel("Difference from 182-day baseline (percentage points)", fontsize=7.2, labelpad=2)
        panel_ax.tick_params(axis="x", labelsize=7.0)
        panel_ax.tick_params(axis="y", labelsize=7.0, length=0)
        panel_ax.grid(axis="x", color="#E8EDF1", lw=0.55)
        panel_ax.set_axisbelow(True)
    handles = [Line2D([], [], color=BLUE, marker="o", lw=0, ms=5, label="Other catalog dates (n=62)"),
               Line2D([], [], color=ORANGE, marker="D", lw=0, ms=5, label="Image-QA scenes (n=11)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 0.995),
               frameon=False, ncol=2, fontsize=7.5)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.88, bottom=0.10)
    save(fig, "Figure_4_scene_sampling")


def figure6() -> None:
    data = pd.read_csv(CONTINUOUS)
    groups = ["All Sentinel-2 catalog dates", "Cloud cover <= 50%", "Cloud cover <= 20%",
              "Cloud cover <= 10%", "Processed ROI-QA scenes"]
    labels = {groups[0]: "Catalog", groups[1]: "Cloud ≤50%", groups[2]: "Cloud ≤20%",
              groups[3]: "Cloud ≤10%", groups[4]: "ROI-QA 11"}
    panels = [(0.10, "wave_only", "A  D₅₀ 0.10 mm · waves"),
              (0.10, "waves_plus_daily_currents", "B  D₅₀ 0.10 mm · waves + currents"),
              (0.25, "wave_only", "C  D₅₀ 0.25 mm · waves"),
              (0.25, "waves_plus_daily_currents", "D  D₅₀ 0.25 mm · waves + currents"),
              (1.00, "wave_only", "E  D₅₀ 1.00 mm · waves"),
              (1.00, "waves_plus_daily_currents", "F  D₅₀ 1.00 mm · waves + currents")]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.5), sharex=True, sharey=True, facecolor="white")
    y = np.arange(len(groups))
    for ax, (d50, forcing, title) in zip(axes.flat, panels):
        part = data[np.isclose(data["D50_scenario_mm_not_measured"], d50)
                    & data["forcing_case"].eq(forcing)].set_index("group").reindex(groups)
        ax.axvline(0, color="#4C5964", lw=0.8, ls=(0, (3, 2)), zorder=0)
        total = pd.to_numeric(part["total_difference_log10_units"], errors="coerce").to_numpy()
        within = pd.to_numeric(part["within_month_difference_log10_units"], errors="coerce").to_numpy()
        ax.scatter(total, y - 0.11, s=31, color=BLUE, edgecolor="white", linewidth=0.5, label="Total difference", zorder=3)
        ax.scatter(within, y + 0.11, s=31, marker="D", color=ORANGE, edgecolor="white", linewidth=0.5, label="Within-month difference", zorder=3)
        ax.set_title(title, loc="left", fontsize=8.5, pad=7, color=NAVY)
        ax.grid(axis="x", color="#E8EDF1", lw=0.6)
        ax.set_yticks(y, [labels[g] for g in groups])
        ax.tick_params(axis="y", labelsize=7.4)
        if ax in axes[1]:
            ax.set_xlabel("Difference in mean log₁₀(P₉₅ M) from all-weather reference", fontsize=7.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Sentinel-2 sampling set", fontsize=8)
    axes[1, 0].set_ylabel("Sentinel-2 sampling set", fontsize=8)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.060), fontsize=8)
    fig.text(0.5, 0.018,
             "Daily P₉₅(M) is a scenario metric; D₅₀ is unmeasured and regional forcing does not resolve the surf zone.",
             ha="center", fontsize=7.7, color="#5F6B76")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.16, wspace=0.22, hspace=0.28)
    save(fig, "Figure_6_continuous_sampling_sensitivity")


def write_captions() -> None:
    text = """Figure 3. Frequency of modeled incipient-motion-threshold exceedance (M ≥ 1) by survey depth and assumed D50. Panels compare wave-only with wave plus daily regional currents, each under the unadjusted local hydrographic zero and the unverified +2.159 m datum-offset sensitivity. Grain sizes are scenarios, not local measurements. The horizontal axis begins at 0.45 to resolve differences among high frequencies. Curves are conditional frequencies; the axes do not show confidence intervals, sediment flux, or transport rates.

Figure 4. (A) Sentinel-2 product-tile cloud cover and regional significant wave height at acquisition; diamonds indicate the 11 scenes with image-based ROI quality assessment. (B–E) Difference in the mean depth-equal modeled M ≥ 1 frequency between acquisition-date groups and the full 182-day record, for two unmeasured D50 scenarios and two forcing configurations. Cloud cover is tile metadata; modeled exceedance is not observed SSC, flux, or a causal cloud effect.

Figure 6. Differences in the continuous model-based mobility-intensity score between Sentinel-2 sampling sets and the full 182-day forcing reference. Circles show the total difference; diamonds show the within-month component after standardising the reference to the months represented by each date set. Scores are the mean across fixed depth classes of daily log10(P95(M)). Grain size is unmeasured and the regional forcing does not resolve the surf zone; values are not sediment flux or observed mobility.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Figure_3_4_6_captions_EN.txt").write_text(text, encoding="utf-8")


def main() -> None:
    missing = [p for p in (MOBILITY, SCENES, ROI_QA, SAMPLING, CONTINUOUS) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required saved analysis tables are missing:\n" + "\n".join(map(str, missing)))
    figure3()
    figure4()
    figure6()
    write_captions()
    print(f"Redrew Figures 3, 4 and 6 from saved tables in {OUT}; no model fitting was repeated.")


if __name__ == "__main__":
    main()
