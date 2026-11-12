from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
OUT = R / "ECSS_Submission_Package" / "Supporting_Data"
FIG = R / "Additional_Analyses" / "ECSS_Supplementary_Figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sampling = R / "Additional_Analyses" / "Scene_Sampling_Representativeness"
mobility = R / "Sediment_Mobility_Scenarios"
wave = R / "Additional_Analyses" / "Within_Cell_Wave_Response"
cmems = R / "CMEMS_Ocean_Colour_Audit"

dates = pd.read_csv(sampling / "sentinel2_date_sampling_audit.csv", parse_dates=["date_utc"])
daily = pd.read_csv(mobility / "mobility_daily_by_depth_grain_datum_scenario.csv.gz", parse_dates=["date_utc"])
sel = daily[(daily.datum_case == "ZH_unadjusted")
            & (daily.D50_scenario_mm_not_measured.round(3) == 0.250)
            & daily.survey_depth_bin_below_local_ZH_m.between(3.5, 23.0)
            & (daily.wave_valid_hours >= 12)].copy()
sel = sel[np.isfinite(sel.wave_M_p95) & (sel.wave_M_p95 > 0)]
sel["log10_p95_M"] = np.log10(sel.wave_M_p95)
score = sel.groupby("date_utc", as_index=False).agg(
    mean_log10_p95_M=("log10_p95_M", "mean"),
    supported_depth_bins=("survey_depth_bin_below_local_ZH_m", "nunique"))
score = score[score.supported_depth_bins == 40]
if score.empty:
    raise RuntimeError("No daily score met the manuscript's 40-bin fixed-support criterion.")

dates["processed_roi_qa"] = dates.processed_roi_qa.astype(bool)
all_dates = dates.drop_duplicates("date_utc")
qa = all_dates[all_dates.processed_roi_qa]
fig, axes = plt.subplots(2, 1, figsize=(11.3, 7.2), gridspec_kw={"height_ratios": [2.2, 1.0]})
ax = axes[0]
ax.plot(score.date_utc, score.mean_log10_p95_M, color="#263B53", lw=1.5, label="Daily model-conditional score")
ymin, ymax = ax.get_ylim(); yr = ymax - ymin
for d in all_dates.date_utc:
    ax.vlines(d, ymax - 0.07 * yr, ymax - 0.02 * yr, color="#9AA8B5", lw=0.75, alpha=0.8)
ax.scatter(qa.date_utc, np.full(len(qa), ymax - 0.045 * yr), s=45, marker="D",
           facecolor="#D55E00", edgecolor="white", linewidth=0.55, label="11 scenes with ROI image QA", zorder=4)
ax.set_ylim(ymin, ymax + 0.04 * yr)
ax.set_ylabel("Mean log10(P95(M)) across 40 fixed depth bins")
ax.set_title("A  Acquisition dates against the six-month modeled reference", loc="left", fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_xlim(score.date_utc.min(), score.date_utc.max())
ax.grid(axis="y", color="#DCE3E8", lw=0.7); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)

dates["month_period"] = dates.date_utc.dt.to_period("M")
qa = dates[dates.processed_roi_qa]
catalog_counts = dates.drop_duplicates("date_utc").groupby("month_period").size()
qa_counts = qa.drop_duplicates("date_utc").groupby("month_period").size()
periods = pd.period_range(dates.date_utc.min().to_period("M"), dates.date_utc.max().to_period("M"), freq="M")
x = np.arange(len(periods)); w = 0.38; ax2 = axes[1]
ax2.bar(x - w/2, [catalog_counts.get(p, 0) for p in periods], w, color="#4C78A8", label="Catalog dates (n=73)")
ax2.bar(x + w/2, [qa_counts.get(p, 0) for p in periods], w, color="#D55E00", label="ROI-QA scenes (n=11)")
ax2.set_xticks(x, [p.strftime("%b %Y") for p in periods]); ax2.set_ylabel("Unique dates")
ax2.set_title("B  Monthly distribution of catalog and ROI-QA dates", loc="left", fontweight="bold")
ax2.grid(axis="y", color="#DCE3E8", lw=0.7); ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
fig.suptitle("Sentinel-2 temporal support and conditional seabed-mobility context at Anza", fontsize=14, fontweight="bold", y=0.99)
fig.text(0.5, 0.01, "Unmeasured D50 = 0.25 mm; unadjusted local hydrographic zero. The score is not observed sediment transport.",
         ha="center", fontsize=8.4, color="#4C5965")
fig.tight_layout(rect=[0.035, 0.045, 0.78, 0.95])
png = FIG / "Figure_S10_scene_calendar_and_mobility_context.png"
pdf = FIG / "Figure_S10_scene_calendar_and_mobility_context.pdf"
fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white"); plt.close(fig)

dates.drop(columns=["month_period"]).to_csv(OUT / "Table_S2_sentinel2_catalog_73_dates.csv", index=False)
pd.read_csv(mobility / "D50_scenario_provenance.csv").to_csv(OUT / "Table_S3_D50_scenario_provenance.csv", index=False)
sens = pd.read_csv(sampling / "continuous_mobility_sampling_sensitivity.csv")
sens[sens.group.str.contains("ROI", case=False, na=False)].to_csv(OUT / "Table_S4_ROI_QA_continuous_sensitivity.csv", index=False)
pd.read_csv(cmems / "cmems_hr_oc_selected_scene_water_support.csv").to_csv(OUT / "Table_S5_HR_OC_selected_scene_support.csv", index=False)
pd.read_csv(sampling / "fixed_depth_support.csv").to_csv(OUT / "Table_S6_fixed_depth_support.csv", index=False)
pd.read_csv(wave / "wave_response_robustness_sensitivity.csv").to_csv(OUT / "Table_S7_wave_response_robustness.csv", index=False)
pd.read_csv(wave / "wave_lag_response_models.csv").to_csv(OUT / "Table_S8_wave_lag_models.csv", index=False)

caption = ("Figure S10. Temporal support of the Sentinel-2 acquisition catalog and the 11 scenes with image-based ROI quality assessment, "
           "shown against the saved daily model-conditional mobility score. Panel A shows the daily mean log10 of the 95th percentile of M "
           "across the 40 fixed depth bins used in the main analysis for the unadjusted local hydrographic zero and the unmeasured D50 = 0.25 mm "
           "scenario. Gray ticks mark unique catalog dates; orange diamonds identify the 11 ROI-QA scenes. Panel B gives monthly counts. "
           "The score is model output, not observed seabed mobility, suspended-sediment concentration, or flux.")
(FIG / "Figure_S10_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
readme = """# Supporting tables for the Anza manuscript\n\nThese CSV files are derived from saved analysis outputs. They do not include the raw survey XYZ or full satellite/model grids. Grain-size values are scenarios, not local measurements. HR-OC is satellite-derived, not independent field validation. The mobility score is conditional on the selected equations, regional forcing products, local hydrographic zero, and support rules.\n\n- Table S2: the 73 unique Sentinel-2 catalog dates, product-level cloud metadata, and ROI-QA flag.\n- Table S3: provenance and limits for six D50 scenarios.\n- Table S4: continuous-score sensitivity results for the 11 ROI-QA dates.\n- Table S5: HR-OC water-support summaries.\n- Table S6: fixed depth-support records.\n- Tables S7-S8: within-cell wave-response robustness and lag-window models; confidence intervals crossing zero indicate imprecise associations.\n\nDo not describe any table as measured sediment flux.\n"""
(OUT / "README_supporting_tables_EN.md").write_text(readme, encoding="utf-8")
meta = {"catalog_dates": int(all_dates.shape[0]), "roi_qa_dates": int(qa.drop_duplicates("date_utc").shape[0]),
        "fixed_support_daily_scores": int(score.shape[0]), "depth_bins_per_score": 40,
        "D50_mm_not_measured": 0.25, "datum_case": "ZH_unadjusted", "source_outputs_read_only": True}
(OUT / "Figure_S10_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"Created {png} and {pdf}; catalog={meta['catalog_dates']}, ROI-QA={meta['roi_qa_dates']}, daily scores={meta['fixed_support_daily_scores']}.")
