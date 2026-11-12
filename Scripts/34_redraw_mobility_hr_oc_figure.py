"""Redraw the mobility–HR-OC association figure from the saved model table only."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Sediment_Mobility_Scenarios" / "HR_OC_Association"
TABLE = OUT / "mobility_HR_OC_fixed_cell_models.csv"


def main() -> None:
    if not TABLE.is_file():
        raise SystemExit(f"Model table not found: {TABLE}. Run Script 33 first.")
    data = pd.read_csv(TABLE)
    required = {
        "datum_case", "D50_scenario_mm_not_measured", "model",
        "beta_log_spm_per_1sd_mobility", "block_bootstrap_95ci_low",
        "block_bootstrap_95ci_high",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise SystemExit("The saved model table lacks the corrected coefficient fields: " + ", ".join(missing))
    if data.empty:
        raise SystemExit("The saved model table contains no model rows.")

    colors = {"Wave_only": "#287C8E", "Wave_plus_daily_current": "#D55E00"}
    model_labels = {"Wave_only": "wave-only", "Wave_plus_daily_current": "waves + daily currents"}
    model_order = {"Wave_only": 0, "Wave_plus_daily_current": 1}
    datum_order = ("ZH_unadjusted", "IHO_2024_candidate_offset_sensitivity")
    datum_titles = {
        "ZH_unadjusted": "Local hydrographic zero, unadjusted",
        "IHO_2024_candidate_offset_sensitivity": "Candidate +2.159 m sensitivity",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 8.0), sharex=True, sharey=True)
    left_labels: list[str] = []
    max_abs = 0.0
    for datum in datum_order:
        subset = data[data["datum_case"].eq(datum)].copy()
        subset["_model_order"] = subset["model"].map(model_order).fillna(99)
        subset = subset.sort_values(["D50_scenario_mm_not_measured", "_model_order"])
        for _, row in subset.iterrows():
            max_abs = max(max_abs, abs(float(row["beta_log_spm_per_1sd_mobility"])))
            for bound in ("block_bootstrap_95ci_low", "block_bootstrap_95ci_high"):
                if pd.notna(row[bound]):
                    max_abs = max(max_abs, abs(float(row[bound])))

    for panel_index, (ax, datum) in enumerate(zip(axes, datum_order)):
        subset = data[data["datum_case"].eq(datum)].copy()
        subset["_model_order"] = subset["model"].map(model_order).fillna(99)
        subset = subset.sort_values(["D50_scenario_mm_not_measured", "_model_order"]).reset_index(drop=True)
        y_positions = np.arange(len(subset))
        for y, row in zip(y_positions, subset.to_dict("records")):
            model = row["model"]
            color = colors.get(model, "#687582")
            beta = float(row["beta_log_spm_per_1sd_mobility"])
            low = float(row["block_bootstrap_95ci_low"])
            high = float(row["block_bootstrap_95ci_high"])
            ax.plot([low, high], [y, y], color=color, lw=1.7, solid_capstyle="round", zorder=2)
            ax.plot([low, low], [y - 0.13, y + 0.13], color=color, lw=1.0, zorder=2)
            ax.plot([high, high], [y - 0.13, y + 0.13], color=color, lw=1.0, zorder=2)
            ax.scatter(beta, y, color=color, s=27, zorder=3, edgecolor="white", linewidth=0.45)
            if panel_index == 0:
                d50 = float(row["D50_scenario_mm_not_measured"])
                left_labels.append(f"{d50:g} mm · {model_labels.get(model, model)}")

        ax.axvline(0, color="#334155", lw=1.15, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(datum_titles.get(datum, datum), fontsize=11.2, pad=10)
        ax.set_ylim(len(subset) - 0.35, -0.65)
        ax.grid(axis="x", color="#DCE4EA", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8.4)

    axes[0].set_yticks(np.arange(len(left_labels)), labels=left_labels)
    axes[0].set_ylabel("Grain-size scenario and forcing", labelpad=8)
    axes[1].tick_params(axis="y", labelleft=False)
    limit = max(0.05, np.ceil(max_abs / 0.05) * 0.05 + 0.01)
    axes[0].set_xlim(-limit, limit)

    legend_handles = [
        Line2D([0], [0], color=colors["Wave_only"], marker="o", lw=1.8, label="Wave-only forcing"),
        Line2D([0], [0], color=colors["Wave_plus_daily_current"], marker="o", lw=1.8, label="Waves + daily currents"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.63, 0.935),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Modeled sediment mobility and satellite-derived HR-OC SPM", fontsize=13, y=0.985)
    fig.supxlabel("Change in natural log(SPM) per 1-SD increase in mobility (95% block-bootstrap interval)",
                  fontsize=10, y=0.085)
    fig.text(0.5, 0.025,
             "Exploratory within-cell association; HR-OC is satellite-derived, not independent field validation. "
             "The source granule match is unverified; no causal or transport-rate inference is made.",
             ha="center", va="bottom", fontsize=8.0, color="#475569")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.87, bottom=0.17, wspace=0.08)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Figure_mobility_HR_OC_association.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_mobility_HR_OC_association.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Redrew mobility–HR-OC association figure from saved model estimates; no model or bootstrap was rerun.")
    print(f"Updated figure files: {OUT / 'Figure_mobility_HR_OC_association.png'} and PDF")


if __name__ == "__main__":
    main()
