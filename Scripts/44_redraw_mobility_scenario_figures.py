"""Redraw Script 32 publication figures from saved scenario outputs only.

No tide prediction, hydrodynamic calculation, or model fitting is performed.
This is useful when the licensed GOT5.6 input is not present in a release copy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Sediment_Mobility_Scenarios"
D50_MM = (0.063, 0.100, 0.125, 0.250, 0.500, 1.000)


def main() -> None:
    summary_path = OUT / "mobility_by_depth_grain_datum_scenario.csv"
    point_path = OUT / "mobility_point_map_D50_0p10mm_ZH_unadjusted.csv.gz"
    metadata_path = OUT / "mobility_scenario_run_metadata.json"
    for path in (summary_path, point_path, metadata_path):
        if not path.is_file():
            raise SystemExit(f"Required saved scenario output is missing: {path}")

    summary = pd.read_csv(summary_path)
    point_table = pd.read_csv(point_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_time = int(metadata["wave_records"])
    depth_bins = summary["survey_depth_bin_below_local_ZH_m"].astype(float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    # Figure A: retain the local survey geometry and scenario scale, but no title.
    fig, ax = plt.subplots(figsize=(7.1, 6.0), constrained_layout=True)
    sc = ax.scatter(
        point_table["easting_utm29n_candidate_m"],
        point_table["northing_utm29n_candidate_m"],
        c=point_table["combined_exceedance_fraction"], s=2.2,
        cmap="viridis", vmin=0.82, vmax=1.0, linewidths=0,
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.82)
    cb.set_label("Fraction of valid hours with modeled M ≥ 1 (display range: 0.82–1.00)")
    ax.set_xlabel("Easting (UTM 29N; m)")
    ax.set_ylabel("Northing (UTM 29N; m)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02, 0.02,
        "Regional forcing held spatially constant; pattern mainly follows depth.\n"
        "No local wave transformation.",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
    )
    for stem in ("Figure_mobility_scenario_map",):
        fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure B: retain panel headings, which distinguish physical scenarios.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    for ax in axes.flat:
        ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.19, hspace=0.28, wspace=0.13)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(D50_MM)))
    panels = [
        (axes[0, 0], "ZH_unadjusted", "wave_only_exceedance_fraction", "Wave-only; local ZH"),
        (axes[0, 1], "ZH_unadjusted", "combined_exceedance_fraction", "Wave + currents; local ZH"),
        (axes[1, 0], "IHO_2024_candidate_offset_sensitivity", "wave_only_exceedance_fraction", "Wave-only; +2.159 m scenario"),
        (axes[1, 1], "IHO_2024_candidate_offset_sensitivity", "combined_exceedance_fraction", "Wave + currents; +2.159 m scenario"),
    ]
    for ax, case, metric, title in panels:
        subset = summary[summary["datum_case"] == case]
        for color, d50 in zip(colors, D50_MM):
            line = subset[subset["D50_scenario_mm_not_measured"] == d50].sort_values(
                "survey_depth_bin_below_local_ZH_m"
            )
            ax.plot(
                line[metric], line["survey_depth_bin_below_local_ZH_m"],
                marker="o", ms=2.5, lw=1.4, color=color, label=f"{d50:g} mm",
            )
        ax.set_title(title, fontsize=9.4, pad=5)
        ax.grid(True, color="#D9E2E8", linewidth=0.55)
        ax.set_xlim(0.45, 1.0)
        ax.set_xticks(np.arange(0.5, 1.01, 0.1))
        ax.invert_yaxis()
    fig.text(0.025, 0.54, "Survey depth below local ZH (m; increasing downward)",
             rotation=90, ha="center", va="center", fontsize=8.8)
    fig.text(0.56, 0.115, "Fraction of valid hourly records with M ≥ 1 (axis shown from 0.45)",
             ha="center", va="center", fontsize=8.8)
    axes[0, 1].legend(title="Assumed D50", frameon=True, fontsize=7.6, title_fontsize=7.6, loc="lower right")
    fig.text(0.55, 0.035,
             "Valid-hour support varies by depth; see the supporting figure. "
             "Scenario frequencies are not probabilities or transport rates.",
             ha="center", fontsize=7.5)
    fig.savefig(OUT / "Figure_mobility_depth_grain_sensitivity.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_mobility_depth_grain_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure C: denominator/support diagnostic, title remains in the external caption.
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
    fig.savefig(OUT / "Figure_mobility_forcing_support.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_mobility_forcing_support.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Redrew three mobility figures from the saved scenario tables; no tide or mobility calculations were rerun.")
    print(f"Output folder: {OUT}")


if __name__ == "__main__":
    main()
