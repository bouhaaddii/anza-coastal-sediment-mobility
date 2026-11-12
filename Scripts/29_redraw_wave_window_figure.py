from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Additional_Analyses" / "Within_Cell_Wave_Response"
TABLE = OUT / "wave_lag_response_models.csv"


def main() -> None:
    if not TABLE.is_file():
        raise FileNotFoundError(f"Run Script 28 first; missing model table: {TABLE}")
    data = pd.read_csv(TABLE)
    required = {"model", "lag_window", "standardized_beta_log_spm_per_1sd_wave", "block_bootstrap_95ci_low", "block_bootstrap_95ci_high"}
    if not required.issubset(data.columns):
        raise ValueError(f"Model table is missing required columns: {sorted(required.difference(data.columns))}")
    data = data.dropna(subset=["standardized_beta_log_spm_per_1sd_wave"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No estimable coefficients are available to plot.")

    palette = {"0–24 h": "#176B87", "24–48 h": "#D97732", "48–72 h": "#7567A8"}
    fig, ax = plt.subplots(figsize=(12.0, max(5.5, 0.58 * len(data) + 1.2)), constrained_layout=True)
    y = np.arange(len(data))[::-1]
    for idx, row in data.iterrows():
        beta = float(row["standardized_beta_log_spm_per_1sd_wave"])
        low = pd.to_numeric(pd.Series([row["block_bootstrap_95ci_low"]]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([row["block_bootstrap_95ci_high"]]), errors="coerce").iloc[0]
        color = palette.get(str(row["lag_window"]), "#334155")
        if pd.notna(low) and pd.notna(high):
            ax.plot([float(low), float(high)], [y[idx], y[idx]], color=color, lw=1.7, solid_capstyle="round")
            ax.plot([float(low), float(low)], [y[idx] - .08, y[idx] + .08], color=color, lw=1.2)
            ax.plot([float(high), float(high)], [y[idx] - .08, y[idx] + .08], color=color, lw=1.2)
        ax.plot(beta, y[idx], "o", color=color, ms=6)
    ax.axvline(0, color="#475569", linestyle="--", linewidth=1)
    labels = [f"{row.model} | {row.lag_window}" for row in data.itertuples()]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Within-cell change in log(daily HR-OC SPM) per 1-SD increase in mean IBI Hs")
    ax.set_title("Daily HR-OC SPM associations with nominal antecedent wave windows at Anza", loc="left", weight="bold")
    ax.grid(axis="x", color="#CBD5E1", alpha=.6, linewidth=.7)
    ax.text(
        0.0, -0.17,
        "Timing caveat: HR-OC L3 aggregates daily Sentinel-2 tile mosaics; source overpass time per SPM pixel is unavailable. "
        "Windows are referenced to the local scene time and are not verified physical response lags.",
        transform=ax.transAxes, fontsize=8.5, wrap=True,
    )
    ax.margins(y=.08)
    fig.savefig(OUT / "Figure_wave_lag_response.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "Figure_wave_lag_response.pdf", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure. Within-cell fixed-effect associations between daily Copernicus Marine HR-OC L3 SPM and mean IBI significant wave height "
        "summarized over nominal 0–24 h, 24–48 h, and 48–72 h windows referenced to a local Sentinel-2 acquisition time. "
        "Separate-window models estimate each wave window individually; joint distributed-lag models estimate conditional coefficients where estimable. "
        "Points show standardized coefficients and bars show exploratory two-axis block-bootstrap 95% intervals (500 m spatial blocks and moving blocks of three eligible acquisitions). "
        "The HR-OC L3 daily product aggregates Sentinel-2 tile mosaics and the source acquisition time per SPM pixel is unavailable in the downloaded subset; "
        "therefore these are nominal forcing-window associations, not verified physical response lags. HR-OC SPM was not validated against local in-situ SPM in this study, "
        "and no causal sediment-transport inference is made."
    )
    (OUT / "wave_lag_response_figure_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
    print("Redrew Figure wave-lag response with the daily-product timing caveat; no regression or bootstrap was rerun.")
    print(f"  Figure: {OUT / 'Figure_wave_lag_response.png'}")
    print(f"  Vector PDF: {OUT / 'Figure_wave_lag_response.pdf'}")
    print(f"  Updated caption: {OUT / 'wave_lag_response_figure_caption_EN.txt'}")


if __name__ == "__main__":
    main()
