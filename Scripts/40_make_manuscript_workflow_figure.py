"""Create a staged input-to-processing-to-analysis workflow figure."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Manuscript_Figures"


def card(ax, x, y, w, h, title, body, face, edge="#577789"):
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.05, edgecolor=edge, facecolor=face)
    ax.add_patch(box)
    ax.text(x + 0.014, y + h - 0.023, title, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color="#153B51")
    ax.text(x + 0.014, y + h - 0.050, body, ha="left", va="top",
            fontsize=8.0, color="#263C48", linespacing=1.15)


def group(ax, xywh, label, edge="#A6B8C2"):
    x, y, w, h = xywh
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.85, edgecolor=edge, facecolor="#FBFCFD",
        linestyle=(0, (3, 2)))
    ax.add_patch(box)
    ax.text(x + 0.012, y + h - 0.010, label, ha="left", va="top",
            fontsize=8.7, fontweight="bold", color="#3E6478")


def arrow(ax, start, end, color="#557386"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=9.5, linewidth=1.05, color=color,
        connectionstyle="arc3,rad=0", shrinkA=1, shrinkB=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#203442"})
    fig, ax = plt.subplots(figsize=(11.8, 8.1), facecolor="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # The stage headings are inside their group borders; every card is self-contained.
    input_box = (0.025, 0.735, 0.95, 0.235)
    process_box = (0.025, 0.475, 0.95, 0.225)
    analysis_box = (0.025, 0.175, 0.95, 0.265)
    limit_box = (0.025, 0.025, 0.95, 0.115)
    group(ax, input_box, "INPUT DATA")
    group(ax, process_box, "PREPROCESSING AND HARMONISATION")
    group(ax, analysis_box, "ANALYSES")
    group(ax, limit_box, "INTERPRETATION BOUNDARY")

    xs = [0.045, 0.285, 0.525, 0.765]
    w = 0.19
    input_cards = [
        ("Sentinel-2 L2A",
         "73 catalog dates\n11 ROI-QA scenes\nB02/B03/B04/B08; SCL masks",
         "#E8F2F7"),
        ("Local bathymetry",
         "24,508 XYZ soundings\n22,986 supported submerged points\nPort of Agadir hydrographic zero",
         "#E8F3F0"),
        ("Regional forcing",
         "IBI-WAV: 4,368 hourly records\nIBI currents: daily profiles\nGOT5.6: relative tide anomaly",
         "#F2F0FA"),
        ("Ocean-colour product",
         "Copernicus HR-OC L3\nSPM and TUR; 151 daily dates\nWater support varies by date",
         "#FBF3E6"),
    ]
    for x, (title, body, face) in zip(xs, input_cards):
        card(ax, x, 0.765, w, 0.155, title, body, face)

    process_cards = [
        ("Optical screening",
         "SCL cloud/shadow QA\nReflectance and NDTI\nUnique acquisition dates",
         "#F3F7FA"),
        ("Depth support",
         "Horizontal coordinate audit\n0.5 m depth bins\nLocal vertical zero retained",
         "#F1F7F5"),
        ("Forcing and mobility",
         "Depth-matched wave/current\nSwart–Soulsby stress screen\nSix unmeasured D50 scenarios",
         "#F5F4FA"),
        ("HR-OC matching",
         "Daily water-support audit\nSame-date product pairing\nNo field calibration",
         "#FCF7EF"),
    ]
    for x, (title, body, face) in zip(xs, process_cards):
        card(ax, x, 0.505, w, 0.135, title, body, face)
        arrow(ax, (x + w / 2, 0.755), (x + w / 2, 0.651))

    # Stage boundary arrows show the flow into the analysis block without crossing.
    for x in xs:
        arrow(ax, (x + w / 2, 0.495), (x + w / 2, 0.447), color="#247BA0")

    card(ax, 0.05, 0.215, 0.565, 0.16,
         "PRIMARY ANALYSIS  |  acquisition-date representativeness",
         "Compare 73 catalog dates, cloud-metadata thresholds, and 11 ROI-QA scenes with the 182-day forcing record.\n"
         "Hold 40 supported depth classes (3.5–23 m) fixed; report binary and continuous mobility scores and\n"
         "separate month composition from within-month contrasts.",
         "#E9F2F7", edge="#247BA0")
    card(ax, 0.645, 0.215, 0.305, 0.16,
         "SECONDARY ANALYSES  |  product associations",
         "Same-date NDTI–HR-OC with depth/spatial controls.\n"
         "Repeated-cell wave association; support, lag,\n"
         "and block-size sensitivity.",
         "#F7F3EA", edge="#BD8A45")

    ax.text(0.045, 0.094,
        "Modeled, scenario-conditional bed-mobility indicator only. No measured SSC or sediment flux, "
        "transport direction, or validated surf-zone simulation.",
        ha="left", va="center", fontsize=8.7, color="#334B5B", wrap=True)
    ax.text(0.955, 0.047, "Data layers are not vertically fused.",
            ha="right", va="bottom", fontsize=7.4, color="#607784")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.015)
    png = OUT / "Figure_Anza_analysis_workflow.png"
    pdf = OUT / "Figure_Anza_analysis_workflow.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    caption = (
        "Figure 2. Staged workflow for the Anza analysis. Inputs are grouped above their preprocessing and "
        "harmonisation steps. The primary analysis compares satellite acquisition-date groups with the full "
        "182-day forcing record over a fixed bathymetric support. Secondary analyses evaluate satellite-product "
        "associations and repeated-cell wave response. All mobility results are model-conditional; the workflow "
        "does not represent field-calibrated suspended sediment concentration, sediment flux, transport direction, "
        "or a validated surf-zone simulation."
    )
    (OUT / "Figure_Anza_analysis_workflow_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Workflow figure saved: {png}")
    print(f"Vector PDF saved: {pdf}")
    print("Inputs, preprocessing, analyses, and interpretation limits are separated in labelled groups.")


if __name__ == "__main__":
    main()
