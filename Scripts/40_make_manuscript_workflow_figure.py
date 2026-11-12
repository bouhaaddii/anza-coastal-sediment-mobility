"""Create the publication workflow schematic from the executed Anza study design."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Manuscript_Figures"


def card(ax, x, y, w, h, title, body, face, edge="#42647A", title_color="#173B52"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + 0.018, y + h - 0.038, title, ha="left", va="top",
            fontsize=10.2, fontweight="bold", color=title_color, wrap=True)
    ax.text(x + 0.018, y + h - 0.082, body, ha="left", va="top",
            fontsize=8.7, color="#243746", linespacing=1.3, wrap=True)


def arrow(ax, start, end, color="#557386", lw=1.35):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color, connectionstyle="arc3,rad=0", shrinkA=2, shrinkB=3,
    ))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#203442"})
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    card(ax, 0.05, 0.88, 0.90, 0.10,
         "Data streams and analytical sequence at Anza",
         "Temporal sampling audit of a bathymetry-conditioned, modelled seabed-mobility indicator",
         "#F2F5F7", edge="#A6B5BE", title_color="#173B52")

    xs = [0.035, 0.2825, 0.53, 0.7775]
    w = 0.205
    # Keep the source cards clear of the title/subtitle card above.
    source_y, source_h = 0.675, 0.155
    proc_y, proc_h = 0.425, 0.16
    source_cards = [
        ("Satellite optical data",
         "Sentinel-2 L2A\n73 catalogue dates; 11 ROI-QA scenes\nRed/green NDTI; tile cloud metadata",
         "#E8F2F7"),
        ("Local bathymetric survey",
         "24,508 XYZ soundings\n22,986 submerged points\nDepth retained below Port of Agadir ZH",
         "#E8F3F0"),
        ("Regional marine forcing",
         "IBI-WAV hourly; 4,368 records\nIBI currents: daily regional means\nGOT5.6: modelled tide anomalies",
         "#F2F0FA"),
        ("Ocean-colour product",
         "Copernicus HR-OC daily SPM/TUR\n151 dates; variable water support\nSatellite-derived, not field validation",
         "#FBF3E6"),
    ]
    for x, (title, body, face) in zip(xs, source_cards):
        card(ax, x, source_y, w, source_h, title, body, face)

    proc_cards = [
        ("Optical quality and support",
         "Scene catalogue and ROI-QA subset\nCloud-threshold groups\nNDTI and common-date support",
         "#F5F8FA"),
        ("Depth harmonisation",
         "Project survey coordinates\nAssign 0.5-m depth classes\nPreserve local hydrographic zero",
         "#F3F8F6"),
        ("Forcing and mobility calculation",
         "Match wave/current forcing to depth\nWave and combined bed stress\nSix unmeasured D50 scenarios",
         "#F7F6FA"),
        ("HR-OC product screening",
         "Daily water-cell support audit\nPair eligible S2/HR-OC dates\nNo independent in-situ calibration",
         "#FCF8F1"),
    ]
    for x, (title, body, face) in zip(xs, proc_cards):
        card(ax, x, proc_y, w, proc_h, title, body, face)
        arrow(ax, (x + w/2, source_y - 0.008), (x + w/2, proc_y + proc_h + 0.008),
              color="#557386")

    # Each source stream is processed in its own lane, then enters the integrated analysis block.
    analysis_box = (0.055, 0.145, 0.89, 0.235)
    card(ax, *analysis_box,
         "Integrated analyses",
         "PRIMARY · Scene-date representativeness: compare 73 catalogue dates, cloud thresholds, and 11 ROI-QA scenes "
         "against the 182-day baseline on 40 fixed depth classes (3.5–23 m); evaluate binary M ≥ 1 and continuous "
         "mean log10(P95(M)), with month and within-month decomposition.\n"
         "SECONDARY · Optical-product association: same-day NDTI–HR-OC comparison, depth/spatial-trend confounding "
         "checks, and repeated-cell wave-association sensitivity.",
         "#E8F2F7", edge="#247BA0", title_color="#155B79")
    for x in xs:
        arrow(ax, (x + w/2, proc_y - 0.008), (x + w/2, analysis_box[1] + analysis_box[3] + 0.008),
              color="#247BA0")

    final_box = (0.08, 0.025, 0.84, 0.095)
    card(ax, *final_box,
         "Interpretation boundary",
         "Model-conditional sampling diagnostics only  ·  No measured SSC, sediment flux, transport direction, or validated surf-zone simulation",
         "#F2F4F6", edge="#788B98", title_color="#334B5B")
    arrow(ax, (0.5, analysis_box[1] - 0.008), (0.5, final_box[1] + final_box[3] + 0.008),
          color="#788B98")

    fig.subplots_adjust(left=0.025, right=0.975, top=0.99, bottom=0.025)
    png = OUT / "Figure_Anza_analysis_workflow.png"
    pdf = OUT / "Figure_Anza_analysis_workflow.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    caption = (
        "Figure 2. Workflow of the Anza study. Sentinel-2 scenes, local soundings, regional wave/current forcing, "
        "modelled tide anomalies, and the satellite-derived HR-OC product are screened and harmonised for separate "
        "primary and secondary analyses. The primary analysis compares acquisition-date groups against the 182-day "
        "forcing reference on 40 fixed depth classes, using binary and continuous model-conditional mobility metrics. "
        "The secondary branch evaluates optical-product associations and their sensitivity to depth and spatial structure. "
        "Neither branch is an in-situ sediment-concentration or sediment-flux validation."
    )
    (OUT / "Figure_Anza_analysis_workflow_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"Workflow figure saved: {png}")
    print(f"Vector PDF saved: {pdf}")
    print("Caption saved. The schematic represents executed analyses and their interpretation limits.")


if __name__ == "__main__":
    main()
