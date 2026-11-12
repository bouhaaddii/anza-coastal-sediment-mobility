from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Results" / "Sediment_Proxy" / "Spatial_Support"
STRICT = OUT_DIR / "ndti_strict_temporal_comparison.csv"
FORCING = ROOT / "Results" / "scene_forcing_and_quality.csv"
WAVE_CONTEXT = ROOT / "Results" / "Additional_Analyses" / "sentinel2_antecedent_wave_context.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return None
    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(values.size, dtype=float)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return ranks

    xr = average_ranks(x[valid])
    yr = average_ranks(y[valid])
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    strict_rows = read_csv(STRICT)
    forcing_by_product = {row.get("product_name", ""): row for row in read_csv(FORCING)}
    wave_by_product = {row.get("product_name", ""): row for row in read_csv(WAVE_CONTEXT)}
    if not strict_rows:
        raise RuntimeError("Strict temporal summary contains no scene rows.")

    joined: list[dict[str, object]] = []
    missing = []
    for row in strict_rows:
        product = row["product_name"]
        forcing = forcing_by_product.get(product)
        wave = wave_by_product.get(product)
        if forcing is None or wave is None:
            missing.append(product)
            continue
        strict_median = number(row.get("median_ndti_strict_common_pixels"))
        if strict_median is None:
            raise RuntimeError(f"Strict common-support NDTI is missing for {product}.")
        uo = number(forcing.get("ibi_uo_m_s"))
        vo = number(forcing.get("ibi_vo_m_s"))
        current_speed = float(np.hypot(uo, vo)) if uo is not None and vo is not None else None
        joined.append({
            "product_name": product,
            "acquisition_utc": row.get("acquisition_utc", ""),
            "strict_common_pixels_n": int(number(row.get("strict_all_scene_common_pixels")) or 0),
            "ndti_median_same_pixels": strict_median,
            "ndti_p25_same_pixels": number(row.get("median_ndti_p25_strict_common_pixels")),
            "ndti_p75_same_pixels": number(row.get("median_ndti_p75_strict_common_pixels")),
            "ibi_wave_hs_at_acquisition_m": number(forcing.get("ibi_vhm0_m")),
            "ibi_wave_peak_period_s": number(forcing.get("ibi_vtpk_s")),
            "ibi_wave_mean_direction_deg": number(forcing.get("ibi_vmdr_deg")),
            "ibi_preceding_24h_max_hs_m": number(wave.get("preceding_24h_max_hs_m")),
            "ibi_preceding_24h_mean_hs_m": number(wave.get("preceding_24h_mean_hs_m")),
            "wave_nearest_time_offset_min": number(wave.get("nearest_wave_time_offset_min")),
            "wave_grid_distance_from_target_km": number(wave.get("wave_grid_distance_from_target_km")),
            "ibi_surface_current_speed_m_s": current_speed,
            "current_grid_distance_from_target_km": number(forcing.get("ibi_current_grid_distance_km")),
            "got56_tide_anomaly_from_utc_day_mean_m": number(forcing.get("got56_anomaly_from_utc_day_mean_m")),
            "tide_estimate_status": forcing.get("tide_estimate_status", ""),
        })
    if missing:
        raise RuntimeError("Missing paired forcing records for: " + ", ".join(missing))

    joined.sort(key=lambda row: str(row["acquisition_utc"]))
    variables = [
        ("ibi_preceding_24h_max_hs_m", "Maximum Hs in preceding 24 h (m)"),
        ("ibi_wave_hs_at_acquisition_m", "Hs at acquisition (m)"),
        ("ibi_surface_current_speed_m_s", "Regional surface-current speed (m s⁻¹)"),
        ("got56_tide_anomaly_from_utc_day_mean_m", "GOT5.6 modeled tide anomaly (m)"),
    ]
    y_all = np.asarray([float(row["ndti_median_same_pixels"]) for row in joined])
    association_rows: list[dict[str, object]] = []
    for key, label in variables:
        x = np.asarray([float(row[key]) if row[key] is not None else np.nan for row in joined])
        rho = spearman_rho(x, y_all)
        n = int((np.isfinite(x) & np.isfinite(y_all)).sum())
        association_rows.append({
            "predictor": key,
            "predictor_label": label,
            "paired_scene_count": n,
            "spearman_rho_descriptive_only": rho if rho is not None else "",
            "p_value": "NOT_COMPUTED",
            "interpretation": "Exploratory association only; 11 scenes, seasonal confounding, regional forcing grid, uncalibrated optical proxy.",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired_path = OUT_DIR / "ndti_same_pixel_wave_tide_matchups.csv"
    association_path = OUT_DIR / "ndti_wave_tide_descriptive_associations.csv"
    write_csv(paired_path, joined)
    write_csv(association_path, association_rows)

    rho_by_predictor = {
        str(row["predictor"]): row["spearman_rho_descriptive_only"]
        for row in association_rows
    }

    def fr_rho(key: str) -> str:
        value = rho_by_predictor.get(key)
        return "non défini" if value == "" else f"{float(value):.2f}".replace(".", ",")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
    fig.suptitle("Exploratory NDTI and hydrodynamic context at Anza", fontsize=14, fontweight="bold")
    dates = [datetime.fromisoformat(str(row["acquisition_utc"]).replace("Z", "+00:00")) for row in joined]
    color_values = mdates.date2num(dates)
    scatter_for_colorbar = None
    for ax, ((key, label), assoc) in zip(axes.flat, zip(variables, association_rows)):
        x = np.asarray([float(row[key]) if row[key] is not None else np.nan for row in joined])
        valid = np.isfinite(x) & np.isfinite(y_all)
        rho = assoc["spearman_rho_descriptive_only"]
        title_rho = f"ρs={float(rho):.2f}" if rho != "" else "ρs not defined"
        scatter_for_colorbar = ax.scatter(x[valid], y_all[valid], c=color_values[valid], cmap="plasma", s=45,
                                          edgecolor="white", linewidth=0.55, zorder=3)
        ax.set_xlabel(label)
        ax.set_ylabel("Median NDTI on the same pixels (unitless)")
        ax.set_title(f"{chr(65 + list(axes.flat).index(ax))}  {label}\n{title_rho}; n={int(valid.sum())} scenes")
        ax.grid(color="#CBD5E1", linewidth=0.5, alpha=0.75)

    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=list(axes.flat), shrink=0.82, pad=0.02)
        cbar.set_ticks(color_values)
        cbar.ax.set_yticklabels([date.strftime("%d %b") for date in dates], fontsize=7)
        cbar.set_label("Acquisition date (UTC; exact dates in paired table)")

    figure_path = OUT_DIR / "Figure_NDTI_hydrodynamic_context.png"
    fig.savefig(figure_path, dpi=350, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure_NDTI_hydrodynamic_context.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "scene_count": len(joined),
        "NDTI_support": "Strict intersection of 50,128 persistent-water pixels valid in all 11 selected scenes.",
        "wave_and_current_source": "Copernicus Marine IBI regional model values matched to the nearest grid point/hour as recorded in project outputs.",
        "tide_source": "GOT5.6 modeled astronomical tide anomaly relative to UTC-day mean; no local gauge validation or port-zero tie.",
        "association_method": "Spearman rank correlation coefficient; coefficients shown descriptively; p-values not computed.",
        "limitations": [
            "Only 11 temporally irregular Sentinel-2 scenes are paired; time and season can confound associations.",
            "IBI grid values are regional context and do not resolve the surf zone or nearshore circulation at the study scale.",
            "NDTI is unitless and uncalibrated to suspended-sediment concentration; it is affected by water optics, glint, foam, and atmosphere.",
            "GOT5.6 anomaly is modeled, not validated by a local tide gauge and not tied to the local hydrographic zero.",
            "No p-values, causal effects, sediment fluxes, or transport rates are estimated.",
        ],
        "outputs": [
            "ndti_same_pixel_wave_tide_matchups.csv",
            "ndti_wave_tide_descriptive_associations.csv",
            "Figure_NDTI_hydrodynamic_context.png/.pdf",
        ],
    }
    (OUT_DIR / "ndti_hydrodynamic_context_settings.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "Figure_NDTI_hydrodynamic_context_caption_EN.txt").write_text(
        "Figure. Exploratory associations between Sentinel-2 NDTI and regional hydrodynamic context at acquisition dates. Panels show the median "
        "NDTI on the same 50,128 pixels observed in all 11 selected scenes against (A) maximum significant wave height during the preceding 24 h, "
        "(B) significant wave height at acquisition, (C) regional surface-current speed, and (D) the modeled GOT5.6 tide anomaly relative to the UTC-day "
        "mean. Point color and labels indicate acquisition date. Spearman coefficients are descriptive only; no p-values or causal interpretation are "
        "provided. The IBI fields are regional model context, the tide estimate has no local gauge validation, and NDTI is not calibrated to suspended-sediment concentration.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "ndti_hydrodynamic_context_interpretation_FR.txt").write_text(
        "Lecture prudente des résultats exploratoires (11 scènes)\n\n"
        f"Sur l'intersection fixe de {int(joined[0]['strict_common_pixels_n']):,} pixels, le coefficient de rang de Spearman entre la médiane NDTI et le maximum de Hs des 24 heures précédentes est de {fr_rho('ibi_preceding_24h_max_hs_m')}. "
        f"L'association avec Hs à l'acquisition est de {fr_rho('ibi_wave_hs_at_acquisition_m')}; celle avec la vitesse du courant régional IBI est de {fr_rho('ibi_surface_current_speed_m_s')} et celle avec l'anomalie GOT5.6 est de {fr_rho('got56_tide_anomaly_from_utc_day_mean_m')}. "
        "Ces coefficients sont descriptifs, sans valeurs p ni test d'hypothèse.\n\n"
        "Le signe négatif avec le maximum de houle antérieure peut motiver l'hypothèse que les conditions précédant l'image sont plus informatives que la houle instantanée pour expliquer la variabilité du signal optique. "
        "Il ne démontre ni hausse ni baisse de la concentration de matières en suspension : le NDTI est non calibré, et les dates couvrent une période saisonnière irrégulière. "
        "La houle et les courants IBI sont des champs régionaux qui ne résolvent pas le déferlement côtier; l'anomalie de marée est modélisée sans validation locale.\n\n"
        "Pour une interprétation physique, il faudrait davantage de dates réparties entre conditions calmes et énergétiques, une validation optique par mesures de terrain ou un proxy MES validé, ainsi qu'une meilleure caractérisation des vagues et courants près du rivage.\n",
        encoding="utf-8",
    )

    print("Exploratory same-pixel hydrodynamic matchup complete; source rasters were not changed.")
    print(f"  Paired scenes: {len(joined)}; strict spatial support: {joined[0]['strict_common_pixels_n']:,} pixels")
    print(f"  Scene matchups: {paired_path}")
    print(f"  Descriptive associations: {association_path}")
    print(f"  Figure: {figure_path}")
    print(f"  Interpretation: {OUT_DIR / 'ndti_hydrodynamic_context_interpretation_FR.txt'}")
    print("Spearman values are descriptive only; no p-values, causal claims, SSC concentrations, or sediment-transport rates are produced.")


if __name__ == "__main__":
    main()
