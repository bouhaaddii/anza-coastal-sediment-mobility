from __future__ import annotations

import csv
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
CATALOG = ROOT / "Data" / "Sentinel2" / "scene_catalog.csv"
MODEL_DIR = ROOT / "Data" / "Tide_Model"
OUT = ROOT / "Results" / "sentinel2_GOT5.6_tide_estimates.csv"


def predict(lon: float, lat: float, times: pd.DatetimeIndex) -> np.ndarray:
    times_naive_utc = times.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    result = pyTMD.compute.tide_elevations(
        np.full(len(times_naive_utc), lon, dtype=float),
        np.full(len(times_naive_utc), lat, dtype=float),
        times_naive_utc,
        directory=MODEL_DIR,
        model=CONFIG["tide_model"]["name"],
        crs="4326",
        standard="datetime",
        type="trajectory",
        method="linear",
        crop=True,
        bounds=[lon - 0.5, lon + 0.5, lat - 0.5, lat + 0.5],
        buffer=0.25,
        extrapolate=True,
        cutoff=float(CONFIG["tide_model"]["maximum_wet_node_distance_km"]),
    )
    values = result.values if hasattr(result, "values") else result
    return np.asarray(values, dtype=float).reshape(-1)


def main() -> None:
    if not CATALOG.is_file():
        raise FileNotFoundError("Run Scripts/01_catalog_sentinel2.py first.")
    if not (MODEL_DIR / "GOT5.5").is_dir() or not (MODEL_DIR / "GOT5.6").is_dir():
        raise FileNotFoundError(f"GOT5.5/GOT5.6 model grids not found under {MODEL_DIR}.")

    with CATALOG.open(newline="", encoding="utf-8-sig") as stream:
        scenes = [row for row in csv.DictReader(stream) if row.get("acquisition_utc")]
    if not scenes:
        raise ValueError("No Sentinel-2 acquisition timestamps are present in the catalogue.")

    model = CONFIG["tide_model"]
    lon, lat = map(float, model["point_wgs84"])
    timestamps = pd.to_datetime([row["acquisition_utc"] for row in scenes], utc=True)
    unique_dates = sorted({stamp.date() for stamp in timestamps})
    interval = int(model["within_day_sampling_minutes"])
    day_means: dict[str, float] = {}

    print(f"Stage 3: predicting modeled astronomical tide for {len(scenes)} Sentinel-2 acquisitions.")
    print(f"  Model: {model['name']}; point: {lon:.5f}, {lat:.5f} (WGS84).")
    print(f"  Point note: {model['point_note']}")
    for day in unique_dates:
        start = pd.Timestamp(day, tz="UTC")
        end = start + pd.Timedelta(days=1)
        sample_times = pd.date_range(start, end, freq=f"{interval}min", inclusive="left")
        estimates = predict(lon, lat, sample_times)
        finite = estimates[np.isfinite(estimates)]
        day_means[day.isoformat()] = float(np.mean(finite)) if finite.size else float("nan")
        if len(day_means) % 20 == 0:
            print(f"  Daily reference means calculated: {len(day_means)}/{len(unique_dates)}")

    scene_values = predict(lon, lat, timestamps)
    rows = []
    for scene, stamp, value in zip(scenes, timestamps, scene_values):
        day_mean = day_means.get(stamp.date().isoformat(), float("nan"))
        rows.append({
            "scene_id": scene.get("scene_id", ""),
            "product_name": scene.get("product_name", ""),
            "acquisition_utc": stamp.isoformat(),
            "longitude_wgs84": lon,
            "latitude_wgs84": lat,
            "model": model["name"],
            "model_reference_tidal_elevation_m": float(value),
            "modeled_utc_day_mean_m": day_mean,
            "anomaly_relative_to_modeled_utc_day_mean_m": float(value - day_mean),
            "wet_node_cutoff_km": float(model["maximum_wet_node_distance_km"]),
            "status": "modeled astronomical estimate; no local tide-gauge validation or port-zero tie",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f", encoding="utf-8-sig")
    readme = ROOT / "Data" / "Tides" / "README_GOT5.6_Sentinel2.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# GOT5.6 estimates for Sentinel-2 overpasses\n\n"
        "The output contains model-reference astronomical tidal elevations and an anomaly relative to the modelled mean of each UTC day. "
        "This is a phase/context diagnostic. It is not an observed water level, it does not include meteorological surge or wave setup, "
        "and it is not a vertical tie to the Port of Agadir hydrographic zero. The point coordinate is inherited from the preceding Anza project "
        "and must be checked against the coastline and wet model nodes. GOT is a global, coarse tide model; coastal extrapolation is limited to the configured 10 km cutoff.\n\n"
        "Model components: GOT5.5 base constituents plus the GOT5.6 supplementary constituents supplied in Data/Tide_Model. Cite NASA GSFC tide-model documentation and pyTMD in any manuscript.\n\n"
        "Official sources: https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models ; https://pytmd.readthedocs.io/en/stable/ ; https://doi.org/10.5281/zenodo.22970362\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} tide estimates: {OUT}")
    print("Interpret these as modeled astronomical context, not local gauge observations or absolute port-datum levels.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
