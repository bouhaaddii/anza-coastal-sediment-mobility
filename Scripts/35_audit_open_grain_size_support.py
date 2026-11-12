"""Download and spatially audit open NW Africa grain-size samples near Anza.

The PANGAEA dataset is regional/offshore context. Its grain fractions must not
be treated as local Anza D50 unless the sample locations and seabed settings
are demonstrably representative.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data" / "External_Sediment" / "PANGAEA_207974"
OUT_DIR = ROOT / "Results" / "External_Sediment_Audit"
URL = "https://doi.pangaea.de/10.1594/PANGAEA.207974?format=textfile"
DOI = "10.1594/PANGAEA.207974"
SITE_LON = -9.67552
SITE_LAT = 30.45000


def download_bytes(path: Path, overwrite: bool) -> bytes:
    if path.is_file() and not overwrite:
        print(f"Using existing open dataset file: {path}")
        return path.read_bytes()
    request = Request(URL, headers={"User-Agent": "AgadirCoastalStudy/1.0 (open research data audit)"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
    if not payload or b"<html" in payload[:1000].lower() or b"<!doctype html" in payload[:1000].lower():
        raise RuntimeError("PANGAEA returned an HTML page instead of the tab-delimited dataset.")
    if not ("text" in content_type.lower() or "octet-stream" in content_type.lower() or payload[:1] in (b"#", b"E")):
        print(f"Note: PANGAEA returned Content-Type={content_type!r}; checking tabular content.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    print(f"Downloaded PANGAEA dataset ({len(payload):,} bytes): {path}")
    return payload


def read_pangaea_table(payload: bytes) -> pd.DataFrame:
    text = payload.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        cells = [cell.strip().casefold() for cell in line.split("\t")]
        if any(cell.startswith("latitude") for cell in cells) and any(cell.startswith("longitude") for cell in cells):
            header_index = index
            break
    if header_index is None:
        raise ValueError("Could not identify Latitude and Longitude columns in the downloaded PANGAEA table.")
    table_text = "\n".join(lines[header_index:])
    frame = pd.read_csv(io.StringIO(table_text), sep="\t", quoting=csv.QUOTE_MINIMAL, dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def first_column(columns: list[str], prefix: str) -> str:
    matches = [column for column in columns if column.casefold().startswith(prefix)]
    if not matches:
        raise ValueError(f"No column beginning with {prefix!r} was found.")
    return matches[0]


def haversine_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon1 = np.radians(lon)
    lat1 = np.radians(lat)
    lon0 = math.radians(SITE_LON)
    lat0 = math.radians(SITE_LAT)
    dlon = lon1 - lon0
    dlat = lat1 - lat0
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="download the PANGAEA table again")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "PANGAEA_207974_grain_size_NW_Africa.tsv"
    payload = download_bytes(raw_path, args.overwrite)
    frame = read_pangaea_table(payload)

    lat_col = first_column(list(frame.columns), "latitude")
    lon_col = first_column(list(frame.columns), "longitude")
    frame["_latitude_deg"] = pd.to_numeric(frame[lat_col].str.replace(",", ".", regex=False), errors="coerce")
    frame["_longitude_deg"] = pd.to_numeric(frame[lon_col].str.replace(",", ".", regex=False), errors="coerce")
    valid = frame.dropna(subset=["_latitude_deg", "_longitude_deg"]).copy()
    if valid.empty:
        raise ValueError("No rows with numeric latitude and longitude were parsed.")
    valid["distance_to_Anza_km"] = haversine_km(
        valid["_longitude_deg"].to_numpy(float), valid["_latitude_deg"].to_numpy(float)
    )
    valid = valid.sort_values("distance_to_Anza_km").reset_index(drop=True)

    all_path = OUT_DIR / "PANGAEA_207974_all_samples_with_distance.csv"
    nearest_path = OUT_DIR / "PANGAEA_207974_nearest_samples_to_Anza.csv"
    summary_path = OUT_DIR / "PANGAEA_207974_Anza_spatial_support.json"
    note_path = OUT_DIR / "PANGAEA_207974_interpretation_FR.txt"
    figure_path = OUT_DIR / "Figure_PANGAEA_207974_spatial_support.png"

    valid.to_csv(all_path, index=False, encoding="utf-8-sig")
    valid.head(25).to_csv(nearest_path, index=False, encoding="utf-8-sig")
    distances = valid["distance_to_Anza_km"].to_numpy(float)
    summary = {
        "source_doi": DOI,
        "source_url": URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "site_reference": {"name": "Anza / Agadir", "longitude": SITE_LON, "latitude": SITE_LAT},
        "rows_parsed": int(len(frame)),
        "rows_with_numeric_coordinates": int(len(valid)),
        "nearest_sample_distance_km": float(distances.min()),
        "median_sample_distance_km": float(np.median(distances)),
        "sample_counts_within_km": {str(radius): int(np.count_nonzero(distances <= radius)) for radius in (1, 2, 5, 10, 25, 50, 100)},
        "latitude_column": lat_col,
        "longitude_column": lon_col,
        "all_parsed_columns": list(frame.columns),
        "interpretation": "Regional/offshore context only; not local Anza D50 or validation without matching habitat, sedimentology, depth, and sample position.",
        "license_reported_by_PANGAEA": "CC-BY-3.0",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axes[0].scatter(valid["_longitude_deg"], valid["_latitude_deg"], s=10, color="#3B728F", alpha=0.62,
                    edgecolors="none", label="PANGAEA samples")
    axes[0].scatter([SITE_LON], [SITE_LAT], s=95, color="#D55E00", marker="*", edgecolor="white",
                    linewidth=0.6, zorder=5, label="Anza survey reference")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_title("NW African margin sample locations")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    axes[0].grid(color="#D8E1E8", linewidth=0.55, alpha=0.8)
    axes[1].hist(distances, bins=25, color="#2A788E", edgecolor="white", linewidth=0.5)
    axes[1].axvline(10, color="#D55E00", linestyle="--", linewidth=1.2, label="10 km")
    axes[1].set_xlabel("Distance from Anza reference (km)")
    axes[1].set_ylabel("Number of samples")
    axes[1].set_title(f"Nearest station: {distances.min():.1f} km")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#D8E1E8", linewidth=0.55, alpha=0.8)
    fig.suptitle("Spatial support of open grain-size observations", fontsize=12)
    fig.text(0.5, 0.015, "Regional context only; distance does not establish sedimentological representativeness.",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    note = [
        "Audit of PANGAEA dataset 10.1594/PANGAEA.207974 against the Anza / Agadir survey reference.",
        f"Parsed {len(frame):,} table rows; {len(valid):,} rows had usable coordinates.",
        f"Nearest coordinate is {distances.min():.2f} km from the Anza reference; {summary['sample_counts_within_km']['10']} samples fall within 10 km.",
        "The dataset reports sand, silt, and clay fractions from the NW African continental margin. Its broad regional coverage and older sampling period do not make it a local beach or nearshore D50 measurement.",
        "Use this dataset only as regional sedimentological context unless the nearest sample's depth, seabed setting, and sampling location are shown to match the Anza survey domain.",
        "Do not replace the assumed D50 scenarios in the mobility model with these regional data automatically.",
        f"Source and required citation: Holz, C.; Stuut, J.-B. W.; Henrich, R. (2004), PANGAEA dataset, DOI {DOI}; license CC BY 3.0.",
        "Outputs: PANGAEA_207974_all_samples_with_distance.csv, PANGAEA_207974_nearest_samples_to_Anza.csv, spatial-support JSON, and figure.",
    ]
    note_path.write_text("\n".join(note) + "\n", encoding="utf-8")

    print("Open grain-size spatial-support audit complete; this dataset was not fused with survey or mobility inputs.")
    print(f"Rows parsed: {len(frame):,}; coordinates: {len(valid):,}")
    print(f"Nearest PANGAEA sample: {distances.min():.2f} km from Anza; within 10 km: {summary['sample_counts_within_km']['10']}")
    print(f"Raw data: {raw_path}")
    print(f"Audit outputs: {OUT_DIR}")
    print("Treat as regional context only; do not use as local D50 without a sedimentological match.")


if __name__ == "__main__":
    main()
