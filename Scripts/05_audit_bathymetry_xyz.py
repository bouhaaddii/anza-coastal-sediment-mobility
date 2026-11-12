from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points.txt"
OUT = ROOT / "Data" / "Local_Bathymetry" / "Anza_Agadir" / "Anza_Agadir_bathy_points_normalized.csv"
REPORT = ROOT / "Results" / "bathymetry_xyz_audit.txt"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Survey point file not found: {SOURCE}")
    records = []
    malformed = []
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        for line_no, line in enumerate(stream, start=1):
            parts = line.strip().replace(",", ".").split()
            if len(parts) < 4:
                malformed.append((line_no, line.rstrip()))
                continue
            try:
                records.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                malformed.append((line_no, line.rstrip()))
    if not records:
        raise ValueError("No valid ID/X/Y/Z rows were parsed; the input format may differ from expected tab-separated points.")
    duplicate_xy = sum(count - 1 for count in Counter((r[1], r[2]) for r in records).values() if count > 1)
    xs, ys, zs = [r[1] for r in records], [r[2] for r in records], [r[3] for r in records]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_id", "x_source_m", "y_source_m", "z_source_m"])
        writer.writerows(records)
    lines = [
        "Anza bathymetry point audit",
        f"Source file: {SOURCE}",
        f"Normalized copy: {OUT}",
        f"Records parsed: {len(records)}",
        f"Malformed rows: {len(malformed)}",
        f"Repeated XY rows after the first occurrence: {duplicate_xy}",
        f"X min/max (source units): {min(xs):.3f} / {max(xs):.3f}",
        f"Y min/max (source units): {min(ys):.3f} / {max(ys):.3f}",
        f"Z min/max (m as documented): {min(zs):.3f} / {max(zs):.3f}",
        f"Z median (m): {statistics.median(zs):.3f}",
        "Horizontal coordinate system: not transformed; confirm the survey projection/EPSG and axis convention from the plan before mapping.",
        "Vertical reference: hydrographic zero of the Port of Agadir, a local reference. No absolute datum conversion was performed.",
        "No outliers were deleted; no gridding, interpolation, vertical adjustment, mesh generation, or hydrodynamic simulation was performed.",
    ]
    if malformed:
        lines.append("First malformed rows: " + repr(malformed[:5]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Bathymetry audit complete; raw point file was not changed.")
    print(f"  Points parsed: {len(records):,}; repeated XY rows: {duplicate_xy:,}; malformed rows: {len(malformed):,}")
    print(f"  Z range as supplied: {min(zs):.3f} to {max(zs):.3f} m")
    print(f"  Report: {REPORT}")
    print("  No coordinate or vertical datum transformation and no mesh were generated.")


if __name__ == "__main__":
    main()
