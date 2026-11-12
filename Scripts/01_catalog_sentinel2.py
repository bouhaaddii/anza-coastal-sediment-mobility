from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
S2 = CONFIG["sentinel2"]
OUT_DIR = ROOT / "Data" / "Sentinel2"
CATALOG = OUT_DIR / "scene_catalog.csv"
SELECTION = OUT_DIR / "download_selection.csv"
STAC_JSON = OUT_DIR / "scene_catalog_stac.json"


def local_status(product: str) -> str:
    products = OUT_DIR / "Products"
    options = [products / f"{product}.zip", products / f"{product}.SAFE", products / product]
    return "already available" if any(path.exists() for path in options) else "not downloaded"


def get_product_name(item: dict[str, Any]) -> str:
    props = item.get("properties", {})
    private = props.get("_private", {})
    for value in (private.get("product_name"), props.get("s2:product_uri"),
                  props.get("productIdentifier"), props.get("product_uri"),
                  props.get("title")):
        if value:
            return str(value).removesuffix(".SAFE")
    return str(item.get("id", "")).removesuffix(".SAFE")


def get_private(item: dict[str, Any], field: str, default: Any = "") -> Any:
    return item.get("properties", {}).get("_private", {}).get(field, default)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "Products").mkdir(parents=True, exist_ok=True)
    bbox = CONFIG["site_bbox_wgs84"]
    payload: dict[str, Any] = {
        "collections": [S2["collection"]],
        "bbox": bbox,
        "datetime": f"{S2['start_datetime']}/{S2['end_datetime']}",
        "limit": 100
    }
    print("Stage 1: querying the public Copernicus Data Space STAC catalogue.")
    print(f"  Approximate Anza box: {bbox}")
    print(f"  Time range: {payload['datetime']}")
    print("  Cloud limit is 100%; cloudy scenes remain in the catalogue for availability analysis.")
    features: list[dict[str, Any]] = []
    url = S2["stac_url"]
    method = "POST"
    pages = 0
    while url:
        pages += 1
        response = requests.post(url, json=payload, timeout=120) if method == "POST" else requests.get(url, timeout=120)
        response.raise_for_status()
        page = response.json()
        features.extend(page.get("features", []))
        next_link = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
        if next_link:
            url = next_link.get("href")
            method = next_link.get("method", "GET").upper()
            if method == "POST" and isinstance(next_link.get("body"), dict):
                payload = next_link["body"]
        else:
            url = ""
        print(f"  Catalogue pages read: {pages}; scenes so far: {len(features)}")

    rows = []
    for item in features:
        props = item.get("properties", {})
        product = get_product_name(item)
        rows.append({
            "scene_id": item.get("id", ""),
            "product_name": product,
            "product_uuid": get_private(item, "product_uuid"),
            "product_size_bytes": get_private(item, "product_size"),
            "acquisition_utc": props.get("datetime", ""),
            "cloud_cover_pct": props.get("eo:cloud_cover", ""),
            "mgrs_tile": props.get("grid:code", props.get("s2:mgrs_tile", props.get("mgrs:utm_zone", ""))),
            "published_utc": props.get("published", ""),
            "local_status": local_status(product),
            "stac_item_url": item.get("links", [{}])[0].get("href", "") if item.get("links") else ""
        })
    rows.sort(key=lambda r: (r["acquisition_utc"], r["product_name"]))
    fields = ["scene_id", "product_name", "product_uuid", "product_size_bytes", "acquisition_utc",
              "cloud_cover_pct", "mgrs_tile", "published_utc", "local_status", "stac_item_url"]
    with CATALOG.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    STAC_JSON.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")

    # Keep previous user selections when recataloguing; new scenes start unchecked.
    selected: dict[str, str] = {}
    if SELECTION.exists():
        with SELECTION.open(newline="", encoding="utf-8-sig") as stream:
            selected = {r.get("product_name", ""): r.get("download_selected", "NO") for r in csv.DictReader(stream)}
    with SELECTION.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["product_name", "product_uuid", "product_size_bytes",
                                                     "acquisition_utc", "cloud_cover_pct", "mgrs_tile",
                                                     "published_utc", "download_selected"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"product_name": row["product_name"], "product_uuid": row["product_uuid"],
                             "product_size_bytes": row["product_size_bytes"],
                             "acquisition_utc": row["acquisition_utc"],
                             "cloud_cover_pct": row["cloud_cover_pct"],
                             "mgrs_tile": row["mgrs_tile"], "published_utc": row["published_utc"],
                             "download_selected": selected.get(row["product_name"], "NO")})

    ready = sum(row["local_status"] == "already available" for row in rows)
    print(f"Catalogue saved: {CATALOG} ({len(rows)} scenes; {ready} product(s) already available locally).")
    print(f"Selection sheet saved: {SELECTION}")
    print("Open download_selection.csv and change NO to YES only for scenes you want to download.")
    print("No satellite product was downloaded by this cataloguing step.")


if __name__ == "__main__":
    main()
