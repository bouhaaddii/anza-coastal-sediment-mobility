from __future__ import annotations

import argparse
import csv
import getpass
import json
import sys
import zipfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SELECTION = ROOT / "Data" / "Sentinel2" / "download_selection.csv"
STAC_JSON = ROOT / "Data" / "Sentinel2" / "scene_catalog_stac.json"
PRODUCTS = ROOT / "Data" / "Sentinel2" / "Products"


def ask_token() -> str:
    print("Copernicus Data Space account required for product downloads.")
    print("Credentials are entered only here, kept in memory for this run, and never written to disk.")
    username = input("CDSE username/email: ").strip()
    password = getpass.getpass("CDSE password (input hidden): ")
    response = requests.post(TOKEN_URL, data={"client_id": "cdse-public", "username": username,
                                              "password": password, "grant_type": "password"}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"CDSE token request failed ({response.status_code}). Check account access and try again.")
    return response.json()["access_token"]


def stac_product_index() -> dict[str, dict]:
    if not STAC_JSON.is_file():
        return {}
    payload = json.loads(STAC_JSON.read_text(encoding="utf-8"))
    index = {}
    for item in payload.get("features", []):
        private = item.get("properties", {}).get("_private", {})
        name = (private.get("product_name") or item.get("id", "")).removesuffix(".SAFE")
        if name:
            index[name] = {
                "product_uuid": private.get("product_uuid", ""),
                "product_size_bytes": private.get("product_size", ""),
                "mgrs_tile": item.get("properties", {}).get("grid:code", ""),
                "published_utc": item.get("properties", {}).get("published", ""),
                "acquisition_utc": item.get("properties", {}).get("datetime", ""),
            }
    return index


def find_product(product_name: str, token: str, product_uuid: str = "") -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    if product_uuid:
        # STAC returns the stable CDSE product UUID. Resolve that UUID directly;
        # the OData Name field includes the .SAFE suffix omitted in selection CSVs.
        response = requests.get(f"{BASE}/Products({product_uuid})",
                                params={"$select": "Id,Name,ContentLength,Online"},
                                headers=headers, timeout=60)
        if response.status_code == 200:
            product = response.json()
            expected = product_name.removesuffix(".SAFE")
            actual = str(product.get("Name", "")).removesuffix(".SAFE")
            if actual != expected:
                raise RuntimeError(f"STAC/OData product-name mismatch for UUID {product_uuid}: {actual}")
            return product
        if response.status_code not in {404, 410}:
            response.raise_for_status()

    # Backward-compatible fallback for older selection CSVs without a UUID.
    safe_name = product_name if product_name.endswith(".SAFE") else product_name + ".SAFE"
    escaped = safe_name.replace("'", "''")
    response = requests.get(f"{BASE}/Products", params={"$filter": f"Name eq '{escaped}'",
                                                         "$select": "Id,Name,ContentLength,Online"},
                            headers=headers, timeout=60)
    response.raise_for_status()
    matches = response.json().get("value", [])
    if not matches:
        raise RuntimeError(f"No CDSE product matched {safe_name}; rerun Scripts/01_catalog_sentinel2.py to refresh UUIDs.")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only Sentinel-2 scenes marked YES in the selection CSV.")
    parser.add_argument("--selection-file", type=Path, default=SELECTION)
    parser.add_argument("--max-scenes", type=int, default=5, help="Safety cap; default downloads at most five selected scenes.")
    parser.add_argument("--allow-large-selection", action="store_true",
                        help="Allow more than 20 unique acquisition/tile selections after checking the reported size.")
    args = parser.parse_args()
    if not args.selection_file.is_file():
        raise FileNotFoundError(f"Selection file not found: {args.selection_file}. Run 01_catalog_sentinel2.py first.")
    PRODUCTS.mkdir(parents=True, exist_ok=True)
    stac_index = stac_product_index()
    with args.selection_file.open(newline="", encoding="utf-8-sig") as stream:
        chosen = [row for row in csv.DictReader(stream) if row.get("download_selected", "").strip().upper() in {"YES", "Y", "1", "TRUE"}]
    if not chosen:
        print(f"No scene is marked YES in {args.selection_file}. Nothing downloaded.")
        return
    # Keep the latest published processing of each physical acquisition/tile.
    # STAC can expose more than one reprocessed SAFE for the same overpass.
    latest_by_acquisition = {}
    duplicates = []
    for row in chosen:
        name = row["product_name"].removesuffix(".SAFE")
        metadata = stac_index.get(name, {})
        for field in ("product_uuid", "product_size_bytes", "mgrs_tile", "published_utc", "acquisition_utc"):
            if not row.get(field):
                row[field] = metadata.get(field, "")
        acquisition = row.get("acquisition_utc", "").strip()
        tile = row.get("mgrs_tile", "").strip()
        key = (acquisition, tile) if acquisition and tile else (name,)
        current = latest_by_acquisition.get(key)
        if current is None:
            latest_by_acquisition[key] = row
        else:
            current_published = current.get("published_utc", "")
            new_published = row.get("published_utc", "")
            if new_published > current_published:
                duplicates.append(current["product_name"])
                latest_by_acquisition[key] = row
            else:
                duplicates.append(name)
    for skipped in duplicates:
        print(f"DUPLICATE ACQUISITION; skipped older/duplicate processing: {skipped}")

    if len(latest_by_acquisition) > 20 and not args.allow_large_selection:
        estimated_bytes = sum(
            int(row.get("product_size_bytes") or stac_index.get(row["product_name"].removesuffix(".SAFE"), {}).get("product_size_bytes") or 0)
            for row in latest_by_acquisition.values()
        )
        print(f"SAFETY STOP: {len(chosen)} rows are YES, representing {len(latest_by_acquisition)} unique acquisition/tile products.")
        if estimated_bytes:
            print(f"Estimated total archive size: {estimated_bytes / 1e9:.1f} GB (catalog estimates; not downloaded).")
        print("No product was downloaded. Set NO for unneeded rows, or add --allow-large-selection only after reviewing the volume.")
        return

    pending = []
    for row in latest_by_acquisition.values():
        name = row["product_name"].removesuffix(".SAFE")
        if (PRODUCTS / f"{name}.zip").exists() or (PRODUCTS / f"{name}.SAFE").exists():
            print(f"ALREADY AVAILABLE; skipped: {name}")
        else:
            pending.append((name, row))
    pending = pending[: max(0, args.max_scenes)]
    if not pending:
        print("All selected scenes are already available locally.")
        return
    print(f"Selected scenes still missing: {len(pending)} (safety cap {args.max_scenes}).")
    for name, row in pending:
        cloud = row.get("cloud_cover_pct", "")
        warning = " [CLOUDY: inspect water pixels]" if cloud and float(cloud) >= 30 else ""
        print(f"  {name} | {row.get('acquisition_utc','')} | tile={row.get('mgrs_tile','')} | cloud={cloud}%{warning}")
    if any(not row.get("product_uuid") and name not in stac_index for name, row in pending):
        print("WARNING: at least one selection has no product UUID in the local STAC catalog; will try exact .SAFE OData-name lookup.")
    if input("Type DOWNLOAD to continue: ").strip() != "DOWNLOAD":
        print("Cancelled by user; no product downloaded.")
        return

    token = ask_token()
    for index, (name, row) in enumerate(pending, start=1):
        existing_zip = PRODUCTS / f"{name}.zip"
        if existing_zip.exists():
            print(f"ALREADY AVAILABLE; skipped: {existing_zip.name}")
            continue
        product_uuid = row.get("product_uuid") or stac_index.get(name, {}).get("product_uuid", "")
        product = find_product(name, token, product_uuid)
        if not product.get("Online", True):
            print(f"NOT ONLINE; skipped: {product.get('Name', name)}")
            continue
        product_id = product["Id"]
        size = int(product.get("ContentLength") or 0)
        output = PRODUCTS / f"{name}.zip"
        partial = output.with_suffix(output.suffix + ".part")
        print(f"Downloading {index}/{len(pending)}: {name} ({size / 1e6:.1f} MB)")
        url = f"{DOWNLOAD_BASE}/Products({product_id})/$value"
        with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=(60, 180)) as response:
            response.raise_for_status()
            total = 0
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        total += len(chunk)
                        print(f"  received {total / 1e6:.1f} MB", end="\r", flush=True)
        if size and total != size:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Size mismatch for {name}; incomplete file removed.")
        try:
            with zipfile.ZipFile(partial) as archive:
                bad_member = archive.testzip()
        except zipfile.BadZipFile as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded product is not a valid ZIP for {name}; incomplete file removed.") from exc
        if bad_member:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded ZIP failed CRC validation at {bad_member}; incomplete file removed.")
        partial.replace(output)
        print(f"\nSaved: {output} ({total:,} bytes)")
    print("Selected downloads complete. Credentials were not saved by this script.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
