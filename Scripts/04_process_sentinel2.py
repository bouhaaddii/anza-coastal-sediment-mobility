from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import reproject, transform_geom


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "Config" / "project_config.json").read_text(encoding="utf-8"))
PRODUCTS = ROOT / "Data" / "Sentinel2" / "Products"
SAFE_DIR = ROOT / "Data" / "Sentinel2" / "SAFE"
OUT_DIR = ROOT / "Data" / "Sentinel2" / "Processed"
EXCLUDED_SCL = {0, 1, 3, 8, 9, 10, 11}
BAND_ID = {"B02": 0, "B03": 1, "B04": 2, "B08": 7}


def extract_zip(path: Path) -> Path:
    safe_name = path.name.removesuffix(".zip")
    destination = SAFE_DIR / f"{safe_name}.SAFE"
    if destination.exists():
        print(f"  ALREADY EXTRACTED; skipped: {destination.name}")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {path.name} ...")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(SAFE_DIR)
    # Product ZIPs generally include a top-level .SAFE directory.
    nested = SAFE_DIR / f"{safe_name}.SAFE"
    return nested if nested.exists() else destination


def product_files(safe: Path) -> tuple[dict[str, Path], Path, Path]:
    granules = list(safe.glob("GRANULE/*"))
    if not granules:
        raise FileNotFoundError(f"No granule folder found in {safe}")
    granule = granules[0]
    img = granule / "IMG_DATA"
    bands = {}
    for code in BAND_ID:
        matches = list(img.glob(f"R10m/*_{code}_10m.jp2"))
        if not matches:
            raise FileNotFoundError(f"Missing {code} 10 m image in {safe}")
        bands[code] = matches[0]
    scl_matches = list(img.glob("R20m/*_SCL_20m.jp2"))
    if not scl_matches:
        raise FileNotFoundError(f"Missing SCL 20 m image in {safe}")
    metadata = list(safe.glob("MTD_MSIL2A.xml"))
    if not metadata:
        raise FileNotFoundError(f"Missing MTD_MSIL2A.xml in {safe}")
    return bands, scl_matches[0], metadata[0]


def read_reflectance_offsets(xml_path: Path) -> tuple[float, dict[int, float]]:
    root = ElementTree.parse(xml_path).getroot()
    quantification = 10000.0
    offsets = {i: 0.0 for i in range(13)}
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag == "BOA_QUANTIFICATION_VALUE" and element.text:
            quantification = float(element.text)
        elif tag == "BOA_ADD_OFFSET" and element.text:
            band_id = element.attrib.get("band_id")
            if band_id is not None:
                offsets[int(band_id)] = float(element.text)
    return quantification, offsets


def process_one(safe: Path) -> None:
    bands, scl_path, xml_path = product_files(safe)
    scene = safe.name.removesuffix(".SAFE")
    out_ref = OUT_DIR / f"{scene}_reflectance_10m.tif"
    out_qa = OUT_DIR / f"{scene}_qa_10m.tif"
    if out_ref.exists() and out_qa.exists():
        print(f"  ALREADY PROCESSED; skipped: {scene}")
        return

    quantification, offsets = read_reflectance_offsets(xml_path)
    bbox = CONFIG["site_bbox_wgs84"]
    polygon_wgs84 = {"type": "Polygon", "coordinates": [[[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                      [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]]}
    with rasterio.open(bands["B02"]) as reference:
        geom = transform_geom("EPSG:4326", reference.crs, polygon_wgs84)
        ref_arr, out_transform = mask(reference, [geom], crop=True, filled=False)
        height, width = ref_arr.shape[1:]
        out_crs = reference.crs
        profile = reference.profile.copy()
    profile.update(driver="GTiff", height=height, width=width, count=4, dtype="float32",
                   transform=out_transform, crs=out_crs, nodata=np.nan,
                   compress="deflate", predictor=3, tiled=True)

    reflectance = np.full((4, height, width), np.nan, dtype=np.float32)
    scl_dst = np.zeros((height, width), dtype=np.uint8)
    for band_index, code in enumerate(("B02", "B03", "B04", "B08"), start=1):
        with rasterio.open(bands[code]) as src:
            destination = np.full((height, width), np.nan, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=destination,
                      src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                      dst_transform=out_transform, dst_crs=out_crs, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            valid = np.isfinite(destination) & (destination != 0)
            destination[valid] = (destination[valid] + offsets[BAND_ID[code]]) / quantification
            destination[~valid] = np.nan
            reflectance[band_index - 1] = destination
    with rasterio.open(scl_path) as src:
        reproject(source=rasterio.band(src, 1), destination=scl_dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=out_transform, dst_crs=out_crs,
                  resampling=Resampling.nearest)

    clear = (~np.isin(scl_dst, list(EXCLUDED_SCL))).astype(np.uint8)
    water = (scl_dst == 6).astype(np.uint8)
    with rasterio.open(out_ref, "w", **profile) as dst:
        dst.write(reflectance)
        dst.set_band_description(1, "B02_blue_surface_reflectance")
        dst.set_band_description(2, "B03_green_surface_reflectance")
        dst.set_band_description(3, "B04_red_surface_reflectance")
        dst.set_band_description(4, "B08_nir_surface_reflectance")
        dst.update_tags(product=scene, source_level="Sentinel-2 L2A", reflectance="(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE",
                        bbox_note=CONFIG["bbox_note"], glint_correction="not applied", turbidity_calibration="not available")
    qa_profile = profile.copy()
    qa_profile.update(count=2, dtype="uint8", nodata=0, predictor=1)
    with rasterio.open(out_qa, "w", **qa_profile) as dst:
        dst.write(clear, 1)
        dst.write(water, 2)
        dst.set_band_description(1, "SCL_clear_nonexcluded_class")
        dst.set_band_description(2, "SCL_class_6_water")
        dst.update_tags(excluded_scl_classes=",".join(map(str, sorted(EXCLUDED_SCL))),
                        water_mask_note="SCL class 6 only; not a verified coastal water polygon")
    print(f"  Wrote {out_ref.name} and {out_qa.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAFE_DIR.mkdir(parents=True, exist_ok=True)
    products = []
    if PRODUCTS.exists():
        for zip_path in sorted(PRODUCTS.glob("*.zip")):
            products.append(extract_zip(zip_path))
        products.extend(sorted(PRODUCTS.glob("*.SAFE")))
    products.extend(sorted(SAFE_DIR.glob("*.SAFE")))
    products = list(dict.fromkeys(path for path in products if path.exists()))
    if not products:
        print(f"No local SAFE or ZIP products found in {PRODUCTS}. Download selected scenes first.")
        return
    print(f"Stage 2: processing {len(products)} local Sentinel-2 L2A product(s); nothing is downloaded here.")
    for safe in products:
        try:
            process_one(safe)
        except Exception as exc:
            print(f"  FAILED {safe.name}: {exc}", file=sys.stderr)
    print(f"Processed outputs are in {OUT_DIR}")
    print("This step applies the SCL cloud/shadow mask and clips to the approximate working box. It does not correct sunglint, foam, or estimate sediment concentration.")


if __name__ == "__main__":
    main()
