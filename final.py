"""
final.py

Pipeline:
  - Run a lightweight clothing detector on `cloth.png` to find the primary item
  - Crop the primary box and remove its background (transparent PNG)
  - Run CLIP-based classification (gender, type, style, fit, material)
  - Extract two dominant colors (primary and secondary)
  - Infer a simple `season` using material heuristics
  - Save a concise wardrobe entry JSON to `color_detection/wardrobe_db.json`

Usage: python final.py --image cloth.png
"""

from __future__ import annotations

import os
import json
import time
import argparse
from typing import Tuple, Optional

from PIL import Image

# Try to import detection and color helpers available in the repo
try:
    from ultralyticsplus import YOLO
except Exception:
    try:
        from ultralytics import YOLO
    except Exception:
        YOLO = None

try:
    from color_detection.color import (
        load_clip_model,
        classify_all,
        extract_pixels_for_kmeans,
        _PALETTE_LAB,
        COLOR_PALETTE,
    )
except Exception:
    load_clip_model = None
    classify_all = None
    extract_pixels_for_kmeans = None
    _PALETTE_LAB = None
    COLOR_PALETTE = None

# sklearn + skimage are used for color clustering/mapping
try:
    from sklearn.cluster import KMeans
    from skimage.color import rgb2lab, lab2rgb
    import numpy as np
except Exception:
    KMeans = None
    rgb2lab = None
    lab2rgb = None
    np = None

# optional background removal
try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None


def load_detector() -> Optional[object]:
    candidates = ["kesimeg/yolov8n-clothing-detection", "yolov8n", "yolov8n.pt"]
    if YOLO is None:
        print("YOLO import not available. Detection will be skipped.")
        return None
    for w in candidates:
        try:
            print("Trying to load model:", w)
            return YOLO(w)
        except Exception as e:
            print(f"Failed to load '{w}': {e}")
    return None


def detect_primary_box(model, image_path: str) -> Optional[Tuple[float, float, float, float, str, float]]:
    """Return (x1, y1, x2, y2, label, conf) for the most confident detection or None."""
    if model is None:
        return None
    try:
        results = model.predict(image_path)
        if len(results) == 0:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
        except Exception:
            xyxy = boxes.xyxy.numpy()
            confs = boxes.conf.numpy()
        # heuristic labels like in detect/det.py based on box geometry
        im = Image.open(image_path).convert("RGBA")
        W, H = im.size
        mapped = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            conf = float(confs[i]) if i < len(confs) else 0.0
            y_center = (y1 + y2) / 2.0
            height = (y2 - y1)
            width = (x2 - x1)
            aspect = height / (width + 1e-6)
            if height < H * 0.08 and width < W * 0.25 and y_center > H * 0.80:
                label = "shoe"
            elif (height < H * 0.12 and y_center < H * 0.35) or (height * width < (W * H) * 0.02):
                label = "accessory"
            else:
                if y_center > H * 0.55:
                    label = "pant"
                elif y_center < H * 0.45:
                    label = "shirt"
                else:
                    label = "pant" if aspect > 1.2 else "shirt"
            mapped.append((x1, y1, x2, y2, label, conf))
        # pick the most confident
        best = max(mapped, key=lambda t: t[5])
        return tuple(best)
    except Exception as e:
        print("Detection failed:", e)
        return None


def crop_and_cutout(image_path: str, box: Tuple[float, float, float, float], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(image_path).convert("RGBA")
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = im.crop((x1, y1, x2, y2))
    crop_path = os.path.join(out_dir, "crop.png")
    crop.save(crop_path)
    # try to remove background to get transparent PNG
    cutout_path = os.path.join(out_dir, "cutout.png")
    if rembg_remove is not None:
        try:
            out = rembg_remove(crop)
            out.save(cutout_path)
            return cutout_path
        except Exception as e:
            print("rembg failed, saving plain crop:", e)
    # fallback: save original crop (may have background)
    crop.save(cutout_path)
    return cutout_path


def map_lab_center_to_name(center_lab: np.ndarray) -> str:
    # Use palette precomputed in color_detection.color (_PALETTE_LAB)
    if _PALETTE_LAB is None:
        # fallback to naive mapping to 'unknown'
        return "unknown"
    def lab_dist(a, b):
        return np.linalg.norm(a - b)
    nearest = min(_PALETTE_LAB.keys(), key=lambda name: lab_dist(center_lab, _PALETTE_LAB[name]))
    return nearest


def dominant_colors(image_path: str, n_colors: int = 1) -> list:
    """Return a list of dominant colors as dicts: {name, hex, proportion}.

    Algorithm improvements:
    - Use KMeans in Lab space with at least 3 clusters for robustness
    - Compute cluster chroma (sqrt(a^2+b^2)) and prefer colorful clusters when present
    - Return hex color and proportion (cluster pixel fraction)
    """
    if extract_pixels_for_kmeans is None or KMeans is None or rgb2lab is None or lab2rgb is None or np is None:
        return [{"name": "unknown", "hex": None, "proportion": 0.0}] * n_colors
    img = Image.open(image_path).convert("RGBA").resize((150, 150))
    pixels = extract_pixels_for_kmeans(img)
    if len(pixels) == 0:
        return [{"name": "unknown", "hex": None, "proportion": 0.0}] * n_colors

    # sample for speed
    max_sample = 15000
    if len(pixels) > max_sample:
        idx = np.random.choice(len(pixels), max_sample, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    rgb01 = pixels_sample / 255.0
    lab = rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)

    # Use at least 3 clusters to allow separating background/shadow/actual color
    n_clusters = min(max(3, n_colors + 1), len(lab))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(lab)
    centers_lab = kmeans.cluster_centers_
    counts = np.bincount(labels)
    proportions = counts / counts.sum()

    # Compute chroma (colorfulness) from Lab a,b channels
    chroma = np.sqrt(centers_lab[:, 1] ** 2 + centers_lab[:, 2] ** 2)

    # If any cluster is colorful, weight by chroma*proportion to prefer colorful clusters
    chroma_threshold = 8.0
    if (chroma > chroma_threshold).any():
        scores = proportions * chroma
    else:
        # otherwise prefer by proportion only (avoid picking tiny colorful specks)
        scores = proportions

    order = np.argsort(scores)[::-1]

    results = []
    for idx in order[:n_colors]:
        center_lab = centers_lab[idx]
        prop = float(proportions[idx])
        # convert Lab center back to RGB (0..1) then to 0..255 ints
        rgb01_center = lab2rgb(center_lab.reshape(1, 1, 3)).reshape(3)
        rgb255 = np.clip((rgb01_center * 255.0).round(), 0, 255).astype(int)
        hexv = "#{:02x}{:02x}{:02x}".format(int(rgb255[0]), int(rgb255[1]), int(rgb255[2]))
        name = map_lab_center_to_name(center_lab)
        results.append({"name": name, "hex": hexv, "proportion": prop})

    while len(results) < n_colors:
        results.append({"name": "unknown", "hex": None, "proportion": 0.0})
    return results

# Backwards compatible wrapper for older code
def two_dominant_colors(image_path: str) -> Tuple[str, str]:
    cols = dominant_colors(image_path, n_colors=2)
    return (cols[0]["name"], cols[1]["name"])


def infer_season(classification: dict) -> str:
    material = classification.get("material", "").lower() if classification else ""
    if not material:
        return "unknown"
    winter = {"wool", "leather", "denim"}
    summer = {"cotton", "linen"}
    if any(m in material for m in winter):
        return "winter"
    if any(m in material for m in summer):
        return "summer"
    return "all-season"


def append_to_db(entry: dict, db_path: str):
    if os.path.isfile(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        except Exception:
            data = []
    else:
        data = []
    data.append(entry)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create a wardrobe DB entry for an image")
    parser.add_argument("--image", "-i", default="cloth.png", help="Input image path (default: cloth.png)")
    # default DB location moved to project root next to this script
    parser.add_argument("--out-db", default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "wardrobe_db.json"), help="Output DB JSON path (default: project root)")
    parser.add_argument("--out-images", default=os.path.join("color_detection", "images"), help="Folder to save cutouts")
    parser.add_argument("--colors", "-c", default=1, type=int, choices=[1,2], help="Number of dominant colors to extract (1 or 2). Default 1")
    args = parser.parse_args()

    # Move or merge existing DB from old location (color_detection/wardrobe_db.json) to new project root location
    try:
        old_db = os.path.join(os.path.abspath(os.path.dirname(__file__)), "color_detection", "wardrobe_db.json")
        new_db = args.out_db
        if os.path.isfile(old_db) and os.path.abspath(old_db) != os.path.abspath(new_db):
            try:
                with open(old_db, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    if not isinstance(old_data, list):
                        old_data = [old_data]
            except Exception:
                old_data = []
            # read new data if exists
            if os.path.isfile(new_db):
                try:
                    with open(new_db, "r", encoding="utf-8") as f:
                        new_data = json.load(f)
                        if not isinstance(new_data, list):
                            new_data = [new_data]
                except Exception:
                    new_data = []
            else:
                new_data = []
            # merge and write
            merged = new_data + old_data
            os.makedirs(os.path.dirname(new_db), exist_ok=True)
            with open(new_db, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            try:
                os.remove(old_db)
            except Exception:
                pass
            print(f"Moved wardrobe DB from {old_db} to {new_db}")
    except Exception as e:
        print("Could not move/merge old DB:", e)

    img_path = args.image
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    detector = load_detector()
    primary = detect_primary_box(detector, img_path)
    if primary is None:
        print("No primary detection — using whole image as crop.")
        im = Image.open(img_path).convert("RGBA")
        W, H = im.size
        box = (0, 0, W, H)
        label = "unknown"
    else:
        x1, y1, x2, y2, label, conf = primary
        box = (x1, y1, x2, y2)
        print(f"Primary detection: {label} conf={conf:.3f} box={box}")

    # create unique folder per run
    run_dir = os.path.join(args.out_images, str(int(time.time())))
    cutout_path = crop_and_cutout(img_path, box, run_dir)

    # Run CLIP classification on the cutout if available
    classification = {}
    if load_clip_model is not None and classify_all is not None:
        try:
            model, preprocess, device = load_clip_model()
            classification = classify_all(cutout_path, {"gender": ["men's clothing", "women's clothing"], "type": ["t-shirt", "shirt", "jacket", "sweater", "hoodie", "kurta", "jeans", "trousers", "shorts", "skirt", "dress"], "style": ["formal wear", "casual wear", "streetwear", "party wear", "gym wear"], "fit": ["slim fit", "regular fit", "oversized"], "material": ["denim", "cotton", "leather", "wool", "polyester"] }, model, preprocess, device, topk=1)
        except Exception as e:
            print("CLIP classification failed:", e)
    else:
        print("CLIP functions not available — skipping attribute classification.")

    # get dominant colors (respecting --colors)
    colors = dominant_colors(cutout_path, n_colors=args.colors)
    primary = colors[0] if len(colors) >= 1 else {"name": "unknown", "hex": None, "proportion": 0.0}
    primary_color = primary.get("name", "unknown")
    primary_hex = primary.get("hex")
    primary_prop = primary.get("proportion", 0.0)
    secondary = colors[1] if args.colors > 1 and len(colors) > 1 else None
    secondary_color = None
    secondary_hex = None
    secondary_prop = 0.0
    if secondary:
        secondary_color = secondary.get("name")
        secondary_hex = secondary.get("hex")
        secondary_prop = secondary.get("proportion", 0.0)

    # choose final item_type prefer CLIP type if available, otherwise the detector label
    item_type = None
    if classification.get("type"):
        item_type = classification.get("type")
    else:
        item_type = label

    # normalize gender
    gender = classification.get("gender") if classification.get("gender") else "unknown"
    if isinstance(gender, str):
        if "men" in gender.lower():
            gender = "men"
        elif "women" in gender.lower():
            gender = "women"

    season = infer_season(classification)

    # image field: file:// path to stored cutout
    image_url = os.path.abspath(cutout_path)
    if os.name == 'nt':
        image_url = "file:///" + image_url.replace('\\', '/')
    else:
        image_url = "file://" + image_url

    entry = {
        "item_type": item_type,
        "gender": gender,
        "primary_color": primary_color,
        "primary_color_hex": primary_hex,
        "primary_color_proportion": primary_prop,
        "season": season,
        "image": image_url,
    }
    if secondary_color:
        entry["secondary_color"] = secondary_color
        entry["secondary_color_hex"] = secondary_hex
        entry["secondary_color_proportion"] = secondary_prop
    # Persist a compact entry and also echo detailed CLIP output to a helper file
    out_db_path = args.out_db
    os.makedirs(os.path.dirname(out_db_path), exist_ok=True)
    append_to_db(entry, out_db_path)

    # Save a per-run detailed JSON for traceability
    detailed_path = os.path.join(run_dir, "detailed_classification.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"entry": entry, "clip_classification": classification}, f, indent=2)

    print("Saved wardrobe entry:")
    print(json.dumps(entry, indent=2))
    print(f"Appended entry to {out_db_path}")


if __name__ == "__main__":
    main()
