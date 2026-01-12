"""
Simple script to extract primary and secondary clothing colors using K-Means.
Reads `imgg.png` from the current folder by default and prints a dict like:
{ "primary_color": "navy", "secondary_color": "beige" }

Dependencies: numpy, opencv-python (cv2), scikit-learn
"""
from __future__ import annotations

import json
import os
from typing import Tuple, List

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Optional/perceptual naming dependencies (used only for CSS/fuzzy naming)
try:
    from skimage import color as skcolor
except Exception:
    skcolor = None

try:
    import webcolors
except Exception:
    webcolors = None

import colorsys

# A small palette of common fashion color names with representative RGB values
PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "navy": (0, 0, 128),
    "beige": (245, 245, 220),
    "cream": (255, 253, 208),
    "red": (220, 30, 30),
    "maroon": (128, 0, 0),
    "blue": (30, 120, 200),
    "sky": (135, 206, 235),
    "teal": (0, 128, 128),
    "green": (50, 160, 80),
    "olive": (128, 128, 0),
    "brown": (150, 75, 0),
    "tan": (210, 180, 140),
    "gray": (128, 128, 128),
    "charcoal": (54, 69, 79),
    "pink": (255, 192, 203),
    "purple": (128, 0, 128),
    "magenta": (255, 0, 255),
    "yellow": (240, 220, 0),
    "coral": (255, 127, 80),
    # additional fashion-focused names
    "khaki": (195, 176, 145),
    "mustard": (199, 163, 31),
    "burgundy": (128, 0, 32),
    "indigo": (75, 0, 130),
    "mint": (152, 255, 152),
    "lavender": (230, 230, 250),
    "olive_drab": (107, 142, 35),
}

# Extra fashion palette to augment CSS names if desired (name -> hex)
EXTRA_FASHION_COLORS = {
    "khaki": "#C3B091",
    "mustard": "#C7A31F",
    "burgundy": "#800020",
    "indigo": "#4B0082",
    "mint": "#98FF98",
    "lavender": "#E6E6FA",
    "olive_drab": "#6B8E23",
}


def load_image_pixels(path: str, remove_background: bool = True) -> np.ndarray:
    """Load an image and return an (N,3) array of RGB pixels.

    Handles PNG transparency by removing fully transparent pixels. If remove_background=True,
    the function samples the four corners to detect a uniform background color and removes
    pixels close to that background in Lab space (deltaE < 12) to avoid background bias.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    # Convert BGR(A) to RGB(A)
    if img.shape[2] == 4:
        # keep only non-transparent pixels
        alpha = img[:, :, 3]
        mask = alpha > 0
        rgb = img[:, :, :3]
        pixels = rgb[mask]
        img_rgb = rgb
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
        pixels = img_rgb.reshape((-1, 3))

    if pixels.size == 0:
        raise ValueError("No visible pixels found in the image.")

    if remove_background and skcolor is not None:
        # sample corners (small patch) to estimate background color
        h, w = img_rgb.shape[:2]
        patch = img_rgb
        corners = np.array([
            patch[0:10, 0:10].reshape(-1, 3).mean(axis=0),
            patch[0:10, max(0, w-10):w].reshape(-1, 3).mean(axis=0),
            patch[max(0, h-10):h, 0:10].reshape(-1, 3).mean(axis=0),
            patch[max(0, h-10):h, max(0, w-10):w].reshape(-1, 3).mean(axis=0),
        ])
        # pick the most similar corner as background estimate
        bg_rgb = np.round(corners.mean(axis=0)).astype(int)

        # convert pixels and bg to Lab and remove close pixels
        try:
            pixel_lab = skcolor.rgb2lab((pixels.reshape(-1, 3) / 255.0).reshape(-1, 3))
            bg_lab = skcolor.rgb2lab(np.array([[[bg_rgb[0]/255.0, bg_rgb[1]/255.0, bg_rgb[2]/255.0]]]))[0,0]
            dists = np.linalg.norm(pixel_lab - bg_lab, axis=1)
            mask = dists > 12.0  # threshold (deltaE)
            if mask.sum() == 0:
                # fallback: keep original pixels
                pass
            else:
                pixels = pixels[mask]
        except Exception:
            # if Lab conversion failed, skip background removal
            pass

    return pixels.astype(np.float32)


def run_kmeans(pixels: np.ndarray, n_clusters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Run KMeans on the pixels and return (centers, labels).

    centers are shape (n_clusters, 3) in RGB (float), labels length N.
    """
    if len(pixels) < n_clusters:
        raise ValueError("Not enough pixels to form the requested number of clusters")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_, kmeans.labels_


def rgb_to_name(rgb: Tuple[float, float, float], palette: dict = PALETTE, return_distance: bool = False) -> str | Tuple[str, float]:
    """Map an RGB triple to the nearest named color in the palette using perceptual (Lab) distance when available.

    If scikit-image is available, this computes Lab distances for better perceptual matching. Returns (name, distance) when return_distance=True where distance is the Euclidean distance in Lab space (deltaE-like) if available or RGB euclidean otherwise.
    """
    r, g, b = [float(x) for x in rgb]
    best_name = None
    best_dist = float("inf")

    if skcolor is not None:
        # precompute palette Lab values
        names = []
        rgbs = []
        for n, (pr, pg, pb) in palette.items():
            names.append(n)
            rgbs.append([pr/255.0, pg/255.0, pb/255.0])
        lab_palette = skcolor.rgb2lab(np.array(rgbs))
        lab_target = skcolor.rgb2lab(np.array([[[r/255.0, g/255.0, b/255.0]]]))[0,0]
        for i, name in enumerate(names):
            d = float(np.linalg.norm(lab_target - lab_palette[i]))
            if d < best_dist:
                best_dist = d
                best_name = name
    else:
        best_d2 = float("inf")
        for name, (pr, pg, pb) in palette.items():
            d2 = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_name = name
        best_dist = float(np.sqrt(best_d2))

    if return_distance:
        return best_name or "neutral", best_dist
    return best_name or "neutral"


# ----------------- CSS / perceptual naming helpers -----------------

def _build_css_lab_palette():
    """Build a cache of CSS3 names -> (rgb tuple, lab array)."""
    if webcolors is None or skcolor is None:
        raise ImportError("CSS naming requires 'webcolors' and 'scikit-image' packages. Install with: pip install webcolors scikit-image")
    css_rgb = {}
    css_lab = {}

    # webcolors API varies by version; try common mappings
    names_to_hex = None
    if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        names_to_hex = webcolors.CSS3_NAMES_TO_HEX
    elif hasattr(webcolors, "css3_names_to_hex"):
        names_to_hex = webcolors.css3_names_to_hex
    elif hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
        # invert hex->name mapping
        inv = webcolors.CSS3_HEX_TO_NAMES
        names_to_hex = {v: k for k, v in inv.items()}
    else:
        # Fallback: try to build using available functions (slow)
        # webcolors has a large list of named colors we can iterate through
        try:
            # iterate a subset of common names
            fallback_names = ["white", "black", "red", "green", "blue", "navy", "beige", "gray", "pink", "purple", "yellow", "brown"]
            names_to_hex = {n: webcolors.name_to_hex(n) for n in fallback_names}
        except Exception as e:
            raise ImportError("Unable to locate CSS3 name mapping in webcolors. Upgrade/install webcolors.")

    for name, hexval in names_to_hex.items():
        # hexval like '#rrggbb'
        r = int(hexval[1:3], 16)
        g = int(hexval[3:5], 16)
        b = int(hexval[5:7], 16)
        css_rgb[name] = (r, g, b)
    # convert to Lab using skimage (expects RGB in [0,1])
    names = []
    rgbs = []
    for n, (r, g, b) in css_rgb.items():
        names.append(n)
        rgbs.append([r / 255.0, g / 255.0, b / 255.0])
    lab = skcolor.rgb2lab(np.array(rgbs))
    for i, n in enumerate(names):
        css_lab[n] = {
            "rgb": css_rgb[n],
            "lab": lab[i],
            "hex": names_to_hex[n],
        }
    return css_lab


_CSS_LAB_CACHE = None

def css_nearest_name(rgb: Tuple[float, float, float]):
    """Return nearest CSS3 name and deltaE distance (CIE76) for given RGB triple."""
    global _CSS_LAB_CACHE
    if _CSS_LAB_CACHE is None:
        _CSS_LAB_CACHE = _build_css_lab_palette()

    if skcolor is None:
        raise ImportError("scikit-image is required for CSS/perceptual naming. Install with: pip install scikit-image")

    r, g, b = rgb
    lab = skcolor.rgb2lab(np.array([[[r / 255.0, g / 255.0, b / 255.0]]]))[0, 0]
    best_name = None
    best_d = float("inf")
    for name, data in _CSS_LAB_CACHE.items():
        d = np.linalg.norm(lab - data["lab"])
        if d < best_d:
            best_d = float(d)
            best_name = name
            best_hex = data["hex"]
            best_rgb = data["rgb"]
    return {"name": best_name, "hex": best_hex, "rgb": best_rgb, "deltaE": best_d}


def fuzzy_buckets(rgb: Tuple[float, float, float]) -> List[str]:
    """Return a list of fuzzy attributes like light/dark and muted/vibrant.

    Tuned thresholds (empirically chosen):
      - light if L* >= 65, dark if L* <= 35
      - muted if chroma < 20 or sat < 0.28
      - vibrant if chroma > 35 or sat > 0.55
    """
    # Use Lab L* for light/dark and chroma / HSV saturation for muted/vibrant
    r, g, b = rgb
    # Lab
    if skcolor is None:
        raise ImportError("scikit-image is required for fuzzy buckets. Install with: pip install scikit-image")
    lab = skcolor.rgb2lab(np.array([[[r / 255.0, g / 255.0, b / 255.0]]]))[0, 0]
    L, a, b_ = lab
    chroma = float(np.sqrt(a * a + b_ * b_))

    # HSV saturation
    hsv = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    sat = hsv[1]

    attrs = []
    if L >= 65:
        attrs.append("light")
    elif L <= 35:
        attrs.append("dark")
    else:
        attrs.append("medium")

    if chroma < 20 or sat < 0.28:
        attrs.append("muted")
    elif chroma > 35 or sat > 0.55:
        attrs.append("vibrant")
    else:
        attrs.append("balanced")

    # Extra hint: if very low saturation, mark 'near-neutral'
    if sat < 0.08:
        attrs.append("near-neutral")

    return attrs


def extract_primary_secondary(image_path: str = "imgg.png", naming: str = "css") -> dict:
    """Extract primary and secondary color names and numeric centers from the image.

    naming: 'simple' = basic palette, 'css' = nearest CSS3 name (perceptual),
            'fuzzy' = CSS name + fuzzy buckets (light/dark/muted/vibrant)

    Returns a dict like:
      {
        "primary_color": "navy",             # legacy/simple name (if available)
        "secondary_color": "beige",
        "primary_rgb": [r,g,b],
        "secondary_rgb": [r,g,b],
        "primary_distance": 12.3,             # distance to simple palette name
        "secondary_distance": 45.1,
        "primary_css": {...},                 # present when naming in ['css','fuzzy']
        "secondary_css": {...},
        "primary_fuzzy": [...],               # present for 'fuzzy'
      }
    """
    pixels = load_image_pixels(image_path)
    centers, labels = run_kmeans(pixels, n_clusters=2)

    # count labels to decide dominant and secondary
    unique, counts = np.unique(labels, return_counts=True)
    # sort by count descending
    order = unique[np.argsort(-counts)]

    dominant = np.round(centers[order[0]]).astype(int)
    secondary = np.round(centers[order[1]]).astype(int)

    primary_name, primary_dist = rgb_to_name(tuple(dominant.tolist()), return_distance=True)
    secondary_name, secondary_dist = rgb_to_name(tuple(secondary.tolist()), return_distance=True)

    out = {
        "primary_color": primary_name,
        "secondary_color": secondary_name,
        "primary_rgb": dominant.tolist(),
        "secondary_rgb": secondary.tolist(),
        "primary_distance": primary_dist,
        "secondary_distance": secondary_dist,
    }

    if naming in ("css", "fuzzy"):
        try:
            # merge EXTRA_FASHION_COLORS into names_to_hex if available
            global _CSS_LAB_CACHE
            if _CSS_LAB_CACHE is None:
                # build palette and then inject extras
                _CSS_LAB_CACHE = _build_css_lab_palette()
                # inject extra fashion colors
                for n, hexv in EXTRA_FASHION_COLORS.items():
                    if n not in _CSS_LAB_CACHE:
                        r = int(hexv[1:3], 16)
                        g = int(hexv[3:5], 16)
                        b = int(hexv[5:7], 16)
                        labv = skcolor.rgb2lab(np.array([[[r / 255.0, g / 255.0, b / 255.0]]]))[0, 0]
                        _CSS_LAB_CACHE[n] = {"rgb": (r, g, b), "lab": labv, "hex": hexv}

            out["primary_css"] = css_nearest_name(tuple(dominant.tolist()))
            out["secondary_css"] = css_nearest_name(tuple(secondary.tolist()))
        except Exception as exc:
            out["css_error"] = str(exc)

    if naming == "fuzzy":
        try:
            out["primary_fuzzy"] = fuzzy_buckets(tuple(dominant.tolist()))
            out["secondary_fuzzy"] = fuzzy_buckets(tuple(secondary.tolist()))
        except Exception as exc:
            out["fuzzy_error"] = str(exc)

    return out


if __name__ == "__main__":
    import argparse
    import csv
    import glob
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Extract primary and secondary clothing colors from an image or folder.")
    parser.add_argument("image", nargs="?", default=None, help="Path to the image (default: none). Use --input-dir to process a folder.")
    parser.add_argument("--input-dir", default=None, help="Process all images in a folder (jpg/png).")
    parser.add_argument("--output-csv", default=None, help="Write batch results to CSV file (when --input-dir is used).")
    parser.add_argument("--naming", choices=["simple", "css", "fuzzy"], default="css", help="Naming strategy: simple (basic palette), css (nearest CSS3 name, perceptual), fuzzy (css + fuzzy buckets)")
    args = parser.parse_args()

    def process_image(path: str, naming: str) -> dict:
        data = extract_primary_secondary(path, naming=naming)
        data["image"] = path
        return data

    try:
        if args.input_dir:
            pattern = str(Path(args.input_dir) / "**" / "*")
            files = [f for f in glob.glob(pattern, recursive=True) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            results = []
            for f in sorted(files):
                try:
                    results.append(process_image(f, args.naming))
                except Exception as e:
                    results.append({"image": f, "error": str(e)})

            if args.output_csv:
                # Flatten results and write CSV
                fieldnames = [
                    "image",
                    "primary_color",
                    "secondary_color",
                    "primary_rgb",
                    "secondary_rgb",
                    "primary_distance",
                    "secondary_distance",
                    "primary_css_name",
                    "primary_css_hex",
                    "primary_css_deltaE",
                    "primary_fuzzy",
                ]
                with open(args.output_csv, "w", newline='', encoding='utf-8') as csvf:
                    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in results:
                        row = {
                            "image": r.get("image"),
                            "primary_color": r.get("primary_color"),
                            "secondary_color": r.get("secondary_color"),
                            "primary_rgb": ",".join(map(str, r.get("primary_rgb", []))) if r.get("primary_rgb") else "",
                            "secondary_rgb": ",".join(map(str, r.get("secondary_rgb", []))) if r.get("secondary_rgb") else "",
                            "primary_distance": r.get("primary_distance"),
                            "secondary_distance": r.get("secondary_distance"),
                            "primary_css_name": (r.get("primary_css") or {}).get("name"),
                            "primary_css_hex": (r.get("primary_css") or {}).get("hex"),
                            "primary_css_deltaE": (r.get("primary_css") or {}).get("deltaE"),
                            "primary_fuzzy": ";".join(r.get("primary_fuzzy", [])) if r.get("primary_fuzzy") else "",
                        }
                        writer.writerow(row)
                print(json.dumps({"processed": len(results), "csv": args.output_csv}))
            else:
                print(json.dumps(results))
        else:
            if args.image is None:
                print(json.dumps({"error": "No image provided. Use positional image or --input-dir."}))
                sys.exit(2)
            result = process_image(args.image, args.naming)
            print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
