"""
color.py

Performs fashion attribute classification using OpenAI CLIP and extracts dominant color using KMeans.

Usage:
  1) Install dependencies:
     pip install torch torchvision ftfy regex tqdm scikit-learn scikit-image
     pip install git+https://github.com/openai/CLIP.git

  2) Run:
     python color.py --image shirt_cutout.png

The script will output a JSON wardrobe entry, e.g.:
{
  "type": "jacket",
  "gender": "men's clothing",
  "style": "streetwear",
  "fit": "oversized",
  "material": "denim",
  "color": "black"
}
"""

from __future__ import annotations

import json
import os
import argparse
from typing import Dict, List, Tuple, Any

# Optional imports (provide helpful error messages if missing)
try:
    import clip
    import torch
    from PIL import Image
    import numpy as np
    from sklearn.cluster import KMeans
    from skimage.color import rgb2lab
except Exception as e:  # pragma: no cover - helpful message on missing deps
    raise ImportError(
        "Missing required packages. Install with:\n"
        "pip install torch torchvision ftfy regex tqdm scikit-learn scikit-image\n"
        "pip install git+https://github.com/openai/CLIP.git\n"
        f"Original error: {e}"
    )


# --- VALLUGE Fashion Vocabulary ---
CATEGORIES: Dict[str, List[str]] = {
    "gender": [
        "men's clothing", "women's clothing"
    ],
    "type": [
        "t-shirt", "shirt", "jacket", "sweater", "hoodie", "kurta",
        "jeans", "trousers", "shorts", "skirt", "dress"
    ],
    "style": [
        "formal wear", "casual wear", "streetwear", "party wear", "gym wear"
    ],
    "fit": [
        "slim fit", "regular fit", "oversized"
    ],
    "material": [
        "denim", "cotton", "leather", "wool", "polyester"
    ]
}

# A comprehensive color palette (R,G,B) to map cluster centers to human-friendly names
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "navy": (0, 0, 128),
    "blue": (0, 0, 255),
    "royal_blue": (65, 105, 225),
    "red": (255, 0, 0),
    "maroon": (128, 0, 0),
    "green": (0, 128, 0),
    "olive": (128, 128, 0),
    "yellow": (255, 255, 0),
    "brown": (150, 75, 0),
    "gray": (128, 128, 128),
    "charcoal": (54, 69, 79),
    "beige": (245, 245, 220),
    "cream": (255, 253, 208),
    "pink": (255, 192, 203),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0)
}

# Precompute palette in Lab space for better distance matching
_PALETTE_LAB = None

def _compute_palette_lab():
    global _PALETTE_LAB
    rgb_arr = np.array(list(COLOR_PALETTE.values()), dtype=float) / 255.0
    lab = rgb2lab(rgb_arr.reshape(1, -1, 3)).reshape(-1, 3)
    _PALETTE_LAB = {name: lab[i] for i, name in enumerate(COLOR_PALETTE.keys())}

_compute_palette_lab()


def load_clip_model(device: str = None):
    """Load CLIP model and preprocess pipeline."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def classify(image_tensor: torch.Tensor, labels: List[str], model, device: str, topk: int = 1) -> Any:
    """Return the best label or top-k labels with probabilities from CLIP.

    If topk == 1 returns a single label str, otherwise returns a list of (label, prob) tuples.
    """
    text = clip.tokenize(labels).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().ravel()
    sorted_idx = probs.argsort()[::-1]
    if topk == 1:
        return labels[sorted_idx[0]]
    else:
        top = []
        for idx in sorted_idx[:topk]:
            top.append((labels[idx], float(probs[idx])))
        return top


def classify_all(image_path: str, categories: Dict[str, List[str]], model, preprocess, device: str, topk: int = 1) -> Dict[str, Any]:
    """Classify the image for each category in categories.

    Returns a mapping where values are either single label strings (topk==1) or lists of (label, prob) when topk>1.
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    result: Dict[str, Any] = {}
    for key, labels in categories.items():
        result[key] = classify(image, labels, model, device, topk=topk)
    return result


def _remove_background_by_border(img: Image.Image, tol: int = 30) -> np.ndarray:
    """Estimate background color from image borders and remove pixels close to that color.

    Returns an (N,3) array of RGB pixels not considered background.
    """
    arr = np.array(img.convert('RGB'))
    h, w, _ = arr.shape
    # Sample a 3-pixel border (top, bottom, left, right)
    border_pixels = np.vstack([
        arr[0:3, :, :].reshape(-1, 3),
        arr[h-3:h, :, :].reshape(-1, 3),
        arr[:, 0:3, :].reshape(-1, 3),
        arr[:, w-3:w, :].reshape(-1, 3),
    ])
    bg_color = border_pixels.mean(axis=0)
    # Compute distance from border color
    flat = arr.reshape(-1, 3).astype(float)
    d = np.linalg.norm(flat - bg_color, axis=1)
    mask = d > tol  # keep pixels sufficiently different from border color
    return flat[mask]


def extract_pixels_for_kmeans(img: Image.Image) -> np.ndarray:
    """Return an (N,3) array of RGB pixels, ignoring transparent pixels if present and removing border-like background."""
    # First, remove fully transparent pixels if any
    if img.mode == "RGBA":
        arr = np.array(img)
        alpha = arr[:, :, 3]
        mask = alpha > 0
        rgb = arr[:, :, :3]
        pixels = rgb[mask]
        # If after removing transparency we still have many background-like pixels, further remove border background
        if len(pixels) > 0:
            # Reconstruct an image from masked pixels is hard; instead fallback to border removal from original img
            border_removed = _remove_background_by_border(img)
            if len(border_removed) > 0:
                pixels = border_removed
    else:
        pixels = _remove_background_by_border(img)
    return pixels.astype(float)


def dominant_color_name(image_path: str, n_clusters: int = 2, sample_size: int = 10000) -> str:
    """Run KMeans on the image pixels (in Lab space) and map the dominant cluster center to a named color.

    Uses Lab distance and simple L thresholds to detect black/white extremes.
    """
    img = Image.open(image_path).convert("RGBA").resize((100, 100))
    pixels = extract_pixels_for_kmeans(img)
    if len(pixels) == 0:
        return "unknown"

    # If there are too many pixels, sample for performance
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    # Convert to 0..1 and to Lab
    rgb01 = pixels_sample / 255.0
    lab = rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)

    # KMeans in Lab space
    kmeans = KMeans(n_clusters=min(n_clusters, len(lab)), random_state=42)
    labels = kmeans.fit_predict(lab)
    centers_lab = kmeans.cluster_centers_

    # Choose the dominant cluster by pixel counts
    counts = np.bincount(labels)
    dominant_idx = counts.argmax()
    dominant_center_lab = centers_lab[dominant_idx]

    L = dominant_center_lab[0]
    # Detect black/white by lightness thresholds
    if L < 20:
        return "black"
    if L > 92:
        return "white"

    # Map to nearest palette color in Lab space
    def lab_dist(a, b):
        return np.linalg.norm(a - b)

    nearest_name = min(_PALETTE_LAB.keys(), key=lambda name: lab_dist(dominant_center_lab, _PALETTE_LAB[name]))
    return nearest_name


def make_wardrobe_entry(classification: Dict[str, Any], color: str) -> Dict[str, Any]:
    entry = {}
    # If classification values are lists (topk), pick the top label for the main field but keep the topk as auxiliary data
    for k, v in classification.items():
        if isinstance(v, list) and len(v) > 0:
            # v is a list of (label, prob) tuples
            top_label = v[0][0]
            entry[k] = top_label
            entry[f"{k}_top"] = [{"label": lbl, "prob": p} for lbl, p in v]
        else:
            entry[k] = v
    # Normalize gender
    if "gender" in entry:
        g = entry["gender"]
        if isinstance(g, str):
            if "men" in g.lower():
                entry["gender"] = "men"
            elif "women" in g.lower():
                entry["gender"] = "women"
    entry["color"] = color
    return entry


def main():
    parser = argparse.ArgumentParser(description="Fashion attribute and color extractor using CLIP + KMeans")
    parser.add_argument("--image", "-i", required=True, help="Path to the cutout PNG image (no background)")
    parser.add_argument("--device", "-d", default=None, help="Device: cpu or cuda (default: auto)")
    parser.add_argument("--clusters", "-k", default=2, type=int, help="KMeans number of clusters for colors")
    parser.add_argument("--topk", default=1, type=int, help="Return top-K labels per category (default 1)")
    parser.add_argument("--out", default="wardrobe_entry.json", help="Output JSON file path to save the result")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, preprocess, device = load_clip_model(args.device)

    print(f"Running CLIP classification on {image_path} (device={device}, topk={args.topk})")
    classification = classify_all(image_path, CATEGORIES, model, preprocess, device, topk=args.topk)

    print("Extracting dominant color using improved Lab KMeans...")
    color = dominant_color_name(image_path, n_clusters=args.clusters)

    wardrobe_entry = make_wardrobe_entry(classification, color)

    print(json.dumps(wardrobe_entry, indent=2))

    # Save to file
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wardrobe_entry, f, indent=2)
    print(f"Saved wardrobe entry to {out_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
