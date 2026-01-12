# Wardrobe — Clothing Detection & Color/Attribute Extraction 🔎👗

**Wardrobe** is a small toolkit for detecting and extracting clothing items and their visual attributes from images. It uses a lightweight YOLO-based clothes detector and an OpenAI CLIP + KMeans pipeline to infer clothing attributes (type, gender, style, fit, material) and dominant colors, then saves a compact wardrobe JSON entry.

---

## Features ✅

- Detects the primary clothing item in an image (YOLO-based) and crops it. 🔲
- Removes background (optional, via `rembg`) and extracts dominant colors using KMeans in Lab space. 🎨
- Infers clothing attributes using CLIP zero-shot classification (gender, type, style, fit, material). 🧠
- Produces compact JSON wardrobe entries and appends them to a simple DB (`wardrobe_db.json`). 📦

---

## Quick start — Setup 🔧

Recommended: use a virtual environment.

Windows PowerShell example:

```powershell
# create + activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install core deps
pip install -r requirements.txt  # optional if you create one, see below
```

Direct installs (core commands referenced in the code):

```bash
pip install torch torchvision ftfy regex tqdm scikit-learn scikit-image
pip install git+https://github.com/openai/CLIP.git
# Optional helpers (detection, background removal)
pip install ultralytics rembg pillow
```

Notes:
- CLIP is installed via Git URL in the code comments (openai/CLIP).
- Detection uses `ultralytics` (YOLO). The repository bundles a `yolov8n.pt` fallback model.
- `rembg` is optional and used to remove backgrounds when available.

---

## Usage — scripts 🔁

1) Extract attributes and dominant color from a prepared cutout (transparent PNG):

```bash
python color_detection/color.py --image path/to/cutout.png --out wardrobe_entry.json
```

- This script runs CLIP + KMeans and writes a `wardrobe_entry.json` file with fields such as `type`, `gender`, `style`, `fit`, `material`, `color`.

2) Full pipeline (detect, cutout, classify, color, save DB):

```bash
python final.py --image path/to/image.png
```

- `final.py` will try to run a YOLO detector to find the primary item, crop and optionally remove background, run CLIP classification and color extraction, infer a `season` from material heuristics, and append a compact entry to `wardrobe_db.json`.
- Use `--colors 2` to extract two dominant colors.

3) Run the detector (standalone):

```bash
python detect/det.py
```

- `det.py` uses `ultralytics` to load a preferred clothing detection checkpoint and will save an annotated render if successful.

---

## Tests ✅

A small test suite exists for color detection.

Install pytest and run:

```bash
pip install pytest
pytest -q
```

---

## Notes & Tips 💡

- If models fail to load, check your network and optional model files (the repo includes `yolov8n.pt` as a fallback). Consider using Git LFS for large model weights when collaborating.
- CLIP model selection and device (CPU vs GPU) are auto-selected; pass `--device cuda` to force GPU usage if available.
- If you want reproducible color extraction, use the `--clusters` option in `color.py` or `--colors` in `final.py`.

---

## Example output

A `final.py` run produces a compact entry like:

```json
{
  "item_type": "jacket",
  "gender": "men",
  "primary_color": "black",
  "primary_color_hex": "#000000",
  "primary_color_proportion": 0.86,
  "season": "winter",
  "image": "file:///abs/path/to/color_detection/images/169.../cutout.png"
}
```

---

## Contributing & License

Feel free to open issues or PRs. There's no explicit license file in the repo; add one if you plan to share widely.

---

If you'd like, I can also add a `requirements.txt` or GitHub Actions workflow to run tests on push. 🔧