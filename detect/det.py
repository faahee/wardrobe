from ultralyticsplus import YOLO, render_result
import os
import sys
import traceback
from PIL import Image, ImageDraw, ImageFont
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "pant.png")
preferred_model = "kesimeg/yolov8n-clothing-detection"
fallback_model = "yolov8n.pt"

# Try to register common classes as safe globals with torch (PyTorch >=2.6) to allow loading checkpoints.
try:
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    from torch.nn.modules.module import Module
    # Try to include ultralytics-specific classes required for unpickling
    safe_list = [DetectionModel, Sequential, Module]
    try:
        import ultralytics.nn.modules as unmod
        extra = []
        for name in dir(unmod):
            if not name.startswith('_'):
                obj = getattr(unmod, name)
                # add classes (types) — functions/modules shouldn't be necessary
                if isinstance(obj, type):
                    extra.append(obj)
        if extra:
            safe_list.extend(extra)
    except Exception:
        # If we can't import/inspect ultralytics modules, continue with basic safe_list
        pass

    # NOTE: Some checkpoints require allowing non-safe globals when unpickling. In PyTorch >=2.6 the
    # default of torch.load uses weights_only=True which restricts globals. To support older checkpoints
    # we can set weights_only=False by default when torch.load is called. This can execute arbitrary
    # code from the checkpoint, so only enable it if you trust the source of the weights.
    try:
        _orig_torch_load = torch.load
        def _torch_load_weights_ok(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _orig_torch_load(*args, **kwargs)
        torch.load = _torch_load_weights_ok
        print('Patched torch.load to default to weights_only=False (unsafe — only do this for trusted checkpoints).')
    except Exception as e:
        print('Could not monkeypatch torch.load:', e)
    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals(safe_list)
            print("Added safe globals:", [c.__name__ for c in safe_list])
        except Exception as e:
            print("Warning: failed to add safe globals:", e)
    else:
        print("torch.serialization.add_safe_globals not available in this torch build.")
except Exception as e:
    print("Could not import torch/DetectionModel for safe globals:", e)

# Try preferred model and a set of fallbacks
candidates = [preferred_model, "yolov8n", "yolov8n.pt"]
model = None
for w in candidates:
    try:
        print("Trying to load model:", w)
        model = YOLO(w)
        print("Loaded model:", w)
        break
    except Exception as e:
        print(f"Failed to load '{w}': {e}")
        traceback.print_exc()
if model is None:
    print("No models could be loaded. See errors above.", file=sys.stderr)
    sys.exit(1)

try:
    print("Running prediction on:", img_path)
    results = model.predict(img_path)
    print("Prediction finished. Results length:", len(results))
    if len(results) == 0:
        print("No results returned.")
    else:
        out_path = os.path.join(script_dir, "upload_rendered2.png")
        try:
            # Load the original image as the canvas for our annotations (so we don't draw the
            # original model's 'clothing' box).
            im = Image.open(img_path).convert("RGBA")
            W, H = im.size

            # Post-process predictions to map "clothing" -> specific types (shirt, pant, shoe, accessory)
            try:
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    print("No boxes to post-process.")
                    im.convert("RGB").save(out_path)
                    print("Saved image to", out_path)
                else:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                    except Exception:
                        xyxy = boxes.xyxy.numpy()
                    try:
                        confs = boxes.conf.cpu().numpy()
                    except Exception:
                        confs = boxes.conf.numpy()

                    draw = ImageDraw.Draw(im)
                    font = ImageFont.load_default()
                    mapped_labels = []
                    accessories = []

                    # Load CLIP processor/model lazily (if available)
                    use_clip = False
                    clip_model = None
                    clip_processor = None
                    clip_labels = ["shirt", "pant", "shoe", "accessory"]
                    clip_threshold = 0.25  # tuneable: minimum probability to accept CLIP label
                    try:
                        from transformers import CLIPProcessor, CLIPModel
                        import torch as _torch
                        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                        clip_model.eval()
                        use_clip = True
                        print("CLIP zero-shot classifier loaded.")
                    except Exception as e:
                        # If CLIP isn't available or fails to load, we'll fall back to heuristics
                        print("CLIP not available, falling back to heuristics:", e)

                    for i, (x1, y1, x2, y2) in enumerate(xyxy):
                        conf = float(confs[i]) if i < len(confs) else 0.0
                        y_center = (y1 + y2) / 2.0
                        height = (y2 - y1)
                        width = (x2 - x1)
                        aspect = height / (width + 1e-6)

                        # Initial heuristic label
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

                        # If CLIP is available, run zero-shot classification on the cropped region
                        clip_used = False
                        clip_label = None
                        clip_score = 0.0
                        if use_clip:
                            try:
                                # Crop the region and convert to RGB PIL image
                                crop = im.crop((int(x1), int(y1), int(x2), int(y2))).convert("RGB")
                                inputs = clip_processor(text=clip_labels, images=crop, return_tensors="pt", padding=True)
                                with _torch.no_grad():
                                    outputs = clip_model(**inputs)
                                # outputs.logits_per_image shape: (batch=1, text_labels)
                                logits = outputs.logits_per_image[0].cpu()
                                probs = _torch.softmax(logits, dim=0)
                                best_idx = int(_torch.argmax(probs).item())
                                clip_score = float(probs[best_idx].item())
                                clip_label = clip_labels[best_idx]
                                if clip_score >= clip_threshold:
                                    label = clip_label
                                    clip_used = True
                            except Exception as e:
                                print("CLIP classification failed for crop:", e)

                        mapped_labels.append((label, conf, (x1, y1, x2, y2)))
                        if label == "accessory":
                            accessories.append((label, conf, (x1, y1, x2, y2)))

                        # Draw our own box and label (no original "clothing" box will remain)
                        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                        # Use a different color when CLIP changed the label
                        box_color = (0, 200, 0, 255) if clip_used else (255, 0, 0, 255)
                        draw.rectangle([x1i, y1i, x2i, y2i], outline=box_color, width=3)
                        text = f"{label} {conf:.2f}"
                        if clip_used:
                            text += f" (clip {clip_score:.2f})"
                        try:
                            w, h = font.getsize(text)
                        except Exception:
                            bbox = draw.textbbox((0, 0), text, font=font)
                            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        tx, ty = x1i, max(0, y1i - h - 4)
                        draw.rectangle([tx - 1, ty - 1, tx + w + 1, ty + h + 1], fill=(0, 0, 0, 160))
                        draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)

                    # Save annotated image
                    im.convert("RGB").save(out_path)
                    print("Saved rendered image to", out_path)

                    # Print mapped labels to console
                    for label, conf, box in mapped_labels:
                        print(f"Mapped detection -> {label} (confidence={conf:.3f}) at box={box}")

                    # Print a clear terminal summary of detected types and accessories
                    from collections import Counter
                    labels_only = [l for (l, _, _) in mapped_labels]
                    counts = Counter(labels_only)
                    counts_str = ', '.join(f"{k}({v})" for k, v in counts.items())
                    primary = max(mapped_labels, key=lambda x: x[1])
                    print(f"Detected clothing types: {counts_str}")
                    print(f"Primary detected clothing type: {primary[0]} (confidence={primary[1]:.3f})")
                    if accessories:
                        acc_str = ', '.join(f"accessory(conf={a[1]:.3f})" for a in accessories)
                        print(f"Accessories detected: {acc_str}")
                    else:
                        print("Accessories detected: none")
            except Exception as e:
                print("Post-processing/annotation failed:", e)
        except Exception as e:
            print("Could not process/save annotated image:", e)
            try:
                im.show()
            except Exception:
                print("Could not show image either.")
except Exception:
    print("Exception during prediction:", file=sys.stderr)
    traceback.print_exc()
