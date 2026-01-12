from rembg import remove
from PIL import Image
import os
import sys
import traceback

print("cwd:", os.getcwd())
print("__file__:", __file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
print("script_dir:", script_dir)
img_path = os.path.join(script_dir, "shirt.jpg")
print("Trying to open image at:", img_path)
try:
    img = Image.open(img_path)
    print("Image opened:", getattr(img, 'size', None), getattr(img, 'mode', None))
    out = remove(img)
    print("remove returned:", type(out))
    out_path = os.path.join(script_dir, "shirt_cutout.png")
    out.save(out_path)
    print("Saved", out_path)
except Exception:
    print("Exception during processing:", file=sys.stderr)
    traceback.print_exc()
