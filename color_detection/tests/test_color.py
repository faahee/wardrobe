import json
from PIL import Image
import os
from color import dominant_color_name


def make_image(path, color, size=(50, 50), mode="RGB"):
    img = Image.new(mode, size, color)
    img.save(path)


def test_black_detected(tmp_path):
    p = tmp_path / "black.png"
    make_image(str(p), (0, 0, 0))
    assert dominant_color_name(str(p), n_clusters=1) == "black"


def test_white_detected(tmp_path):
    p = tmp_path / "white.png"
    make_image(str(p), (255, 255, 255))
    assert dominant_color_name(str(p), n_clusters=1) == "white"


def test_red_detected(tmp_path):
    p = tmp_path / "red.png"
    make_image(str(p), (255, 0, 0))
    # Allow either 'red' or 'maroon' depending on mapping
    name = dominant_color_name(str(p), n_clusters=1)
    assert name in ("red", "maroon")
