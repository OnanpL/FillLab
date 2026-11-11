"""samples/ 의 PNG 파일로 명세의 [관찰] 항목을 다시 확인한다."""

import os

import numpy as np
import pytest
from PIL import Image

import fill_core as fc

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def load(name):
    path = os.path.join(SAMPLES, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 없음")
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"))


def test_sample_diagonal_touch_8_connected():
    img = load("diagonal_touch.png")
    r = fc.compute_region(img, 20, 20, fc.FillOptions())
    assert r[100, 100]


@pytest.mark.parametrize("tol, white_included", [(83, False), (84, True)])
def test_sample_black_white(tol, white_included):
    img = load("black_white.png")
    r = fc.compute_region(img, 10, 10, fc.FillOptions(tolerance=tol))
    assert r[:, :80].all()
    assert bool(r[:, 80:].all()) is white_included
    assert bool(r[:, 80:].any()) is white_included


def test_sample_ring_square_shift_fills_inside_and_outside():
    img = load("ring_square.png")
    contiguous = fc.compute_region(img, 100, 100, fc.FillOptions())
    assert contiguous[100, 100] and not contiguous[10, 10]
    whole = fc.compute_region(img, 100, 100, fc.FillOptions(), mode=fc.MODE_GLOBAL)
    assert whole[100, 100] and whole[10, 10] and not whole[60, 60]


def test_sample_transparent_premultiplied_default():
    img = load("transparent.png")
    r = fc.compute_region(img, 2, 2, fc.FillOptions(tolerance=0))
    assert r[0, 0] and not r[50, 50]
