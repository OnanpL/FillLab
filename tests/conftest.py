import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
RED = (255, 0, 0, 255)


def canvas(w, h, color=WHITE):
    img = np.empty((h, w, 4), dtype=np.uint8)
    img[...] = color
    return img


def ring_image(size=41, radius=14, thickness=1.6):
    """흰 배경에 검은 닫힌 원(테두리). 8방향으로도 새지 않을 만큼 두껍게 그린다."""
    img = canvas(size, size)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    d = np.hypot(xx - c, yy - c)
    ring = np.abs(d - radius) <= thickness
    img[ring] = BLACK
    inside = d < radius - thickness
    outside = d > radius + thickness
    return img, inside, outside


def two_squares():
    """흰 배경에 떨어진 빨간 사각형 2개."""
    img = canvas(40, 20)
    a = np.zeros((20, 40), bool)
    b = np.zeros((20, 40), bool)
    a[5:15, 3:13] = True
    b[5:15, 25:35] = True
    img[a] = RED
    img[b] = RED
    return img, a, b


def gradient(w=256, h=8):
    """왼쪽 0에서 오른쪽 255로 변하는 불투명 회색 그라데이션."""
    g = np.round(np.linspace(0, 255, w)).astype(np.uint8)
    img = np.empty((h, w, 4), dtype=np.uint8)
    img[..., 0] = g
    img[..., 1] = g
    img[..., 2] = g
    img[..., 3] = 255
    return img, g


def black_white():
    """왼쪽 절반 불투명 검정, 오른쪽 절반 불투명 흰색."""
    img = canvas(20, 10)
    img[:, :10] = BLACK
    return img
