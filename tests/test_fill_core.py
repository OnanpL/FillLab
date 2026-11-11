"""채우기 순수 로직 단위 시험 (명세 4~6장)."""

import numpy as np
import pytest

import fill_core as fc
from conftest import BLACK, RED, WHITE, canvas

TRANSPARENT_BLACK = (0, 0, 0, 0)


# --- 5.2절 계산 예시 -------------------------------------------------------

@pytest.mark.parametrize("a, b, expected", [
    (BLACK, WHITE, 221),
    (BLACK, TRANSPARENT_BLACK, 128),
    (TRANSPARENT_BLACK, WHITE, 255),
])
def test_straight_distance_examples(a, b, expected):
    assert fc.color_distance_pair(a, b, fc.ALPHA_STRAIGHT) == expected


def test_distance_same_color_is_zero():
    assert fc.color_distance_pair(RED, RED) == 0


def test_distance_minimum_one_when_different():
    assert fc.color_distance_pair((10, 10, 10, 255), (10, 10, 11, 255), fc.ALPHA_STRAIGHT) == 1


def test_premultiplied_ignores_color_of_transparent_pixels():
    assert fc.color_distance_pair((255, 0, 0, 0), (0, 0, 255, 0), fc.ALPHA_PREMULTIPLIED) == 0
    assert fc.color_distance_pair((255, 0, 0, 0), (0, 0, 255, 0), fc.ALPHA_STRAIGHT) > 0


def test_premultiplied_opaque_equals_straight():
    assert fc.color_distance_pair(BLACK, WHITE, fc.ALPHA_PREMULTIPLIED) == 221


# --- 5.3절 허용 오차 변환 ---------------------------------------------------

def test_tolerance_threshold_boundary():
    assert fc.tolerance_to_threshold(83) < 221 <= fc.tolerance_to_threshold(84)
    assert fc.tolerance_to_threshold(0) == 0
    assert fc.tolerance_to_threshold(100) == 255


def test_tolerance_zero_only_exact_color():
    img = canvas(3, 1)
    img[0, 1] = (254, 255, 255, 255)
    region = fc.compute_region(img, 0, 0, fc.FillOptions(tolerance=0, mode=fc.MODE_GLOBAL))
    assert region.tolist() == [[True, False, True]]


def test_tolerance_max_includes_everything():
    img = canvas(4, 1)
    img[0, 1] = BLACK
    img[0, 2] = TRANSPARENT_BLACK
    region = fc.compute_region(img, 0, 0, fc.FillOptions(tolerance=100, mode=fc.MODE_GLOBAL))
    assert region.all()


# --- 4장 채우기 모드 --------------------------------------------------------

def test_contiguous_is_8_connected_through_diagonal():
    # 두 흰 영역이 대각선 한 점에서만 닿는다 [관찰]
    img = canvas(4, 4, BLACK)
    img[0:2, 0:2] = WHITE
    img[2:4, 2:4] = WHITE
    opts = fc.FillOptions(tolerance=0)
    region = fc.compute_region(img, 0, 0, opts)
    assert region[3, 3]
    # 접점 한 칸을 막으면 갈라진다
    img[1, 1] = BLACK
    region = fc.compute_region(img, 0, 0, opts)
    assert not region[3, 3]


def test_global_mode_ignores_connectivity():
    img = canvas(5, 1)
    img[0, 2] = BLACK
    region = fc.compute_region(img, 0, 0, fc.FillOptions(tolerance=0), mode=fc.MODE_GLOBAL)
    assert region.tolist() == [[True, True, False, True, True]]


def test_contiguous_compares_with_seed_color_not_neighbor():
    # 한 칸씩 10씩 밝아지는 행: 이웃끼리는 비슷해도 기준색과 멀어지면 멈춘다
    img = np.zeros((1, 20, 4), np.uint8)
    img[0, :, :3] = (np.arange(20) * 10)[:, None]
    img[..., 3] = 255
    opts = fc.FillOptions(tolerance=10)  # 임계값 26
    region = fc.compute_region(img, 0, 0, opts)
    dist = fc.color_distance(img, img[0, 0])
    expected = dist[0] <= fc.tolerance_to_threshold(10)
    assert region[0].tolist() == expected.tolist()
    assert not region[0, -1]


def test_seed_outside_canvas_paints_nothing():
    img = canvas(5, 5)
    for x, y in [(-1, 2), (2, -0.5), (5, 2), (2, 5.2)]:
        assert not fc.compute_region(img, x, y, fc.FillOptions()).any()


def test_seed_floors_fractional_coordinates():
    img = canvas(3, 1)
    img[0, 1] = BLACK
    region = fc.compute_region(img, 1.99, 0.5, fc.FillOptions(tolerance=0), mode=fc.MODE_GLOBAL)
    assert region.tolist() == [[False, True, False]]


@pytest.mark.parametrize("mode", [fc.MODE_CONTIGUOUS, fc.MODE_GLOBAL])
def test_seed_outside_selection_paints_nothing(mode):
    img = canvas(10, 10)
    sel = np.zeros((10, 10), bool)
    sel[2:5, 2:5] = True
    assert not fc.compute_region(img, 8, 8, fc.FillOptions(), mode, sel).any()
    r = fc.compute_region(img, 3, 3, fc.FillOptions(), mode, sel)
    assert np.array_equal(r, sel)


def test_contiguous_large_image_speed():
    import time
    img = canvas(1200, 900)
    t = time.perf_counter()
    r = fc.compute_region(img, 10, 10, fc.FillOptions())
    assert r.all()
    assert time.perf_counter() - t < 2.0


# --- 6장 결과 ---------------------------------------------------------------

def test_mask_without_antialias_is_binary():
    region = np.zeros((5, 5), bool)
    region[2, 2] = True
    m = fc.build_mask(region, antialias=False)
    assert set(np.unique(m).tolist()) == {0, 255}


def test_antialias_fringe_values():
    region = np.zeros((7, 7), bool)
    region[2:5, 2:5] = True          # 3×3 정사각형
    m = fc.build_mask(region, antialias=True)
    assert m[3, 3] == 255
    assert m[1, 3] == fc.AA_EDGE_ONE        # 변으로 맞닿음 (≈10%)
    assert m[1, 1] == fc.AA_CORNER_ONLY     # 모서리만 (≈3%)
    assert m[0, 0] == 0                     # 둘레 1픽셀 밖은 0
    # 두 변이 맞닿는 픽셀 (오목한 모서리)
    region2 = np.zeros((5, 5), bool)
    region2[1, 1:4] = True
    region2[1:4, 1] = True
    m2 = fc.build_mask(region2, antialias=True)
    assert m2[2, 2] == fc.AA_EDGE_TWO


def test_antialias_red_on_black_matches_observation():
    img = canvas(7, 7, BLACK)
    img[2:5, 2:5] = WHITE
    res = fc.flood_fill(img, 3, 3, RED, WHITE, fc.FillOptions(tolerance=0))
    assert tuple(res.image[3, 3]) == RED
    assert res.image[1, 3, 0] == 26
    assert res.image[1, 1, 0] == 8


def test_composite_keeps_unmasked_pixels():
    img = canvas(3, 3, (10, 20, 30, 0))
    res = fc.flood_fill(img, 0, 0, RED, WHITE, fc.FillOptions(antialias=False),
                        selection=np.eye(3, dtype=bool))
    assert tuple(res.image[0, 1]) == (10, 20, 30, 0)
    assert tuple(res.image[0, 0]) == RED


def test_hatch_uses_fg_and_bg():
    src = fc.paint_source((6, 6), fc.FILL_HATCH_HORIZONTAL, RED, WHITE)
    assert tuple(src[0, 0]) == RED
    assert tuple(src[1, 0]) == WHITE


def test_multiply_blend():
    img = canvas(1, 1, (128, 128, 128, 255))
    res = fc.flood_fill(img, 0, 0, (255, 0, 0, 255), WHITE,
                        fc.FillOptions(antialias=False, blend=fc.BLEND_MULTIPLY))
    assert tuple(res.image[0, 0]) == (128, 0, 0, 255)
