"""명세 9장 인수 시험 10개.

대부분의 시험은 안티앨리어싱(기본값: 사용)을 끈 상태와 켠 상태 모두에서
'칠해진 영역(마스크 255)'을 검사한다. 안티앨리어싱을 켜면 영역 바로 바깥
1픽셀에 색이 옅게 섞이므로, 그 둘레는 '칠해지지 않음'에서 제외하고 본다.
"""

import numpy as np
import pytest

import fill_core as fc
from editor import Editor, BUTTON_LEFT, BUTTON_RIGHT, PHASE_EDITING
from conftest import (BLACK, RED, WHITE, black_white, gradient, ring_image,
                      two_squares)

PRIMARY = (0, 0, 255, 255)     # 기본색: 파랑
SECONDARY = (0, 200, 0, 255)   # 보조색: 초록


def make_editor(img, **opts):
    ed = Editor(image=img)
    ed.set_colors(PRIMARY, SECONDARY)
    if opts:
        ed.set_fill_option(**opts)
    return ed


def click(ed, x, y, button=BUTTON_LEFT, shift=False):
    ed.fill_press(x, y, button, shift)
    ed.fill_release()


def painted(img, color):
    return np.all(img == np.array(color, np.uint8), axis=-1)


@pytest.mark.parametrize("aa", [False, True])
def test_01_circle_left_click_primary(aa):
    img, inside, outside = ring_image()
    ed = make_editor(img, tolerance=0.0, mode=fc.MODE_CONTIGUOUS, antialias=aa)
    click(ed, 20.5, 20.5, BUTTON_LEFT)
    out = ed.display_image()
    region = ed.fill_result().region
    assert region[inside].all()
    assert not region[outside].any()
    assert painted(out, PRIMARY)[inside].all()
    assert not painted(out, PRIMARY)[outside].any()
    assert np.array_equal(out[outside], img[outside])      # 바깥은 그대로


@pytest.mark.parametrize("aa", [False, True])
def test_02_circle_right_click_secondary(aa):
    img, inside, outside = ring_image()
    ed = make_editor(img, tolerance=0.0, antialias=aa)
    click(ed, 20.5, 20.5, BUTTON_RIGHT)
    out = ed.display_image()
    assert painted(out, SECONDARY)[inside].all()
    assert not painted(out, PRIMARY).any()
    assert np.array_equal(out[outside], img[outside])


@pytest.mark.parametrize("aa", [False, True])
def test_03_shift_click_fills_both_and_keeps_mode(aa):
    img, a, b = two_squares()
    ed = make_editor(img, mode=fc.MODE_CONTIGUOUS, antialias=aa)
    click(ed, 7, 9, shift=True)
    out = ed.display_image()
    assert painted(out, PRIMARY)[a].all()
    assert painted(out, PRIMARY)[b].all()
    assert ed.fill_options.mode == fc.MODE_CONTIGUOUS      # 옵션의 모드는 그대로
    # 다음 클릭은 다시 연속 모드
    ed.commit_fill()
    img2, a2, b2 = two_squares()
    ed.load_image(img2)
    click(ed, 7, 9)
    assert not painted(ed.display_image(), PRIMARY)[b2].any()


@pytest.mark.parametrize("aa", [False, True])
def test_04_plain_click_fills_only_clicked_square(aa):
    img, a, b = two_squares()
    ed = make_editor(img, mode=fc.MODE_CONTIGUOUS, antialias=aa)
    click(ed, 7, 9)
    out = ed.display_image()
    assert painted(out, PRIMARY)[a].all()
    assert not painted(out, PRIMARY)[b].any()
    assert np.array_equal(out[b], img[b])


def test_05_gradient_stops_at_tolerance():
    img, g = gradient()
    tol = 30.0
    ed = make_editor(img, tolerance=tol, antialias=False)
    click(ed, 0.5, 3.5)
    region = ed.fill_result().region
    thr = fc.tolerance_to_threshold(tol)
    # 기준색(검정)과 각 열 회색의 거리: 명세 5.2절의 식으로 따로 계산한다.
    expected_cols = []
    for v in g:
        d = np.sqrt(3 * (v / 255.0) ** 2) * 0.5 * 255.0
        d = np.floor(d + 0.5)
        if v != 0:
            d = max(d, 1)
        expected_cols.append(d <= thr)
    expected_cols = np.array(expected_cols)
    # 처음 기준색과 비교하므로 왼쪽부터 이어진 구간만 칠해진다.
    assert expected_cols[0]
    stop = int(np.argmin(expected_cols)) if not expected_cols.all() else len(g)
    assert 0 < stop < len(g)
    assert region[:, :stop].all()
    assert not region[:, stop:].any()


@pytest.mark.parametrize("aa", [False, True])
def test_06_black_white_boundary_83_84(aa):
    img = black_white()
    black = np.zeros(img.shape[:2], bool)
    black[:, :10] = True
    white = ~black

    ed = make_editor(img, tolerance=83.0, mode=fc.MODE_CONTIGUOUS, antialias=aa)
    click(ed, 2, 5)
    r83 = ed.fill_result()
    assert np.array_equal(r83.region, black)
    assert not (r83.mask[white] == 255).any()

    ed = make_editor(img, tolerance=84.0, mode=fc.MODE_CONTIGUOUS, antialias=aa)
    click(ed, 2, 5)
    r84 = ed.fill_result()
    assert r84.region.all()
    assert painted(ed.display_image(), PRIMARY).all()


@pytest.mark.parametrize("aa", [False, True])
def test_07_selection_limits_fill(aa):
    img = np.full((20, 20, 4), 255, np.uint8)
    ed = make_editor(img, antialias=aa)
    ed.set_selection((5, 5, 12, 12))
    click(ed, 8, 8)
    out = ed.display_image()
    sel = ed.selection_mask()
    assert painted(out, PRIMARY)[sel].all()
    assert np.array_equal(out[~sel], img[~sel])
    # 선택 밖을 클릭하면 아무 일도 일어나지 않는다 [관찰]
    ed.cancel_fill()
    click(ed, 1, 1)
    assert np.array_equal(ed.display_image(), img)


def test_08_drag_moves_seed():
    img, a, b = two_squares()
    ed = make_editor(img, antialias=False)
    ed.fill_press(7, 9)
    assert painted(ed.display_image(), PRIMARY)[a].all()
    ed.fill_drag(15, 9)          # 흰 배경 위
    ed.fill_drag(29.7, 9.2)      # 오른쪽 사각형
    ed.fill_release()
    out = ed.display_image()
    assert painted(out, PRIMARY)[b].all()           # 놓은 위치의 결과가 남는다
    assert not painted(out, PRIMARY)[a].any()       # 처음 위치의 결과는 남지 않는다
    assert ed.fill_state.seed_x == pytest.approx(29.7)
    assert ed.fill_state.phase == PHASE_EDITING


def test_09_editing_tolerance_recomputes():
    img, g = gradient()
    ed = make_editor(img, tolerance=10.0, antialias=False)
    click(ed, 0.5, 3.5)
    before = ed.fill_result().region.sum()
    ed.set_fill_option(tolerance=40.0)
    after_res = ed.fill_result()
    expected = fc.compute_region(img, 0.5, 3.5, ed.fill_options)
    assert after_res.region.sum() > before
    assert np.array_equal(after_res.region, expected)
    assert ed.fill_state.phase == PHASE_EDITING
    assert ed.fill_state.options.tolerance == 40.0


def test_10_click_elsewhere_commits_and_starts_new():
    img, a, b = two_squares()
    ed = make_editor(img, antialias=False)
    click(ed, 7, 9)
    first = ed.display_image().copy()
    ed.fill_press(30, 9, BUTTON_RIGHT)     # 다른 곳 (핸들에서 먼 곳)
    # 이전 결과가 확정되어 레이어에 들어갔다
    assert np.array_equal(ed.image, first)
    assert "채우기 확정" in ed.history_names()
    # 새 채우기가 그리기 상태로 시작되었다
    assert ed.fill_active and ed.fill_state.phase != PHASE_EDITING
    ed.fill_release()
    out = ed.display_image()
    assert painted(out, PRIMARY)[a].all()
    assert painted(out, SECONDARY)[b].all()
