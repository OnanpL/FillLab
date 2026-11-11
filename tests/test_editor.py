"""편집기 흐름: 입력, 작업 기록, 선택/이동, 브러시 (명세 3장, 6.3절, 7장)."""

import numpy as np

import fill_core as fc
from editor import (Editor, BUTTON_LEFT, BUTTON_RIGHT, PHASE_DRAWING, PHASE_EDITING,
                    TOOL_BRUSH, TOOL_FILL)
from conftest import BLACK, WHITE, canvas, two_squares

BLUE = (0, 0, 255, 255)


def new_editor(img=None):
    ed = Editor(image=canvas(20, 20) if img is None else img)
    ed.set_colors(BLUE, WHITE)
    ed.set_fill_option(antialias=False)
    return ed


def test_release_enters_editing_and_records():
    ed = new_editor()
    ed.fill_press(3, 3)
    assert ed.fill_state.phase == PHASE_DRAWING
    assert ed.history_names() == ["새 이미지"]
    ed.fill_release()
    assert ed.fill_state.phase == PHASE_EDITING
    assert ed.history_names()[-1] == "채우기"
    # 확정 전에는 레이어가 바뀌지 않는다
    assert (ed.image == 255).all()


def test_shift_decided_at_press_only():
    img, a, b = two_squares()
    ed = new_editor(img)
    ed.fill_press(7, 9, shift=False)
    ed.fill_drag(8, 9)
    assert ed.fill_state.mode == fc.MODE_CONTIGUOUS
    ed.fill_release()
    assert not np.all(ed.display_image()[b] == BLUE, axis=-1).any()


def test_press_while_drawing_cancels():
    ed = new_editor()
    ed.fill_press(3, 3)
    ed.fill_press(5, 5, BUTTON_RIGHT)
    assert not ed.fill_active
    assert np.array_equal(ed.display_image(), ed.image)


def test_esc_cancel_and_enter_commit():
    ed = new_editor()
    ed.fill_press(3, 3)
    ed.fill_release()
    ed.cancel_fill()
    assert not ed.fill_active and (ed.image == 255).all()
    ed.fill_press(3, 3)
    ed.fill_release()
    ed.commit_fill()
    assert np.all(ed.image == BLUE, axis=-1).all()


def test_undo_redo_step_by_step():
    img, a, b = two_squares()
    ed = new_editor(img)
    ed.fill_press(7, 9)
    ed.fill_release()                         # 채우기
    ed.set_fill_option(tolerance=100.0)      # 허용 오차 변경 → 전부
    assert np.all(ed.display_image() == BLUE, axis=-1).all()
    ed.commit_fill()                          # 채우기 확정
    assert ed.history_names() == ["새 이미지", "채우기", "허용 오차 변경", "채우기 확정"]

    ed.undo()   # 확정 취소 → 편집 대기로
    assert ed.fill_state.phase == PHASE_EDITING
    assert np.array_equal(ed.image, img)
    assert ed.fill_options.tolerance == 100.0
    ed.undo()   # 허용 오차 50으로
    assert ed.fill_state.options.tolerance == 50.0
    assert ed.fill_options.tolerance == 50.0
    assert np.all(ed.display_image() == BLUE, axis=-1)[a].all()
    assert not np.all(ed.display_image() == BLUE, axis=-1)[b].any()
    ed.undo()   # 채우기 전
    assert not ed.fill_active
    assert np.array_equal(ed.display_image(), img)
    assert not ed.undo()

    ed.redo()
    ed.redo()
    ed.redo()
    assert not ed.fill_active
    assert np.all(ed.image == BLUE, axis=-1).all()


def test_mode_change_while_editing_recomputes_and_records():
    img, a, b = two_squares()
    ed = new_editor(img)
    ed.fill_press(7, 9)
    ed.fill_release()
    ed.set_fill_option(mode=fc.MODE_GLOBAL)
    assert ed.history_names()[-1] == "채우기 모드 변경"
    assert np.all(ed.display_image() == BLUE, axis=-1)[b].all()


def test_handle_drag_moves_seed():
    img, a, b = two_squares()
    ed = new_editor(img)
    ed.fill_press(7, 9)
    ed.fill_release()
    ed.fill_press(7.5, 9.5)          # 핸들 근처
    ed.fill_drag(29, 9)
    ed.fill_release()
    out = ed.display_image()
    assert np.all(out == BLUE, axis=-1)[b].all()
    assert not np.all(out == BLUE, axis=-1)[a].any()
    assert ed.history_names()[-1] == "채우기 시작점 이동"
    assert "채우기 확정" not in ed.history_names()


def test_tool_switch_commits_fill():
    ed = new_editor()
    ed.fill_press(3, 3)
    ed.fill_release()
    ed.set_tool(TOOL_BRUSH)
    assert not ed.fill_active
    assert np.all(ed.image == BLUE, axis=-1).all()


def test_slider_drag_records_once():
    img, a, b = two_squares()
    ed = new_editor(img)
    ed.fill_press(7, 9)
    ed.fill_release()
    for t in (55.0, 70.0, 90.0, 100.0):
        ed.set_fill_option(record=False, tolerance=t)
    assert np.all(ed.display_image() == BLUE, axis=-1).all()   # 미리 보기는 바로 갱신
    assert ed.history_names()[-1] == "채우기"
    ed.flush_option_record()
    assert ed.history_names()[-2:] == ["채우기", "허용 오차 변경"]
    ed.undo()
    assert ed.fill_state.options.tolerance == 50.0


def test_pending_record_kept_before_commit():
    ed = new_editor()
    ed.fill_press(3, 3)
    ed.fill_release()
    ed.set_fill_option(record=False, tolerance=80.0)
    ed.commit_fill()
    assert ed.history_names()[-3:] == ["채우기", "허용 오차 변경", "채우기 확정"]
    ed.undo()
    assert ed.fill_state.options.tolerance == 80.0


def test_option_change_when_idle_not_recorded():
    ed = new_editor()
    n = len(ed.history)
    ed.set_fill_option(tolerance=10.0)
    assert len(ed.history) == n


def test_brush_stroke_and_undo():
    ed = new_editor()
    ed.set_tool(TOOL_BRUSH)
    ed.brush_size = 3
    ed.brush_press(2, 10)
    ed.brush_drag(17, 10)
    ed.brush_release()
    row = np.all(ed.image[10] == BLUE, axis=-1)
    assert row[3:16].all()
    assert (ed.image[0] == 255).all()
    ed.undo()
    assert (ed.image == 255).all()


def test_brush_right_button_uses_secondary_and_respects_selection():
    ed = new_editor()
    ed.set_colors(BLUE, BLACK)
    ed.set_tool(TOOL_BRUSH)
    ed.set_selection((0, 0, 10, 20))
    ed.brush_press(2, 10, BUTTON_RIGHT)
    ed.brush_drag(17, 10)
    ed.brush_release()
    row = np.all(ed.image[10] == BLACK, axis=-1)
    assert row[3:10].all()
    assert not row[10:].any()


def test_select_and_move():
    img = canvas(20, 20)
    img[2:5, 2:5] = BLACK
    ed = new_editor(img)
    ed.select_press(2, 2)
    ed.select_drag(5, 5)
    ed.select_release()
    assert ed.selection == (2, 2, 5, 5)
    ed.move_press(3, 3)
    ed.move_drag(13, 8)
    ed.move_release()
    assert ed.selection == (12, 7, 15, 10)
    assert np.all(ed.image[7:10, 12:15] == BLACK, axis=-1).all()
    assert (ed.image[2:5, 2:5, 3] == 0).all()   # 옮긴 자리는 투명
    ed.undo()
    assert ed.selection == (2, 2, 5, 5)
    assert np.array_equal(ed.image, img)


def test_click_without_drag_clears_selection():
    ed = new_editor()
    ed.set_selection((1, 1, 5, 5))
    ed.select_press(8, 8)
    ed.select_release()
    assert ed.selection is None


def test_release_with_nothing_to_paint_does_nothing():
    ed = new_editor()
    ed.set_selection((0, 0, 5, 5))
    n = len(ed.history)
    ed.fill_press(10, 10)      # 선택 밖
    ed.fill_release()
    assert not ed.fill_active and len(ed.history) == n
    ed.fill_press(-3, 4)       # 캔버스 밖
    ed.fill_release()
    assert not ed.fill_active and len(ed.history) == n


def test_drag_outside_canvas_follows_and_paints_nothing():
    ed = new_editor()
    ed.fill_press(3, 3)
    ed.fill_drag(-5, 3)
    assert ed.fill_state.seed_x == -5
    assert np.array_equal(ed.display_image(), ed.image)
    ed.fill_drag(4, 4)
    ed.fill_release()
    assert np.all(ed.display_image() == BLUE, axis=-1).all()


def test_fill_keeps_selection_captured_at_press():
    ed = new_editor()
    ed.set_selection((0, 0, 5, 5))
    ed.fill_press(2, 2)
    ed.fill_release()
    assert ed.fill_state.selection[:5, :5].all()
    assert not ed.fill_state.selection[6:, 6:].any()
