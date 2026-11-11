"""편집기 상태와 동작 (UI 없음).

이미지 한 장(레이어 하나), 사각형 선택, 채우기 작업 흐름(명세 7장), 브러시,
선택 영역 이동, 단계별 실행 취소/다시 실행을 담당한다.
tkinter 코드는 app.py에 있고, 여기 있는 동작을 호출만 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

import fill_core as fc

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

TOOL_FILL = "fill"
TOOL_BRUSH = "brush"
TOOL_SELECT = "select"
TOOL_MOVE = "move"

PHASE_DRAWING = "drawing"   # 누른 채 드래그 중
PHASE_EDITING = "editing"   # 놓은 뒤 편집 대기

BUTTON_LEFT = 1
BUTTON_RIGHT = 3

# 편집 대기 상태에서 핸들을 잡았다고 보는 거리 (이미지 픽셀). app.py가 배율에 맞춰 넘긴다.
DEFAULT_HANDLE_RADIUS = 4.0


@dataclass(frozen=True)
class FillState:
    """진행 중인 채우기 하나. 작업 기록에 그대로 저장되므로 바꾸지 않는다."""

    seed_x: float
    seed_y: float
    fg: tuple
    bg: tuple
    mode: str                     # 실제로 쓰는 모드 (Shift면 전체 모드)
    options: fc.FillOptions
    selection: np.ndarray | None  # 그리기 시작 때 묶은 선택 영역 (7장 [확인])
    forced_global: bool = False   # Shift로 이번만 전체 모드
    phase: str = PHASE_DRAWING
    button: int = BUTTON_LEFT


@dataclass(frozen=True)
class Snapshot:
    image: np.ndarray                   # 확정된 레이어 (바꾸지 않는다)
    selection: tuple | None             # (x0, y0, x1, y1), 끝은 제외
    fill: FillState | None


@dataclass
class HistoryEntry:
    name: str
    snap: Snapshot


def rect_mask(rect: tuple | None, w: int, h: int) -> np.ndarray | None:
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    m = np.zeros((h, w), dtype=bool)
    m[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = True
    return m


def normalize_rect(x0, y0, x1, y1, w, h) -> tuple | None:
    """두 꼭짓점(소수 좌표)을 캔버스 안의 정수 사각형으로 바꾼다. 비면 None."""
    ax, bx = sorted((x0, x1))
    ay, by = sorted((y0, y1))
    ax, ay = max(0, math.floor(ax)), max(0, math.floor(ay))
    bx, by = min(w, math.ceil(bx)), min(h, math.ceil(by))
    if bx - ax < 1 or by - ay < 1:
        return None
    return (ax, ay, bx, by)


class Editor:
    def __init__(self, width: int = 640, height: int = 480, image: np.ndarray | None = None):
        if image is None:
            image = np.full((height, width, 4), 255, dtype=np.uint8)
        self.primary = BLACK      # 기본색
        self.secondary = WHITE    # 보조색
        self.fill_options = fc.FillOptions()
        self.brush_size = 5
        self.tool = TOOL_FILL
        self.history: list[HistoryEntry] = []
        self.index = -1
        self._image = np.ascontiguousarray(image, dtype=np.uint8)
        self._selection: tuple | None = None
        self._fill: FillState | None = None
        self._preview_cache: tuple | None = None
        self._stroke: dict | None = None   # 브러시, 선택, 이동, 핸들 드래그 중 임시 상태
        self.on_change = None              # 화면 갱신용 콜백
        self._pending_record = None        # 미뤄 둔 옵션 변경 기록 (이름, 상태)
        self._push("새 이미지")

    # ------------------------------------------------------------------
    # 기본 상태
    # ------------------------------------------------------------------
    @property
    def width(self) -> int:
        return self._image.shape[1]

    @property
    def height(self) -> int:
        return self._image.shape[0]

    @property
    def image(self) -> np.ndarray:
        """확정된 레이어."""
        return self._image

    @property
    def selection(self) -> tuple | None:
        return self._selection

    @property
    def fill_state(self) -> FillState | None:
        return self._fill

    @property
    def fill_active(self) -> bool:
        return self._fill is not None

    def selection_mask(self) -> np.ndarray | None:
        return rect_mask(self._selection, self.width, self.height)

    def load_image(self, image: np.ndarray, name: str = "열기") -> None:
        """새 이미지로 바꾼다. 작업 기록은 새로 시작한다."""
        self._image = np.ascontiguousarray(image, dtype=np.uint8)
        self._selection = None
        self._fill = None
        self._stroke = None
        self._preview_cache = None
        self._pending_record = None
        self.history.clear()
        self.index = -1
        self._push(name)

    def new_image(self, width: int, height: int, color=WHITE) -> None:
        img = np.empty((height, width, 4), dtype=np.uint8)
        img[...] = color
        self.load_image(img, "새 이미지")

    # ------------------------------------------------------------------
    # 작업 기록 (6.3절, 7장)
    # ------------------------------------------------------------------
    def _snapshot(self) -> Snapshot:
        return Snapshot(self._image, self._selection, self._fill)

    def _push(self, name: str) -> None:
        pending = self._pending_record
        self._pending_record = None
        if pending is not None and pending[0] != name:
            # 미뤄 둔 옵션 변경 기록을 먼저 남긴다 (단계별 되돌리기 유지).
            del self.history[self.index + 1:]
            self.history.append(HistoryEntry(*pending))
            self.index = len(self.history) - 1
        del self.history[self.index + 1:]
        self.history.append(HistoryEntry(name, self._snapshot()))
        self.index = len(self.history) - 1
        self._changed()

    def _restore(self, snap: Snapshot) -> None:
        self._image = snap.image
        self._selection = snap.selection
        self._fill = snap.fill
        self._stroke = None
        if snap.fill is not None:
            # 되돌린 채우기의 옵션을 옵션 바에도 반영한다.
            self.fill_options = snap.fill.options
        self._changed()

    @property
    def can_undo(self) -> bool:
        return self.index > 0

    @property
    def can_redo(self) -> bool:
        return self.index < len(self.history) - 1

    def undo(self) -> bool:
        """한 단계씩 되돌린다 [관찰]."""
        self.flush_option_record()
        if self._stroke is not None:
            self._abort_stroke()
        if self._fill is not None and self._fill.phase == PHASE_DRAWING:
            self.cancel_fill(record=False)
            return True
        if not self.can_undo:
            return False
        self.index -= 1
        self._restore(self.history[self.index].snap)
        return True

    def redo(self) -> bool:
        if self._stroke is not None or not self.can_redo:
            return False
        if self._fill is not None and self._fill.phase == PHASE_DRAWING:
            return False
        self.index += 1
        self._restore(self.history[self.index].snap)
        return True

    def history_names(self) -> list[str]:
        return [e.name for e in self.history]

    # ------------------------------------------------------------------
    # 화면에 보일 이미지
    # ------------------------------------------------------------------
    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def fill_result(self) -> fc.FillResult | None:
        """진행 중인 채우기의 결과. 같은 상태면 다시 계산하지 않는다."""
        f = self._fill
        if f is None:
            return None
        key = (math.floor(f.seed_x), math.floor(f.seed_y), f.fg, f.bg, f.mode,
               f.options.with_(mode=f.mode))
        c = self._preview_cache
        if c is not None and c[0] == key and c[1] is self._image and c[2] is f.selection:
            return c[3]
        # 칠하는 대상은 활성 레이어다 (추측, 6.1절). 이 편집기는 레이어가 하나뿐이라
        # 샘플링의 레이어/이미지 결과가 같다 (DECISIONS.md).
        result = fc.flood_fill(self._image, f.seed_x, f.seed_y, f.fg, f.bg,
                               f.options, f.mode, f.selection)
        self._preview_cache = (key, self._image, f.selection, result)
        return result

    def display_image(self) -> np.ndarray:
        if self._stroke is not None and "preview" in self._stroke:
            return self._stroke["preview"]
        r = self.fill_result()
        return r.image if r is not None else self._image

    # ------------------------------------------------------------------
    # 채우기 도구 (3장, 7장)
    # ------------------------------------------------------------------
    def fill_press(self, x: float, y: float, button: int = BUTTON_LEFT,
                   shift: bool = False, handle_radius: float = DEFAULT_HANDLE_RADIUS) -> None:
        f = self._fill
        if f is not None and f.phase == PHASE_DRAWING:
            # 드래그 중 다시 누름: 진행 중인 그리기를 취소한다 [추측]
            self.cancel_fill(record=False)
            return
        if f is not None and f.phase == PHASE_EDITING:
            if math.hypot(x - f.seed_x, y - f.seed_y) <= handle_radius:
                # 편집 대기에서 핸들 드래그: 시작점을 옮긴다 [추측]
                self._stroke = {"kind": "handle"}
                return
            # 편집 대기에서 캔버스를 누름: 기존 결과 확정 후 새 채우기 [추측]
            self.commit_fill()
        # 색과 Shift는 누르는 순간에 정한다 [확인]
        if button == BUTTON_RIGHT:
            fg, bg = self.secondary, self.primary
        else:
            fg, bg = self.primary, self.secondary
        mode = fc.MODE_GLOBAL if shift else self.fill_options.mode
        self._fill = FillState(
            seed_x=float(x), seed_y=float(y), fg=tuple(fg), bg=tuple(bg), mode=mode,
            options=self.fill_options,    # options.mode는 옵션 바의 모드 (Shift와 무관)
            selection=self.selection_mask(), forced_global=bool(shift),
            phase=PHASE_DRAWING, button=button)
        self._changed()

    def fill_drag(self, x: float, y: float) -> None:
        f = self._fill
        if f is None:
            return
        handle = self._stroke is not None and self._stroke.get("kind") == "handle"
        if f.phase != PHASE_DRAWING and not handle:
            return
        old_px = (math.floor(f.seed_x), math.floor(f.seed_y))
        # 같은 색과 모드로 시작점만 바꾼다 [확인]. 캔버스 밖으로 나가도 시작점은
        # 따라가고, 밖에서는 칠하지 않는다 (추측, 8장).
        self._fill = replace(f, seed_x=float(x), seed_y=float(y))
        # 달라진 내용이 없으면 갱신하지 않는다 [확인]
        if (math.floor(x), math.floor(y)) != old_px:
            self._changed()

    def fill_release(self) -> None:
        f = self._fill
        if f is None:
            return
        if self._stroke is not None and self._stroke.get("kind") == "handle":
            self._stroke = None
            self._push("채우기 시작점 이동")
            return
        if f.phase == PHASE_DRAWING:
            if not self.fill_result().mask.any():
                # 칠할 곳이 없으면(캔버스 밖, 선택 밖) 아무 일도 일어나지 않는다 [관찰]
                self.cancel_fill(record=False)
                return
            self._fill = replace(f, phase=PHASE_EDITING)
            self._push("채우기")          # 그리기 종료 시 기록 [확인]

    def set_fill_option(self, record: bool = True, **kw) -> None:
        """옵션을 바꾼다. 편집 대기 중이면 결과를 다시 만들고 기록한다 (7장).

        record=False 이면 결과만 다시 만들고 기록은 미룬다 (슬라이더를 끄는 동안).
        미룬 기록은 flush_option_record()로 남긴다.
        """
        old = self.fill_options
        self.fill_options = old.with_(**kw)
        f = self._fill
        if f is None or f.phase != PHASE_EDITING or self.fill_options == f.options:
            self._changed()
            return
        if "mode" in kw:
            # 편집 중 모드를 고르면 Shift로 정한 모드 대신 고른 모드를 쓴다.
            f = replace(f, mode=self.fill_options.mode, forced_global=False)
        self._fill = replace(f, options=self.fill_options)
        name = _option_record_name(kw)
        if record:
            self._push(name)
        else:
            self._pending_record = (name, self._snapshot())
            self._changed()

    def flush_option_record(self) -> None:
        """미뤄 둔 옵션 변경 기록을 남긴다."""
        pending = self._pending_record
        if pending is None:
            return
        self._pending_record = None
        del self.history[self.index + 1:]
        self.history.append(HistoryEntry(*pending))
        self.index = len(self.history) - 1
        self._changed()

    def set_fill_colors(self) -> None:
        """기본색/보조색이 바뀌었을 때 편집 중인 채우기에 반영한다 (6.3절)."""
        f = self._fill
        if f is None or f.phase != PHASE_EDITING:
            return
        # 처음 누른 버튼의 역할(전경/배경)을 유지한다.
        if f.button == BUTTON_RIGHT:
            fg, bg = self.secondary, self.primary
        else:
            fg, bg = self.primary, self.secondary
        if (fg, bg) == (f.fg, f.bg):
            return
        self._fill = replace(f, fg=tuple(fg), bg=tuple(bg))
        self._push("채우기 색 변경")

    def commit_fill(self) -> bool:
        """확정: 결과를 레이어에 쓰고 기록한다 [확인]."""
        f = self._fill
        if f is None:
            return False
        result = self.fill_result()
        self._image = result.image
        self._fill = None
        self._preview_cache = None
        self._push("채우기 확정")
        return True

    def cancel_fill(self, record: bool = True) -> bool:
        """취소: 진행 중인 결과를 버린다."""
        f = self._fill
        if f is None:
            return False
        self._fill = None
        self._stroke = None
        self._preview_cache = None
        if record and f.phase == PHASE_EDITING:
            self._push("채우기 취소")
        else:
            self._changed()
        return True

    # ------------------------------------------------------------------
    # 도구 전환
    # ------------------------------------------------------------------
    def set_tool(self, tool: str) -> None:
        if tool == self.tool:
            return
        if self._stroke is not None:
            self._abort_stroke()
        if self._fill is not None:
            # 도구 전환 시 처리는 [UNKNOWN] → 확정한다 (DECISIONS.md)
            if self._fill.phase == PHASE_DRAWING:
                self.cancel_fill(record=False)
            else:
                self.commit_fill()
        self.tool = tool
        self._changed()

    def _abort_stroke(self) -> None:
        self._stroke = None
        self._changed()

    # ------------------------------------------------------------------
    # 브러시
    # ------------------------------------------------------------------
    def brush_press(self, x: float, y: float, button: int = BUTTON_LEFT) -> None:
        color = self.secondary if button == BUTTON_RIGHT else self.primary
        canvas = self._image.copy()
        self._stroke = {"kind": "brush", "preview": canvas, "color": color,
                        "last": (x, y), "clip": self.selection_mask()}
        self._stamp_line(x, y, x, y)
        self._changed()

    def brush_drag(self, x: float, y: float) -> None:
        s = self._stroke
        if s is None or s["kind"] != "brush":
            return
        lx, ly = s["last"]
        self._stamp_line(lx, ly, x, y)
        s["last"] = (x, y)
        self._changed()

    def brush_release(self) -> None:
        s = self._stroke
        if s is None or s["kind"] != "brush":
            return
        self._image = s["preview"]
        self._stroke = None
        self._push("브러시")

    def _stamp_line(self, x0, y0, x1, y1) -> None:
        s = self._stroke
        img = s["preview"]
        h, w = img.shape[:2]
        r = max(0.5, self.brush_size / 2.0)
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist / max(0.5, r / 2)))
        bx0 = max(0, math.floor(min(x0, x1) - r))
        by0 = max(0, math.floor(min(y0, y1) - r))
        bx1 = min(w, math.ceil(max(x0, x1) + r) + 1)
        by1 = min(h, math.ceil(max(y0, y1) + r) + 1)
        if bx1 <= bx0 or by1 <= by0:
            return
        yy, xx = np.mgrid[by0:by1, bx0:bx1]
        cx, cy = xx + 0.5, yy + 0.5
        hit = np.zeros(cx.shape, dtype=bool)
        for i in range(steps + 1):
            t = i / steps
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            hit |= (cx - px) ** 2 + (cy - py) ** 2 <= r * r
        clip = s["clip"]
        if clip is not None:
            hit &= clip[by0:by1, bx0:bx1]
        region = img[by0:by1, bx0:bx1]
        region[hit] = s["color"]

    # ------------------------------------------------------------------
    # 사각형 선택과 이동
    # ------------------------------------------------------------------
    def select_press(self, x: float, y: float) -> None:
        self._stroke = {"kind": "select", "start": (x, y), "end": (x, y)}

    def select_drag(self, x: float, y: float) -> None:
        s = self._stroke
        if s is None or s["kind"] != "select":
            return
        s["end"] = (x, y)
        self._changed()

    def select_release(self) -> None:
        s = self._stroke
        if s is None or s["kind"] != "select":
            return
        self._stroke = None
        (x0, y0), (x1, y1) = s["start"], s["end"]
        rect = normalize_rect(x0, y0, x1, y1, self.width, self.height)
        if math.hypot(x1 - x0, y1 - y0) < 1:
            rect = None            # 그냥 클릭하면 선택 해제
        if rect == self._selection:
            self._changed()
            return
        self._selection = rect
        self._push("사각형 선택" if rect else "선택 해제")

    def pending_selection_rect(self) -> tuple | None:
        """드래그 중인 선택 사각형 (화면 표시용)."""
        s = self._stroke
        if s is None or s["kind"] != "select":
            return None
        (x0, y0), (x1, y1) = s["start"], s["end"]
        return normalize_rect(x0, y0, x1, y1, self.width, self.height)

    def set_selection(self, rect: tuple | None, name: str | None = None) -> None:
        if rect is not None:
            rect = normalize_rect(*rect, self.width, self.height)
        if rect == self._selection:
            return
        self._selection = rect
        self._push(name or ("사각형 선택" if rect else "선택 해제"))

    def select_all(self) -> None:
        self.set_selection((0, 0, self.width, self.height), "전체 선택")

    def deselect(self) -> None:
        self.set_selection(None, "선택 해제")

    def move_press(self, x: float, y: float) -> None:
        if self._selection is None:
            return
        x0, y0, x1, y1 = self._selection
        piece = self._image[y0:y1, x0:x1].copy()
        base = self._image.copy()
        base[y0:y1, x0:x1] = 0      # 옮긴 자리는 투명 (DECISIONS.md)
        self._stroke = {"kind": "move", "start": (x, y), "piece": piece, "base": base,
                        "rect": self._selection, "offset": (0, 0), "preview": self._image}

    def move_drag(self, x: float, y: float) -> None:
        s = self._stroke
        if s is None or s["kind"] != "move":
            return
        sx, sy = s["start"]
        off = (round(x - sx), round(y - sy))
        if off == s["offset"]:
            return
        s["offset"] = off
        s["preview"] = self._paste(s["base"], s["piece"], s["rect"], off)
        self._changed()

    def move_release(self) -> None:
        s = self._stroke
        if s is None or s["kind"] != "move":
            return
        self._stroke = None
        if s["offset"] == (0, 0):
            self._changed()
            return
        dx, dy = s["offset"]
        x0, y0, x1, y1 = s["rect"]
        self._image = s["preview"]
        self._selection = normalize_rect(x0 + dx, y0 + dy, x1 + dx, y1 + dy,
                                         self.width, self.height)
        self._push("선택 영역 이동")

    def moving_rect(self) -> tuple | None:
        s = self._stroke
        if s is None or s["kind"] != "move":
            return None
        dx, dy = s["offset"]
        x0, y0, x1, y1 = s["rect"]
        return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    @staticmethod
    def _paste(base: np.ndarray, piece: np.ndarray, rect: tuple, off: tuple) -> np.ndarray:
        out = base.copy()
        h, w = out.shape[:2]
        x0, y0 = rect[0] + off[0], rect[1] + off[1]
        ph, pw = piece.shape[:2]
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(w, x0 + pw), min(h, y0 + ph)
        if dx1 <= dx0 or dy1 <= dy0:
            return out
        src = piece[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
        mask = np.full(src.shape[:2], 255, dtype=np.uint8)
        out[dy0:dy1, dx0:dx1] = fc.composite(out[dy0:dy1, dx0:dx1], src, mask)
        return out

    # ------------------------------------------------------------------
    # 색
    # ------------------------------------------------------------------
    def set_colors(self, primary=None, secondary=None) -> None:
        if primary is not None:
            self.primary = tuple(primary)
        if secondary is not None:
            self.secondary = tuple(secondary)
        self._changed()

    def swap_colors(self) -> None:
        self.primary, self.secondary = self.secondary, self.primary
        self._changed()


_OPTION_NAMES = {
    "mode": "채우기 모드 변경",
    "tolerance": "허용 오차 변경",
    "alpha_mode": "알파 처리 변경",
    "sampling": "샘플링 변경",
    "antialias": "안티앨리어싱 변경",
    "fill_type": "채우기 유형 변경",
    "blend": "혼합 모드 변경",
    "selection_quality": "선택 영역 품질 변경",
}


def _option_record_name(kw: dict) -> str:
    if len(kw) == 1:
        return _OPTION_NAMES.get(next(iter(kw)), "채우기 옵션 변경")
    return "채우기 옵션 변경"
