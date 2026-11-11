"""채우기 도구의 순수 로직 (UI 없음).

모든 함수는 numpy 배열을 입력받는다. 이미지 배열은 (높이, 너비, 4) 모양의
uint8 RGBA 배열이고, 선택 영역은 (높이, 너비) 모양의 bool 배열 또는 None 이다.

명세: docs/SPEC_fill.md 3~8장.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

# ---------------------------------------------------------------------------
# 옵션 값
# ---------------------------------------------------------------------------

MODE_CONTIGUOUS = "contiguous"   # 연속 모드
MODE_GLOBAL = "global"           # 전체 모드

ALPHA_PREMULTIPLIED = "premultiplied"
ALPHA_STRAIGHT = "straight"

SAMPLING_LAYER = "layer"
SAMPLING_IMAGE = "image"

FILL_SOLID = "solid"
FILL_HATCH_HORIZONTAL = "hatch_h"
FILL_HATCH_VERTICAL = "hatch_v"
FILL_HATCH_DIAGONAL = "hatch_d"
FILL_HATCH_CROSS = "hatch_x"
FILL_TYPES = (FILL_SOLID, FILL_HATCH_HORIZONTAL, FILL_HATCH_VERTICAL,
              FILL_HATCH_DIAGONAL, FILL_HATCH_CROSS)

BLEND_NORMAL = "normal"
BLEND_MULTIPLY = "multiply"

# 명세 5.3절 [권장]: 거리 <= 화면 허용 오차(%) / 100 × 264
TOLERANCE_SCALE = 264.0


@dataclass(frozen=True)
class FillOptions:
    """채우기 도구 옵션. 기본값은 명세 3.2절의 [관찰] 값이다."""

    mode: str = MODE_CONTIGUOUS
    tolerance: float = 50.0            # 화면 표시 값(%), 실수로 보관 (5.3절 [확인])
    alpha_mode: str = ALPHA_PREMULTIPLIED
    sampling: str = SAMPLING_LAYER
    antialias: bool = True
    fill_type: str = FILL_SOLID
    blend: str = BLEND_NORMAL
    selection_quality: str = "smooth"  # UI만 있음 (DECISIONS.md)

    def with_(self, **kw) -> "FillOptions":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# 허용 오차와 색 거리 (5장)
# ---------------------------------------------------------------------------

def tolerance_to_threshold(percent: float) -> int:
    """화면 허용 오차(%)를 0~255 정수 임계값으로 바꾼다.

    5.3절 [권장]: percent/100 × 264. 비교할 때 0~255 정수로 변환한다 [확인].
    정수 변환은 내림, 255를 넘으면 255로 자른다 (DECISIONS.md).
    """
    p = min(100.0, max(0.0, float(percent)))
    value = p / 100.0 * TOLERANCE_SCALE
    return int(min(255, math.floor(value + 1e-9)))


def _round_half_away(x: np.ndarray) -> np.ndarray:
    """0에서 먼 쪽으로 반올림 (5.2절 [확인]). 입력은 0 이상."""
    return np.floor(x + 0.5)


def _to_unit(pixels: np.ndarray, alpha_mode: str) -> np.ndarray:
    """RGBA uint8 → 0~1 실수 (5.2절, 0~1 범위는 [추측])."""
    f = pixels.astype(np.float64) / 255.0
    if alpha_mode == ALPHA_PREMULTIPLIED:
        # 5.2절 [권장]: R, G, B에 각자의 알파를 곱한다.
        f[..., :3] *= f[..., 3:4]
    return f


def color_distance(image: np.ndarray, ref, alpha_mode: str = ALPHA_PREMULTIPLIED) -> np.ndarray:
    """각 픽셀과 기준색의 거리(0~255 정수 배열)를 구한다.

    image: (..., 4) uint8, ref: 길이 4의 RGBA.
    """
    img = np.asarray(image)
    ref_arr = np.asarray(ref, dtype=np.uint8).reshape((1,) * (img.ndim - 1) + (4,))
    a = _to_unit(img, alpha_mode)
    b = _to_unit(ref_arr, alpha_mode)
    diff = a - b
    dist_f = np.sqrt(np.sum(diff * diff, axis=-1))          # 유클리드 거리 [확인]
    dist = _round_half_away(dist_f * 0.5 * 255.0)            # ×0.5×255, 반올림 [확인]
    # 5.2절 [권장]: 두 색이 조금이라도 다르면 거리를 최소 1로 둔다.
    different = np.any(diff != 0.0, axis=-1)
    dist = np.where(different, np.maximum(dist, 1.0), dist)
    return np.clip(dist, 0, 255).astype(np.int32)


def color_distance_pair(c1, c2, alpha_mode: str = ALPHA_PREMULTIPLIED) -> int:
    """두 색 사이의 거리 (시험과 디버그용)."""
    img = np.asarray(c1, dtype=np.uint8).reshape(1, 4)
    return int(color_distance(img, c2, alpha_mode)[0])


def match_mask(sample: np.ndarray, ref, tolerance_percent: float,
               alpha_mode: str = ALPHA_PREMULTIPLIED) -> np.ndarray:
    """기준색과의 거리가 허용 오차 이하인 픽셀 (5.1절: 같은 경우도 포함 [확인])."""
    thr = tolerance_to_threshold(tolerance_percent)
    return color_distance(sample, ref, alpha_mode) <= thr


# ---------------------------------------------------------------------------
# 채우기 모드 (4장)
# ---------------------------------------------------------------------------

def contiguous_region(match: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """시작점과 8방향으로 이어진 match 영역 (4.1절).

    명세의 행 단위 확장(스캔라인)과 같은 결과를 내도록, 각 행의 연속 구간(run)을
    단위로 넓혀 간다. 위/아래 행의 구간은 대각선까지 포함해(8방향 [관찰])
    한 칸이라도 닿으면 이어진 것으로 본다.
    """
    h, w = match.shape
    out = np.zeros((h, w), dtype=bool)
    if not (0 <= sx < w and 0 <= sy < h) or not match[sy, sx]:
        return out

    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = match
    d = np.diff(padded, axis=1)
    s_rows, s_cols = np.nonzero(d == 1)    # 구간 시작 (포함)
    _, e_cols = np.nonzero(d == -1)        # 구간 끝 (제외)
    row_ptr = np.searchsorted(s_rows, np.arange(h + 1))

    lo, hi = row_ptr[sy], row_ptr[sy + 1]
    k = lo + int(np.searchsorted(s_cols[lo:hi], sx, side="right")) - 1
    visited = np.zeros(len(s_cols), dtype=bool)
    visited[k] = True
    stack = [k]
    while stack:
        r = stack.pop()
        y = int(s_rows[r])
        s, e = int(s_cols[r]), int(e_cols[r])
        out[y, s:e] = True
        for ny in (y - 1, y + 1):
            if ny < 0 or ny >= h:
                continue
            lo, hi = row_ptr[ny], row_ptr[ny + 1]
            if lo == hi:
                continue
            # 이웃 구간 [s2, e2)가 [s-1, e] 범위와 겹치면 8방향으로 이어진다.
            j0 = lo + int(np.searchsorted(e_cols[lo:hi], s, side="left"))
            j1 = lo + int(np.searchsorted(s_cols[lo:hi], e, side="right"))
            for j in range(j0, j1):
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
    return out


def compute_region(sample: np.ndarray, seed_x: float, seed_y: float,
                   options: FillOptions, mode: str | None = None,
                   selection: np.ndarray | None = None) -> np.ndarray:
    """칠할 영역(포함 여부 bool 마스크)을 구한다.

    - 시작점은 소수 좌표이며 x, y를 각각 내림한 픽셀을 쓴다 (3.1절 [확인]).
    - 시작점이 캔버스 밖이면 칠하지 않는다 (8장 [추측]).
    - 선택 영역이 있으면 그 안에서만 칠하고, 시작점이 선택 밖이면 아무것도
      칠하지 않는다 (연속 모드 [관찰], 전체 모드 [권장]).
    """
    mode = mode or options.mode
    h, w = sample.shape[:2]
    px, py = math.floor(seed_x), math.floor(seed_y)
    empty = np.zeros((h, w), dtype=bool)
    if not (0 <= px < w and 0 <= py < h):
        return empty
    if selection is not None and not selection[py, px]:
        return empty

    # 기준색 = 시작 픽셀 [확인]. 투명한 시작 픽셀도 네 채널로 비교한다 (추측, 8장).
    # 시작 픽셀은 거리 0이므로 항상 포함된다 (추측, 5.1절).
    # 허용 오차 100%면 임계값 255로 모든 픽셀이 포함된다 (추측, 8장).
    ref = sample[py, px]
    match = match_mask(sample, ref, options.tolerance, options.alpha_mode)
    # 선택 영역이 없으면 캔버스 전체가 대상이다 (추측, 4.1절).
    if selection is not None:
        match &= selection
    if mode == MODE_GLOBAL:
        return match
    return contiguous_region(match, px, py)


# ---------------------------------------------------------------------------
# 경계 처리 (6.2절)
# ---------------------------------------------------------------------------

# 6.2절 [관찰] 값 (검정 위에 빨강): 변 하나 ≈ 26, 모서리만 ≈ 8, 두 변 ≈ 33.
AA_EDGE_ONE = 26
AA_EDGE_TWO = 33
AA_CORNER_ONLY = 8


def _neighbor_counts(inc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(inc, 1).astype(np.int8)
    edges = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    corners = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    return edges, corners


def build_mask(region: np.ndarray, antialias: bool,
               selection: np.ndarray | None = None) -> np.ndarray:
    """포함 여부 마스크로부터 0~255 적용 비율 마스크를 만든다.

    - 안티앨리어싱 끔 [추측]: 포함 255, 나머지 0.
    - 안티앨리어싱 켬: 영역 바로 바깥 1픽셀 둘레에 부분 불투명도를 준다.
      값은 6.2절 [관찰] 수치를 그대로 쓴다 (세부 공식은 [UNKNOWN],
      DECISIONS.md 참고).
    """
    mask = np.where(region, 255, 0).astype(np.uint8)
    if not antialias or not region.any():
        return mask
    edges, corners = _neighbor_counts(region)
    outside = ~region
    fringe = np.zeros(region.shape, dtype=np.uint8)
    fringe[(edges == 0) & (corners > 0)] = AA_CORNER_ONLY
    fringe[edges == 1] = AA_EDGE_ONE
    fringe[edges >= 2] = AA_EDGE_TWO
    fringe[~outside] = 0
    if selection is not None:
        fringe[~selection] = 0
    return np.maximum(mask, fringe)


# ---------------------------------------------------------------------------
# 칠하기 (6.1절)
# ---------------------------------------------------------------------------

def paint_source(shape: tuple[int, int], fill_type: str, fg, bg) -> np.ndarray:
    """칠할 색 배열 (높이, 너비, 4). 해치 무늬는 전경색과 배경색을 쓴다 [추측]."""
    h, w = shape
    fg = np.asarray(fg, dtype=np.uint8)
    bg = np.asarray(bg, dtype=np.uint8)
    if fill_type == FILL_SOLID:
        return np.broadcast_to(fg, (h, w, 4)).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    period = 6
    if fill_type == FILL_HATCH_HORIZONTAL:
        on = yy % period == 0
    elif fill_type == FILL_HATCH_VERTICAL:
        on = xx % period == 0
    elif fill_type == FILL_HATCH_DIAGONAL:
        on = (xx + yy) % period == 0
    elif fill_type == FILL_HATCH_CROSS:
        on = (yy % period == 0) | (xx % period == 0)
    else:
        raise ValueError(f"알 수 없는 채우기 유형: {fill_type}")
    return np.where(on[..., None], fg, bg).astype(np.uint8)


def composite(dst: np.ndarray, src: np.ndarray, mask: np.ndarray,
              blend: str = BLEND_NORMAL) -> np.ndarray:
    """src를 mask(0~255)를 적용 비율로 dst 위에 합성한다 (6.1절 [권장]).

    합성 공식은 [UNKNOWN] 이므로 일반 알파 합성(source-over)을 쓴다.
    """
    d = dst.astype(np.float64) / 255.0
    s = src.astype(np.float64) / 255.0
    m = mask.astype(np.float64)[..., None] / 255.0
    s_rgb, s_a = s[..., :3], s[..., 3:4] * m
    d_rgb, d_a = d[..., :3], d[..., 3:4]
    if blend == BLEND_MULTIPLY:
        # 바탕이 불투명한 만큼 곱한 색을 쓴다.
        s_rgb = s_rgb * (d_rgb * d_a + (1.0 - d_a))
    elif blend != BLEND_NORMAL:
        raise ValueError(f"알 수 없는 혼합 모드: {blend}")
    out_a = s_a + d_a * (1.0 - s_a)
    safe = np.where(out_a > 0, out_a, 1.0)
    out_rgb = (s_rgb * s_a + d_rgb * d_a * (1.0 - s_a)) / safe
    out = np.concatenate([out_rgb, out_a], axis=-1)
    out = np.where(out_a > 0, out, 0.0)
    out = np.clip(np.floor(out * 255.0 + 0.5), 0, 255).astype(np.uint8)
    # 마스크가 0인 픽셀은 원래 값을 그대로 둔다.
    return np.where(mask[..., None] > 0, out, dst).astype(np.uint8)


@dataclass
class FillResult:
    region: np.ndarray   # 포함 여부 (bool)
    mask: np.ndarray     # 적용 비율 (uint8)
    image: np.ndarray    # 칠한 결과 (uint8 RGBA)


def flood_fill(target: np.ndarray, seed_x: float, seed_y: float, fg, bg,
               options: FillOptions = FillOptions(), mode: str | None = None,
               selection: np.ndarray | None = None,
               sample: np.ndarray | None = None) -> FillResult:
    """채우기 한 번을 계산한다.

    target: 칠할 레이어. sample: 색을 비교할 이미지 (샘플링 옵션에 따라
    레이어 또는 합친 이미지). 생략하면 target을 쓴다.
    """
    sample = target if sample is None else sample
    region = compute_region(sample, seed_x, seed_y, options, mode, selection)
    mask = build_mask(region, options.antialias, selection)
    if not mask.any():
        return FillResult(region, mask, target.copy())
    src = paint_source(target.shape[:2], options.fill_type, fg, bg)
    return FillResult(region, mask, composite(target, src, mask, options.blend))
