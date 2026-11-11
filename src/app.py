"""FillLab의 tkinter 화면.

위쪽: 도구 이름과 옵션이 한 줄로 이어지는 옵션 바 (오른쪽 끝에 확정 버튼)
왼쪽: 도구 상자와 기본색/보조색
가운데: 캔버스 (스크롤, 확대/축소)
아래쪽: 상태 표시줄 (커서 좌표, 이미지 크기, 배율)

그림 상태와 동작은 editor.Editor에 있고, 이 파일은 입력을 전달하고 화면을 그린다.
"""

from __future__ import annotations

import math
import os
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import fill_core as fc
from editor import (BUTTON_LEFT, BUTTON_RIGHT, PHASE_EDITING, TOOL_BRUSH, TOOL_FILL,
                    TOOL_MOVE, TOOL_SELECT, Editor)

APP_NAME = "FillLab"

ZOOM_LEVELS = [0.125, 0.25, 1 / 3, 0.5, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]

TOOLS = [
    # (도구, 표시 이름, 단축키)
    (TOOL_FILL, "채우기", "F"),       # F 키 [추측]
    (TOOL_BRUSH, "브러시", "B"),
    (TOOL_SELECT, "사각형 선택", "S"),
    (TOOL_MOVE, "선택 이동", "M"),
]

CURSORS = {
    TOOL_FILL: "crosshair",   # 페인트통 커서는 만들지 않는다 (DECISIONS.md)
    TOOL_BRUSH: "circle",
    TOOL_SELECT: "tcross",
    TOOL_MOVE: "fleur",
}

# 옵션 바의 드롭다운: (표시 문구, 내부 값)
MODE_CHOICES = [("연속 영역", fc.MODE_CONTIGUOUS), ("전체 이미지", fc.MODE_GLOBAL)]
ALPHA_CHOICES = [("프리멀티플라이드", fc.ALPHA_PREMULTIPLIED), ("스트레이트", fc.ALPHA_STRAIGHT)]
SAMPLING_CHOICES = [("레이어", fc.SAMPLING_LAYER), ("이미지", fc.SAMPLING_IMAGE)]
AA_CHOICES = [("부드럽게", True), ("또렷하게", False)]   # 안티앨리어싱 사용/해제
FILL_TYPE_CHOICES = [("단색", fc.FILL_SOLID), ("가로줄", fc.FILL_HATCH_HORIZONTAL),
                     ("세로줄", fc.FILL_HATCH_VERTICAL), ("빗금", fc.FILL_HATCH_DIAGONAL),
                     ("격자", fc.FILL_HATCH_CROSS)]
BLEND_CHOICES = [("표준", fc.BLEND_NORMAL), ("곱하기", fc.BLEND_MULTIPLY)]
QUALITY_CHOICES = [("부드럽게", "smooth"), ("보통", "normal"), ("빠르게", "fast")]

CHECKER = 8   # 투명 배경 바둑판 한 칸 (화면 픽셀)


def _hex(color) -> str:
    return "#%02x%02x%02x" % tuple(color[:3])


class Dropdown(ttk.Combobox):
    """(표시 문구, 값) 목록을 고르는 읽기 전용 드롭다운."""

    def __init__(self, master, choices, on_change, width=None):
        self.choices = choices
        labels = [c[0] for c in choices]
        # 한글은 글자 폭이 넓으므로 2칸으로 센다.
        text_w = max(sum(1.6 if ord(ch) > 127 else 1 for ch in s) for s in labels)
        super().__init__(master, values=labels, state="readonly",
                         width=width or math.ceil(text_w) + 1)
        self._on_change = on_change
        self.bind("<<ComboboxSelected>>", self._selected)

    def _selected(self, _event=None):
        self.selection_clear()
        self._on_change(self.choices[self.current()][1])

    def set_value(self, value):
        for i, (_, v) in enumerate(self.choices):
            if v == value:
                if self.current() != i:
                    self.current(i)
                return


class FillLabApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.editor = Editor(800, 600)
        self.editor.on_change = self.request_render
        self.zoom = 1.0
        self.path: str | None = None
        self.saved_index = self.editor.index
        self._render_pending = False
        self._photo = None
        self._checker_cache: dict = {}
        self._syncing = False
        self._last_mouse = None
        self._pressed_button = None

        root.title(APP_NAME)
        root.geometry("1440x860")
        root.minsize(720, 480)
        root.protocol("WM_DELETE_WINDOW", self.quit)

        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Tool.TButton", padding=(6, 6))
        style.configure("ActiveTool.TButton", padding=(6, 6), font=("Segoe UI", 9, "bold"))

        self._build_menu()
        self._build_option_bar()
        self._build_status_bar()
        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        self._build_toolbox(body)
        self._build_canvas(body)
        self._bind_keys()

        self.select_tool(TOOL_FILL)
        self.update_title()
        self.request_render()

    # ------------------------------------------------------------------
    # 화면 구성
    # ------------------------------------------------------------------
    def _build_menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=False)
        f.add_command(label="새 이미지...", accelerator="Ctrl+N", command=self.new_image)
        f.add_command(label="열기...", accelerator="Ctrl+O", command=self.open_image)
        f.add_command(label="저장", accelerator="Ctrl+S", command=self.save)
        f.add_command(label="다른 이름으로 저장...", accelerator="Ctrl+Shift+S", command=self.save_as)
        f.add_separator()
        f.add_command(label="끝내기", command=self.quit)
        m.add_cascade(label="파일", menu=f)

        e = tk.Menu(m, tearoff=False)
        e.add_command(label="실행 취소", accelerator="Ctrl+Z", command=self.undo)
        e.add_command(label="다시 실행", accelerator="Ctrl+Y", command=self.redo)
        e.add_separator()
        e.add_command(label="전체 선택", accelerator="Ctrl+A", command=self.editor.select_all)
        e.add_command(label="선택 해제", accelerator="Ctrl+D", command=self.editor.deselect)
        e.add_separator()
        e.add_command(label="채우기 확정", accelerator="Enter", command=self.commit_fill)
        e.add_command(label="채우기 취소", accelerator="Esc", command=self.cancel_fill)
        m.add_cascade(label="편집", menu=e)

        v = tk.Menu(m, tearoff=False)
        v.add_command(label="확대", accelerator="Ctrl++", command=lambda: self.step_zoom(+1))
        v.add_command(label="축소", accelerator="Ctrl+-", command=lambda: self.step_zoom(-1))
        v.add_command(label="실제 크기", accelerator="Ctrl+0", command=lambda: self.set_zoom(1.0))
        v.add_command(label="창에 맞추기", accelerator="Ctrl+B", command=self.zoom_to_fit)
        m.add_cascade(label="보기", menu=v)
        self.root.config(menu=m)

    def _build_option_bar(self):
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(side="top", fill="x")
        ttk.Separator(self.root, orient="horizontal").pack(side="top", fill="x")
        self.tool_label = ttk.Label(bar, text="", width=7, font=("Segoe UI", 10, "bold"))
        self.tool_label.pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        # 오른쪽 끝의 확정 버튼 (채우기 전에는 비활성 [관찰])
        self.commit_btn = ttk.Button(bar, text="✓ 확정", width=7, command=self.commit_fill,
                                     state="disabled")
        self.commit_btn.pack(side="right")
        self.options_holder = ttk.Frame(bar)
        self.options_holder.pack(side="left", fill="x", expand=True)

        self.fill_opts = self._build_fill_options(self.options_holder)
        self.brush_opts = self._build_brush_options(self.options_holder)
        self.select_opts = self._build_select_options(self.options_holder)
        self.move_opts = ttk.Frame(self.options_holder)
        ttk.Label(self.move_opts, text="선택 영역 안을 끌어서 옮깁니다. 옮긴 자리는 투명해집니다."
                  ).pack(side="left")

    def _opt_label(self, parent, text):
        ttk.Label(parent, text=text).pack(side="left", padx=(7, 2))

    def _build_fill_options(self, parent):
        fr = ttk.Frame(parent)
        self._opt_label(fr, "방식")
        self.dd_mode = Dropdown(fr, MODE_CHOICES, lambda v: self.set_fill_option(mode=v))
        self.dd_mode.pack(side="left")

        self._opt_label(fr, "허용 오차")
        ttk.Button(fr, text="−", width=2, command=lambda: self.nudge_tolerance(-1)).pack(side="left")
        self.tol_var = tk.DoubleVar(value=self.editor.fill_options.tolerance)
        self.tol_scale = ttk.Scale(fr, from_=0, to=100, orient="horizontal", length=96,
                                   variable=self.tol_var, command=self._tolerance_moved)
        self.tol_scale.pack(side="left", padx=2)
        self.tol_scale.bind("<ButtonRelease-1>", lambda e: self.editor.flush_option_record())
        ttk.Button(fr, text="+", width=2, command=lambda: self.nudge_tolerance(+1)).pack(side="left")
        self.tol_text = ttk.Label(fr, text="50%", width=4, anchor="e")
        self.tol_text.pack(side="left")

        self._opt_label(fr, "알파")
        self.dd_alpha = Dropdown(fr, ALPHA_CHOICES, lambda v: self.set_fill_option(alpha_mode=v),
                                 width=16)
        self.dd_alpha.pack(side="left")
        self._opt_label(fr, "색 읽기")
        self.dd_sampling = Dropdown(fr, SAMPLING_CHOICES, lambda v: self.set_fill_option(sampling=v))
        self.dd_sampling.pack(side="left")
        self._opt_label(fr, "경계")
        self.dd_aa = Dropdown(fr, AA_CHOICES, lambda v: self.set_fill_option(antialias=v))
        self.dd_aa.pack(side="left")
        self._opt_label(fr, "유형")
        self.dd_type = Dropdown(fr, FILL_TYPE_CHOICES, lambda v: self.set_fill_option(fill_type=v))
        self.dd_type.pack(side="left")
        self._opt_label(fr, "혼합")
        self.dd_blend = Dropdown(fr, BLEND_CHOICES, lambda v: self.set_fill_option(blend=v))
        self.dd_blend.pack(side="left")
        self._opt_label(fr, "선택 품질")
        self.dd_quality = Dropdown(fr, QUALITY_CHOICES,
                                   lambda v: self.set_fill_option(selection_quality=v))
        self.dd_quality.pack(side="left")
        return fr

    def _build_brush_options(self, parent):
        fr = ttk.Frame(parent)
        self._opt_label(fr, "브러시 크기")
        self.brush_var = tk.IntVar(value=self.editor.brush_size)
        sp = ttk.Spinbox(fr, from_=1, to=200, width=5, textvariable=self.brush_var,
                         command=self._brush_size_changed)
        sp.pack(side="left")
        sp.bind("<Return>", lambda e: self._brush_size_changed())
        sp.bind("<FocusOut>", lambda e: self._brush_size_changed())
        ttk.Scale(fr, from_=1, to=100, orient="horizontal", length=140,
                  variable=self.brush_var,
                  command=lambda v: self._brush_size_changed(round(float(v)))).pack(side="left", padx=4)
        ttk.Label(fr, text="왼쪽 버튼: 기본색, 오른쪽 버튼: 보조색").pack(side="left", padx=8)
        return fr

    def _build_select_options(self, parent):
        fr = ttk.Frame(parent)
        ttk.Button(fr, text="전체 선택", command=self.editor.select_all).pack(side="left", padx=2)
        ttk.Button(fr, text="선택 해제", command=self.editor.deselect).pack(side="left", padx=2)
        ttk.Label(fr, text="끌어서 사각형을 선택합니다. 그냥 클릭하면 선택이 풀립니다."
                  ).pack(side="left", padx=8)
        return fr

    def _build_toolbox(self, parent):
        box = ttk.Frame(parent, padding=4)
        box.pack(side="left", fill="y")
        ttk.Separator(parent, orient="vertical").pack(side="left", fill="y")
        self.tool_buttons = {}
        for tool, label, key in TOOLS:
            b = ttk.Button(box, text=f"{label} ({key})", width=13, style="Tool.TButton",
                           command=lambda t=tool: self.select_tool(t))
            b.pack(side="top", pady=2, fill="x")
            self.tool_buttons[tool] = b

        ttk.Separator(box, orient="horizontal").pack(side="top", fill="x", pady=8)
        ttk.Label(box, text="색").pack(side="top")
        sw = tk.Canvas(box, width=64, height=64, highlightthickness=0, bg=self.root.cget("bg"))
        sw.pack(side="top", pady=4)
        self.swatch = sw
        sw.bind("<Button-1>", self._swatch_click)
        row = ttk.Frame(box)
        row.pack(side="top")
        ttk.Button(row, text="교체", width=5, command=self.swap_colors).pack(side="left")
        ttk.Button(row, text="초기화", width=6, command=self.reset_colors).pack(side="left")
        ttk.Label(box, text="앞: 기본색\n뒤: 보조색", justify="center",
                  foreground="#666").pack(side="top", pady=4)

    def _build_canvas(self, parent):
        fr = ttk.Frame(parent)
        fr.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(fr, bg="#8a8a8a", highlightthickness=0)
        hs = ttk.Scrollbar(fr, orient="horizontal", command=self._xview)
        vs = ttk.Scrollbar(fr, orient="vertical", command=self._yview)
        self.canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        fr.rowconfigure(0, weight=1)
        fr.columnconfigure(0, weight=1)

        c = self.canvas
        for b in (1, 3):
            c.bind(f"<ButtonPress-{b}>", lambda e, b=b: self.on_press(e, b))
            c.bind(f"<B{b}-Motion>", self.on_drag)
            c.bind(f"<ButtonRelease-{b}>", lambda e, b=b: self.on_release(e, b))
        c.bind("<Motion>", self.on_motion)
        c.bind("<Leave>", lambda e: self._set_cursor_status(None))
        c.bind("<Configure>", lambda e: self.request_render())
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Shift-MouseWheel>", self.on_wheel)
        c.bind("<Control-MouseWheel>", self.on_wheel)

    def _build_status_bar(self):
        bar = ttk.Frame(self.root, padding=(8, 2))
        bar.pack(side="bottom", fill="x")
        ttk.Separator(self.root, orient="horizontal").pack(side="bottom", fill="x")
        self.st_cursor = ttk.Label(bar, text="", width=18)
        self.st_cursor.pack(side="left")
        self.st_size = ttk.Label(bar, text="", width=18)
        self.st_size.pack(side="left")
        self.st_message = ttk.Label(bar, text="", foreground="#555")
        self.st_message.pack(side="left", padx=10)
        ttk.Button(bar, text="+", width=2, command=lambda: self.step_zoom(+1)).pack(side="right")
        self.st_zoom = ttk.Label(bar, text="100%", width=7, anchor="center")
        self.st_zoom.pack(side="right")
        ttk.Button(bar, text="−", width=2, command=lambda: self.step_zoom(-1)).pack(side="right")
        ttk.Label(bar, text="배율").pack(side="right", padx=4)

    def _bind_keys(self):
        r = self.root
        r.bind("<Control-n>", lambda e: self.new_image())
        r.bind("<Control-o>", lambda e: self.open_image())
        r.bind("<Control-s>", lambda e: self.save())
        r.bind("<Control-S>", lambda e: self.save_as())
        r.bind("<Control-z>", lambda e: self.undo())
        r.bind("<Control-y>", lambda e: self.redo())
        r.bind("<Control-Z>", lambda e: self.redo())
        r.bind("<Control-a>", lambda e: self._unless_typing(e, self.editor.select_all))
        r.bind("<Control-d>", lambda e: self.editor.deselect())
        r.bind("<Control-plus>", lambda e: self.step_zoom(+1))
        r.bind("<Control-equal>", lambda e: self.step_zoom(+1))
        r.bind("<Control-minus>", lambda e: self.step_zoom(-1))
        r.bind("<Control-0>", lambda e: self.set_zoom(1.0))
        r.bind("<Control-b>", lambda e: self.zoom_to_fit())
        r.bind("<Escape>", lambda e: self.cancel_fill())
        r.bind("<Return>", lambda e: self._unless_typing(e, self.commit_fill))
        r.bind("<KP_Enter>", lambda e: self._unless_typing(e, self.commit_fill))
        for tool, _, key in TOOLS:
            for k in (key.lower(), key.upper()):
                r.bind(f"<KeyPress-{k}>", lambda e, t=tool: self._unless_typing(e, lambda: self.select_tool(t)))
        r.bind("<KeyPress-x>", lambda e: self._unless_typing(e, self.swap_colors))

    @staticmethod
    def _unless_typing(event, action):
        w = event.widget
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Spinbox)):
            return None
        if event.state & 0x0004 and event.keysym.lower() in ("f", "b", "s", "m", "x"):
            return None   # Ctrl과 함께 누른 경우는 단축키가 처리
        action()
        return "break"

    # ------------------------------------------------------------------
    # 도구와 옵션
    # ------------------------------------------------------------------
    def select_tool(self, tool):
        self.editor.set_tool(tool)
        for t, b in self.tool_buttons.items():
            b.configure(style="ActiveTool.TButton" if t == tool else "Tool.TButton")
            b.state(["pressed"] if t == tool else ["!pressed"])
        label = next(lbl for t, lbl, _ in TOOLS if t == tool)
        self.tool_label.configure(text=label)
        for fr in (self.fill_opts, self.brush_opts, self.select_opts, self.move_opts):
            fr.pack_forget()
        {TOOL_FILL: self.fill_opts, TOOL_BRUSH: self.brush_opts,
         TOOL_SELECT: self.select_opts, TOOL_MOVE: self.move_opts}[tool].pack(side="left")
        self.canvas.configure(cursor=CURSORS[tool])
        if tool == TOOL_FILL:
            self.commit_btn.pack(side="right", before=self.options_holder)
        else:
            self.commit_btn.pack_forget()
        self.request_render()

    def set_fill_option(self, record=True, **kw):
        if self._syncing:
            return
        self.editor.set_fill_option(record=record, **kw)

    def _tolerance_moved(self, value):
        v = round(float(value))
        self.tol_text.configure(text=f"{v}%")
        if self._syncing or v == round(self.editor.fill_options.tolerance):
            return
        # 끄는 동안은 결과만 다시 만들고, 놓을 때 한 번 기록한다.
        self.set_fill_option(record=False, tolerance=float(v))

    def nudge_tolerance(self, delta):
        v = min(100.0, max(0.0, round(self.editor.fill_options.tolerance) + delta))
        self.set_fill_option(tolerance=v)

    def _brush_size_changed(self, value=None):
        try:
            v = int(value if value is not None else self.brush_var.get())
        except (tk.TclError, ValueError):
            v = self.editor.brush_size
        v = max(1, min(200, v))
        self.editor.brush_size = v
        if self.brush_var.get() != v:
            self.brush_var.set(v)

    def sync_options(self):
        """편집기의 옵션 값(실행 취소로 바뀔 수 있음)을 옵션 바에 반영한다."""
        o = self.editor.fill_options
        self._syncing = True
        try:
            self.dd_mode.set_value(o.mode)
            self.dd_alpha.set_value(o.alpha_mode)
            self.dd_sampling.set_value(o.sampling)
            self.dd_aa.set_value(o.antialias)
            self.dd_type.set_value(o.fill_type)
            self.dd_blend.set_value(o.blend)
            self.dd_quality.set_value(o.selection_quality)
            if abs(self.tol_var.get() - o.tolerance) > 1e-9:
                self.tol_var.set(o.tolerance)
            self.tol_text.configure(text=f"{round(o.tolerance)}%")
        finally:
            self._syncing = False
        editing = self.editor.fill_active and self.editor.fill_state.phase == PHASE_EDITING
        self.commit_btn.configure(state="normal" if editing else "disabled")

    # ------------------------------------------------------------------
    # 색
    # ------------------------------------------------------------------
    def draw_swatch(self):
        c = self.swatch
        c.delete("all")
        c.create_rectangle(24, 24, 60, 60, fill=_hex(self.editor.secondary), outline="#333")
        c.create_rectangle(4, 4, 40, 40, fill=_hex(self.editor.primary), outline="#333")

    def _swatch_click(self, event):
        primary = event.x < 40 and event.y < 40
        cur = self.editor.primary if primary else self.editor.secondary
        title = "기본색 고르기" if primary else "보조색 고르기"
        rgb, _ = colorchooser.askcolor(color=_hex(cur), title=title, parent=self.root)
        if rgb is None:
            return
        color = tuple(int(round(x)) for x in rgb) + (255,)
        if primary:
            self.editor.set_colors(primary=color)
        else:
            self.editor.set_colors(secondary=color)
        self.editor.set_fill_colors()

    def swap_colors(self):
        self.editor.swap_colors()
        self.editor.set_fill_colors()

    def reset_colors(self):
        self.editor.set_colors((0, 0, 0, 255), (255, 255, 255, 255))
        self.editor.set_fill_colors()

    # ------------------------------------------------------------------
    # 마우스
    # ------------------------------------------------------------------
    def to_image(self, event) -> tuple[float, float]:
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ox, oy = self._origin()
        return (cx - ox) / self.zoom, (cy - oy) / self.zoom

    def on_press(self, event, button):
        if self._pressed_button is not None and self._pressed_button != button:
            # 다른 버튼을 누른 채 다시 누름: 진행 중인 그리기 취소 [추측]
            if self.editor.tool == TOOL_FILL:
                self.editor.fill_press(*self.to_image(event), button)
            self._pressed_button = None
            return
        self._pressed_button = button
        self.canvas.focus_set()
        x, y = self.to_image(event)
        shift = bool(event.state & 0x0001)
        ed = self.editor
        if ed.tool == TOOL_FILL:
            ed.fill_press(x, y, button, shift, handle_radius=max(1.0, 6 / self.zoom))
        elif ed.tool == TOOL_BRUSH:
            ed.brush_press(x, y, button)
        elif ed.tool == TOOL_SELECT and button == BUTTON_LEFT:
            ed.select_press(x, y)
        elif ed.tool == TOOL_MOVE and button == BUTTON_LEFT:
            ed.move_press(x, y)

    def on_drag(self, event):
        x, y = self.to_image(event)
        self._set_cursor_status((x, y))
        if self._pressed_button is None:
            return
        ed = self.editor
        if ed.tool == TOOL_FILL:
            ed.fill_drag(x, y)
        elif ed.tool == TOOL_BRUSH:
            ed.brush_drag(x, y)
        elif ed.tool == TOOL_SELECT:
            ed.select_drag(x, y)
        elif ed.tool == TOOL_MOVE:
            ed.move_drag(x, y)

    def on_release(self, event, button):
        if self._pressed_button != button:
            return
        self._pressed_button = None
        ed = self.editor
        if ed.tool == TOOL_FILL:
            ed.fill_release()
        elif ed.tool == TOOL_BRUSH:
            ed.brush_release()
        elif ed.tool == TOOL_SELECT:
            ed.select_release()
        elif ed.tool == TOOL_MOVE:
            ed.move_release()
        self.update_title()

    def on_motion(self, event):
        self._set_cursor_status(self.to_image(event))

    def on_wheel(self, event):
        if event.state & 0x0004:
            self.step_zoom(+1 if event.delta > 0 else -1, anchor=(event.x, event.y))
        elif event.state & 0x0001:
            self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
            self.request_render()
        else:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            self.request_render()

    def _xview(self, *args):
        self.canvas.xview(*args)
        self.request_render()

    def _yview(self, *args):
        self.canvas.yview(*args)
        self.request_render()

    def _set_cursor_status(self, pos):
        if pos is None:
            self.st_cursor.configure(text="")
            return
        x, y = math.floor(pos[0]), math.floor(pos[1])
        self.st_cursor.configure(text=f"커서: {x}, {y}")

    # ------------------------------------------------------------------
    # 채우기 확정/취소, 실행 취소
    # ------------------------------------------------------------------
    def commit_fill(self):
        self.editor.commit_fill()
        self.update_title()

    def cancel_fill(self):
        self.editor.cancel_fill()
        self.update_title()

    def undo(self):
        self.editor.undo()
        self.update_title()

    def redo(self):
        self.editor.redo()
        self.update_title()

    # ------------------------------------------------------------------
    # 확대와 축소
    # ------------------------------------------------------------------
    def set_zoom(self, z, anchor=None):
        z = max(ZOOM_LEVELS[0], min(ZOOM_LEVELS[-1], z))
        c = self.canvas
        if anchor is None:
            anchor = (c.winfo_width() / 2, c.winfo_height() / 2)
        ax, ay = anchor
        ix = (c.canvasx(ax) - self._origin()[0]) / self.zoom
        iy = (c.canvasy(ay) - self._origin()[1]) / self.zoom
        self.zoom = z
        self._update_scrollregion()
        ox, oy = self._origin()
        tw, th = self._scroll_size()
        if tw > 0:
            c.xview_moveto(max(0.0, (ix * z + ox - ax) / tw))
        if th > 0:
            c.yview_moveto(max(0.0, (iy * z + oy - ay) / th))
        self.request_render()

    def step_zoom(self, direction, anchor=None):
        z = self.zoom
        if direction > 0:
            bigger = [l for l in ZOOM_LEVELS if l > z + 1e-9]
            if bigger:
                self.set_zoom(bigger[0], anchor)
        else:
            smaller = [l for l in ZOOM_LEVELS if l < z - 1e-9]
            if smaller:
                self.set_zoom(smaller[-1], anchor)

    def zoom_to_fit(self):
        c = self.canvas
        cw, ch = max(1, c.winfo_width() - 20), max(1, c.winfo_height() - 20)
        fit = min(cw / self.editor.width, ch / self.editor.height)
        cands = [l for l in ZOOM_LEVELS if l <= fit] or [ZOOM_LEVELS[0]]
        self.set_zoom(cands[-1])

    def _scroll_size(self):
        c = self.canvas
        w = self.editor.width * self.zoom
        h = self.editor.height * self.zoom
        return max(w + 2 * 16, c.winfo_width()), max(h + 2 * 16, c.winfo_height())

    def _origin(self):
        """캔버스 좌표에서 이미지 왼쪽 위의 위치 (작으면 가운데 정렬)."""
        tw, th = self._scroll_size()
        w = self.editor.width * self.zoom
        h = self.editor.height * self.zoom
        return math.floor((tw - w) / 2), math.floor((th - h) / 2)

    def _update_scrollregion(self):
        tw, th = self._scroll_size()
        self.canvas.configure(scrollregion=(0, 0, tw, th))

    # ------------------------------------------------------------------
    # 그리기
    # ------------------------------------------------------------------
    def request_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.root.after_idle(self._render)

    def _checker(self, w, h):
        key = (w, h)
        img = self._checker_cache.get(key)
        if img is None:
            yy, xx = np.mgrid[0:h, 0:w]
            on = ((xx // CHECKER) + (yy // CHECKER)) % 2 == 0
            arr = np.where(on[..., None], 255, 204).astype(np.uint8)
            arr = np.concatenate([np.repeat(arr, 3, axis=2), np.full((h, w, 1), 255, np.uint8)], 2)
            img = Image.fromarray(arr, "RGBA")
            if len(self._checker_cache) > 8:
                self._checker_cache.clear()
            self._checker_cache[key] = img
        return img

    def _render(self):
        self._render_pending = False
        c = self.canvas
        ed = self.editor
        self._update_scrollregion()
        c.delete("all")
        z = self.zoom
        ox, oy = self._origin()
        vw, vh = c.winfo_width(), c.winfo_height()
        vx0, vy0 = c.canvasx(0), c.canvasy(0)

        # 보이는 부분만 잘라서 확대한다.
        x0 = max(0, math.floor((vx0 - ox) / z))
        y0 = max(0, math.floor((vy0 - oy) / z))
        x1 = min(ed.width, math.ceil((vx0 + vw - ox) / z) + 1)
        y1 = min(ed.height, math.ceil((vy0 + vh - oy) / z) + 1)
        if x1 > x0 and y1 > y0:
            crop = ed.display_image()[y0:y1, x0:x1]
            sx0, sy0 = round(x0 * z), round(y0 * z)
            sw = max(1, round(x1 * z) - sx0)
            sh = max(1, round(y1 * z) - sy0)
            pil = Image.fromarray(np.ascontiguousarray(crop), "RGBA")
            if (sw, sh) != pil.size:
                resample = Image.NEAREST if z >= 1 else Image.BOX
                pil = pil.resize((sw, sh), resample)
            bg = self._checker(sw, sh).copy()
            bg.alpha_composite(pil)
            self._photo = ImageTk.PhotoImage(bg)
            c.create_image(ox + sx0, oy + sy0, image=self._photo, anchor="nw")
        c.create_rectangle(ox - 1, oy - 1, ox + ed.width * z, oy + ed.height * z, outline="#444")

        # 선택 영역
        for rect, dash in ((ed.selection if ed.moving_rect() is None else None, (4, 4)),
                           (ed.pending_selection_rect(), (4, 4)),
                           (ed.moving_rect(), (2, 2))):
            if rect is None:
                continue
            rx0, ry0, rx1, ry1 = rect
            coords = (ox + rx0 * z, oy + ry0 * z, ox + rx1 * z - 1, oy + ry1 * z - 1)
            c.create_rectangle(*coords, outline="white", width=1)
            c.create_rectangle(*coords, outline="black", dash=dash, width=1)

        # 채우기 핸들: 시작점 부근의 흰 테두리 작은 사각형 (7장 [확인], 모양은 [UNKNOWN])
        f = ed.fill_state
        if f is not None and f.phase == PHASE_EDITING:
            hx, hy = ox + f.seed_x * z, oy + f.seed_y * z
            c.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, outline="black", width=1)
            c.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4, outline="white", width=2)

        self.draw_swatch()
        self.sync_options()
        self.st_size.configure(text=f"크기: {ed.width} × {ed.height}")
        self.st_zoom.configure(text=f"{z * 100:.0f}%")
        self._update_message()
        self.update_title()

    def _update_message(self):
        ed = self.editor
        if ed.tool == TOOL_FILL and ed.fill_active:
            if ed.fill_state.phase == PHASE_EDITING:
                msg = "편집 중: 옵션을 바꾸거나 핸들을 끌어 고칩니다. Enter=확정, Esc=취소"
            else:
                msg = "놓으면 편집할 수 있습니다."
        elif ed.tool == TOOL_FILL:
            msg = "클릭: 기본색, 오른쪽 클릭: 보조색, Shift+클릭: 이번만 전체 이미지"
        else:
            msg = ""
        if ed.history:
            msg = (msg + "   |   " if msg else "") + f"마지막 작업: {ed.history[ed.index].name}"
        self.st_message.configure(text=msg)

    # ------------------------------------------------------------------
    # 파일
    # ------------------------------------------------------------------
    def is_dirty(self):
        return self.editor.index != self.saved_index or self.editor.fill_active

    def update_title(self):
        name = os.path.basename(self.path) if self.path else "제목 없음"
        mark = "*" if self.is_dirty() else ""
        self.root.title(f"{mark}{name} - {APP_NAME}")

    def confirm_discard(self) -> bool:
        if not self.is_dirty():
            return True
        ans = messagebox.askyesnocancel(APP_NAME, "바뀐 내용을 저장할까요?", parent=self.root)
        if ans is None:
            return False
        if ans:
            return self.save()
        return True

    def new_image(self):
        if not self.confirm_discard():
            return
        size = NewImageDialog(self.root, self.editor.width, self.editor.height).result
        if size is None:
            return
        self.editor.new_image(*size)
        self.path = None
        self.saved_index = self.editor.index
        self.zoom_to_fit()
        self.update_title()

    def open_image(self):
        if not self.confirm_discard():
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="열기",
            filetypes=[("PNG 이미지", "*.png"), ("모든 이미지", "*.png *.jpg *.jpeg *.bmp *.gif"),
                       ("모든 파일", "*.*")])
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            with Image.open(path) as im:
                arr = np.array(im.convert("RGBA"))
        except Exception as exc:  # noqa: BLE001 - 사용자에게 알린다
            messagebox.showerror(APP_NAME, f"이미지를 열 수 없습니다.\n{exc}", parent=self.root)
            return
        self.editor.load_image(arr)
        self.path = path
        self.saved_index = self.editor.index
        self.zoom_to_fit()
        self.update_title()

    def save(self) -> bool:
        if self.path is None or not self.path.lower().endswith(".png"):
            return self.save_as()
        return self._write(self.path)

    def save_as(self) -> bool:
        path = filedialog.asksaveasfilename(
            parent=self.root, title="다른 이름으로 저장", defaultextension=".png",
            filetypes=[("PNG 이미지", "*.png")],
            initialfile=os.path.splitext(os.path.basename(self.path))[0] + ".png"
            if self.path else "제목 없음.png")
        if not path:
            return False
        return self._write(path)

    def _write(self, path) -> bool:
        # 편집 중인 채우기는 확정한 뒤 저장한다 (DECISIONS.md)
        if self.editor.fill_active:
            self.editor.commit_fill()
        try:
            Image.fromarray(self.editor.image, "RGBA").save(path, "PNG")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(APP_NAME, f"저장하지 못했습니다.\n{exc}", parent=self.root)
            return False
        self.path = path
        self.saved_index = self.editor.index
        self.update_title()
        return True

    def quit(self):
        if self.confirm_discard():
            self.root.destroy()


class NewImageDialog(tk.Toplevel):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.title("새 이미지")
        self.resizable(False, False)
        self.transient(master)
        self.result = None
        fr = ttk.Frame(self, padding=12)
        fr.pack(fill="both", expand=True)
        self.w = tk.StringVar(value=str(width))
        self.h = tk.StringVar(value=str(height))
        ttk.Label(fr, text="너비 (픽셀)").grid(row=0, column=0, sticky="w", pady=3)
        ew = ttk.Spinbox(fr, from_=1, to=8000, textvariable=self.w, width=8)
        ew.grid(row=0, column=1, pady=3)
        ttk.Label(fr, text="높이 (픽셀)").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Spinbox(fr, from_=1, to=8000, textvariable=self.h, width=8).grid(row=1, column=1, pady=3)
        ttk.Label(fr, text="흰 배경으로 만듭니다.", foreground="#666").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        btns = ttk.Frame(fr)
        btns.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(btns, text="확인", command=self._ok).pack(side="left", padx=3)
        ttk.Button(btns, text="취소", command=self.destroy).pack(side="left")
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        ew.focus_set()
        ew.selection_range(0, "end")
        self.grab_set()
        self.wait_window(self)

    def _ok(self):
        try:
            w, h = int(self.w.get()), int(self.h.get())
        except ValueError:
            messagebox.showerror("새 이미지", "숫자를 입력하세요.", parent=self)
            return
        if not (1 <= w <= 8000 and 1 <= h <= 8000):
            messagebox.showerror("새 이미지", "1~8000 사이로 입력하세요.", parent=self)
            return
        self.result = (w, h)
        self.destroy()


def run(argv=None):
    try:  # 고해상도 화면에서 흐려지지 않게 (Windows)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        pass
    root = tk.Tk()
    app = FillLabApp(root)
    if argv:
        root.after(50, lambda: app.load_path(argv[0]))
    else:
        root.after(50, app.zoom_to_fit)
    root.mainloop()
