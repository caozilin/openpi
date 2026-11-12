"""Capture and restore the three Windows GUI rectangles for this project."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

from ..config import PROJECT_ROOT


MAIN_WINDOW_TITLE = "Franka Panda Teleoperation"
CAMERAS_WINDOW_TITLE = "Robot cameras"
VIEWER_WINDOW_TITLE = "MuJoCo : Franka Panda teleoperation"
WINDOW_TITLES = {
    "main": MAIN_WINDOW_TITLE,
    "cameras": CAMERAS_WINDOW_TITLE,
    "viewer": VIEWER_WINDOW_TITLE,
}
LAYOUT_PATH = PROJECT_ROOT / "window_layout.json"
_HWND_NOTOPMOST = ctypes.c_void_p(-2)
_SWP_NOACTIVATE = 0x0010


class _RECT(ctypes.Structure):
    _fields_ = (("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long))


def _user32() -> ctypes.WinDLL:
    if sys.platform != "win32":
        raise RuntimeError("Window-layout capture is only available on Windows.")
    user32 = ctypes.windll.user32
    user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(_RECT))
    user32.GetWindowRect.restype = ctypes.c_bool
    user32.SetWindowPos.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    user32.SetWindowPos.restype = ctypes.c_bool
    return user32


def load_layout() -> dict[str, dict[str, int]]:
    """Return a validated saved outer-window layout, or an empty mapping."""
    try:
        raw = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    layout: dict[str, dict[str, int]] = {}
    for name in WINDOW_TITLES:
        rect = raw.get(name)
        if not isinstance(rect, dict):
            continue
        try:
            values = {key: int(rect[key]) for key in ("x", "y", "width", "height")}
        except (KeyError, TypeError, ValueError):
            continue
        if values["width"] > 0 and values["height"] > 0:
            layout[name] = values
    return layout


def apply_saved_layout(
    hwnd: int,
    name: str,
    *,
    minimum_width: int = 1,
    minimum_height: int = 1,
) -> bool:
    """Apply a saved rectangle, clamped to the window's usable minimum size."""
    rect = load_layout().get(name)
    if rect is None or not hwnd or sys.platform != "win32":
        return False
    width = max(int(rect["width"]), int(minimum_width))
    height = max(int(rect["height"]), int(minimum_height))
    user32 = _user32()
    return bool(user32.SetWindowPos(
        ctypes.c_void_p(hwnd),
        _HWND_NOTOPMOST,
        rect["x"],
        rect["y"],
        width,
        height,
        _SWP_NOACTIVATE,
    ))


def find_window(title: str) -> int:
    """Return a top-level window handle for an exact title, or zero."""
    if sys.platform != "win32":
        return 0
    return int(_user32().FindWindowW(None, title) or 0)


def window_rect(hwnd: int) -> dict[str, int] | None:
    """Return an outer window rectangle, so neighbouring windows can touch exactly."""
    if not hwnd or sys.platform != "win32":
        return None
    rect = _RECT()
    if not _user32().GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(rect)):
        return None
    return {
        "x": int(rect.left),
        "y": int(rect.top),
        "width": int(rect.right - rect.left),
        "height": int(rect.bottom - rect.top),
    }


def place_window_right_of(reference_hwnd: int, target_hwnd: int) -> bool:
    """Place target flush against the reference window's right outer edge."""
    reference = window_rect(reference_hwnd)
    target = window_rect(target_hwnd)
    if reference is None or target is None:
        return False
    return bool(
        _user32().SetWindowPos(
            ctypes.c_void_p(target_hwnd),
            _HWND_NOTOPMOST,
            reference["x"] + reference["width"],
            reference["y"],
            target["width"],
            target["height"],
            _SWP_NOACTIVATE,
        )
    )


def capture_current_layout() -> dict[str, dict[str, int]]:
    """Read all three visible project windows from the current interactive desktop."""
    user32 = _user32()
    layout: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    for name, title in WINDOW_TITLES.items():
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            missing.append(title)
            continue
        rect = _RECT()
        if not user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(rect)):
            missing.append(title)
            continue
        layout[name] = {
            "x": int(rect.left),
            "y": int(rect.top),
            "width": int(rect.right - rect.left),
            "height": int(rect.bottom - rect.top),
        }
    if missing:
        raise RuntimeError("Could not find: " + ", ".join(missing))
    LAYOUT_PATH.write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return layout
