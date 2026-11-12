"""Windows placement for this project's native MuJoCo viewer window."""

from __future__ import annotations

import ctypes
import sys
import threading
import time

from .window_layout import CAMERAS_WINDOW_TITLE, MAIN_WINDOW_TITLE, apply_saved_layout, find_window, window_rect


_VIEWER_TITLE = "MuJoCo : Franka Panda teleoperation"
_SPI_GETWORKAREA = 48
# Explicitly remove any previous topmost state while retaining the requested
# initial bottom-right placement.  Other desktop applications should always
# be able to cover the simulator.
_HWND_NOTOPMOST = ctypes.c_void_p(-2)
_SWP_NOACTIVATE = 0x0010
# The camera panel starts at y=10 and has a 390 px client height.  Reserve
# 430 px at the top of the working area for its title bar and a small gap.
_CAMERA_PANEL_RESERVED_HEIGHT = 430


class _RECT(ctypes.Structure):
    _fields_ = (("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long))


def place_mujoco_viewer_bottom_right() -> bool:
    """Place the child-process MuJoCo viewer at the work area's bottom-right."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    # Explicit pointer-sized signatures are important on 64-bit Windows.
    user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.SetWindowPos.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint)
    user32.SetWindowPos.restype = ctypes.c_bool
    user32.SystemParametersInfoW.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(_RECT), ctypes.c_uint)
    user32.SystemParametersInfoW.restype = ctypes.c_bool

    hwnd = user32.FindWindowW(None, _VIEWER_TITLE)
    if not hwnd:
        return False
    if apply_saved_layout(int(hwnd), "viewer"):
        return True
    work = _RECT()
    if not user32.SystemParametersInfoW(_SPI_GETWORKAREA, 0, ctypes.byref(work), 0):
        return False
    # Stack both view windows flush against the operation panel's right edge.
    main_rect = window_rect(find_window(MAIN_WINDOW_TITLE))
    cameras_rect = window_rect(find_window(CAMERAS_WINDOW_TITLE))
    if main_rect is not None and cameras_rect is not None:
        x = main_rect["x"] + main_rect["width"]
        y = cameras_rect["y"] + cameras_rect["height"]
        available_width = work.right - x
        available_height = work.bottom - y
        if available_width > 0 and available_height > 0:
            width = min(1050, available_width)
            height = min(round(width * 3 / 4), available_height)
            width = round(height * 4 / 3)
            return bool(user32.SetWindowPos(ctypes.c_void_p(hwnd), _HWND_NOTOPMOST, x, y, width, height, _SWP_NOACTIVATE))

    # Fallback when running without the operation/camera UI.
    width = min(1050, work.right - work.left)
    available_height = max(1, work.bottom - work.top - _CAMERA_PANEL_RESERVED_HEIGHT)
    height = min(round(width * 3 / 4), available_height)
    width = round(height * 4 / 3)
    return bool(user32.SetWindowPos(ctypes.c_void_p(hwnd), _HWND_NOTOPMOST, work.right - width, work.bottom - height, width, height, _SWP_NOACTIVATE))


def schedule_mujoco_viewer_placement() -> None:
    """Retry while MuJoCo creates its separate GUI-process window."""
    def wait_for_viewer() -> None:
        for _ in range(60):  # up to 15 seconds
            if place_mujoco_viewer_bottom_right():
                return
            time.sleep(0.25)

    threading.Thread(target=wait_for_viewer, name="mujoco-window-placement", daemon=True).start()
