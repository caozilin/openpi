"""Run from a second PowerShell while the Panda application is open."""

from __future__ import annotations

import json

from .window_layout import LAYOUT_PATH, capture_current_layout


def main() -> None:
    layout = capture_current_layout()
    print(json.dumps(layout, ensure_ascii=False, indent=2))
    print(f"Saved to {LAYOUT_PATH}")


if __name__ == "__main__":
    main()
