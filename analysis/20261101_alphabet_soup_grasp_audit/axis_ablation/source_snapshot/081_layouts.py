"""Versioned, reusable custom scene layouts.

The UI and offline tools deliberately share this module.  A custom layout is
an explicit replacement for seeded object placement, not a second simulation
or task-execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

import mujoco
import numpy as np

from .scenes import SceneManager


LAYOUT_FORMAT = "tasktol.custom-layout"
LAYOUT_VERSION = 1
CUSTOM_LAYOUT_ROOT = Path(__file__).resolve().parents[2] / "custom_layouts"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class CustomLayout:
    scene: str
    robot: str
    source_seed: int
    variant_index: int
    objects: tuple[Mapping[str, Any], ...]
    created_at: str
    format: str = LAYOUT_FORMAT
    version: int = LAYOUT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "scene": self.scene,
            "robot": self.robot,
            "source_seed": self.source_seed,
            "variant_index": self.variant_index,
            "created_at": self.created_at,
            "coordinate_frame": "mujoco_world",
            "editable_dofs": ["x", "y", "yaw"],
            "objects": [dict(item) for item in self.objects],
        }


def _normalised_quaternion(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(result))
    if result.shape != (4,) or not np.isfinite(result).all() or norm <= 1e-12:
        raise ValueError("Object quaternion must contain four finite values")
    return result / norm


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, _normalised_quaternion(quaternion))
    rotation = matrix.reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def quaternion_with_yaw(quaternion: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Replace world yaw while preserving the object's authored tilt."""
    original = _normalised_quaternion(quaternion)
    old_yaw = yaw_from_quaternion(original)
    half = 0.5 * (float(yaw_rad) - old_yaw)
    delta = np.array((np.cos(half), 0.0, 0.0, np.sin(half)))
    result = np.empty(4)
    mujoco.mju_mulQuat(result, delta, original)
    return _normalised_quaternion(result)


def capture_custom_layout(
    manager: SceneManager,
    data: mujoco.MjData,
    *,
    robot: str,
    source_seed: int,
) -> CustomLayout:
    objects = tuple(
        {
            "name": item.body_name,
            "position": [float(value) for value in item.position],
            "quaternion_wxyz": [float(value) for value in item.quaternion],
            "yaw_rad": yaw_from_quaternion(item.quaternion),
        }
        for item in manager.layout_objects(data)
    )
    return CustomLayout(
        scene=manager.current_name,
        robot=str(robot),
        source_seed=int(source_seed),
        variant_index=manager.selected_variant_index,
        objects=objects,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def parse_custom_layout(document: Mapping[str, Any]) -> CustomLayout:
    if document.get("format") != LAYOUT_FORMAT:
        raise ValueError(f"Unsupported layout format: {document.get('format')!r}")
    if int(document.get("version", -1)) != LAYOUT_VERSION:
        raise ValueError(f"Unsupported layout version: {document.get('version')!r}")
    raw_objects = document.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("Layout must contain at least one object")
    objects: list[Mapping[str, Any]] = []
    names: set[str] = set()
    for raw in raw_objects:
        if not isinstance(raw, Mapping):
            raise ValueError("Every layout object must be a mapping")
        name = str(raw.get("name", ""))
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate layout object name: {name!r}")
        position = np.asarray(raw.get("position"), dtype=float)
        quaternion = _normalised_quaternion(raw.get("quaternion_wxyz"))
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Invalid position for layout object {name!r}")
        names.add(name)
        objects.append({
            "name": name,
            "position": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "yaw_rad": float(raw.get("yaw_rad", yaw_from_quaternion(quaternion))),
        })
    return CustomLayout(
        scene=str(document.get("scene", "")),
        robot=str(document.get("robot", "")),
        source_seed=int(document.get("source_seed", -1)),
        variant_index=int(document.get("variant_index", 0)),
        objects=tuple(objects),
        created_at=str(document.get("created_at", "")),
    )


def apply_custom_layout(
    manager: SceneManager,
    data: mujoco.MjData,
    layout: CustomLayout | Mapping[str, Any],
    *,
    robot: str | None = None,
    restore_variant: bool = True,
) -> None:
    """Apply a saved layout for UI use or repeatable offline validation."""
    parsed = layout if isinstance(layout, CustomLayout) else parse_custom_layout(layout)
    if parsed.scene != manager.current_name:
        raise ValueError(
            f"Layout scene {parsed.scene!r} does not match active scene "
            f"{manager.current_name!r}"
        )
    # ``robot`` is accepted for API compatibility and provenance-aware
    # callers, but layouts are intentionally robot-independent. Their object
    # poses belong to the shared MuJoCo world/scene coordinate system.
    if restore_variant:
        manager.select_variant_index(parsed.variant_index)
        manager.activate(data, parsed.scene)
    active = set(manager.active_body_names)
    supplied = {str(item["name"]) for item in parsed.objects}
    if supplied != active:
        missing = sorted(active - supplied)
        unknown = sorted(supplied - active)
        raise ValueError(
            f"Layout objects do not match active scene (missing={missing}, unknown={unknown})"
        )
    for item in parsed.objects:
        name = str(item["name"])
        manager.set_layout_pose(
            data,
            name,
            np.asarray(item["position"], dtype=float),
            np.asarray(item["quaternion_wxyz"], dtype=float),
        )
        manager.reset_layout_articulation(data, name)
    mujoco.mj_forward(manager.model, data)


def save_custom_layout(path: str | Path, layout: CustomLayout) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(layout.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_custom_layout(path: str | Path) -> CustomLayout:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("Layout document root must be an object")
    return parse_custom_layout(document)


def _path_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    component = component.strip("._-")
    if not component:
        raise ValueError("Layout scene and robot names must not be empty")
    return component


def validate_custom_layout_name(name: str) -> str:
    result = str(name).strip()
    if not _SAFE_NAME.fullmatch(result):
        raise ValueError(
            "Layout name must be 1-64 characters using letters, numbers, ".strip()
            + "dot, underscore, or hyphen"
        )
    return result


def custom_layout_directory(
    scene: str,
    *,
    root: str | Path = CUSTOM_LAYOUT_ROOT,
) -> Path:
    return Path(root) / _path_component(scene)


def named_custom_layout_path(
    scene: str,
    name: str,
    *,
    root: str | Path = CUSTOM_LAYOUT_ROOT,
) -> Path:
    return custom_layout_directory(scene, root=root) / (
        validate_custom_layout_name(name) + ".json"
    )


def list_named_custom_layouts(
    scene: str,
    *,
    root: str | Path = CUSTOM_LAYOUT_ROOT,
) -> tuple[str, ...]:
    directory = custom_layout_directory(scene, root=root)
    if not directory.is_dir():
        return ()
    return tuple(sorted(
        path.stem for path in directory.glob("*.json")
        if _SAFE_NAME.fullmatch(path.stem)
    ))


def save_named_custom_layout(
    name: str,
    layout: CustomLayout,
    *,
    root: str | Path = CUSTOM_LAYOUT_ROOT,
) -> Path:
    path = named_custom_layout_path(
        layout.scene, name, root=root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_custom_layout(path, layout)
    return path


def load_named_custom_layout(
    scene: str,
    name: str,
    *,
    root: str | Path = CUSTOM_LAYOUT_ROOT,
) -> CustomLayout:
    return load_custom_layout(
        named_custom_layout_path(scene, name, root=root),
    )
