"""Main-view-only markers for gripper contact with the work table."""

from __future__ import annotations

import mujoco
import numpy as np

from ..simulation.robot import RobotHandles


CONTACT_MARKER_COLOR = np.array((1.0, 0.05, 0.03, 0.95), dtype=np.float32)
CONTACT_MARKER_RADIUS_M = 0.005
CONTACT_MARKER_PERSISTENCE_S = 0.10
MAX_CONTACT_MARKERS = 16
_CONTACT_CLUSTER_DISTANCE_M = 0.003


def _body_subtree(model: mujoco.MjModel, root_id: int) -> set[int]:
    result: set[int] = set()
    for candidate in range(model.nbody):
        current = candidate
        while current > root_id:
            current = int(model.body_parentid[current])
        if current == root_id:
            result.add(candidate)
    return result


def _lowest_common_ancestor(
    model: mujoco.MjModel,
    body_ids: np.ndarray,
) -> int:
    bodies = [int(body_id) for body_id in np.asarray(body_ids).reshape(-1)]
    if not bodies:
        raise ValueError("At least one gripper body is required")

    ancestor_paths: list[list[int]] = []
    for body_id in bodies:
        path = [body_id]
        while path[-1] != 0:
            path.append(int(model.body_parentid[path[-1]]))
        ancestor_paths.append(path)

    common = set(ancestor_paths[0]).intersection(*ancestor_paths[1:])
    return next(body_id for body_id in ancestor_paths[0] if body_id in common)


def _cluster_contact_points(
    points: list[np.ndarray],
    maximum_points: int,
) -> np.ndarray:
    clustered: list[np.ndarray] = []
    for point in points:
        if any(
            np.linalg.norm(point - existing) < _CONTACT_CLUSTER_DISTANCE_M
            for existing in clustered
        ):
            continue
        clustered.append(point)
        if len(clustered) >= maximum_points:
            break
    if not clustered:
        return np.empty((0, 3), dtype=float)
    return np.stack(clustered)


class GripperTableContactMarkers:
    """Track physical gripper-table contacts for a decorative viewer overlay."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        *,
        table_body_name: str = "work_surface",
        persistence_s: float = CONTACT_MARKER_PERSISTENCE_S,
        maximum_points: int = MAX_CONTACT_MARKERS,
    ) -> None:
        if persistence_s < 0.0:
            raise ValueError("Contact-marker persistence must be non-negative")
        if maximum_points <= 0:
            raise ValueError("Contact-marker count must be positive")

        table_root = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, table_body_name,
        )
        if table_root < 0:
            raise RuntimeError(
                f"Contact visualization is missing table body {table_body_name!r}"
            )
        finger_bodies = model.jnt_bodyid[
            np.asarray(handles.finger_joint_ids, dtype=int)
        ]
        gripper_root = _lowest_common_ancestor(model, finger_bodies)
        self.model = model
        self.gripper_bodies = _body_subtree(model, gripper_root)
        self.table_bodies = _body_subtree(model, int(table_root))
        self.persistence_s = float(persistence_s)
        self.maximum_points = int(maximum_points)
        self._points = np.empty((0, 3), dtype=float)
        self._expires_at = -np.inf

    @property
    def points(self) -> np.ndarray:
        return self._points.copy()

    def clear(self) -> None:
        self._points = np.empty((0, 3), dtype=float)
        self._expires_at = -np.inf

    def observe(self, data: mujoco.MjData) -> None:
        """Capture only penetrating/touching gripper-table contact points."""
        points: list[np.ndarray] = []
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > 0.0:
                continue
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            gripper_table_pair = (
                body_a in self.gripper_bodies
                and body_b in self.table_bodies
            ) or (
                body_b in self.gripper_bodies
                and body_a in self.table_bodies
            )
            if gripper_table_pair:
                points.append(np.asarray(contact.pos, dtype=float).copy())

        now = float(data.time)
        if points:
            self._points = _cluster_contact_points(
                points, self.maximum_points,
            )
            self._expires_at = now + self.persistence_s
        elif now > self._expires_at:
            self.clear()

    def append_to_scene(
        self,
        scene: mujoco.MjvScene,
        base_geom_count: int,
    ) -> int:
        """Append red spheres without disturbing existing user-scene geoms."""
        base = int(base_geom_count)
        if not 0 <= base <= int(scene.maxgeom):
            raise ValueError("Invalid base user-scene geometry count")
        scene.ngeom = base
        available = int(scene.maxgeom) - base
        identity = np.eye(3).reshape(-1)
        marker_size = np.array((CONTACT_MARKER_RADIUS_M, 0.0, 0.0))
        for point in self._points[:available]:
            index = int(scene.ngeom)
            mujoco.mjv_initGeom(
                scene.geoms[index],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                marker_size,
                point,
                identity,
                CONTACT_MARKER_COLOR,
            )
            scene.geoms[index].category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1
        return int(scene.ngeom) - base
