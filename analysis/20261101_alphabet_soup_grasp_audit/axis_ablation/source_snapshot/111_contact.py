"""Contact predicates and prohibited-contact monitoring."""

from __future__ import annotations

import mujoco
import numpy as np

from ..types import CollisionEvent


TABLE_COLLISION_CATEGORY = "robot_table_collision"
ROBOT_TABLE_CLEARANCE_M = 0.005


def is_blocking_collision(collision: CollisionEvent | None) -> bool:
    """Return whether a prohibited robot collision should terminate a task."""
    return collision is not None


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Task evaluation object is missing: {name}")
    return int(object_id)


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    result: set[int] = set()
    for candidate in range(model.nbody):
        current = candidate
        while current > root:
            current = int(model.body_parentid[current])
        if current == root:
            result.add(candidate)
    return result


class RobotSafetyMonitor:
    """Block robot collisions and sub-threshold tabletop clearance."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        robot_root_name: str | None = None,
        finger_body_names: tuple[str, str] | None = None,
        table_body_name: str = "work_surface",
        clearance_threshold_m: float = 0.0,
    ) -> None:
        self.model = model
        threshold = float(clearance_threshold_m)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "clearance_threshold_m must be finite and non-negative"
            )
        self.clearance_threshold_m = threshold
        choices = (
            ("link0", ("left_finger", "right_finger")),
            ("link_base", ("left_finger", "right_finger")),
            ("base", ("rb_left_pad", "rb_right_pad")),
        )
        if robot_root_name is None or finger_body_names is None:
            match = next(
                (
                    (root, fingers)
                    for root, fingers in choices
                    if mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, root,
                    ) >= 0
                ),
                None,
            )
            if match is None:
                raise RuntimeError("Cannot identify robot bodies for safety evaluation")
            robot_root_name, finger_body_names = match
        self.robot_bodies = _subtree(
            model, _required_id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name),
        )
        self.finger_bodies = set().union(*(
            _subtree(
                model, _required_id(model, mujoco.mjtObj.mjOBJ_BODY, name),
            )
            for name in finger_body_names
        ))
        self.table_bodies = _subtree(
            model, _required_id(model, mujoco.mjtObj.mjOBJ_BODY, table_body_name),
        )
        self.table_clearance = RobotTableClearanceRecorder(
            model,
            robot_root_name=robot_root_name,
            table_body_name=table_body_name,
        )
        self.collision: CollisionEvent | None = None

    def collision_category(self, body_a: int, body_b: int) -> str | None:
        pair = {int(body_a), int(body_b)}
        if (
            len(pair) == 2
            and pair <= self.robot_bodies
            and not pair <= self.finger_bodies
        ):
            return "robot_self_collision"
        if pair & self.robot_bodies and pair & self.table_bodies:
            return TABLE_COLLISION_CATEGORY
        return None

    def _name(self, object_type: mujoco.mjtObj, object_id: int) -> str:
        return mujoco.mj_id2name(self.model, object_type, object_id) or f"id_{object_id}"

    def observe(self, data: mujoco.MjData, frame_index: int, time_s: float) -> None:
        if is_blocking_collision(self.collision):
            return
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > 0.0:
                continue
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            category = self.collision_category(body_a, body_b)
            if category is None:
                continue
            self.collision = CollisionEvent(
                frame_index=int(frame_index),
                time_s=float(time_s),
                body_a=self._name(mujoco.mjtObj.mjOBJ_BODY, body_a),
                body_b=self._name(mujoco.mjtObj.mjOBJ_BODY, body_b),
                geom_a=self._name(mujoco.mjtObj.mjOBJ_GEOM, geom_a),
                geom_b=self._name(mujoco.mjtObj.mjOBJ_GEOM, geom_b),
                category=category,
                penetration_m=max(0.0, -float(contact.dist)),
                clearance_m=(
                    float(contact.dist)
                    if category == TABLE_COLLISION_CATEGORY else None
                ),
                clearance_threshold_m=(
                    self.clearance_threshold_m
                    if category == TABLE_COLLISION_CATEGORY else None
                ),
            )
            return
        # Ordinary task failure is contact-only. Distance-query CCD can return
        # an invalid zero with zero witness points for separated mesh/box pairs;
        # it must not manufacture a contact absent from MuJoCo's contact list.
        if self.clearance_threshold_m <= 0.0:
            return
        clearance, robot_geom, table_geom = self.table_clearance.minimum(data)
        if clearance >= self.clearance_threshold_m:
            return
        robot_body = int(self.model.geom_bodyid[robot_geom])
        table_body = int(self.model.geom_bodyid[table_geom])
        self.collision = CollisionEvent(
            frame_index=int(frame_index),
            time_s=float(time_s),
            body_a=self._name(mujoco.mjtObj.mjOBJ_BODY, robot_body),
            body_b=self._name(mujoco.mjtObj.mjOBJ_BODY, table_body),
            geom_a=self._name(mujoco.mjtObj.mjOBJ_GEOM, robot_geom),
            geom_b=self._name(mujoco.mjtObj.mjOBJ_GEOM, table_geom),
            category=TABLE_COLLISION_CATEGORY,
            penetration_m=0.0,
            clearance_m=clearance,
            clearance_threshold_m=self.clearance_threshold_m,
        )


def configure_robot_table_clearance_threshold(
    evaluator: object,
    threshold_m: float,
) -> None:
    """Configure every safety monitor owned by a task evaluator tree."""
    threshold = float(threshold_m)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold_m must be finite and non-negative")
    visited: set[int] = set()

    def visit(value: object) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, RobotSafetyMonitor):
            value.clearance_threshold_m = threshold
            return
        for attribute in (
            "contacts", "safety", "evaluators", "open_lid", "pear", "close_lid",
        ):
            child = getattr(value, attribute, None)
            if isinstance(child, (tuple, list)):
                for item in child:
                    visit(item)
            elif child is not None:
                visit(child)

    visit(evaluator)


class RobotTableClearanceRecorder:
    """Measure collision-geometry clearance at the physics-step rate."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        robot_root_name: str | None = None,
        table_body_name: str = "work_surface",
        distance_limit_m: float = 2.0,
    ) -> None:
        if not np.isfinite(distance_limit_m) or distance_limit_m <= 0.0:
            raise ValueError("distance_limit_m must be finite and positive")
        self.model = model
        self.distance_limit_m = float(distance_limit_m)
        roots = ("link0", "link_base", "base")
        if robot_root_name is None:
            robot_root_name = next((
                name for name in roots
                if mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, name,
                ) >= 0
            ), None)
        if robot_root_name is None:
            raise RuntimeError("Cannot identify the robot root body")
        robot_bodies = _subtree(
            model,
            _required_id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name),
        )
        # Clearance is specifically to the tabletop body.  Unlike prohibited
        # contact classification, do not include its descendant leg/support
        # geometry: those are not the horizontal work surface and some mesh
        # pairs do not provide a useful positive separation distance.
        table_bodies = {
            _required_id(model, mujoco.mjtObj.mjOBJ_BODY, table_body_name),
        }
        collidable = (
            (np.asarray(model.geom_contype) != 0)
            | (np.asarray(model.geom_conaffinity) != 0)
        )
        self.robot_geoms = tuple(
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_bodies
            and bool(collidable[geom_id])
        )
        self.table_geoms = tuple(
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in table_bodies
            and bool(collidable[geom_id])
        )
        if not self.robot_geoms or not self.table_geoms:
            raise RuntimeError("Robot or table has no collidable geometry")
        self._values: list[float] = []
        self._phase_values: dict[str, list[float]] = {}
        self._minimum: dict[str, object] | None = None
        self._fromto = np.empty(6, dtype=float)
        self._invalid_pair_queries = 0

    def _name(self, object_type: mujoco.mjtObj, object_id: int) -> str:
        return (
            mujoco.mj_id2name(self.model, object_type, object_id)
            or f"{object_type.name.lower()}_{object_id}"
        )

    def sample(
        self,
        data: mujoco.MjData,
        *,
        phase: str,
        frame_index: int,
    ) -> float:
        minimum, robot_geom, table_geom = self.minimum(data)
        minimum_pair = (robot_geom, table_geom)
        self._values.append(minimum)
        self._phase_values.setdefault(str(phase), []).append(minimum)
        if self._minimum is None or minimum < float(self._minimum["distance_m"]):
            robot_body = int(self.model.geom_bodyid[robot_geom])
            table_body = int(self.model.geom_bodyid[table_geom])
            self._minimum = {
                "distance_m": minimum,
                "phase": str(phase),
                "frame_index": int(frame_index),
                "time_s": float(data.time),
                "robot_body": self._name(
                    mujoco.mjtObj.mjOBJ_BODY, robot_body,
                ),
                "robot_geom": self._name(
                    mujoco.mjtObj.mjOBJ_GEOM, robot_geom,
                ),
                "table_body": self._name(
                    mujoco.mjtObj.mjOBJ_BODY, table_body,
                ),
                "table_geom": self._name(
                    mujoco.mjtObj.mjOBJ_GEOM, table_geom,
                ),
            }
        return minimum

    def minimum(self, data: mujoco.MjData) -> tuple[float, int, int]:
        """Return the valid minimum distance and responsible geometry pair."""
        minimum = self.distance_limit_m
        minimum_pair: tuple[int, int] | None = None
        # Each local geom AABB contains the complete collision geometry. Its
        # transformed world AABB gives a conservative separation lower bound,
        # not a replacement for the actual geometric distance.
        world_bounds = {}
        for geom in (*self.robot_geoms, *self.table_geoms):
            rotation = data.geom_xmat[geom].reshape(3, 3)
            local = self.model.geom_aabb[geom]
            center = data.geom_xpos[geom] + rotation @ local[:3]
            extent = np.abs(rotation) @ local[3:]
            world_bounds[geom] = (center - extent, center + extent)
        for robot_geom in self.robot_geoms:
            for table_geom in self.table_geoms:
                distance = float(mujoco.mj_geomDistance(
                    self.model,
                    data,
                    robot_geom,
                    table_geom,
                    self.distance_limit_m,
                    self._fromto,
                ))
                segment_length = float(np.linalg.norm(
                    self._fromto[3:] - self._fromto[:3]
                ))
                robot_min, robot_max = world_bounds[robot_geom]
                table_min, table_max = world_bounds[table_geom]
                gap = np.maximum(np.maximum(robot_min-table_max, table_min-robot_max), 0.0)
                lower_bound = float(np.linalg.norm(gap))
                if (
                    not np.isfinite(distance)
                    or not np.all(np.isfinite(self._fromto))
                    or (lower_bound > 1.0e-7 and distance < lower_bound - 1.0e-7)
                ):
                    self._invalid_pair_queries += 1
                    continue
                # Some tiny box pairs in MuJoCo return a zero scalar while
                # leaving a clearly nonzero closest-point segment.  Such a
                # pair did not produce a valid distance and must not masquerade
                # as tabletop contact. Genuine separation has |segment|=d.
                # Both zero is not sufficient: the AABB check above must also
                # permit touching (regression: upright cylinder seed 63).
                if distance >= 0.0 and not np.isclose(
                    segment_length, distance, atol=1.0e-7, rtol=1.0e-6,
                ):
                    self._invalid_pair_queries += 1
                    continue
                if distance < minimum:
                    minimum = distance
                    minimum_pair = (robot_geom, table_geom)
        if minimum_pair is None:
            raise RuntimeError("No robot-table geometry pair returned a valid distance")
        return minimum, minimum_pair[0], minimum_pair[1]

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | int | None]:
        array = np.asarray(values, dtype=float)
        return {
            "samples": int(len(array)),
            "minimum_m": float(np.min(array)) if len(array) else None,
            "p01_m": float(np.quantile(array, 0.01)) if len(array) else None,
            "p05_m": float(np.quantile(array, 0.05)) if len(array) else None,
            "p50_m": float(np.quantile(array, 0.50)) if len(array) else None,
        }

    def summary(self) -> dict[str, object]:
        return {
            "minimum": self._minimum,
            "distribution": self._distribution(self._values),
            "by_phase": {
                phase: self._distribution(values)
                for phase, values in self._phase_values.items()
            },
            "robot_geom_count": len(self.robot_geoms),
            "table_geom_count": len(self.table_geoms),
            "invalid_pair_queries": self._invalid_pair_queries,
        }


class PickPlaceContactMonitor(RobotSafetyMonitor):
    """Robot safety plus target/finger contact queries for pick-and-place."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        target_body_name: str,
        receptacle_body_name: str,
        distractor_body_names: tuple[str, ...],
        robot_root_name: str = "link0",
        finger_body_names: tuple[str, str] = ("left_finger", "right_finger"),
        table_body_name: str = "work_surface",
    ) -> None:
        super().__init__(
            model,
            robot_root_name=robot_root_name,
            finger_body_names=finger_body_names,
            table_body_name=table_body_name,
        )
        self.target_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, target_body_name,
        )
        self.receptacle_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, receptacle_body_name,
        )
        self.distractor_bodies = {
            _required_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in distractor_body_names
        }
    def finger_target_contacts(self, data: mujoco.MjData) -> set[int]:
        contacting: set[int] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > 1e-6:
                continue
            bodies = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            if self.target_body in bodies:
                contacting.update(bodies & self.finger_bodies)
        return contacting
