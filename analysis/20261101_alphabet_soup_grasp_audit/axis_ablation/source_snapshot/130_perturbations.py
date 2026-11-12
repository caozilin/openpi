"""Episode-local semantic goal noise, applied before public action compilation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import math

import mujoco
import numpy as np

from ...geometry import rotation_matrix, rotation_vector
from ...simulation.robot import RobotHandles, ee_pose
from ..subtasks import identify_pick_place_subtasks
from .types import TaskFrame
from .wipe_whiteboard import WIPE_COMPRESSION_JITTER_M


@dataclass
class EpisodePerturbations:
    """Cache accepted scalar draws by subtask/node/axis, independent of call order.

    Each axis is N(0, sigma²) conditioned on |epsilon| <= 2 sigma. Extra
    geometric bounds also use rejection, never clipping. Seed -1 is nominal.
    """

    seed: int
    samples: dict[str, float] = field(default_factory=dict)

    def sample(
        self, key: str, sigma: float, *, lower: float = -math.inf,
        upper: float = math.inf,
    ) -> float:
        if key in self.samples:
            return self.samples[key]
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")
        if self.seed == -1 or sigma == 0.0:
            value = 0.0
        else:
            low, high = max(-2.0 * sigma, lower), min(2.0 * sigma, upper)
            if not low < high:
                raise ValueError(f"No legal perturbation for {key}")
            digest = hashlib.blake2b(key.encode(), digest_size=16).digest()
            words = np.frombuffer(digest, dtype="<u4").tolist()
            rng = np.random.default_rng(np.random.SeedSequence((self.seed, *words)))
            for _ in range(100_000):
                value = float(rng.normal(0.0, sigma))
                if low <= value <= high:
                    break
            else:
                raise ValueError(f"Unable to sample a legal perturbation for {key}")
        self.samples[key] = value
        return value


def strict_placement(task: str, phase: str) -> bool:
    """Classify each destination, including mixed multi-subtask episodes."""
    text = f"{task} {phase}".casefold()
    return any(word in text for word in (
        "region", "paper", "narrow box", "insert", "stack", "close pot lid",
        "onto pot", "stable", "without tilting",
    ))


def grasp_height_floor(
    model: mujoco.MjModel, data: mujoco.MjData, handles: RobotHandles,
) -> Callable[[np.ndarray], float]:
    """Lowest legal EE height over the live table for open and closed fingers.

    Transform collision geometry rigidly from EE coordinates; this is a
    geometric planning check and advances no physics or controller state.
    """
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
    table_geoms = np.flatnonzero(model.geom_bodyid == table) if table >= 0 else []
    table_top = max((float(data.geom_xpos[g, 2] +
                          np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
                     for g in table_geoms), default=-math.inf)
    root = int(model.site_bodyid[handles.ee_site_id])
    geom_ids = []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        while body and body != root:
            body = int(model.body_parentid[body])
        if body == root and (model.geom_contype[g] or model.geom_conaffinity[g]):
            geom_ids.append(g)
    geometry = []
    probe = mujoco.MjData(model)
    mujoco.mj_copyData(probe, model, data)
    for opened in (True, False):
        for joint, qpos in zip(handles.finger_joint_ids, handles.finger_qpos_ids):
            upper = opened == handles.spec.gripper_open_at_upper
            probe.qpos[qpos] = model.jnt_range[joint, int(upper)]
        mujoco.mj_forward(model, probe)
        position, rotation = ee_pose(probe, handles)
        for g in geom_ids:
            geometry.append((g, rotation.T @ (probe.geom_xpos[g] - position),
                             rotation.T @ probe.geom_xmat[g].reshape(3, 3)))

    def floor(rotation: np.ndarray) -> float:
        bottom = math.inf
        for g, center, local_rotation in geometry:
            row = (rotation @ local_rotation)[2]
            size = model.geom_size[g]
            kind = model.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = float(np.abs(row) @ size)
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = float(size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = float(size[0] + abs(row[2]) * size[1])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = float(size[0] * np.linalg.norm(row[:2]) + size[1] * abs(row[2]))
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent = float(np.linalg.norm(row * size))
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(model.geom_dataid[g])
                first = int(model.mesh_vertadr[mesh])
                vertices = model.mesh_vert[first:first + int(model.mesh_vertnum[mesh])]
                extent = -float(np.min(vertices @ row))
            else:
                extent = float(model.geom_rbound[g])
            bottom = min(bottom, float((rotation @ center)[2]) - extent)
        return table_top - bottom if geometry else -math.inf

    return floor


def perturb_task_frames(
    frames: Sequence[TaskFrame],
    initial_position: np.ndarray,
    initial_rotation: np.ndarray,
    noise: EpisodePerturbations,
    *,
    task: str,
    subtask_index: int,
    grasp_floor: Callable[[np.ndarray], float],
) -> list[TaskFrame]:
    """Deform the existing motion profiles between semantic goal anchors.

    No noise is drawn per tick. Descent and lift share pickup XY; grasp yaw is
    already reached at pregrasp and persists through descent/closure/lift.
    Constant nominal yaw axes acquire no independent orientation noise.
    """
    if noise.seed == -1:
        return list(frames)
    result = list(frames)
    cycle_start = 0

    def has_yaw(first: int, last: int) -> bool:
        previous = initial_rotation if first == 0 else frames[first - 1].rotation
        for frame in frames[first:last + 1]:
            if abs(float(rotation_vector(frame.rotation @ previous.T)[2])) > 1e-8:
                return True
            previous = frame.rotation
        return False

    for span in identify_pick_place_subtasks(frames):
        close = span.pregrasp_index + 1
        end = span.postgrasp_index
        # Pressing and knob control have no pick/lift node.
        lifts = [i for i in range(close, end + 1)
                 if frames[i].phase.casefold().startswith(("lift ", "raise "))]
        descents = [i for i in range(cycle_start, close)
                    if frames[i].phase.casefold().startswith(("descend ", "lower "))]
        if not lifts or not descents:
            cycle_start = end + 1
            continue
        descent = descents[-1]
        while descent > cycle_start and frames[descent - 1].phase == frames[descent].phase:
            descent -= 1
        pre = descent - 1
        if pre < cycle_start:
            cycle_start = end + 1
            continue
        grasp = frames[span.pregrasp_index]
        prefix = f"subtask/{subtask_index + span.index}"

        def draw(node: str, axis: str, sigma: float, **bounds: float) -> float:
            return noise.sample(f"{prefix}/{node}/{axis}", sigma, **bounds)

        xy = np.array([draw("pregrasp", axis, 0.002) for axis in ("x", "y")])
        yaw_active = has_yaw(cycle_start, pre)
        yaw = draw("grasp", "yaw", math.radians(1.0) if yaw_active else 0.0)
        biased_rotation = rotation_matrix(np.array((0.0, 0.0, yaw))) @ grasp.rotation
        dz = draw("grasp", "z", 0.001,
                  lower=grasp_floor(biased_rotation) - float(grasp.position[2]),
                  upper=float(frames[pre].position[2] - grasp.position[2]) - 0.010)
        grasp_offset = np.array((*xy, dz))
        pre_offset = np.array((*xy, draw("pregrasp", "z", 0.005)))
        lift_offset = np.array((*xy, dz + draw("lift", "z", 0.005)))
        lift_end = lifts[-1]
        # Separate height raises from the subsequent transfer; both inherit XY.
        lowerings = [i for i in range(lift_end + 1, end + 1)
                     if frames[i].phase.casefold().startswith(("lower ", "insert ", "press cloth "))]
        prior_offset = (np.zeros(3) if cycle_start == 0 else
                        result[cycle_start - 1].position - frames[cycle_start - 1].position)
        prior_yaw = (0.0 if cycle_start == 0 else float(rotation_vector(
            result[cycle_start - 1].rotation @ frames[cycle_start - 1].rotation.T)[2]))
        anchors: list[tuple[int, np.ndarray, float]] = [
            (cycle_start - 1, prior_offset, prior_yaw),
            (pre, pre_offset, yaw),
            (span.pregrasp_index, grasp_offset, yaw),
            (lifts[0] - 1, grasp_offset, yaw),
        ]
        anchors.extend((i, lift_offset, yaw) for i in lifts
                       if i == end or frames[i + 1].phase != frames[i].phase)
        # Keep an exceptional initial safety raise intact before approaching.
        approach = next((i for i in range(cycle_start, pre + 1)
                         if frames[i].phase.casefold().startswith("move above ")), cycle_start)
        if approach > cycle_start:
            anchors.append((approach - 1, prior_offset, prior_yaw))
        final_offset, final_yaw = lift_offset, yaw
        if lowerings:
            lower_start = lowerings[0]
            final = frames[end]
            # The destination semantics take precedence over a mixed task name.
            strict = strict_placement("", final.phase + " " + frames[lower_start].phase)
            if not strict:
                strict = strict_placement(task, "") and "then" not in task.casefold()
            sigma_xy = 0.0005 if strict else 0.005
            wiping = (frames[lower_start].phase.casefold().startswith("press cloth ")
                      and final.phase.casefold() == "wipe whiteboard")
            if wiping:
                # A lower/higher pinch changes the EE-to-cloth offset by dz.
                # Preserve that transform, then independently randomize the
                # desired compression, once per episode (4 +/- 1 mm).
                compression_delta = draw("contact", "compression_delta",
                    WIPE_COMPRESSION_JITTER_M / 2,
                    lower=-WIPE_COMPRESSION_JITTER_M, upper=WIPE_COMPRESSION_JITTER_M)
                if abs(compression_delta) > WIPE_COMPRESSION_JITTER_M:
                    raise ValueError("Cached whiteboard compression perturbation is out of range")
                final_z = dz - compression_delta
            else:
                final_z = draw("placement", "z", 0.001)
            final_offset = np.array((
                draw("placement", "x", sigma_xy),
                draw("placement", "y", sigma_xy),
                final_z,
            ))
            place_yaw_active = has_yaw(lift_end + 1, end)
            final_yaw = (draw("placement", "yaw", math.radians(1.0))
                         if place_yaw_active else yaw)
            above_offset = np.array((*final_offset[:2], lift_offset[2]))
            anchors.extend(((lower_start - 1, above_offset, final_yaw),
                            (end, final_offset, final_yaw)))
            if wiping:
                # Finish the descent's perturbation AT first contact, not at
                # the end of the horizontal stroke. All later wipe frames
                # keep the same height and compression, without lift drift.
                lower_end = lower_start
                while lower_end < end and frames[lower_end + 1].phase == frames[lower_start].phase:
                    lower_end += 1
                anchors.append((lower_end, final_offset, final_yaw))
        else:
            anchors.append((end, final_offset, final_yaw))
        release_end = end
        while release_end + 1 < len(frames) and not frames[release_end + 1].gripper_closed:
            if not frames[release_end + 1].phase.casefold().startswith(("release", "hold", "settle")):
                break
            release_end += 1
        anchors.append((release_end, final_offset, final_yaw))
        # Coincident semantic endpoints (e.g. no close dwell) share one anchor.
        unique = {index: (offset, angle) for index, offset, angle in anchors}
        ordered = sorted(unique)
        for first, last in zip(ordered, ordered[1:]):
            p0 = initial_position if first < 0 else frames[first].position
            r0 = initial_rotation if first < 0 else frames[first].rotation
            distance = np.array([
                np.linalg.norm(frames[i].position - (p0 if i == first + 1 else frames[i - 1].position))
                + 0.05 * np.linalg.norm(rotation_vector(
                    frames[i].rotation @ (r0 if i == first + 1 else frames[i - 1].rotation).T))
                for i in range(first + 1, last + 1)
            ])
            progress = np.cumsum(distance)
            if progress.size and progress[-1] > 1e-12:
                progress /= progress[-1]
            else:
                progress = np.ones(last - first)
            offset0, yaw0 = unique[first]
            offset1, yaw1 = unique[last]
            for i, fraction in zip(range(first + 1, last + 1), progress):
                offset = offset0 + fraction * (offset1 - offset0)
                angle = yaw0 + fraction * (yaw1 - yaw0)
                frame = frames[i]
                result[i] = TaskFrame(
                    frame.position + offset,
                    rotation_matrix(np.array((0.0, 0.0, angle))) @ frame.rotation,
                    frame.gripper_closed, frame.phase,
                )
        cycle_start = release_end + 1
    return result
