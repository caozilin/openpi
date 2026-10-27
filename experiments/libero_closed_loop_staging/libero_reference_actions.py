"""Read validated physical reference increments without changing source data."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ..geometry import rotation_vector
from ..tolerance.runtime import advance_reference_pose


@dataclass(frozen=True)
class LiberoReferenceActions:
    actions: np.ndarray
    anchor_position: np.ndarray
    anchor_rotation: np.ndarray
    goal_positions: np.ndarray
    goal_rotations: np.ndarray
    metadata: dict


def load_reference_actions(path, demo_id, *, source_path, source_states, source_actions):
    """Fail closed on stale inputs, wrong action contracts or broken goals."""
    import h5py

    path = Path(path).resolve()
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema") != "libero_osc_independent_reference_actions_v1":
            raise ValueError("Unexpected reference-action schema")
        if Path(file.attrs["source_hdf5"]).resolve() != Path(source_path).resolve():
            raise ValueError("Reference actions belong to a different source HDF5")
        demo = file["data"][demo_id]
        expected_hash = hashlib.sha256(source_states.tobytes() + source_actions.tobytes()).hexdigest()
        if demo.attrs["input_state_action_sha256"] != expected_hash:
            raise ValueError("Reference source state/action hash does not match")
        actions = demo["actions"][()]
        anchor_position = demo["anchor_position_m"][()]
        anchor_rotation = demo["anchor_tool_rotation"][()]
        positions = demo["goal_positions_m"][()]
        rotations = demo["goal_tool_rotations"][()]
        contract = json.loads(file.attrs["action_contract"])
    count = len(source_actions)
    if (
        actions.shape != (count, 7)
        or positions.shape != (count, 3)
        or rotations.shape != (count, 3, 3)
        or anchor_position.shape != (3,)
        or anchor_rotation.shape != (3, 3)
        or count == 0
        or not all(np.all(np.isfinite(a)) for a in (actions, positions, rotations, anchor_position, anchor_rotation))
    ):
        raise ValueError("Invalid reference-action shapes or values")
    if not np.array_equal(actions[:, 6], source_actions[:, 6]):
        raise ValueError("Reference gripper actions differ from the source")
    all_rotations = np.concatenate([anchor_rotation[None], rotations])
    if not np.allclose(np.swapaxes(all_rotations, 1, 2) @ all_rotations, np.eye(3), atol=1e-9, rtol=0):
        raise ValueError("Reference rotations must belong to SO(3)")
    if not np.allclose(np.linalg.det(all_rotations), 1.0, atol=1e-9, rtol=0):
        raise ValueError("Reference rotations must belong to SO(3)")
    position, rotation = anchor_position.copy(), anchor_rotation.copy()
    position_error = rotation_error = 0.0
    for action, goal_position, goal_rotation in zip(actions, positions, rotations, strict=True):
        position, rotation = advance_reference_pose(position, rotation, action)
        position_error = max(position_error, float(np.linalg.norm(position - goal_position)))
        rotation_error = max(rotation_error, float(np.linalg.norm(rotation_vector(rotation @ goal_rotation.T))))
    if position_error > 1e-10 or rotation_error > 1e-9:
        raise ValueError("Reference actions do not reconstruct saved goals")
    return LiberoReferenceActions(
        actions, anchor_position, anchor_rotation, positions, rotations,
        dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), demo=demo_id,
             source_input_sha256=expected_hash, action_contract=contract,
             reconstruction_position_max_m=position_error, reconstruction_rotation_max_rad=rotation_error),
    )


def verify_reference_anchor(reference, actual_position, actual_rotation):
    """Require the saved nominal anchor to match the restored initial robot."""
    if (
        np.linalg.norm(reference.anchor_position - actual_position) > 1e-10
        or np.linalg.norm(rotation_vector(reference.anchor_rotation @ actual_rotation.T)) > 1e-9
    ):
        raise ValueError("Reference anchor differs from the restored actual initial pose")
