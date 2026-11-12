"""TaskTol adapter around the locally maintained OpenPI websocket client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OpenPIClientConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    action_key: str = "actions"


class OpenPIWebsocketPolicy:
    """Adapt ``openpi_client.WebsocketClientPolicy`` to ``ActionPolicy``."""

    def __init__(self, config: OpenPIClientConfig = OpenPIClientConfig()) -> None:
        try:
            from .clientlib.websocket_client_policy import WebsocketClientPolicy
        except ImportError as error:
            raise RuntimeError(
                "Bundled OpenPI client dependencies are unavailable; run `uv sync`."
            ) from error
        self.config = config
        self._client = WebsocketClientPolicy(
            host=config.host,
            port=config.port,
            api_key=config.api_key,
        )

    @property
    def server_metadata(self) -> dict[str, Any]:
        return dict(self._client.get_server_metadata())

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self._client.infer(observation)
        if self.config.action_key not in response:
            raise KeyError(
                f"OpenPI response has no {self.config.action_key!r}: {tuple(response)}"
            )
        actions = np.asarray(response[self.config.action_key], dtype=float)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7:
            raise ValueError("OpenPI actions must have non-empty shape [horizon, 7]")
        if not np.all(np.isfinite(actions)):
            raise ValueError("OpenPI actions must be finite")
        result = dict(response)
        result[self.config.action_key] = actions
        for key in ("stage_target_pose", "tolerance_frame_code", "tolerance_frame"):
            if key in response:
                raise KeyError(
                    f"OpenPI returned obsolete {key!r}; TaskTol requires only "
                    "7D actions and 6D rotation_tolerance"
                )
        if "rotation_tolerance" in response:
            value = np.asarray(response["rotation_tolerance"], dtype=float)
            if value.ndim == 1:
                value = value[None, :]
            if value.shape != (actions.shape[0], 6):
                raise ValueError(
                    f"OpenPI rotation_tolerance must have shape [{actions.shape[0]}, 6]"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError("OpenPI rotation_tolerance must be finite")
            result["rotation_tolerance"] = value
        return result

    def reset(self) -> None:
        self._client.reset()

    def close(self) -> None:
        self._client.close()


def franka_mujoco_observation(
    ee_position_m: np.ndarray,
    ee_rotation_vector_base_rad: np.ndarray,
    finger_positions_m: np.ndarray,
    main_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """Build the exact runtime input used by ``pi05_franka_mujoco``."""
    position = np.asarray(ee_position_m, dtype=np.float32)
    rotation = np.asarray(ee_rotation_vector_base_rad, dtype=np.float32)
    fingers = np.asarray(finger_positions_m, dtype=np.float32)
    if position.shape != (3,) or rotation.shape != (3,) or fingers.shape != (2,):
        raise ValueError("Franka pose must be 3+3 and finger state must contain two values")
    if not all(np.all(np.isfinite(value)) for value in (position, rotation, fingers)):
        raise ValueError("Franka OpenPI state must be finite")
    state = np.concatenate((
        position,
        rotation,
        np.asarray((0.5 * float(np.sum(fingers)),), dtype=np.float32),
    ))
    if state.shape != (7,):
        raise AssertionError("pi05_franka_mujoco state must be seven-dimensional")
    return {
        "observation/state": state,
        "observation/image": np.asarray(main_rgb, dtype=np.uint8),
        "observation/wrist_image": np.asarray(wrist_rgb, dtype=np.uint8),
        "prompt": str(prompt),
    }
