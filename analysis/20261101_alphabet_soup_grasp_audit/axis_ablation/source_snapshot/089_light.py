"""Deterministic, scene-independent MuJoCo light randomization."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ...geometry import axis_angle_matrix


@dataclass(frozen=True)
class LightRandomizationLimits:
    """Conservative illumination variation suitable for RGB policy data."""

    brightness_range: tuple[float, float] = (0.70, 1.30)
    ambient_range: tuple[float, float] = (0.80, 1.20)
    specular_range: tuple[float, float] = (0.70, 1.25)
    warmth: float = 0.08
    direction_tilt_deg: float = 12.0
    position_offset_m: float = 0.15


@dataclass(frozen=True)
class RandomizedLighting:
    positions: np.ndarray
    directions: np.ndarray
    ambient: np.ndarray
    diffuse: np.ndarray
    specular: np.ndarray
    brightness: np.ndarray
    warmth: np.ndarray
    direction_angles_deg: np.ndarray
    position_offsets: np.ndarray


class LightRandomizer:
    """Map one domain seed to reproducible position, direction and colour."""

    _RNG_STREAM_ID = 0x4C49474854  # "LIGHT", independent of camera/table.

    def __init__(self, model: mujoco.MjModel, limits: LightRandomizationLimits | None = None) -> None:
        self.limits = limits or LightRandomizationLimits()
        self._positions = model.light_pos.copy()
        self._directions = model.light_dir.copy()
        self._ambient = model.light_ambient.copy()
        self._diffuse = model.light_diffuse.copy()
        self._specular = model.light_specular.copy()

    @staticmethod
    def _tangent_axes(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direction = direction / np.linalg.norm(direction)
        reference = np.array((0.0, 0.0, 1.0))
        if abs(float(direction @ reference)) > 0.90:
            reference = np.array((0.0, 1.0, 0.0))
        first = np.cross(direction, reference)
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        second /= np.linalg.norm(second)
        return first, second

    def sample(self, seed: int) -> RandomizedLighting:
        light_count = self._directions.shape[0]
        if seed == -1:
            return RandomizedLighting(
                self._positions.copy(), self._directions.copy(), self._ambient.copy(),
                self._diffuse.copy(), self._specular.copy(), np.ones(light_count),
                np.zeros(light_count), np.zeros((light_count, 2)),
                np.zeros((light_count, 3)),
            )
        if seed < -1:
            raise ValueError("Domain seed must be -1 (disabled) or a non-negative integer")

        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((int(seed), self._RNG_STREAM_ID))))
        limits = self.limits
        brightness = rng.uniform(*limits.brightness_range, size=light_count)
        ambient_scale = rng.uniform(*limits.ambient_range, size=light_count)
        specular_scale = rng.uniform(*limits.specular_range, size=light_count)
        warmth = rng.uniform(-limits.warmth, limits.warmth, size=light_count)
        direction_angles_deg = rng.uniform(
            -limits.direction_tilt_deg, limits.direction_tilt_deg, size=(light_count, 2),
        )
        position_offsets = rng.uniform(
            -limits.position_offset_m, limits.position_offset_m, size=(light_count, 3),
        )

        directions = self._directions.copy()
        for index, base_direction in enumerate(self._directions):
            first, second = self._tangent_axes(base_direction)
            first_angle, second_angle = np.radians(direction_angles_deg[index])
            rotated = axis_angle_matrix(second, second_angle) @ axis_angle_matrix(first, first_angle) @ base_direction
            directions[index] = rotated / np.linalg.norm(rotated)

        colour_scale = np.column_stack((1.0 + warmth, np.ones(light_count), 1.0 - warmth))
        diffuse = np.clip(self._diffuse * brightness[:, None] * colour_scale, 0.0, 1.0)
        ambient = np.clip(self._ambient * ambient_scale[:, None] * colour_scale, 0.0, 1.0)
        specular = np.clip(self._specular * specular_scale[:, None] * colour_scale, 0.0, 1.0)
        return RandomizedLighting(
            self._positions + position_offsets, directions, ambient, diffuse, specular,
            brightness, warmth, direction_angles_deg, position_offsets,
        )

    @staticmethod
    def apply(model: mujoco.MjModel, lighting: RandomizedLighting) -> None:
        if lighting.positions.shape != model.light_pos.shape:
            raise ValueError("Randomized lighting does not match this model's light count")
        model.light_pos[:] = lighting.positions
        model.light_dir[:] = lighting.directions
        model.light_ambient[:] = lighting.ambient
        model.light_diffuse[:] = lighting.diffuse
        model.light_specular[:] = lighting.specular
