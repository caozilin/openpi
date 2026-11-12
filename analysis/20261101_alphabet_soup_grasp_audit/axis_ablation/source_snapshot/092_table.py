"""Deterministic position randomization for the work surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RandomizedTablePose:
    position: np.ndarray
    offset: np.ndarray


class TableRandomizer:
    """Uniformly translate a table along forward, lateral and vertical axes."""

    # World X is forward/back, Y is left/right and Z is vertical.
    MAX_OFFSET_M = np.array((0.04, 0.05, 0.03))
    _RNG_STREAM_ID = 0x5441424C45  # "TABLE", independent of the camera stream.

    def __init__(self, center_position: np.ndarray) -> None:
        self.center_position = np.asarray(center_position, dtype=float).copy()

    def sample(self, seed: int) -> RandomizedTablePose:
        if seed == -1:
            return RandomizedTablePose(self.center_position.copy(), np.zeros(3))
        if seed < -1:
            raise ValueError("Domain seed must be -1 (disabled) or a non-negative integer")

        seed_sequence = np.random.SeedSequence((int(seed), self._RNG_STREAM_ID))
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        offset = rng.uniform(-self.MAX_OFFSET_M, self.MAX_OFFSET_M)
        position = self.center_position + offset
        return RandomizedTablePose(position, offset)
