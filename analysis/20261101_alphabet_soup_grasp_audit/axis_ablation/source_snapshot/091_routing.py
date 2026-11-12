"""Central seed routing for object variation versus environment interference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_ENVIRONMENT_SELECTION_STREAM = 0x454E5653454C4543  # "ENVSELEC"


@dataclass(frozen=True)
class RandomizationSeedRouting:
    """Bind every randomizer to either the environment or task seed group.

    Object identity and layout define the task case and always use
    ``episode_seed``. Table pose, RGB camera and lighting are
    visual/environment interference and collectively use ``-1`` when that
    interference is disabled. Semantic goal perturbations use the episode
    seed and are deterministic for that episode, independently of environment.
    """

    episode_seed: int
    environment_randomization_probability: float = 0.0

    def __post_init__(self) -> None:
        probability = float(self.environment_randomization_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Environment randomization probability must be in [0, 1]")
        object.__setattr__(self, "environment_randomization_probability", probability)

    @property
    def environment_randomization(self) -> bool:
        probability = self.environment_randomization_probability
        if self.episode_seed == -1 or probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(self.episode_seed), _ENVIRONMENT_SELECTION_STREAM,
        ))))
        return bool(rng.random() < probability)

    @property
    def environment_seed(self) -> int:
        return int(self.episode_seed) if self.environment_randomization else -1

    @property
    def object_layout_seed(self) -> int:
        return int(self.episode_seed)

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "episode_seed": int(self.episode_seed),
            "environment_randomization_enabled": bool(
                self.environment_randomization
            ),
            "environment_randomization_probability": (
                self.environment_randomization_probability
            ),
            "environment_seed": self.environment_seed,
            "table_seed": self.environment_seed,
            "front_camera_seed": self.environment_seed,
            "lighting_seed": self.environment_seed,
            "object_layout_seed": self.object_layout_seed,
            "semantic_goal_seed": int(self.episode_seed),
        }
