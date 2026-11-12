"""Portable adaptive reachability-map storage and difficulty sampling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReachabilitySample:
    position: np.ndarray
    badness: float
    reachable: bool
    quantile: float
    resolution_m: float
    index: int


@dataclass(frozen=True)
class ReachabilityMap:
    positions: np.ndarray
    badness: np.ndarray
    reachable: np.ndarray
    resolution_m: np.ndarray
    q: np.ndarray
    position_error_m: np.ndarray
    rotation_error_rad: np.ndarray
    minimum_singular_value: np.ndarray
    manipulability: np.ndarray
    joint_margin_fraction: np.ndarray
    self_collision_clearance_m: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        count = int(np.asarray(self.positions).shape[0])
        if np.asarray(self.positions).shape != (count, 3):
            raise ValueError("Reachability positions must have shape [N, 3]")
        for name in (
            "badness", "reachable", "resolution_m", "position_error_m",
            "rotation_error_rad", "minimum_singular_value", "manipulability",
            "joint_margin_fraction", "self_collision_clearance_m",
        ):
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"Reachability field {name} must have shape [N]")
        if np.asarray(self.q).shape[0] != count:
            raise ValueError("Reachability joint configurations must have N rows")

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            positions=self.positions,
            badness=self.badness,
            reachable=self.reachable,
            resolution_m=self.resolution_m,
            q=self.q,
            position_error_m=self.position_error_m,
            rotation_error_rad=self.rotation_error_rad,
            minimum_singular_value=self.minimum_singular_value,
            manipulability=self.manipulability,
            joint_margin_fraction=self.joint_margin_fraction,
            self_collision_clearance_m=self.self_collision_clearance_m,
            metadata_json=np.array(json.dumps(self.metadata, sort_keys=True)),
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> "ReachabilityMap":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            return cls(
                positions=archive["positions"].copy(),
                badness=archive["badness"].copy(),
                reachable=archive["reachable"].astype(bool),
                resolution_m=archive["resolution_m"].copy(),
                q=archive["q"].copy(),
                position_error_m=archive["position_error_m"].copy(),
                rotation_error_rad=archive["rotation_error_rad"].copy(),
                minimum_singular_value=archive["minimum_singular_value"].copy(),
                manipulability=archive["manipulability"].copy(),
                joint_margin_fraction=archive["joint_margin_fraction"].copy(),
                self_collision_clearance_m=archive["self_collision_clearance_m"].copy(),
                metadata=metadata,
            )

    def _candidate_indices(
        self,
        *,
        fixed_z_m: float | None,
        z_tolerance_m: float,
    ) -> np.ndarray:
        indices = np.arange(self.count)
        if fixed_z_m is not None:
            mask = np.abs(self.positions[:, 2] - float(fixed_z_m)) <= z_tolerance_m
            indices = indices[mask]
            if indices.size == 0:
                nearest = float(np.min(np.abs(self.positions[:, 2] - float(fixed_z_m))))
                indices = np.flatnonzero(
                    np.abs(self.positions[:, 2] - float(fixed_z_m)) <= nearest + 1e-9,
                )
        return indices

    def sample(
        self,
        difficulty: float,
        seed: int,
        *,
        fixed_z_m: float | None = None,
        z_tolerance_m: float = 0.006,
        quantile_window: float = 0.04,
    ) -> ReachabilitySample:
        difficulty = float(difficulty)
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("Placement difficulty must be in [0, 1]")
        indices = self._candidate_indices(
            fixed_z_m=fixed_z_m, z_tolerance_m=z_tolerance_m,
        )
        scores = self.badness[indices]
        # Finite sorting keeps +inf unreachable/singular samples at the top.
        order = np.argsort(scores, kind="stable")
        ordered = indices[order]
        ordered_weights = np.maximum(self.resolution_m[ordered], 1e-6) ** 3
        cumulative = np.cumsum(ordered_weights)
        ordered_quantiles = (
            cumulative - 0.5 * ordered_weights
        ) / cumulative[-1]
        if difficulty >= 1.0:
            unreachable = ordered[~self.reachable[ordered]]
            pool = unreachable if unreachable.size else ordered[max(0, int(0.99 * ordered.size)):]
        else:
            lower_q = max(0.0, difficulty - quantile_window)
            upper_q = min(1.0, difficulty + quantile_window)
            pool = ordered[
                (ordered_quantiles >= lower_q)
                & (ordered_quantiles <= upper_q)
            ]
            if pool.size == 0:
                nearest = int(np.argmin(np.abs(ordered_quantiles - difficulty)))
                pool = ordered[nearest:nearest + 1]
        rng_seed = 0 if seed == -1 else int(seed)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            rng_seed, 0x5245414348,
        ))))
        # Each point represents a local leaf volume; compensate for adaptive
        # sampling density so 1 cm danger cells do not gain accidental mass.
        weights = np.maximum(self.resolution_m[pool], 1e-6) ** 3
        weights /= np.sum(weights)
        index = int(rng.choice(pool, p=weights))
        rank = int(np.flatnonzero(ordered == index)[0])
        quantile = float(ordered_quantiles[rank])
        return ReachabilitySample(
            self.positions[index].copy(), float(self.badness[index]),
            bool(self.reachable[index]), float(quantile),
            float(self.resolution_m[index]), index,
        )
