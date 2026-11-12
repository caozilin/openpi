"""Unique reproducible episode-seed selection."""

from __future__ import annotations

import secrets


def sample_batch_seeds(
    batch_size: int,
    *,
    include_nominal: bool = True,
    excluded: set[int] | None = None,
) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    excluded = set() if excluded is None else set(excluded)
    seeds = [-1] if include_nominal else []
    selected = set(excluded)
    while len(seeds) < batch_size:
        seed = secrets.randbelow(1 << 32)
        if seed not in selected:
            selected.add(seed)
            seeds.append(seed)
    return seeds
