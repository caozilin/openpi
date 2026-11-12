"""Wall-clock scheduling for display playback; simulation steps stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class SimulationPlaybackClock:
    physics_dt: float
    refresh_hz: float
    _step_credit: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0.0:
            raise ValueError("Physics timestep must be finite and positive")
        if not math.isfinite(self.refresh_hz) or self.refresh_hz <= 0.0:
            raise ValueError("Display refresh rate must be finite and positive")

    def frame(self, playback_scale: float) -> tuple[int, float]:
        """Return complete physics steps and their desired wall-clock duration."""
        if not math.isfinite(playback_scale) or playback_scale <= 0.0:
            raise ValueError("Playback multiplier must be finite and positive")
        self._step_credit += playback_scale / (self.refresh_hz * self.physics_dt)
        steps = max(1, int(math.floor(self._step_credit + 1e-12)))
        self._step_credit = max(0.0, self._step_credit - steps)
        # At very low speeds a frame still contains one complete physics step.
        # Its longer wall interval preserves the requested multiplier exactly.
        return steps, steps * self.physics_dt / playback_scale
