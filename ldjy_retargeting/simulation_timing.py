"""Deterministic control-tick to MuJoCo-substep scheduling."""

from __future__ import annotations


def physics_steps_for_tick(
    tick_index: int, physics_timestep: float, control_hz: int
) -> int:
    """Return the 4/5-step pattern that preserves average physics time."""
    if tick_index < 0:
        raise ValueError("tick_index must be non-negative")
    if physics_timestep <= 0 or control_hz <= 0:
        raise ValueError("physics_timestep and control_hz must be positive")
    physics_steps_per_tick = 1.0 / (physics_timestep * control_hz)
    return int((tick_index + 1) * physics_steps_per_tick + 1e-12) - int(
        tick_index * physics_steps_per_tick + 1e-12
    )


__all__ = ["physics_steps_for_tick"]
