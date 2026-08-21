"""Shared option-distance helpers for SocioBench evaluation."""

from collections.abc import Iterable


def ordinal_core(options: Iterable[int]) -> tuple[int, ...]:
    """Return the contiguous ordinal scale, excluding special response codes."""
    values = {int(option) for option in options}
    if not values:
        return ()
    start = 0 if 0 in values else 1 if 1 in values else min(values)
    core = []
    value = start
    while value in values:
        core.append(value)
        value += 1
    return tuple(core)


def normalized_option_distance(
    prediction: int,
    target: int,
    valid_options: Iterable[int],
) -> float:
    """Compute bounded ordinal distance while treating special codes categorically."""
    if prediction == target:
        return 0.0
    core = ordinal_core(valid_options)
    if prediction in core and target in core:
        return abs(core.index(prediction) - core.index(target)) / max(len(core) - 1, 1)
    return 1.0
