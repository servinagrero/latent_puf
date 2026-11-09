"""Common latent representation for binary-decision PUFs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryLatentBank:
    """Physical coordinates and signed decision margins for one PUF bank."""

    z: np.ndarray
    margin: np.ndarray

    def __post_init__(self) -> None:
        z = np.asarray(self.z)
        margin = np.asarray(self.margin, dtype=np.float64)
        if z.ndim < 1:
            raise ValueError("z must contain one entry per PUF response")
        if margin.ndim != 1:
            raise ValueError("margin must be one-dimensional")
        if len(z) != len(margin):
            raise ValueError("z and margin must describe the same responses")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "margin", margin)

    @classmethod
    def from_margin(cls, margin: np.ndarray) -> BinaryLatentBank:
        """Create a bank when the physical coordinate is already a signed margin."""

        values = np.asarray(margin, dtype=np.float64)
        return cls(z=values, margin=values)

    @classmethod
    def from_threshold(
        cls,
        z: np.ndarray,
        threshold: float | np.ndarray = 0.0,
        *,
        one_above_threshold: bool = True,
    ) -> BinaryLatentBank:
        """Convert scalar physical coordinates into signed decision margins."""

        values = np.asarray(z, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("threshold decisions require a one-dimensional z")
        try:
            margin = values - threshold
        except ValueError as error:
            raise ValueError("threshold is not compatible with z") from error
        if not one_above_threshold:
            margin = -margin
        return cls(z=values, margin=margin)

    @property
    def valid(self) -> np.ndarray:
        """Identify entries with finite signed margins."""

        return np.isfinite(self.margin)

    @property
    def nominal_response(self) -> np.ndarray:
        """Return nominal responses as 0, 1, or NaN for invalid entries."""

        return np.where(self.valid, self.margin > 0, np.nan).astype(np.float64)


def random_neighbor_indices(
    candidate_indices: np.ndarray,
    targets: np.ndarray,
    neighbors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select reproducible random controls while excluding each target."""

    if candidate_indices.ndim != 1 or len(np.unique(candidate_indices)) != len(
        candidate_indices
    ):
        raise ValueError("candidates must be a vector of unique indices")
    count = len(candidate_indices)
    if not 1 <= neighbors < count:
        raise ValueError("invalid random neighbor count")
    ranks = {int(index): rank for rank, index in enumerate(candidate_indices)}
    if any(int(target) not in ranks for target in targets):
        raise ValueError("every target must be a candidate")

    stride = int(rng.integers(1, count))
    while math.gcd(stride, count) != 1:
        stride = (stride + 1) % count or 1
    starts = rng.integers(0, count, size=len(targets), dtype=np.int64)
    offsets = stride * np.arange(neighbors + 1, dtype=np.int64)
    candidates = (starts[:, None] + offsets[None, :]) % count

    result = np.empty((len(targets), neighbors), dtype=np.int64)
    for row, target in enumerate(targets):
        positions = candidates[row, candidates[row] != ranks[int(target)]][:neighbors]
        result[row] = candidate_indices[positions]
    return result
