"""Shared data access for differential RO PUF experiments."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from latent_puf.latent import BinaryLatentBank

REQUIRED_DATASETS = {
    "frequency_differences",
    "frequency_pairs",
    "device_ids",
    "sample_ids",
}


def split_samples(
    sample_count: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint enrollment and evaluation sample indices."""

    if sample_count < 4:
        raise ValueError("at least four repeated samples are required")
    indices = np.arange(sample_count, dtype=np.int64)
    if method == "alternating":
        return indices[::2], indices[1::2]
    if method == "sequential":
        midpoint = sample_count // 2
        return indices[:midpoint], indices[midpoint:]
    raise ValueError("split must be 'alternating' or 'sequential'")


def validate_ro_group(group: h5py.Group) -> tuple[int, int, int]:
    """Validate the common RO group schema and return ``(D, B, S)``."""

    missing = REQUIRED_DATASETS - set(group.keys())
    if missing:
        raise ValueError(f"{group.name} is missing datasets: {sorted(missing)}")

    differences = group["frequency_differences"]
    pairs = group["frequency_pairs"]
    device_ids = group["device_ids"]
    sample_ids = group["sample_ids"]
    if differences.ndim != 3:
        raise ValueError(f"{differences.name} must have shape (D, B, S)")
    devices, pair_count, samples = differences.shape
    if pairs.shape != (pair_count, 2):
        raise ValueError(f"{pairs.name} must have shape ({pair_count}, 2)")
    if device_ids.shape != (devices,):
        raise ValueError(f"{device_ids.name} must have shape ({devices},)")
    if sample_ids.shape != (samples,):
        raise ValueError(f"{sample_ids.name} must have shape ({samples},)")
    if devices < 1 or pair_count < 2:
        raise ValueError(f"{group.name} does not contain enough devices or pairs")
    return devices, pair_count, samples


def decode_device_id(value: Any) -> str:
    """Convert numeric and byte-string HDF5 identifiers into text."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def finite_mean(
    values: np.ndarray,
    axis: int | tuple[int, ...],
) -> np.ndarray:
    """Compute a finite-value mean without empty-slice warnings."""

    finite = np.isfinite(values)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, values, 0), axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=np.float64)
    return np.divide(total, count, out=result, where=count > 0)


def _row_standard_deviation(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = np.sum(finite, axis=1)
    mean = finite_mean(values, axis=1)
    squared = np.sum(
        np.where(finite, (values - mean[:, None]) ** 2, 0),
        axis=1,
    )
    result = np.full(len(values), np.nan, dtype=np.float64)
    return np.sqrt(
        np.divide(squared, count - 1, out=result, where=count > 1)
    )


def read_device_split(
    dataset: h5py.Dataset,
    device_index: int,
    enrollment: np.ndarray,
    evaluation: np.ndarray,
    block_size: int,
) -> tuple[BinaryLatentBank, np.ndarray, np.ndarray]:
    """Read one device and return its bank, noise, and evaluation responses."""

    pair_count = dataset.shape[1]
    z = np.full(pair_count, np.nan, dtype=np.float64)
    noise_std = np.full(pair_count, np.nan, dtype=np.float64)
    responses = np.full(
        (pair_count, len(evaluation)),
        np.nan,
        dtype=np.float32,
    )

    for start in range(0, pair_count, block_size):
        stop = min(start + block_size, pair_count)
        values = np.asarray(
            dataset[device_index, start:stop, :],
            dtype=np.float64,
        )
        z[start:stop] = finite_mean(values[:, enrollment], axis=1)
        noise_std[start:stop] = _row_standard_deviation(values)
        held_out = values[:, evaluation]
        responses[start:stop] = np.where(
            np.isfinite(held_out),
            held_out > 0,
            np.nan,
        )

    return BinaryLatentBank.from_threshold(z), noise_std, responses


def read_device_values(
    dataset: h5py.Dataset,
    device_index: int,
    block_size: int,
) -> np.ndarray:
    """Read one complete ``(pair, sample)`` device matrix in pair blocks."""

    pair_count, samples = dataset.shape[1:]
    values = np.empty((pair_count, samples), dtype=np.float32)
    for start in range(0, pair_count, block_size):
        stop = min(start + block_size, pair_count)
        values[start:stop] = dataset[device_index, start:stop, :]
    return values
