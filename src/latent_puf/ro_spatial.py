"""Streaming temporal--spatial analysis for differential RO PUF datasets."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from latent_puf.latent import random_neighbor_indices
from latent_puf.ro_data import (
    decode_device_id,
    finite_mean,
    read_device_split,
    split_samples,
    validate_ro_group,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ROSpatialConfig:
    """Configuration for one streaming analysis pass."""

    neighbors: tuple[int, ...] = (4, 8, 16, 32, 64)
    pair_block_size: int = 65_536
    targets_per_device: int = 10_000
    curve_bins: int = 64
    split: str = "alternating"
    seed: int = 123_456


def nearest_latent_indices(
    latent: np.ndarray,
    targets: np.ndarray,
    neighbors: int,
    *,
    pairs: np.ndarray | None = None,
    disjoint_from_target: bool = False,
) -> np.ndarray:
    """Find scalar-latent nearest neighbors, optionally excluding shared ROs."""

    if latent.ndim != 1 or targets.ndim != 1:
        raise ValueError("latent and targets must be vectors")
    if neighbors < 1 or neighbors >= np.isfinite(latent).sum():
        raise ValueError("invalid neighbor count")
    if disjoint_from_target and (pairs is None or pairs.shape != (len(latent), 2)):
        raise ValueError("frequency pairs are required for disjoint selection")
    valid = np.flatnonzero(np.isfinite(latent))
    ordered = valid[np.argsort(latent[valid], kind="stable")]
    ranks = np.full(len(latent), -1, dtype=np.int64)
    ranks[ordered] = np.arange(len(ordered))
    result = np.empty((len(targets), neighbors), dtype=np.int64)

    for row, target in enumerate(targets):
        rank = ranks[target]
        if rank < 0:
            raise ValueError("target has a non-finite latent value")
        left, right = rank - 1, rank + 1
        selected: list[int] = []
        while len(selected) < neighbors and (left >= 0 or right < len(ordered)):
            left_distance = (
                abs(latent[ordered[left]] - latent[target]) if left >= 0 else np.inf
            )
            right_distance = (
                abs(latent[ordered[right]] - latent[target])
                if right < len(ordered)
                else np.inf
            )
            if left_distance <= right_distance:
                candidate = int(ordered[left])
                left -= 1
            else:
                candidate = int(ordered[right])
                right += 1
            if disjoint_from_target:
                assert pairs is not None
                if np.any(pairs[candidate, :, None] == pairs[target, None, :]):
                    continue
            selected.append(candidate)
        if len(selected) != neighbors:
            raise ValueError("not enough eligible neighbors for a target")
        result[row] = selected
    return result


def _shared_fraction(
    pairs: np.ndarray, targets: np.ndarray, selected: np.ndarray
) -> float:
    target_pairs = pairs[targets]
    neighbor_pairs = pairs[selected]
    shared = np.any(
        neighbor_pairs[:, :, :, None] == target_pairs[:, None, None, :], axis=(2, 3)
    )
    return float(shared.mean())


def _spatial_rows(
    *,
    dataset_name: str,
    device_id: str | float,
    selector: str,
    targets: np.ndarray,
    selected: np.ndarray,
    latent: np.ndarray,
    probability: np.ndarray,
    evaluation_responses: np.ndarray,
    pairs: np.ndarray,
    neighbor_counts: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_probability = probability[targets]
    for count in neighbor_counts:
        neighbors = selected[:, :count]
        expected = finite_mean(probability[neighbors], axis=1)
        snapshots = finite_mean(evaluation_responses[neighbors], axis=1)
        bias = expected - target_probability
        total_error = snapshots - target_probability[:, None]
        precision_error = snapshots - expected[:, None]
        radius = np.max(np.abs(latent[neighbors] - latent[targets, None]), axis=1)
        rows.append(
            {
                "dataset": dataset_name,
                "device_id": device_id,
                "selector": selector,
                "neighbors": count,
                "targets": len(targets),
                "mean_latent_radius": float(np.nanmean(radius)),
                "median_latent_radius": float(np.nanmedian(radius)),
                "spatial_bias": float(np.nanmean(bias)),
                "spatial_bias_mae": float(np.nanmean(np.abs(bias))),
                "spatial_bias_rmse": float(np.sqrt(np.nanmean(bias**2))),
                "snapshot_precision_mae": float(np.nanmean(np.abs(precision_error))),
                "snapshot_total_mae": float(np.nanmean(np.abs(total_error))),
                "snapshot_total_rmse": float(np.sqrt(np.nanmean(total_error**2))),
                "shared_oscillator_fraction": _shared_fraction(
                    pairs, targets, neighbors
                ),
            }
        )
    return rows


def _curve_rows(
    *,
    dataset_name: str,
    device_id: str | float,
    latent: np.ndarray,
    probability: np.ndarray,
    noise_std: np.ndarray,
    bins: int,
) -> list[dict[str, object]]:
    valid = np.flatnonzero(np.isfinite(latent) & np.isfinite(probability))
    ordered = valid[np.argsort(latent[valid], kind="stable")]
    rows: list[dict[str, object]] = []
    for bin_index, indices in enumerate(
        np.array_split(ordered, min(bins, len(ordered)))
    ):
        if not len(indices):
            continue
        p = probability[indices].astype(np.float64)
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        entropy = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
        rows.append(
            {
                "dataset": dataset_name,
                "device_id": device_id,
                "bin": bin_index,
                "pairs": len(indices),
                "latent_min": float(latent[indices[0]]),
                "latent_mean": float(np.mean(latent[indices])),
                "latent_max": float(latent[indices[-1]]),
                "response_probability": float(np.mean(p)),
                "reference_reliability": float(np.mean(np.maximum(p, 1 - p))),
                "pairwise_repeatability": float(np.mean(p**2 + (1 - p) ** 2)),
                "binary_entropy": float(np.mean(entropy)),
                "noise_std": float(finite_mean(noise_std[indices], axis=0)),
            }
        )
    return rows


def _graph_summary(pairs: np.ndarray) -> dict[str, int]:
    oscillators, inverse = np.unique(pairs, return_inverse=True)
    edges = inverse.reshape(-1, 2)
    parent = np.arange(len(oscillators), dtype=np.int64)

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in edges:
        first_root, second_root = root(int(first)), root(int(second))
        if first_root != second_root:
            parent[second_root] = first_root
    components = len({root(index) for index in range(len(parent))})
    return {
        "oscillators": len(oscillators),
        "graph_components": components,
        "self_pairs": int(np.sum(pairs[:, 0] == pairs[:, 1])),
        "duplicate_pairs": int(len(pairs) - len(np.unique(pairs, axis=0))),
    }


def analyze_ro_spatial(
    *,
    input_path: Path,
    output_dir: Path,
    config: ROSpatialConfig,
    groups: tuple[str, ...] | None = None,
    limit_devices: int | None = None,
) -> None:
    """Analyze every selected HDF5 group without loading the full file."""

    if tuple(sorted(set(config.neighbors))) != config.neighbors:
        raise ValueError("neighbors must be unique and increasing")
    if min(config.neighbors) < 1 or config.pair_block_size < 1:
        raise ValueError("neighbor counts and pair block size must be positive")
    if config.targets_per_device < 1 or config.curve_bins < 2:
        raise ValueError("targets and curve bins must be positive")
    if limit_devices is not None and limit_devices < 1:
        raise ValueError("limit_devices must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    device_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.seed)

    with h5py.File(input_path, "r") as handle:
        selected_groups = tuple(handle.keys()) if groups is None else groups
        unknown = set(selected_groups) - set(handle.keys())
        if unknown:
            raise ValueError(f"unknown HDF5 groups: {sorted(unknown)}")
        for dataset_name in selected_groups:
            group = handle[dataset_name]
            devices, pair_count, sample_count = validate_ro_group(group)
            if max(config.neighbors) >= pair_count:
                raise ValueError(f"{dataset_name} has too few pairs")
            enrollment, evaluation = split_samples(sample_count, config.split)
            pairs = np.asarray(group["frequency_pairs"])
            graph = _graph_summary(pairs)
            differences = group["frequency_differences"]
            processed_devices = min(devices, limit_devices or devices)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "devices": devices,
                    "processed_devices": processed_devices,
                    "pairs": pair_count,
                    "samples": sample_count,
                    "enrollment_samples": len(enrollment),
                    "evaluation_samples": len(evaluation),
                    "dtype": str(differences.dtype),
                    "chunks": str(differences.chunks),
                    "compression": str(differences.compression),
                    **graph,
                }
            )
            LOGGER.info(
                "%s: %d/%d devices, %d pairs, %d samples",
                dataset_name,
                processed_devices,
                devices,
                pair_count,
                sample_count,
            )
            device_ids = group["device_ids"]
            for device_index in range(processed_devices):
                device_id = decode_device_id(device_ids[device_index])
                LOGGER.info(
                    "%s: reading device %d/%d (%s)",
                    dataset_name,
                    device_index + 1,
                    processed_devices,
                    device_id,
                )
                bank, noise_std, responses = read_device_split(
                    differences,
                    device_index,
                    enrollment,
                    evaluation,
                    config.pair_block_size,
                )
                latent = bank.margin
                probability = finite_mean(responses, axis=1)
                valid = np.flatnonzero(np.isfinite(latent) & np.isfinite(probability))
                if len(valid) <= max(config.neighbors):
                    LOGGER.warning(
                        "%s/%s: insufficient finite pairs", dataset_name, device_id
                    )
                    continue
                target_count = min(config.targets_per_device, len(valid))
                targets = np.sort(rng.choice(valid, size=target_count, replace=False))
                max_neighbors = max(config.neighbors)
                selection_latent = latent.copy()
                selection_latent[~np.isfinite(probability)] = np.nan
                selectors = {
                    "latent_nearest": nearest_latent_indices(
                        selection_latent, targets, max_neighbors
                    ),
                    "latent_nearest_disjoint": nearest_latent_indices(
                        selection_latent,
                        targets,
                        max_neighbors,
                        pairs=pairs,
                        disjoint_from_target=True,
                    ),
                    "random": random_neighbor_indices(
                        valid, targets, max_neighbors, rng
                    ),
                }
                for selector, selected in selectors.items():
                    spatial_rows.extend(
                        _spatial_rows(
                            dataset_name=dataset_name,
                            device_id=device_id,
                            selector=selector,
                            targets=targets,
                            selected=selected,
                            latent=latent,
                            probability=probability,
                            evaluation_responses=responses,
                            pairs=pairs,
                            neighbor_counts=config.neighbors,
                        )
                    )
                curve_rows.extend(
                    _curve_rows(
                        dataset_name=dataset_name,
                        device_id=device_id,
                        latent=latent,
                        probability=probability,
                        noise_std=noise_std,
                        bins=config.curve_bins,
                    )
                )
                p = probability[valid]
                device_rows.append(
                    {
                        "dataset": dataset_name,
                        "device_id": device_id,
                        "valid_pairs": len(valid),
                        "response_balance": float(np.mean(p)),
                        "reference_reliability": float(np.mean(np.maximum(p, 1 - p))),
                        "pairwise_repeatability": float(np.mean(p**2 + (1 - p) ** 2)),
                        "latent_mean": float(np.mean(latent[valid])),
                        "latent_std": float(np.std(latent[valid], ddof=1)),
                        "noise_std_mean": float(
                            finite_mean(noise_std[valid], axis=0)
                        ),
                    }
                )

    pl.DataFrame(summary_rows).write_parquet(output_dir / "dataset_summary.parquet")
    pl.DataFrame(device_rows).write_parquet(output_dir / "device_metrics.parquet")
    pl.DataFrame(curve_rows).write_parquet(output_dir / "latent_curves.parquet")
    pl.DataFrame(spatial_rows).write_parquet(output_dir / "spatial_metrics.parquet")
    manifest = {
        "input": str(input_path.resolve()),
        "groups": list(groups) if groups is not None else None,
        "limit_devices": limit_devices,
        "config": asdict(config),
        "artifacts": [
            "dataset_summary.parquet",
            "device_metrics.parquet",
            "latent_curves.parquet",
            "spatial_metrics.parquet",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote RO artifacts to %s", output_dir)
