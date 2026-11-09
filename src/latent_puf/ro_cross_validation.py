"""Cross-validated one-shot spatial estimation for differential RO PUFs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from latent_puf.latent import BinaryLatentBank, random_neighbor_indices
from latent_puf.ro_data import (
    decode_device_id,
    finite_mean,
    read_device_values,
    split_samples,
    validate_ro_group,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ROCrossValidationConfig:
    """Configuration for fixed-bank or strict leave-one-out validation."""

    mode: str = "fixed"
    neighbors: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    targets_per_stratum: int = 100
    latent_edges: tuple[float, ...] = (
        0.0,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        float("inf"),
    )
    pair_block_size: int = 65_536
    curve_bins: int = 64
    max_folds: int | None = 10
    seed: int = 123_456


def temporal_leave_one_out(responses: np.ndarray, fold: int) -> np.ndarray:
    """Estimate each pair's response probability while excluding one sample."""

    if responses.ndim != 2 or not 0 <= fold < responses.shape[1]:
        raise ValueError("responses must be (B, S) and fold must select a sample")
    keep = np.arange(responses.shape[1]) != fold
    return finite_mean(responses[:, keep], axis=1)


def one_shot_probability(
    held_out_responses: np.ndarray, neighbor_indices: np.ndarray, neighbors: int
) -> np.ndarray:
    """Average one held-out response over each target's spatial neighbors."""

    if held_out_responses.ndim != 1 or neighbor_indices.ndim != 2:
        raise ValueError("invalid held-out responses or neighbor indices")
    if not 1 <= neighbors <= neighbor_indices.shape[1]:
        raise ValueError("invalid neighbor count")
    return finite_mean(
        held_out_responses[neighbor_indices[:, :neighbors]],
        axis=1,
    )


def _response_matrix(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values > 0, np.nan).astype(np.float32)


def _folds(candidates: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None or maximum >= len(candidates):
        return candidates
    positions = np.linspace(0, len(candidates) - 1, maximum, dtype=np.int64)
    return candidates[positions]


def _stratified_targets(
    latent: np.ndarray,
    latent_std: float,
    edges: tuple[float, ...],
    per_stratum: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.abs(latent) / latent_std
    target_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for index, (lower, upper) in enumerate(pairwise(edges)):
        available = np.flatnonzero(
            np.isfinite(normalized) & (normalized >= lower) & (normalized < upper)
        )
        if not len(available):
            continue
        count = min(per_stratum, len(available))
        selected = rng.choice(available, count, replace=False)
        target_parts.append(selected)
        label_parts.append(np.full(count, index, dtype=np.int16))
    if not target_parts:
        raise ValueError("no finite targets fall inside the latent strata")
    targets = np.concatenate(target_parts)
    labels = np.concatenate(label_parts)
    order = np.argsort(targets, kind="stable")
    return targets[order], labels[order]


def _nearest_indices(
    latent: np.ndarray,
    targets: np.ndarray,
    neighbors: int,
    *,
    pairs: np.ndarray | None = None,
    disjoint: bool = False,
) -> np.ndarray:
    """Find sorted scalar neighbors using bounded candidate windows."""

    valid = np.flatnonzero(np.isfinite(latent))
    if neighbors >= len(valid):
        raise ValueError("neighbor count exceeds finite latent population")
    ordered = valid[np.argsort(latent[valid], kind="stable")]
    ranks = np.full(len(latent), -1, dtype=np.int64)
    ranks[ordered] = np.arange(len(ordered))
    result = np.empty((len(targets), neighbors), dtype=np.int64)
    buffer = max(64, neighbors // 16) if disjoint else 0

    for row, target in enumerate(targets):
        rank = int(ranks[target])
        width = neighbors + buffer
        lower = max(0, rank - width)
        upper = min(len(ordered), rank + width + 1)
        if upper - lower < 2 * width + 1:
            if lower == 0:
                upper = min(len(ordered), 2 * width + 1)
            elif upper == len(ordered):
                lower = max(0, len(ordered) - (2 * width + 1))
        candidates = ordered[lower:upper]
        candidates = candidates[candidates != target]
        if disjoint:
            if pairs is None:
                raise ValueError("pairs are required for disjoint neighbors")
            shared = np.any(
                pairs[candidates, :, None] == pairs[target, None, :], axis=(1, 2)
            )
            candidates = candidates[~shared]
        if len(candidates) < neighbors:
            candidates = valid[valid != target]
            if disjoint:
                assert pairs is not None
                shared = np.any(
                    pairs[candidates, :, None] == pairs[target, None, :], axis=(1, 2)
                )
                candidates = candidates[~shared]
        distances = np.abs(latent[candidates] - latent[target])
        chosen = np.argpartition(distances, neighbors - 1)[:neighbors]
        chosen = chosen[np.argsort(distances[chosen], kind="stable")]
        result[row] = candidates[chosen]
    return result


def _shared_matrix(
    pairs: np.ndarray, targets: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    target_pairs = pairs[targets]
    neighbor_pairs = pairs[selected]
    return np.any(
        neighbor_pairs[:, :, :, None] == target_pairs[:, None, None, :], axis=(2, 3)
    )


def _latent_label(index: int, edges: tuple[float, ...]) -> str:
    upper = edges[index + 1]
    upper_text = "inf" if np.isinf(upper) else f"{upper:g}"
    return f"[{edges[index]:g},{upper_text})"


def _ber_masks(probability: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    ber = np.minimum(probability, 1 - probability)
    return (
        ("0", ber == 0),
        ("(0,0.1]", (ber > 0) & (ber <= 0.1)),
        ("(0.1,0.25]", (ber > 0.1) & (ber <= 0.25)),
        ("(0.25,0.5]", (ber > 0.25) & (ber <= 0.5)),
    )


def _metric_row(
    *,
    dataset: str,
    device_id: str,
    fold: int,
    selector: str,
    neighbors: int,
    stratum_type: str,
    stratum: str,
    mask: np.ndarray,
    spatial: np.ndarray,
    temporal: np.ndarray,
    held_out: np.ndarray,
    normalized_radius: np.ndarray,
    shared_fraction: np.ndarray,
) -> dict[str, object] | None:
    valid = mask & np.isfinite(spatial) & np.isfinite(temporal) & np.isfinite(held_out)
    if not valid.any():
        return None
    reconstruction = spatial[valid] - temporal[valid]
    spatial_prediction = spatial[valid] - held_out[valid]
    temporal_prediction = temporal[valid] - held_out[valid]
    spatial_brier = spatial_prediction**2
    temporal_brier = temporal_prediction**2
    return {
        "dataset": dataset,
        "device_id": device_id,
        "fold": fold,
        "selector": selector,
        "neighbors": neighbors,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "targets": int(valid.sum()),
        "reconstruction_bias": float(np.mean(reconstruction)),
        "reconstruction_mae": float(np.mean(np.abs(reconstruction))),
        "reconstruction_rmse": float(np.sqrt(np.mean(reconstruction**2))),
        "spatial_brier": float(np.mean(spatial_brier)),
        "temporal_brier": float(np.mean(temporal_brier)),
        "excess_brier": float(np.mean(spatial_brier - temporal_brier)),
        "spatial_calibration_bias": float(np.mean(spatial_prediction)),
        "temporal_calibration_bias": float(np.mean(temporal_prediction)),
        "normalized_radius": float(np.mean(normalized_radius[valid])),
        "shared_oscillator_fraction": float(np.mean(shared_fraction[valid])),
    }


def _curve_points(
    latent: np.ndarray, probability: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.flatnonzero(np.isfinite(latent) & np.isfinite(probability))
    ordered = valid[np.argsort(latent[valid], kind="stable")]
    groups = [
        group
        for group in np.array_split(ordered, min(bins, len(ordered)))
        if len(group)
    ]
    x = np.array([np.mean(latent[group]) for group in groups])
    p = np.array([np.mean(probability[group]) for group in groups])
    return x, np.maximum.accumulate(p)


def _crossing(x: np.ndarray, probability: np.ndarray, level: float) -> float | None:
    index = int(np.searchsorted(probability, level))
    if index == 0 or index >= len(x):
        return None
    lower, upper = probability[index - 1], probability[index]
    if upper == lower:
        return float((x[index - 1] + x[index]) / 2)
    fraction = (level - lower) / (upper - lower)
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def _curve_features(
    latent: np.ndarray, probability: np.ndarray, bins: int
) -> tuple[dict[str, float | None], np.ndarray, np.ndarray]:
    x, p = _curve_points(latent, probability, bins)
    x10, x50, x90 = (_crossing(x, p, level) for level in (0.1, 0.5, 0.9))
    width = None if x10 is None or x90 is None else x90 - x10
    return {"x10": x10, "x50": x50, "x90": x90, "width": width}, x, p


def _curve_metric_row(
    *,
    dataset: str,
    device_id: str,
    fold: int,
    selector: str,
    neighbors: int,
    latent: np.ndarray,
    spatial: np.ndarray,
    temporal: np.ndarray,
    bins: int,
) -> dict[str, object]:
    spatial_features, spatial_x, spatial_p = _curve_features(latent, spatial, bins)
    temporal_features, temporal_x, temporal_p = _curve_features(latent, temporal, bins)
    common_x = np.unique(np.concatenate((spatial_x, temporal_x)))
    spatial_curve = np.interp(common_x, spatial_x, spatial_p)
    temporal_curve = np.interp(common_x, temporal_x, temporal_p)
    result: dict[str, object] = {
        "dataset": dataset,
        "device_id": device_id,
        "fold": fold,
        "selector": selector,
        "neighbors": neighbors,
        "curve_mae": float(np.mean(np.abs(spatial_curve - temporal_curve))),
        "curve_max_error": float(np.max(np.abs(spatial_curve - temporal_curve))),
    }
    for name in ("x10", "x50", "x90", "width"):
        spatial_value = spatial_features[name]
        temporal_value = temporal_features[name]
        result[f"spatial_{name}"] = spatial_value
        result[f"temporal_{name}"] = temporal_value
        result[f"{name}_error"] = (
            None
            if spatial_value is None or temporal_value is None
            else spatial_value - temporal_value
        )
    return result


def _write_checkpoint(
    output_dir: Path,
    metric_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    pl.DataFrame(metric_rows, infer_schema_length=None).write_parquet(
        output_dir / "cv_metrics.parquet"
    )
    pl.DataFrame(curve_rows, infer_schema_length=None).write_parquet(
        output_dir / "cv_curve_metrics.parquet"
    )
    pl.DataFrame(summary_rows, infer_schema_length=None).write_parquet(
        output_dir / "cv_dataset_summary.parquet"
    )


def run_ro_cross_validation(
    *,
    input_path: Path,
    output_dir: Path,
    config: ROCrossValidationConfig,
    groups: tuple[str, ...] | None = None,
    limit_devices: int | None = None,
) -> None:
    """Run fixed-bank or strict leave-one-out RO validation."""

    if config.mode not in {"fixed", "loo"}:
        raise ValueError("mode must be 'fixed' or 'loo'")
    if tuple(sorted(set(config.neighbors))) != config.neighbors:
        raise ValueError("neighbors must be unique and increasing")
    if config.targets_per_stratum < 1 or config.curve_bins < 4:
        raise ValueError("targets per stratum and curve bins are too small")
    if any(right <= left for left, right in pairwise(config.latent_edges)):
        raise ValueError("latent edges must be strictly increasing")

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
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
            processed_devices = min(devices, limit_devices or devices)
            pairs = np.asarray(group["frequency_pairs"])
            device_ids = group["device_ids"]
            differences = group["frequency_differences"]
            enrollment, evaluation = split_samples(sample_count, "alternating")
            candidate_folds = (
                evaluation if config.mode == "fixed" else np.arange(sample_count)
            )
            folds = _folds(candidate_folds, config.max_folds)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "devices": devices,
                    "processed_devices": processed_devices,
                    "pairs": pair_count,
                    "samples": sample_count,
                    "folds": len(folds),
                    "mode": config.mode,
                }
            )
            LOGGER.info(
                "%s: %d/%d devices, %d folds, mode=%s",
                dataset_name,
                processed_devices,
                devices,
                len(folds),
                config.mode,
            )
            for device_index in range(processed_devices):
                device_id = decode_device_id(device_ids[device_index])
                LOGGER.info(
                    "%s: device %d/%d (%s)",
                    dataset_name,
                    device_index + 1,
                    processed_devices,
                    device_id,
                )
                values = read_device_values(
                    differences, device_index, config.pair_block_size
                )
                responses = _response_matrix(values)
                fixed_bank = BinaryLatentBank.from_threshold(
                    finite_mean(values[:, enrollment], axis=1)
                )
                fixed_latent = fixed_bank.margin
                fixed_std = float(np.nanstd(fixed_latent, ddof=1))
                fixed_targets: np.ndarray | None = None
                fixed_labels: np.ndarray | None = None
                fixed_neighbors: dict[str, np.ndarray] = {}
                if config.mode == "fixed":
                    fixed_targets, fixed_labels = _stratified_targets(
                        fixed_latent,
                        fixed_std,
                        config.latent_edges,
                        config.targets_per_stratum,
                        rng,
                    )
                    fixed_neighbors = {
                        "latent_nearest": _nearest_indices(
                            fixed_latent, fixed_targets, max(config.neighbors)
                        ),
                        "latent_nearest_disjoint": _nearest_indices(
                            fixed_latent,
                            fixed_targets,
                            max(config.neighbors),
                            pairs=pairs,
                            disjoint=True,
                        ),
                        "random": random_neighbor_indices(
                            np.flatnonzero(np.isfinite(fixed_latent)),
                            fixed_targets,
                            max(config.neighbors),
                            rng,
                        ),
                    }

                for fold_value in folds:
                    fold = int(fold_value)
                    if config.mode == "loo":
                        keep = np.arange(sample_count) != fold
                        bank = BinaryLatentBank.from_threshold(
                            finite_mean(values[:, keep], axis=1)
                        )
                        latent = bank.margin
                        latent_std = float(np.nanstd(latent, ddof=1))
                        targets, latent_labels = _stratified_targets(
                            latent,
                            latent_std,
                            config.latent_edges,
                            config.targets_per_stratum,
                            rng,
                        )
                        selectors = {
                            "latent_nearest": _nearest_indices(
                                latent, targets, max(config.neighbors)
                            ),
                            "latent_nearest_disjoint": _nearest_indices(
                                latent,
                                targets,
                                max(config.neighbors),
                                pairs=pairs,
                                disjoint=True,
                            ),
                            "random": random_neighbor_indices(
                                np.flatnonzero(np.isfinite(latent)),
                                targets,
                                max(config.neighbors),
                                rng,
                            ),
                        }
                    else:
                        latent, latent_std = fixed_latent, fixed_std
                        assert fixed_targets is not None and fixed_labels is not None
                        targets, latent_labels = fixed_targets, fixed_labels
                        selectors = fixed_neighbors

                    temporal = temporal_leave_one_out(responses, fold)[targets]
                    held_out = responses[targets, fold].astype(np.float64)
                    target_latent = latent[targets]
                    for selector, selected in selectors.items():
                        shared = _shared_matrix(pairs, targets, selected)
                        distances = np.abs(latent[selected] - target_latent[:, None])
                        cumulative_radius = np.maximum.accumulate(distances, axis=1)
                        cumulative_shared = np.cumsum(shared, axis=1)
                        for neighbors in config.neighbors:
                            spatial = one_shot_probability(
                                responses[:, fold], selected, neighbors
                            )
                            radius = cumulative_radius[:, neighbors - 1] / latent_std
                            shared_fraction = (
                                cumulative_shared[:, neighbors - 1] / neighbors
                            )
                            strata: list[tuple[str, str, np.ndarray]] = [
                                ("all", "all", np.ones(len(targets), dtype=bool))
                            ]
                            for index in np.unique(latent_labels):
                                strata.append(
                                    (
                                        "latent",
                                        _latent_label(int(index), config.latent_edges),
                                        latent_labels == index,
                                    )
                                )
                            strata.extend(
                                ("ber", label, mask)
                                for label, mask in _ber_masks(temporal)
                            )
                            for stratum_type, stratum, mask in strata:
                                row = _metric_row(
                                    dataset=dataset_name,
                                    device_id=device_id,
                                    fold=fold,
                                    selector=selector,
                                    neighbors=neighbors,
                                    stratum_type=stratum_type,
                                    stratum=stratum,
                                    mask=mask,
                                    spatial=spatial,
                                    temporal=temporal,
                                    held_out=held_out,
                                    normalized_radius=radius,
                                    shared_fraction=shared_fraction,
                                )
                                if row is not None:
                                    metric_rows.append(row)
                            curve_rows.append(
                                _curve_metric_row(
                                    dataset=dataset_name,
                                    device_id=device_id,
                                    fold=fold,
                                    selector=selector,
                                    neighbors=neighbors,
                                    latent=target_latent / latent_std,
                                    spatial=spatial,
                                    temporal=temporal,
                                    bins=config.curve_bins,
                                )
                            )
            _write_checkpoint(output_dir, metric_rows, curve_rows, summary_rows)
            LOGGER.info("Checkpointed completed group %s", dataset_name)

    manifest = {
        "input": str(input_path.resolve()),
        "groups": list(groups) if groups is not None else None,
        "limit_devices": limit_devices,
        "config": asdict(config),
        "artifacts": [
            "cv_metrics.parquet",
            "cv_curve_metrics.parquet",
            "cv_dataset_summary.parquet",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote cross-validation artifacts to %s", output_dir)
