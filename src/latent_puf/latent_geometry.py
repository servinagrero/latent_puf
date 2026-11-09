"""Exploratory latent-geometry analysis for modeled APUF challenges."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from hashlib import md5
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from latent_puf.latent import BinaryLatentBank
from latent_puf.margin_bank import read_response_matrix
from latent_puf.room_temperature import (
    InstanceKey,
    atf,
    open_read_only,
    read_instances,
)

LOGGER = logging.getLogger(__name__)
INSTANCE_COLUMNS = (
    "device_id",
    "slice",
    "stages",
    "placement_variant",
    "placement_gap",
    "placement_offset",
)


@dataclass(frozen=True)
class GeometryConfig:
    neighbors: tuple[int, ...] = (5, 10, 25, 50)
    pair_samples: int = 250_000
    umap_neighbors: int = 30
    umap_min_dist: float = 0.1
    target_weight: float = 0.7
    seed: int = 123456


def sample_pairs(
    count: int, pair_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample deterministic ordered pairs with distinct endpoints."""

    if count < 2 or pair_samples < 1:
        raise ValueError("At least two points and one pair sample are required")
    first = rng.integers(0, count, size=pair_samples, dtype=np.int64)
    second = rng.integers(0, count - 1, size=pair_samples, dtype=np.int64)
    second += second >= first
    return np.column_stack((first, second))


def hamming_distances(values: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Return integer Hamming distances for selected row pairs."""

    if values.ndim != 2 or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("values and pairs must be two-dimensional")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(values)):
        raise ValueError("pair index is outside the value array")
    return np.count_nonzero(values[pairs[:, 0]] != values[pairs[:, 1]], axis=1)


def aggregate_pair_distances(
    *,
    distances: np.ndarray,
    margin_differences: np.ndarray,
    probability_differences: np.ndarray,
    label_agreement: np.ndarray,
) -> list[dict[str, object]]:
    """Aggregate latent and response differences by integer distance."""

    arrays = (
        distances,
        margin_differences,
        probability_differences,
        label_agreement,
    )
    if any(array.ndim != 1 for array in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError("pair diagnostic arrays must be equal-length vectors")
    rows: list[dict[str, object]] = []
    for distance in np.unique(distances):
        selected = distances == distance
        rows.append(
            {
                "distance": int(distance),
                "pairs": int(selected.sum()),
                "mean_abs_margin_difference": float(
                    margin_differences[selected].mean()
                ),
                "median_abs_margin_difference": float(
                    np.median(margin_differences[selected])
                ),
                "mean_abs_probability_difference": float(
                    probability_differences[selected].mean()
                ),
                "response_label_agreement": float(label_agreement[selected].mean()),
            }
        )
    return rows


def neighborhood_metric_rows(
    *,
    neighbor_indices: np.ndarray,
    margins: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    neighbor_counts: tuple[int, ...],
) -> list[dict[str, object]]:
    """Measure how well a neighbor ordering preserves physical observables."""

    count = len(margins)
    if not (len(probabilities) == len(labels) == count):
        raise ValueError("observable lengths differ")
    if neighbor_indices.ndim != 2 or len(neighbor_indices) != count:
        raise ValueError("neighbor index shape differs from observables")
    if not neighbor_counts or min(neighbor_counts) < 1:
        raise ValueError("neighbor counts must be positive")
    if max(neighbor_counts) > neighbor_indices.shape[1]:
        raise ValueError("requested neighborhood exceeds available indices")

    rows: list[dict[str, object]] = []
    for neighbors in neighbor_counts:
        selected = neighbor_indices[:, :neighbors]
        rows.append(
            {
                "neighbors": neighbors,
                "mean_abs_neighbor_margin_difference": float(
                    np.mean(np.abs(margins[selected] - margins[:, None]))
                ),
                "mean_abs_neighbor_probability_difference": float(
                    np.mean(np.abs(probabilities[selected] - probabilities[:, None]))
                ),
                "response_label_agreement": float(
                    np.mean(labels[selected] == labels[:, None])
                ),
            }
        )
    return rows


def nearest_indices(
    values: np.ndarray, neighbors: int, metric: str
) -> np.ndarray:
    """Return nearest rows while excluding each query row itself."""

    if values.ndim != 2 or not 1 <= neighbors < len(values):
        raise ValueError("invalid values or neighbor count")
    model = NearestNeighbors(
        n_neighbors=neighbors + 1, metric=metric, algorithm="brute", n_jobs=-1
    ).fit(values)
    indices = model.kneighbors(values, return_distance=False)
    return indices[:, 1:]


def _model_rows(models_path: Path) -> dict[InstanceKey, dict[str, object]]:
    models = pl.read_parquet(models_path)
    required = {*INSTANCE_COLUMNS, "intercept", "coefficients"}
    missing = required - set(models.columns)
    if missing:
        raise ValueError(f"Model artifact is missing columns: {sorted(missing)}")
    result: dict[InstanceKey, dict[str, object]] = {}
    for row in models.iter_rows(named=True):
        key = InstanceKey(*(row[column] for column in INSTANCE_COLUMNS))
        result[key] = row
    return result


def _umap(
    values: np.ndarray,
    config: GeometryConfig,
    *,
    target: np.ndarray | None = None,
    target_metric: str = "categorical",
) -> np.ndarray:
    model = UMAP(
        n_components=2,
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        metric="hamming",
        target_metric=target_metric,
        target_weight=config.target_weight,
        random_state=config.seed,
        transform_seed=config.seed,
        n_jobs=1,
    )
    return model.fit_transform(values, y=target)


def _file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_latent_geometry_analysis(
    *,
    modeling_db: Path,
    modeling_challenges_path: Path,
    models_path: Path,
    output_dir: Path,
    config: GeometryConfig,
    limit_instances: int | None = None,
) -> None:
    """Create unsupervised and supervised APUF challenge embeddings."""

    if tuple(sorted(set(config.neighbors))) != config.neighbors:
        raise ValueError("neighbors must be unique and increasing")
    if config.umap_neighbors < 2 or not 0 <= config.umap_min_dist <= 1:
        raise ValueError("invalid UMAP neighborhood or minimum distance")
    if not 0 <= config.target_weight <= 1:
        raise ValueError("target_weight must be in [0, 1]")

    challenges = np.load(modeling_challenges_path)
    if challenges.ndim != 2 or not np.all((challenges == 0) | (challenges == 1)):
        raise ValueError("modeling challenges must be a binary matrix")
    count, stages = challenges.shape
    if max(config.neighbors) >= count:
        raise ValueError("a neighborhood is not smaller than the dataset")
    raw = challenges.astype(bool)
    transformed = atf(challenges)
    transformed_bits = transformed < 0
    models = _model_rows(models_path)

    LOGGER.info("Computing instance-independent UMAP embeddings")
    raw_embedding = _umap(raw, config)
    atf_embedding = _umap(transformed_bits, config)
    challenge_rows = [
        {
            "challenge_id": challenge_id,
            "raw_umap_x": float(raw_embedding[challenge_id, 0]),
            "raw_umap_y": float(raw_embedding[challenge_id, 1]),
            "atf_umap_x": float(atf_embedding[challenge_id, 0]),
            "atf_umap_y": float(atf_embedding[challenge_id, 1]),
        }
        for challenge_id in range(count)
    ]

    max_neighbors = max(config.neighbors)
    LOGGER.info("Computing shared nearest-neighbor orderings")
    shared_neighbors = {
        "raw_hamming": nearest_indices(raw, max_neighbors, "hamming"),
        "atf_hamming": nearest_indices(transformed_bits, max_neighbors, "hamming"),
        "raw_umap": nearest_indices(raw_embedding, max_neighbors, "euclidean"),
        "atf_umap": nearest_indices(atf_embedding, max_neighbors, "euclidean"),
    }
    rng = np.random.default_rng(config.seed)
    pairs = sample_pairs(count, config.pair_samples, rng)
    pair_distances = {
        "raw_hamming": hamming_distances(raw, pairs),
        "atf_hamming": hamming_distances(transformed_bits, pairs),
    }

    connection = open_read_only(modeling_db)
    try:
        acquisitions = read_instances(connection, stages)
        if set(acquisitions) != set(models):
            raise ValueError("Model artifact and database contain different instances")
        instances = sorted(acquisitions)
        if limit_instances is not None:
            instances = instances[:limit_instances]
        LOGGER.info("Analyzing %d APUF instances", len(instances))

        embedding_rows: list[dict[str, object]] = []
        neighborhood_rows: list[dict[str, object]] = []
        pair_rows: list[dict[str, object]] = []
        for instance_number, key in enumerate(instances, start=1):
            LOGGER.info(
                "[instance %d/%d] %s %s",
                instance_number,
                len(instances),
                key.device_id,
                key.slice,
            )
            row = models[key]
            coefficients = np.asarray(
                cast(list[float], row["coefficients"]), dtype=np.float64
            )
            bank = BinaryLatentBank.from_margin(
                transformed @ coefficients + cast(float, row["intercept"])
            )
            margins = bank.margin
            responses = read_response_matrix(connection, acquisitions[key], count)
            probabilities = responses.mean(axis=0)
            labels = (probabilities >= 0.5).astype(np.int8)
            standardized_margin = (margins - margins.mean()) / margins.std()

            LOGGER.info("Computing response-supervised UMAP")
            response_embedding = _umap(
                raw, config, target=labels, target_metric="categorical"
            )
            LOGGER.info("Computing margin-supervised UMAP")
            margin_embedding = _umap(
                raw,
                config,
                target=standardized_margin,
                target_metric="euclidean",
            )
            identity = asdict(key)
            embedding_rows.extend(
                {
                    **identity,
                    "challenge_id": challenge_id,
                    "response_probability": float(probabilities[challenge_id]),
                    "majority_response": int(labels[challenge_id]),
                    "modeled_margin": float(margins[challenge_id]),
                    "response_supervised_x": float(
                        response_embedding[challenge_id, 0]
                    ),
                    "response_supervised_y": float(
                        response_embedding[challenge_id, 1]
                    ),
                    "margin_supervised_x": float(margin_embedding[challenge_id, 0]),
                    "margin_supervised_y": float(margin_embedding[challenge_id, 1]),
                }
                for challenge_id in range(count)
            )

            instance_neighbors = {
                **shared_neighbors,
                "response_supervised_umap": nearest_indices(
                    response_embedding, max_neighbors, "euclidean"
                ),
                "margin_supervised_umap": nearest_indices(
                    margin_embedding, max_neighbors, "euclidean"
                ),
                "modeled_margin_1d": nearest_indices(
                    margins[:, None], max_neighbors, "euclidean"
                ),
            }
            for representation, indices in instance_neighbors.items():
                neighborhood_rows.extend(
                    {
                        **identity,
                        "representation": representation,
                        **metric_row,
                    }
                    for metric_row in neighborhood_metric_rows(
                        neighbor_indices=indices,
                        margins=margins,
                        probabilities=probabilities,
                        labels=labels,
                        neighbor_counts=config.neighbors,
                    )
                )

            first, second = pairs[:, 0], pairs[:, 1]
            margin_difference = np.abs(margins[first] - margins[second])
            probability_difference = np.abs(
                probabilities[first] - probabilities[second]
            )
            agreement = labels[first] == labels[second]
            for representation, distances in pair_distances.items():
                pair_rows.extend(
                    {
                        **identity,
                        "representation": representation,
                        **distance_row,
                    }
                    for distance_row in aggregate_pair_distances(
                        distances=distances,
                        margin_differences=margin_difference,
                        probability_differences=probability_difference,
                        label_agreement=agreement,
                    )
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(challenge_rows).write_parquet(
            output_dir / "challenge_embeddings.parquet"
        )
        pl.DataFrame(embedding_rows).write_parquet(
            output_dir / "instance_embeddings.parquet"
        )
        pl.DataFrame(neighborhood_rows).write_parquet(
            output_dir / "neighborhood_metrics.parquet"
        )
        pl.DataFrame(pair_rows).write_parquet(
            output_dir / "pair_distance_bins.parquet"
        )
        manifest = {
            "config": asdict(config),
            "instances": len(instances),
            "challenges": count,
            "inputs": {
                "modeling_db": str(modeling_db.expanduser().resolve()),
                "modeling_challenges": str(modeling_challenges_path.resolve()),
                "modeling_challenges_md5": _file_md5(modeling_challenges_path),
                "models": str(models_path.resolve()),
                "models_md5": _file_md5(models_path),
            },
            "notes": {
                "raw_umap": "Unsupervised raw challenge bits with Hamming metric",
                "atf_umap": "Unsupervised suffix-parity bits with Hamming metric",
                "response_supervised": "Raw challenge bits supervised by majority response",
                "margin_supervised": "Raw challenge bits supervised by modeled margin",
                "modeled_margin": "Logistic decision function, not physical time delay",
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote latent-geometry artifacts to %s", output_dir)
    finally:
        connection.close()
