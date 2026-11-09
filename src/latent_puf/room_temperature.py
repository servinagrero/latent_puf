"""Reusable room-temperature spatial/temporal APUF analysis."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import md5
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl
from pypuf.simulation import ArbiterPUF
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import train_test_split

from latent_puf.latent import BinaryLatentBank

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class InstanceKey:
    device_id: str
    slice: str
    stages: int
    placement_variant: str
    placement_gap: int
    placement_offset: int


@dataclass(frozen=True)
class AnalysisConfig:
    stages: int = 64
    dither_bits: int = 16
    k_values: tuple[int, ...] = tuple(range(1, 17))
    test_fraction: float = 0.2
    logistic_c: float = 1.0
    seed: int = 123456
    margin_bin_min: float = -12.0
    margin_bin_max: float = 12.0
    margin_bins: int = 96


def open_read_only(path: Path) -> sqlite3.Connection:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def read_instances(
    connection: sqlite3.Connection, stages: int
) -> dict[InstanceKey, int]:
    rows = connection.execute(
        """
        SELECT s.device_id, s.slice, s.stages, s.placement_variant,
               s.placement_gap, s.placement_offset, a.acquisition_id
        FROM studies AS s JOIN acquisitions AS a USING (study_id)
        WHERE s.stages = ?
        ORDER BY s.device_id, s.slice, a.acquisition_id
        """,
        (stages,),
    ).fetchall()
    if not rows:
        raise ValueError(f"No {stages}-stage acquisitions found")
    instances: dict[InstanceKey, int] = {}
    for *fields, acquisition_id in rows:
        key = InstanceKey(*fields)
        if key in instances:
            raise ValueError(f"Multiple acquisitions found for {key}")
        instances[key] = acquisition_id
    return instances


def read_response_probabilities(
    connection: sqlite3.Connection,
    acquisition_id: int,
    expected_challenges: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = connection.execute(
        """
        SELECT challenge_id, COUNT(*), AVG(response)
        FROM crps WHERE acquisition_id = ?
        GROUP BY challenge_id ORDER BY challenge_id
        """,
        (acquisition_id,),
    ).fetchall()
    if len(rows) != expected_challenges:
        raise ValueError(
            f"Acquisition {acquisition_id}: expected {expected_challenges} challenges, "
            f"found {len(rows)}"
        )
    ids = np.fromiter((row[0] for row in rows), dtype=np.int64)
    if not np.array_equal(ids, np.arange(expected_challenges)):
        raise ValueError(
            f"Acquisition {acquisition_id}: challenge IDs are not contiguous"
        )
    counts = np.fromiter((row[1] for row in rows), dtype=np.int32)
    probabilities = np.fromiter((row[2] for row in rows), dtype=np.float64)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"Acquisition {acquisition_id}: non-binary responses found")
    return counts, probabilities


def read_response_range(
    connection: sqlite3.Connection,
    acquisition_id: int,
    first_challenge: int,
    challenge_count: int,
) -> np.ndarray:
    stop = first_challenge + challenge_count
    rows = connection.execute(
        """
        SELECT sample_id, challenge_id, response FROM crps
        WHERE acquisition_id = ?
          AND sample_id = 0
          AND challenge_id >= ?
          AND challenge_id < ?
        ORDER BY challenge_id
        """,
        (acquisition_id, first_challenge, stop),
    ).fetchall()
    if len(rows) != challenge_count:
        raise ValueError(
            f"Acquisition {acquisition_id}: expected {challenge_count} rows in "
            f"[{first_challenge}, {stop}), found {len(rows)}"
        )
    sample_ids = np.fromiter((row[0] for row in rows), dtype=np.int64)
    challenge_ids = np.fromiter((row[1] for row in rows), dtype=np.int64)
    responses = np.fromiter((row[2] for row in rows), dtype=np.int8)
    if np.any(sample_ids != sample_ids[0]):
        raise ValueError(
            f"Acquisition {acquisition_id}: block contains multiple samples"
        )
    if not np.array_equal(challenge_ids, np.arange(first_challenge, stop)):
        raise ValueError(f"Acquisition {acquisition_id}: block IDs are not contiguous")
    if np.any((responses != 0) & (responses != 1)):
        raise ValueError(f"Acquisition {acquisition_id}: responses are not binary")
    return responses


def atf(challenges: np.ndarray) -> np.ndarray:
    if challenges.ndim != 2 or not np.all((challenges == 0) | (challenges == 1)):
        raise ValueError("Challenges must be a two-dimensional binary array")
    polar = np.where(challenges == 0, 1, -1).astype(np.int8)
    return ArbiterPUF.transform_atf(polar, k=1).reshape(len(challenges), -1)


def fit_instance_model(
    features: np.ndarray,
    probabilities: np.ndarray,
    config: AnalysisConfig,
) -> tuple[LogisticRegression, dict[str, object]]:
    labels = (probabilities >= 0.5).astype(np.int8)
    indices = np.arange(len(labels))
    train_ids, test_ids = train_test_split(
        indices,
        test_size=config.test_fraction,
        random_state=config.seed,
        stratify=labels,
    )
    validation_model = LogisticRegression(
        C=config.logistic_c,
        max_iter=5_000,
        solver="lbfgs",
        random_state=config.seed,
    ).fit(features[train_ids], labels[train_ids])
    predicted = validation_model.predict(features[test_ids])
    predicted_probability = validation_model.predict_proba(features[test_ids])[:, 1]
    metrics: dict[str, object] = {
        "n_modeling": len(labels),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "positive_fraction": float(labels.mean()),
        "test_accuracy": float(accuracy_score(labels[test_ids], predicted)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(labels[test_ids], predicted)
        ),
        "test_log_loss": float(log_loss(labels[test_ids], predicted_probability)),
        "test_brier": float(brier_score_loss(labels[test_ids], predicted_probability)),
    }
    model = LogisticRegression(
        C=config.logistic_c,
        max_iter=5_000,
        solver="lbfgs",
        random_state=config.seed,
    ).fit(features, labels)
    metrics["intercept"] = float(model.intercept_[0])
    metrics["coefficients"] = model.coef_[0].tolist()
    return model, metrics


def validate_gray_nesting(
    first_cluster: np.ndarray,
    dither_bits: int,
    k_values: tuple[int, ...],
) -> None:
    expected_rows = 1 << dither_bits
    if len(first_cluster) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in the first Gray cluster")
    base = first_cluster[0]
    for k in k_values:
        if not 1 <= k <= dither_bits:
            raise ValueError(f"k={k} is outside [1, {dither_bits}]")
        block = first_cluster[: 1 << k]
        if not np.all(block[:, k:] == base[k:]):
            raise ValueError(f"The leading k={k} block changes bits outside its prefix")
        if len(np.unique(block[:, :k], axis=0)) != 1 << k:
            raise ValueError(f"The leading k={k} block is not a complete prefix cube")


def _ordered_selector_metrics(
    name: str,
    order: np.ndarray,
    responses: np.ndarray,
    margins: np.ndarray,
    base_margin: float,
    temporal_probability: float,
    k_values: tuple[int, ...],
) -> list[dict[str, object]]:
    selected_responses = responses[order].astype(np.int64)
    selected_margins = margins[order]
    differences = selected_margins - base_margin
    response_sum = np.cumsum(selected_responses)
    difference_abs_sum = np.cumsum(np.abs(differences))
    difference_square_sum = np.cumsum(differences**2)
    margin_sum = np.cumsum(selected_margins)
    margin_square_sum = np.cumsum(selected_margins**2)
    margin_minimum = np.minimum.accumulate(differences)
    margin_maximum = np.maximum.accumulate(differences)
    margin_radius = np.maximum.accumulate(np.abs(differences))
    below_count = np.cumsum(differences < 0)
    above_count = np.cumsum(differences > 0)
    transition_sum = np.r_[
        0, np.cumsum(selected_responses[1:] != selected_responses[:-1])
    ]
    rows: list[dict[str, object]] = []
    for k in k_values:
        n = (1 << k) - 1
        p_spatial = float(response_sum[n - 1] / n)
        margin_mean = margin_sum[n - 1] / n
        margin_variance = max(margin_square_sum[n - 1] / n - margin_mean**2, 0.0)
        rows.append(
            {
                "selector": name,
                "k": k,
                "n_spatial": n,
                "p_temporal": temporal_probability,
                "p_spatial": p_spatial,
                "base_margin": base_margin,
                "signed_error": p_spatial - temporal_probability,
                "absolute_error": abs(p_spatial - temporal_probability),
                "margin_mae": float(difference_abs_sum[n - 1] / n),
                "margin_rmse": float(np.sqrt(difference_square_sum[n - 1] / n)),
                "margin_std": float(np.sqrt(margin_variance)),
                "margin_bias": float((margin_sum[n - 1] / n) - base_margin),
                "margin_min": float(margin_minimum[n - 1]),
                "margin_max": float(margin_maximum[n - 1]),
                "margin_radius": float(margin_radius[n - 1]),
                "below_fraction": float(below_count[n - 1] / n),
                "balance_shortfall": float(
                    abs(below_count[n - 1] - above_count[n - 1]) / n
                ),
                "transition_rate": (
                    float(transition_sum[n - 1] / (n - 1)) if n > 1 else None
                ),
            }
        )
    return rows


def _balanced_margin_metrics(
    responses: np.ndarray,
    margins: np.ndarray,
    temporal_probability: float,
    k_values: tuple[int, ...],
) -> list[dict[str, object]]:
    """Select the closest available margins on both sides of the reference."""

    candidates = np.arange(1, len(margins), dtype=np.int64)
    differences = margins[candidates] - margins[0]
    below = candidates[differences < 0]
    above = candidates[differences > 0]
    equal = candidates[differences == 0]
    below = below[np.argsort(np.abs(margins[below] - margins[0]), kind="stable")]
    above = above[np.argsort(np.abs(margins[above] - margins[0]), kind="stable")]
    equal = np.sort(equal)
    below_sums = np.r_[0.0, np.cumsum(margins[below] - margins[0])]
    above_sums = np.r_[0.0, np.cumsum(margins[above] - margins[0])]

    rows: list[dict[str, object]] = []
    for k in k_values:
        n = (1 << k) - 1
        equal_count = min(n, len(equal))
        remaining = n - equal_count
        below_count = min(remaining // 2, len(below))
        above_count = min(remaining // 2, len(above))
        missing = remaining - below_count - above_count

        if missing == 1 and below_count < len(below) and above_count < len(above):
            current_sum = below_sums[below_count] + above_sums[above_count]
            below_error = abs(current_sum + margins[below[below_count]] - margins[0])
            above_error = abs(current_sum + margins[above[above_count]] - margins[0])
            if below_error <= above_error:
                below_count += 1
            else:
                above_count += 1
            missing = 0

        if missing:
            below_available = len(below) - below_count
            take_below = min(missing, below_available)
            below_count += take_below
            missing -= take_below
            above_available = len(above) - above_count
            take_above = min(missing, above_available)
            above_count += take_above
            missing -= take_above
        if missing:
            raise ValueError(f"Cannot select {n} balanced candidates")

        selected = np.concatenate(
            (equal[:equal_count], below[:below_count], above[:above_count])
        )
        offsets = np.abs(margins[selected] - margins[0])
        selected = selected[np.argsort(offsets, kind="stable")]
        rows.extend(
            _ordered_selector_metrics(
                "balanced_margin",
                selected,
                responses,
                margins,
                float(margins[0]),
                temporal_probability,
                (k,),
            )
        )
    return rows


def selector_metrics(
    responses: np.ndarray,
    margins: np.ndarray,
    temporal_probability: float,
    k_values: tuple[int, ...],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    if len(responses) != len(margins):
        raise ValueError("Response and margin counts differ")
    if max(k_values) > int(np.log2(len(responses))):
        raise ValueError("A requested k exceeds the response block size")
    candidates = np.arange(1, len(responses), dtype=np.int64)
    base_margin = float(margins[0])
    orders = {
        "gray_prefix": candidates,
        "margin_nearest": candidates[
            np.argsort(np.abs(margins[candidates] - base_margin), kind="stable")
        ],
        "random": rng.permutation(candidates),
    }
    rows: list[dict[str, object]] = []
    for name, order in orders.items():
        rows.extend(
            _ordered_selector_metrics(
                name,
                order,
                responses,
                margins,
                base_margin,
                temporal_probability,
                k_values,
            )
        )
    rows.extend(
        _balanced_margin_metrics(responses, margins, temporal_probability, k_values)
    )
    return rows


def cluster_metrics(
    responses: np.ndarray,
    margins: np.ndarray,
    temporal_probability: float,
) -> dict[str, object]:
    candidates = np.arange(1, len(responses), dtype=np.int64)
    spatial_responses = responses[candidates]
    spatial_margins = margins[candidates]
    p_spatial = float(spatial_responses.mean())
    margin_order = candidates[np.argsort(margins[candidates], kind="stable")]
    ordered_responses = responses[margin_order]
    return {
        "p_temporal": temporal_probability,
        "reliability_pairwise_temporal": (
            temporal_probability**2 + (1 - temporal_probability) ** 2
        ),
        "p_spatial_full": p_spatial,
        "signed_error_full": p_spatial - temporal_probability,
        "absolute_error_full": abs(p_spatial - temporal_probability),
        "base_margin": float(margins[0]),
        "full_margin_mae": float(np.mean(np.abs(spatial_margins - margins[0]))),
        "full_margin_std": float(np.std(spatial_margins)),
        "gray_margin_spearman": float(spearmanr(candidates, spatial_margins).statistic),
        "gray_margin_step_mae": float(np.mean(np.abs(np.diff(spatial_margins)))),
        "gray_transition_rate": float(
            np.mean(spatial_responses[1:] != spatial_responses[:-1])
        ),
        "margin_order_transition_rate": float(
            np.mean(ordered_responses[1:] != ordered_responses[:-1])
        ),
    }


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _instance_fields(key: InstanceKey) -> dict[str, object]:
    return asdict(key)


def _margin_bin_rows(
    accumulators: dict[InstanceKey, tuple[np.ndarray, np.ndarray, np.ndarray]],
    edges: np.ndarray,
) -> Iterator[dict[str, object]]:
    for key, (counts, ones, margin_sums) in accumulators.items():
        for bin_id, count in enumerate(counts):
            if count == 0:
                continue
            yield {
                **_instance_fields(key),
                "bin_id": bin_id,
                "margin_left": float(edges[bin_id]),
                "margin_right": float(edges[bin_id + 1]),
                "n": int(count),
                "ones": int(ones[bin_id]),
                "mean_margin": float(margin_sums[bin_id] / count),
                "mean_response": float(ones[bin_id] / count),
            }


def _temporal_margin_bin_rows(
    key: InstanceKey,
    margins: np.ndarray,
    counts: np.ndarray,
    probabilities: np.ndarray,
    edges: np.ndarray,
) -> Iterator[dict[str, object]]:
    bin_ids = np.searchsorted(edges, margins, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
    for bin_id in range(len(edges) - 1):
        selected = bin_ids == bin_id
        if not np.any(selected):
            continue
        evaluations = int(counts[selected].sum())
        ones = float(np.sum(counts[selected] * probabilities[selected]))
        yield {
            **_instance_fields(key),
            "bin_id": bin_id,
            "margin_left": float(edges[bin_id]),
            "margin_right": float(edges[bin_id + 1]),
            "n_challenges": int(selected.sum()),
            "evaluations": evaluations,
            "ones": ones,
            "mean_margin": float(margins[selected].mean()),
            "mean_response": float(ones / evaluations),
            "mean_challenge_probability": float(probabilities[selected].mean()),
        }


def run_analysis(
    *,
    baseline_db: Path,
    modeling_db: Path,
    ergodicity_db: Path,
    baseline_challenges_path: Path,
    modeling_challenges_path: Path,
    ergodicity_challenges_path: Path,
    output_dir: Path,
    config: AnalysisConfig,
    limit_instances: int | None = None,
    limit_clusters: int | None = None,
    progress_every: int = 1,
) -> None:
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")

    LOGGER.info("Loading challenge arrays")
    baseline_challenges = np.load(baseline_challenges_path)
    modeling_challenges = np.load(modeling_challenges_path)
    ergodicity_challenges = np.load(ergodicity_challenges_path, mmap_mode="r")
    cluster_size = 1 << config.dither_bits
    if baseline_challenges.shape != (200, config.stages):
        raise ValueError(f"Unexpected baseline shape {baseline_challenges.shape}")
    if modeling_challenges.shape != (5_000, config.stages):
        raise ValueError(f"Unexpected modeling shape {modeling_challenges.shape}")
    expected_ergo_shape = (len(baseline_challenges) * cluster_size, config.stages)
    if ergodicity_challenges.shape != expected_ergo_shape:
        raise ValueError(f"Unexpected ergodicity shape {ergodicity_challenges.shape}")
    validate_gray_nesting(
        np.asarray(ergodicity_challenges[:cluster_size]),
        config.dither_bits,
        config.k_values,
    )
    LOGGER.info(
        "Validated %d baseline, %d modeling, and %d ergodicity challenges",
        len(baseline_challenges),
        len(modeling_challenges),
        len(ergodicity_challenges),
    )

    LOGGER.info("Opening SQLite databases in read-only mode")
    connections = {
        "baseline": open_read_only(baseline_db),
        "modeling": open_read_only(modeling_db),
        "ergodicity": open_read_only(ergodicity_db),
    }
    try:
        instance_maps = {
            name: read_instances(connection, config.stages)
            for name, connection in connections.items()
        }
        instance_sets = [set(mapping) for mapping in instance_maps.values()]
        if any(instance_set != instance_sets[0] for instance_set in instance_sets[1:]):
            raise ValueError("The three databases contain different APUF instance sets")
        common_instances = sorted(instance_sets[0])
        if limit_instances is not None:
            common_instances = common_instances[:limit_instances]
        LOGGER.info(
            "Matched %d APUF instances across all databases", len(common_instances)
        )

        LOGGER.info("Computing ATF features for the modeling challenges")
        modeling_features = atf(modeling_challenges)
        edges = np.r_[
            -np.inf,
            np.linspace(
                config.margin_bin_min, config.margin_bin_max, config.margin_bins + 1
            ),
            np.inf,
        ]
        models: dict[InstanceKey, LogisticRegression] = {}
        model_rows: list[dict[str, object]] = []
        temporal_margin_rows: list[dict[str, object]] = []
        baseline_probabilities: dict[InstanceKey, np.ndarray] = {}
        for instance_number, key in enumerate(common_instances, start=1):
            LOGGER.info(
                "[model %d/%d] Aggregating repeats for %s %s",
                instance_number,
                len(common_instances),
                key.device_id,
                key.slice,
            )
            baseline_counts, baseline_p = read_response_probabilities(
                connections["baseline"],
                instance_maps["baseline"][key],
                len(baseline_challenges),
            )
            modeling_counts, modeling_p = read_response_probabilities(
                connections["modeling"],
                instance_maps["modeling"][key],
                len(modeling_challenges),
            )
            model, metrics = fit_instance_model(modeling_features, modeling_p, config)
            modeling_bank = BinaryLatentBank.from_margin(
                model.decision_function(modeling_features)
            )
            LOGGER.info(
                "[model %d/%d] %s %s: accuracy=%.4f, balanced_accuracy=%.4f",
                instance_number,
                len(common_instances),
                key.device_id,
                key.slice,
                metrics["test_accuracy"],
                metrics["test_balanced_accuracy"],
            )
            models[key] = model
            baseline_probabilities[key] = baseline_p
            temporal_margin_rows.extend(
                _temporal_margin_bin_rows(
                    key,
                    modeling_bank.margin,
                    modeling_counts,
                    modeling_p,
                    edges,
                )
            )
            model_rows.append(
                {
                    **_instance_fields(key),
                    "baseline_repeats_min": int(baseline_counts.min()),
                    "baseline_repeats_max": int(baseline_counts.max()),
                    "modeling_repeats_min": int(modeling_counts.min()),
                    "modeling_repeats_max": int(modeling_counts.max()),
                    **metrics,
                }
            )

        bin_count = len(edges) - 1
        margin_accumulators: dict[
            InstanceKey, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {
            key: (
                np.zeros(bin_count, dtype=np.int64),
                np.zeros(bin_count, dtype=np.int64),
                np.zeros(bin_count, dtype=np.float64),
            )
            for key in common_instances
        }
        cluster_rows: list[dict[str, object]] = []
        selector_rows: list[dict[str, object]] = []
        cluster_count = len(baseline_challenges)
        if limit_clusters is not None:
            cluster_count = min(cluster_count, limit_clusters)
        LOGGER.info(
            "Processing %d Gray clusters for %d APUF instances",
            cluster_count,
            len(common_instances),
        )
        cluster_started = monotonic()
        for baseline_id in range(cluster_count):
            first = baseline_id * cluster_size
            features = atf(
                np.asarray(ergodicity_challenges[first : first + cluster_size])
            )
            for key in common_instances:
                responses = read_response_range(
                    connections["ergodicity"],
                    instance_maps["ergodicity"][key],
                    first,
                    cluster_size,
                )
                bank = BinaryLatentBank.from_margin(
                    models[key].decision_function(features)
                )
                margins = bank.margin
                temporal_probability = float(baseline_probabilities[key][baseline_id])
                cluster_rows.append(
                    {
                        **_instance_fields(key),
                        "baseline_id": baseline_id,
                        **cluster_metrics(responses, margins, temporal_probability),
                    }
                )
                rng = np.random.default_rng(config.seed + baseline_id)
                selector_rows.extend(
                    {
                        **_instance_fields(key),
                        "baseline_id": baseline_id,
                        **row,
                    }
                    for row in selector_metrics(
                        responses, margins, temporal_probability, config.k_values, rng
                    )
                )
                bin_ids = np.searchsorted(edges, margins, side="right") - 1
                bin_ids = np.clip(bin_ids, 0, bin_count - 1)
                counts, ones, margin_sums = margin_accumulators[key]
                counts += np.bincount(bin_ids, minlength=bin_count)
                ones += np.bincount(
                    bin_ids, weights=responses, minlength=bin_count
                ).astype(np.int64)
                margin_sums += np.bincount(
                    bin_ids, weights=margins, minlength=bin_count
                )

            completed = baseline_id + 1
            if completed == 1 or completed % progress_every == 0:
                elapsed = monotonic() - cluster_started
                rate = completed / elapsed
                eta = (cluster_count - completed) / rate
                LOGGER.info(
                    "Clusters %d/%d (%.1f%%), elapsed %.1f min, ETA %.1f min",
                    completed,
                    cluster_count,
                    100 * completed / cluster_count,
                    elapsed / 60,
                    eta / 60,
                )

        LOGGER.info("Writing Parquet artifacts to %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(model_rows).write_parquet(output_dir / "models.parquet")
        pl.DataFrame(cluster_rows).write_parquet(output_dir / "cluster_metrics.parquet")
        pl.DataFrame(selector_rows).write_parquet(
            output_dir / "selector_metrics.parquet"
        )
        pl.DataFrame(_margin_bin_rows(margin_accumulators, edges)).write_parquet(
            output_dir / "margin_bins.parquet"
        )
        pl.DataFrame(temporal_margin_rows).write_parquet(
            output_dir / "temporal_margin_bins.parquet"
        )
        manifest = {
            "config": asdict(config),
            "instances": len(common_instances),
            "clusters": cluster_count,
            "inputs": {
                "baseline_db": str(baseline_db.expanduser().resolve()),
                "modeling_db": str(modeling_db.expanduser().resolve()),
                "ergodicity_db": str(ergodicity_db.expanduser().resolve()),
                "baseline_challenges": {
                    "path": str(baseline_challenges_path.resolve()),
                    "md5": _hash_file(baseline_challenges_path),
                },
                "modeling_challenges": {
                    "path": str(modeling_challenges_path.resolve()),
                    "md5": _hash_file(modeling_challenges_path),
                },
                "ergodicity_challenges": {
                    "path": str(ergodicity_challenges_path.resolve()),
                    "md5": _hash_file(ergodicity_challenges_path),
                },
            },
            "notes": {
                "model": "Independent room-temperature modeling CRPs; majority labels",
                "spatial_reference": "Gray index zero excluded from every spatial estimator",
                "temporal_margin_bins": "Modeling challenges with repeated temporal responses",
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Analysis complete")
    finally:
        for connection in connections.values():
            connection.close()
