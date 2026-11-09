"""Fixed modeled-margin challenge-bank analysis."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict
from hashlib import md5
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from latent_puf.latent import BinaryLatentBank
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


def read_response_matrix(
    connection: sqlite3.Connection,
    acquisition_id: int,
    expected_challenges: int,
) -> np.ndarray:
    """Return responses ordered as ``(sample_id, challenge_id)``."""

    summary = connection.execute(
        """
        SELECT MIN(sample_id), MAX(sample_id), COUNT(DISTINCT sample_id), COUNT(*)
        FROM crps WHERE acquisition_id = ?
        """,
        (acquisition_id,),
    ).fetchone()
    if summary is None or summary[0] is None:
        raise ValueError(f"Acquisition {acquisition_id} contains no responses")
    first_sample, last_sample, sample_count, row_count = map(int, summary)
    if first_sample != 0 or last_sample != sample_count - 1:
        raise ValueError(f"Acquisition {acquisition_id}: sample IDs are not contiguous")
    expected_rows = sample_count * expected_challenges
    if row_count != expected_rows:
        raise ValueError(
            f"Acquisition {acquisition_id}: expected {expected_rows} rows, "
            f"found {row_count}"
        )

    rows = connection.execute(
        """
        SELECT sample_id, challenge_id, response FROM crps
        WHERE acquisition_id = ? ORDER BY sample_id, challenge_id
        """,
        (acquisition_id,),
    )

    def ordered_responses() -> Iterator[int]:
        for index, (sample_id, challenge_id, response) in enumerate(rows):
            expected_sample, expected_challenge = divmod(index, expected_challenges)
            if sample_id != expected_sample or challenge_id != expected_challenge:
                raise ValueError(
                    f"Acquisition {acquisition_id}: responses are not a complete "
                    "sample-by-challenge matrix"
                )
            if response not in (0, 1):
                raise ValueError(
                    f"Acquisition {acquisition_id}: responses are not binary"
                )
            yield response

    responses = np.fromiter(ordered_responses(), dtype=np.int8, count=expected_rows)
    return responses.reshape(sample_count, expected_challenges)


def select_margin_bank(
    margins: np.ndarray,
    bin_edges: np.ndarray,
    challenges_per_bin: int,
) -> list[np.ndarray]:
    """Select deterministic, center-nearest challenges in each margin bin."""

    if margins.ndim != 1:
        raise ValueError("margins must be one-dimensional")
    if challenges_per_bin < 1:
        raise ValueError("challenges_per_bin must be positive")
    if bin_edges.ndim != 1 or len(bin_edges) < 2 or np.any(np.diff(bin_edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")

    bin_ids = np.searchsorted(bin_edges, margins, side="right") - 1
    bin_ids[margins == bin_edges[-1]] = len(bin_edges) - 2
    valid = (margins >= bin_edges[0]) & (margins <= bin_edges[-1])
    selected: list[np.ndarray] = []
    for bin_id in range(len(bin_edges) - 1):
        candidates = np.flatnonzero(valid & (bin_ids == bin_id))
        center = (bin_edges[bin_id] + bin_edges[bin_id + 1]) / 2
        order = np.lexsort((candidates, np.abs(margins[candidates] - center)))
        selected.append(candidates[order[:challenges_per_bin]])
    return selected


def monotone_curve_features(
    margins: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | None]:
    """Return monotone 10%, 50%, and 90% crossings and their width."""

    if not (margins.ndim == probabilities.ndim == weights.ndim == 1):
        raise ValueError("curve inputs must be one-dimensional")
    if not (len(margins) == len(probabilities) == len(weights)):
        raise ValueError("curve input lengths differ")
    valid = (
        np.isfinite(margins)
        & np.isfinite(probabilities)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if np.count_nonzero(valid) < 2:
        return {"x10": None, "x50": None, "x90": None, "width_10_90": None}
    x = margins[valid]
    y = probabilities[valid]
    sample_weight = weights[valid]
    order = np.argsort(x, kind="stable")
    x, y, sample_weight = x[order], y[order], sample_weight[order]
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
        x, y, sample_weight=sample_weight
    )

    def crossing(level: float) -> float | None:
        indices = np.flatnonzero(fitted >= level)
        if not len(indices) or fitted[0] > level or fitted[-1] < level:
            return None
        right = int(indices[0])
        if right == 0:
            return float(x[0])
        left = right - 1
        if fitted[right] == fitted[left]:
            return float((x[left] + x[right]) / 2)
        fraction = (level - fitted[left]) / (fitted[right] - fitted[left])
        return float(x[left] + fraction * (x[right] - x[left]))

    x10, x50, x90 = (crossing(level) for level in (0.1, 0.5, 0.9))
    width = None if x10 is None or x90 is None else x90 - x10
    return {"x10": x10, "x50": x50, "x90": x90, "width_10_90": width}


def _file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def curve_error_metrics(
    *,
    snapshot: np.ndarray,
    bank: np.ndarray,
    full: np.ndarray,
) -> dict[str, float]:
    """Decompose snapshot error into precision, enrollment, and total terms."""

    if not (snapshot.ndim == bank.ndim == full.ndim == 1):
        raise ValueError("curve probabilities must be one-dimensional")
    if not (len(snapshot) == len(bank) == len(full)) or not len(snapshot):
        raise ValueError("common curve bins must be nonempty and equal in length")
    if not np.all(np.isfinite(np.concatenate((snapshot, bank, full)))):
        raise ValueError("curve probabilities must be finite")

    result: dict[str, float] = {}
    differences = {
        "precision": snapshot - bank,
        "enrollment": bank - full,
        "total": snapshot - full,
    }
    for name, difference in differences.items():
        result[f"{name}_curve_bias"] = float(np.mean(difference))
        result[f"{name}_curve_mae"] = float(np.mean(np.abs(difference)))
        result[f"{name}_curve_rmse"] = float(np.sqrt(np.mean(difference**2)))
    return result


def _difference(
    first: float | None, second: float | None
) -> float | None:
    return None if first is None or second is None else first - second


def _feature_row(
    *,
    snapshot_probabilities: np.ndarray,
    bank_probabilities: np.ndarray,
    full_probabilities: np.ndarray,
    bin_centers: np.ndarray,
    bank_weights: np.ndarray,
    bank_features: dict[str, float | None],
    full_features: dict[str, float | None],
) -> dict[str, float | None]:
    snapshot_features = monotone_curve_features(
        bin_centers, snapshot_probabilities, bank_weights
    )
    result: dict[str, float | None] = {
        **curve_error_metrics(
            snapshot=snapshot_probabilities,
            bank=bank_probabilities,
            full=full_probabilities,
        )
    }
    for prefix, features in (
        ("snapshot", snapshot_features),
        ("bank", bank_features),
        ("full", full_features),
    ):
        for name, value in features.items():
            result[f"{prefix}_{name}"] = value

    snapshot_x50 = snapshot_features["x50"]
    bank_x50 = bank_features["x50"]
    full_x50 = full_features["x50"]
    snapshot_width = snapshot_features["width_10_90"]
    bank_width = bank_features["width_10_90"]
    full_width = full_features["width_10_90"]
    return {
        **result,
        "x50_precision_error": _difference(snapshot_x50, bank_x50),
        "x50_enrollment_error": _difference(bank_x50, full_x50),
        "x50_total_error": _difference(snapshot_x50, full_x50),
        "width_precision_error": _difference(snapshot_width, bank_width),
        "width_enrollment_error": _difference(bank_width, full_width),
        "width_total_error": _difference(snapshot_width, full_width),
    }


def analyze_margin_bank(
    *,
    modeling_db: Path,
    modeling_challenges_path: Path,
    models_path: Path,
    output_dir: Path,
    challenges_per_bin: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    margin_min: float = -6.0,
    margin_max: float = 6.0,
    margin_bins: int = 48,
) -> None:
    """Evaluate fixed margin banks using each repeat as a single-shot snapshot."""

    if not challenges_per_bin or any(value < 1 for value in challenges_per_bin):
        raise ValueError("challenges_per_bin must contain positive values")
    if tuple(sorted(set(challenges_per_bin))) != challenges_per_bin:
        raise ValueError("challenges_per_bin must be unique and increasing")
    if margin_min >= margin_max or margin_bins < 3:
        raise ValueError("invalid margin range or bin count")

    challenges = np.load(modeling_challenges_path)
    if challenges.ndim != 2:
        raise ValueError("modeling challenges must be a two-dimensional array")
    features = atf(challenges)
    stages = challenges.shape[1]
    edges = np.linspace(margin_min, margin_max, margin_bins + 1)
    models = _model_rows(models_path)
    connection = open_read_only(modeling_db)
    try:
        acquisitions = read_instances(connection, stages)
        common_instances = sorted(set(acquisitions) & set(models))
        if set(acquisitions) != set(models):
            raise ValueError("Model artifact and database contain different instances")
        LOGGER.info("Analyzing %d APUF instances", len(common_instances))
        definition_rows: list[dict[str, object]] = []
        reference_rows: list[dict[str, object]] = []
        curve_rows: list[dict[str, object]] = []
        feature_rows: list[dict[str, object]] = []

        for instance_number, key in enumerate(common_instances, start=1):
            LOGGER.info(
                "[instance %d/%d] %s %s",
                instance_number,
                len(common_instances),
                key.device_id,
                key.slice,
            )
            model = models[key]
            coefficients = np.asarray(
                cast(list[float], model["coefficients"]), dtype=np.float64
            )
            if len(coefficients) != features.shape[1]:
                raise ValueError(f"Coefficient count differs for {key}")
            bank = BinaryLatentBank.from_margin(
                features @ coefficients + cast(float, model["intercept"])
            )
            margins = bank.margin
            responses = read_response_matrix(
                connection, acquisitions[key], len(challenges)
            )
            identity = asdict(key)
            full_bins = select_margin_bank(
                margins, edges, challenges_per_bin=len(challenges)
            )
            full_reference: dict[int, tuple[float, float, int]] = {}
            for bin_id, challenge_ids in enumerate(full_bins):
                if not len(challenge_ids):
                    continue
                bin_center = float((edges[bin_id] + edges[bin_id + 1]) / 2)
                temporal_probability = float(responses[:, challenge_ids].mean())
                full_reference[bin_id] = (
                    bin_center,
                    temporal_probability,
                    len(challenge_ids),
                )
                reference_rows.append(
                    {
                        **identity,
                        "bin_id": bin_id,
                        "bin_center": bin_center,
                        "n_challenges": len(challenge_ids),
                        "evaluations": int(responses.shape[0] * len(challenge_ids)),
                        "mean_margin": float(margins[challenge_ids].mean()),
                        "p_temporal_full": temporal_probability,
                    }
                )

            full_bin_ids = sorted(full_reference)
            full_centers = np.asarray(
                [full_reference[bin_id][0] for bin_id in full_bin_ids]
            )
            full_probabilities_all = np.asarray(
                [full_reference[bin_id][1] for bin_id in full_bin_ids]
            )
            full_weights_all = np.asarray(
                [full_reference[bin_id][2] for bin_id in full_bin_ids],
                dtype=np.float64,
            )
            full_features = monotone_curve_features(
                full_centers, full_probabilities_all, full_weights_all
            )

            for count_per_bin in challenges_per_bin:
                selected_bins = select_margin_bank(
                    margins, edges, challenges_per_bin=count_per_bin
                )
                nonempty = [ids for ids in selected_bins if len(ids)]
                total_challenges = sum(map(len, nonempty))
                if len(nonempty) < 2:
                    raise ValueError(f"Fewer than two occupied margin bins for {key}")

                curve_centers: list[float] = []
                curve_bank: list[float] = []
                curve_full: list[float] = []
                bank_weights: list[float] = []
                full_weights: list[float] = []
                per_bin_responses: list[np.ndarray] = []
                occupied_bin_ids: list[int] = []
                for bin_id, challenge_ids in enumerate(selected_bins):
                    if not len(challenge_ids):
                        continue
                    if bin_id not in full_reference:
                        raise ValueError(f"Selected bin {bin_id} has no full reference")
                    bin_center, full_probability, full_count = full_reference[bin_id]
                    snapshot_probabilities = responses[:, challenge_ids].mean(axis=1)
                    bank_probability = float(snapshot_probabilities.mean())
                    curve_centers.append(bin_center)
                    curve_bank.append(bank_probability)
                    curve_full.append(full_probability)
                    bank_weights.append(float(len(challenge_ids)))
                    full_weights.append(float(full_count))
                    per_bin_responses.append(snapshot_probabilities)
                    occupied_bin_ids.append(bin_id)
                    for challenge_id in challenge_ids:
                        definition_rows.append(
                            {
                                **identity,
                                "challenges_per_bin": count_per_bin,
                                "challenge_id": int(challenge_id),
                                "bin_id": bin_id,
                                "modeled_margin": float(margins[challenge_id]),
                            }
                        )

                x = np.asarray(curve_centers)
                bank = np.asarray(curve_bank)
                full = np.asarray(curve_full)
                selected_weights = np.asarray(bank_weights)
                reference_weights = np.asarray(full_weights)
                snapshots = np.column_stack(per_bin_responses)
                bank_features = monotone_curve_features(x, bank, selected_weights)
                for sample_id, snapshot in enumerate(snapshots):
                    for index, bin_id in enumerate(occupied_bin_ids):
                        curve_rows.append(
                            {
                                **identity,
                                "challenges_per_bin": count_per_bin,
                                "sample_id": sample_id,
                                "bin_id": bin_id,
                                "bin_center": float(x[index]),
                                "n_challenges": int(selected_weights[index]),
                                "full_n_challenges": int(reference_weights[index]),
                                "mean_margin": float(
                                    margins[selected_bins[bin_id]].mean()
                                ),
                                "p_single_shot": float(snapshot[index]),
                                "p_temporal_bank": float(bank[index]),
                                "p_temporal_full": float(full[index]),
                            }
                        )
                    feature_rows.append(
                        {
                            **identity,
                            "challenges_per_bin": count_per_bin,
                            "sample_id": sample_id,
                            "total_challenges": total_challenges,
                            "occupied_bins": len(nonempty),
                            "comparison_bins": len(nonempty),
                            "full_reference_challenges": int(
                                sum(full_reference[bin_id][2] for bin_id in occupied_bin_ids)
                            ),
                            **_feature_row(
                                snapshot_probabilities=snapshot,
                                bank_probabilities=bank,
                                full_probabilities=full,
                                bin_centers=x,
                                bank_weights=selected_weights,
                                bank_features=bank_features,
                                full_features=full_features,
                            ),
                        }
                    )

        output_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(definition_rows).write_parquet(
            output_dir / "bank_definition.parquet"
        )
        pl.DataFrame(reference_rows).write_parquet(
            output_dir / "reference_curves.parquet"
        )
        pl.DataFrame(curve_rows).write_parquet(output_dir / "bank_curves.parquet")
        pl.DataFrame(feature_rows).write_parquet(output_dir / "bank_features.parquet")
        manifest = {
            "challenges_per_bin": challenges_per_bin,
            "margin_min": margin_min,
            "margin_max": margin_max,
            "margin_bins": margin_bins,
            "instances": len(common_instances),
            "samples_per_instance": int(responses.shape[0]),
            "inputs": {
                "modeling_db": str(modeling_db.expanduser().resolve()),
                "modeling_challenges": str(modeling_challenges_path.resolve()),
                "modeling_challenges_md5": _file_md5(modeling_challenges_path),
                "models": str(models_path.resolve()),
                "models_md5": _file_md5(models_path),
            },
            "notes": {
                "snapshot": "One sample_id response from every selected challenge",
                "temporal_reference": (
                    "Mean over every sample_id for the same selected challenges"
                ),
                "full_temporal_reference": (
                    "Mean over every sample_id and every challenge in each margin bin"
                ),
                "selection": "Closest challenges to each equal-width bin center",
                "curve_coordinate": "Fixed equal-width margin-bin centers",
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote fixed-bank artifacts to %s", output_dir)
    finally:
        connection.close()
