"""Extract reusable room-temperature analysis artifacts from SQLite."""

import argparse
import logging
from pathlib import Path

from latent_puf.room_temperature import AnalysisConfig, run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-db", type=Path, required=True)
    parser.add_argument("--modeling-db", type=Path, required=True)
    parser.add_argument("--ergodicity-db", type=Path, required=True)
    parser.add_argument("--baseline-challenges", type=Path, required=True)
    parser.add_argument("--modeling-challenges", type=Path, required=True)
    parser.add_argument("--ergodicity-challenges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", type=int, default=64)
    parser.add_argument("--dither-bits", type=int, default=16)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--limit-clusters", type=int)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Report cluster progress every N completed clusters",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = AnalysisConfig(
        stages=args.stages,
        dither_bits=args.dither_bits,
        k_values=tuple(args.k_values),
        test_fraction=args.test_fraction,
        logistic_c=args.logistic_c,
        seed=args.seed,
    )
    run_analysis(
        baseline_db=args.baseline_db,
        modeling_db=args.modeling_db,
        ergodicity_db=args.ergodicity_db,
        baseline_challenges_path=args.baseline_challenges,
        modeling_challenges_path=args.modeling_challenges,
        ergodicity_challenges_path=args.ergodicity_challenges,
        output_dir=args.output,
        config=config,
        limit_instances=args.limit_instances,
        limit_clusters=args.limit_clusters,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
