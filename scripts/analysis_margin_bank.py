"""Analyze fixed modeled-margin banks using repeated room-temperature data."""

import argparse
import logging
from pathlib import Path

from latent_puf.margin_bank import analyze_margin_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-db", type=Path, required=True)
    parser.add_argument("--modeling-challenges", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--challenges-per-bin",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--margin-min", type=float, default=-6.0)
    parser.add_argument("--margin-max", type=float, default=6.0)
    parser.add_argument("--margin-bins", type=int, default=48)
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
    analyze_margin_bank(
        modeling_db=args.modeling_db,
        modeling_challenges_path=args.modeling_challenges,
        models_path=args.models,
        output_dir=args.output,
        challenges_per_bin=tuple(args.challenges_per_bin),
        margin_min=args.margin_min,
        margin_max=args.margin_max,
        margin_bins=args.margin_bins,
    )


if __name__ == "__main__":
    main()
