"""Run fixed-bank or strict leave-one-out validation for analog RO PUFs."""

import argparse
import logging
from pathlib import Path

from latent_puf.ro_cross_validation import (
    ROCrossValidationConfig,
    run_ro_cross_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--mode", choices=("fixed", "loo"), default="fixed")
    parser.add_argument(
        "--neighbors",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--targets-per-stratum", type=int, default=100)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--pair-block-size", type=int, default=65_536)
    parser.add_argument("--curve-bins", type=int, default=64)
    parser.add_argument("--limit-devices", type=int)
    parser.add_argument("--seed", type=int, default=123_456)
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
    run_ro_cross_validation(
        input_path=args.input,
        output_dir=args.output,
        groups=tuple(args.groups) if args.groups else None,
        limit_devices=args.limit_devices,
        config=ROCrossValidationConfig(
            mode=args.mode,
            neighbors=tuple(args.neighbors),
            targets_per_stratum=args.targets_per_stratum,
            pair_block_size=args.pair_block_size,
            curve_bins=args.curve_bins,
            max_folds=args.max_folds,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
