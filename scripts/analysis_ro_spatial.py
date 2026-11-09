"""Analyze temporal--spatial equivalence in analog RO PUF datasets."""

import argparse
import logging
from pathlib import Path

from latent_puf.ro_spatial import ROSpatialConfig, analyze_ro_spatial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--neighbors", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--pair-block-size", type=int, default=65_536)
    parser.add_argument("--targets-per-device", type=int, default=10_000)
    parser.add_argument("--curve-bins", type=int, default=64)
    parser.add_argument(
        "--split", choices=("alternating", "sequential"), default="alternating"
    )
    parser.add_argument("--seed", type=int, default=123_456)
    parser.add_argument("--limit-devices", type=int)
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
    analyze_ro_spatial(
        input_path=args.input,
        output_dir=args.output,
        groups=tuple(args.groups) if args.groups else None,
        limit_devices=args.limit_devices,
        config=ROSpatialConfig(
            neighbors=tuple(args.neighbors),
            pair_block_size=args.pair_block_size,
            targets_per_device=args.targets_per_device,
            curve_bins=args.curve_bins,
            split=args.split,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
