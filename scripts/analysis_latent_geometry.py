"""Compute exploratory UMAP and neighborhood geometry for APUF challenges."""

import argparse
import logging
from pathlib import Path

from latent_puf.latent_geometry import GeometryConfig, run_latent_geometry_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-db", type=Path, required=True)
    parser.add_argument("--modeling-challenges", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--pair-samples", type=int, default=250_000)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--target-weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--limit-instances", type=int)
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
    run_latent_geometry_analysis(
        modeling_db=args.modeling_db,
        modeling_challenges_path=args.modeling_challenges,
        models_path=args.models,
        output_dir=args.output,
        config=GeometryConfig(
            neighbors=tuple(args.neighbors),
            pair_samples=args.pair_samples,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            target_weight=args.target_weight,
            seed=args.seed,
        ),
        limit_instances=args.limit_instances,
    )


if __name__ == "__main__":
    main()
