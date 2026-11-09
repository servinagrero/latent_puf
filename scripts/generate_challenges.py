from pathlib import Path

import numpy as np

DATA_DIR = Path("./data").resolve()
CHALLENGES_DIR = DATA_DIR / "challenges"
CHALLENGES_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 123456

BASELINE_CHALLENGES = 200
MAX_STAGES = 64

MODEL_CHALLENGES = 5_000
MODEL_RNG_SEED = RNG_SEED + 1


def ints_to_bits(x, nbits, *, bitorder="big", dtype=np.uint8):
    if bitorder not in ("big", "little"):
        raise ValueError("bitorder must be 'big' or 'little'")

    x = np.asarray(x, dtype=np.uint64)

    shifts = (
        np.arange(nbits - 1, -1, -1, dtype=np.uint64)
        if bitorder == "big"
        else np.arange(nbits, dtype=np.uint64)
    )

    return ((x[..., None] >> shifts) & 1).astype(dtype)


def generate_gray_code(n_bits):
    """Generate binary-reflected Gray-code integers."""
    if not 1 <= n_bits <= 32:
        raise ValueError("n_bits must be between 1 and 32")

    values = np.arange(2**n_bits, dtype=np.uint64)
    return values ^ (values >> 1)


def create_challenges_modeling():
    rng = np.random.default_rng(MODEL_RNG_SEED)
    return rng.integers(
        0,
        2,
        size=(MODEL_CHALLENGES, MAX_STAGES),
        dtype=np.uint8,
    )

def create_challenges_baseline():
    """Generate random baseline challenges."""
    rng = np.random.default_rng(RNG_SEED)

    return rng.integers(
        0,
        2,
        size=(BASELINE_CHALLENGES, MAX_STAGES),
        dtype=np.uint8,
    )


def create_challenges_ergodicity_clusters(
    crps_baseline: np.ndarray,
    dither_bits: int = 10,
):
    """
    Generate mirrored Gray-code neighborhoods around baseline challenges.

    Only the leftmost dither_bits are varied.
    The remaining challenge bits remain unchanged.

    Each cluster contains 2**dither_bits challenges.
    The first challenge of each cluster is the original baseline.
    """
    if crps_baseline.ndim != 2:
        raise ValueError("crps_baseline must be a 2D array")

    n_clusters, n_stages = crps_baseline.shape

    if n_stages != MAX_STAGES:
        raise ValueError(f"Expected {MAX_STAGES} stages")

    if not 1 <= dither_bits <= min(n_stages, 32):
        raise ValueError("Invalid dither_bits")

    if not np.all((crps_baseline == 0) | (crps_baseline == 1)):
        raise ValueError("Baseline challenges must contain only 0 and 1")

    # Generate mirrored Gray code.
    gray_ints = generate_gray_code(dither_bits)

    gray_bits = ints_to_bits(
        gray_ints,
        dither_bits,
        bitorder="big",
    )[:, ::-1].copy()

    cluster_size = 2**dither_bits

    # Allocate the output once.
    crps = np.empty(
        (n_clusters * cluster_size, n_stages),
        dtype=np.uint8,
    )

    for cluster, base_challenge in enumerate(crps_baseline):
        start = cluster * cluster_size
        end = start + cluster_size

        block = crps[start:end]

        # Copy the base challenge into the entire cluster.
        block[:] = base_challenge

        # Apply mirrored Gray-code variations to the prefix.
        block[:, :dither_bits] ^= gray_bits

    return crps


if __name__ == "__main__":
    crps_modeling = create_challenges_modeling()
    np.save(CHALLENGES_DIR / "modeling.npy", crps_modeling)
    print(f"Saved {len(crps_modeling)} modeling challenges.")

    # crps_baseline = create_challenges_baseline()
    # np.save(CHALLENGES_DIR / "baseline.npy", crps_baseline)
    # print(f"Saved {len(crps_baseline)} baseline challenges.")

    # crps_ergodicity = create_challenges_ergodicity_clusters(
    #     crps_baseline=crps_baseline,
    #     dither_bits=16,
    # )
    # np.save(CHALLENGES_DIR / "ergodicity.npy", crps_ergodicity)
    # print(f"Saved {len(crps_ergodicity)} ergodicity challenges.")
