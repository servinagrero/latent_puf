# Latent PUF Analysis

Reusable latent-margin analysis for binary-decision PUFs, with Arbiter PUF sensing and differential RO PUF reliability as the current applications.

## Setup

The project requires Python 3.11 or newer and uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync
```

## APUF workflows

`run_all.sh` runs the selected studies in the order provided:

```bash
./run_all.sh room_temperature
./run_all.sh latent_geometry margin_bank
./run_all.sh room_temperature latent_geometry margin_bank
```

The room-temperature study expects the baseline, modeling, and ergodicity databases under `~/Programming/APUF_CMODS7/data`. Override that location with:

```bash
APUF_DATA_ROOT=/path/to/data ./run_all.sh room_temperature
```

The large ergodicity challenge array and generated analysis artifacts are local inputs and are intentionally excluded from version control.

## RO PUF workflows

RO analyses use the HDF5 schema documented by the analysis scripts:

```bash
uv run python scripts/analysis_ro_spatial.py --help
uv run python scripts/analysis_ro_cross_validation.py --help
```

The local `data/ro_frequency_differences.h5` link is intentionally excluded because the underlying dataset is not part of the repository.

## Verification

```bash
uv run pytest
uv run ruff check src scripts tests
uv run ty check src/latent_puf
```
