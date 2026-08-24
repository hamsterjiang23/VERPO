# VERPO-ZPD

Portable VERPO-ZPD training code with two backends:

- native veRL for the formal SDPO Section 3 protocols and matrix launchers;
- a lightweight TRL-compatible path for JSONL/math-text experiments.

The repository keeps the VERPO objective, Fixed/CTR/FEC displacement modes,
Forward- and Reverse-KL variants, top-k support, frozen/snapshot/EMA Teacher
state, length-aware rewards, gradient audits, checkpoint handling, and one
shared semantic configuration projection.

Datasets, model weights, credentials, caches, checkpoints, and run artifacts
are intentionally outside this repository. The examples below show launcher
interfaces only; provide private data and model locations through the host
environment or your local configuration.

## Repository layout

| Path | Purpose |
| --- | --- |
| `risk_aware_opsd/` | Shared VERPO math, rewards, Teacher state, data contracts, and TRL trainer. |
| `pipeline/trl/run.sh` | JSONL/math-text TRL entry point. |
| `pipeline/verl_math/run.sh` | Native veRL entry point for SDPO Section 3. |
| `configs/verpo/` | Model, hardware, protocol, Teacher, arm, and matrix definitions. |
| `verl/` | Vendored veRL source with the VERPO hooks included. |
| `archive/rlcsd/` | Historical RLCSD material; excluded from the public launch path. |
| `provenance/` | Migration metadata and file-level provenance. |

## Native veRL

Resolve one cell without side effects or using a GPU:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_1_7b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --protocol sdpo_section3_biology \
  --teacher frozen \
  --arm fixed_fkl \
  --print-config
```

Inspect a complete matrix before launching it:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_1_7b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --matrix sdpo_five_dataset_teacher_arm \
  --print-command
```

Remove the print flag only after reviewing the resolved cells. Formal training
requires `SWANLAB_API_KEY` from the host environment. Credentials must never
be placed in YAML files, shell scripts, commands, manifests, or logs.

## TRL JSONL smoke

The TRL path accepts local JSONL/math-text records. Each row contains `prompt`
and `completion` (or `response`):

```bash
bash pipeline/trl/run.sh \
  --train-file path/to/private/train.jsonl \
  --output-dir outputs/trl-smoke
```

The input file is deliberately not included in this repository. Keep local
fixtures, checkpoints, and metrics under ignored output directories.

## Data and model locations

The launchers support host-specific paths without changing experiment identity.
Set only the variables needed by your environment:

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | Existing local model snapshot. |
| `MODEL_CACHE` or `HF_HOME` | Model cache location. |
| `SDPO_DATA_DIR` | Private local root containing the protocol data files. |
| `TRAIN_FILE` / `VAL_FILE` | Explicit training or validation file overrides. |
| `OUTPUT_ROOT` | Output directory for a single-cell launch. |
| `MATRIX_OUTPUT_ROOT` | Matrix output root. |
| `SWANLAB_API_KEY` | SwanLab credential injected by the host. |

Data preparation and distribution details are intentionally omitted from this
public README. Keep private manifest or storage instructions outside the
repository or in an access-controlled document.

## Configuration and provenance

The semantic schema is documented in
[`configs/verpo/README.md`](configs/verpo/README.md). The migration
manifest records the source snapshot, vendored veRL baseline, file hashes, and
whether each custom file was retained, rewritten, or archived:

[`provenance/migration_manifest.json`](provenance/migration_manifest.json)

RLCSD files under `archive/rlcsd/` are archival reproducibility material and
are not accepted by the public launcher.

## Environment

Use `uv` with the committed lockfile. CPU/development dependencies are kept
separate from the GPU training profile:

```bash
uv sync --extra cpu --extra dev
```

The GPU environment is intended for a host with the matching CUDA, veRL, and
rollout runtime stack:

```bash
uv sync --extra gpu
```

Do not commit `.env` files, model weights, datasets, checkpoints, or generated
experiment outputs.
