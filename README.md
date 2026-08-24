# VERPO: VERIFIED EVIDENCE REGULARIZED POLICY OPTIMIZATION

Evidence-aware policy optimization with Teacher-guided, token-level
distribution corrections.

VERPO combines a GRPO policy objective with a signed correction derived from
Teacher distributions under different evidence conditions. The correction is
controlled at two levels: a prompt-group gate decides whether a group contains
useful reward variation, and a token-level ZPD controller scales each evidence
direction by its local benefit and policy-movement cost.

The implementation is organized around one semantic configuration and two
training backends. Native veRL provides the Section 3 protocol launcher;
the TRL path provides a small JSONL/math-text interface for portable VERPO
experiments.

## Method

```text
prompt
  -> Student rollout
  -> Teacher scoring under evidence conditions
  -> signed Teacher displacement
  -> ZPD benefit/cost controller
  -> GRPO objective + VERPO correction
  -> updated Student policy
```

For a fixed rollout prefix, the Teacher branches are represented as token
distributions:

- `q0`: evidence-free Teacher distribution;
- `qe`: evidence-conditioned Teacher distribution;
- `q+`, `q-`: positive and negative contrastive Teacher distributions.

The signed Teacher displacement is the only difference between the main
VERPO variants:

```text
Fixed:  Delta_t = q_e,t - q_0,t
CTR:    Delta_t = q_+,t - q_-,t
FEC:    task direction minus the Fisher projection of the nuisance direction
```

For the default Forward-KL controller, the evidence correction is weighted by

```text
w_t = h_t / (h_t + tau * (c_t + rho))
```

`h_t` is the local evidence benefit and `c_t` is the local policy-movement
cost. The FEC direction removes the Student-local Fisher projection of the
evidence-presence nuisance before computing the signed correction.

The backend-neutral trainer combines the reference and evidence terms as

```text
L_VERPO = lambda_ref * L_ref + lambda_evi * L_evi
```

The exact full-vocabulary definitions and the selected-support Top-K
approximation are implemented in `risk_aware_opsd/verpo_zpd.py`.

## Variants and controls

| Variant | Teacher displacement | Purpose |
| --- | --- | --- |
| Fixed | `q_e - q_0` | Evidence-conditioned correction |
| CTR | `q_+ - q_-` | Positive/negative contrastive correction |
| FEC | Contrastive task direction minus nuisance projection | Evidence-confound removal |
| Forward-KL | Probability-space correction | Forward divergence path |
| Reverse-KL | Geometric Teacher path and Student-local Fisher tangent | Reverse divergence path |
| Top-K | Selected-support approximation | Memory-conscious vocabulary computation |

Teacher state is configurable as:

- `frozen`: fixed initial Teacher;
- `snapshot10`: synchronize after the configured optimizer-update interval;
- `ema_095`: update the Teacher with exponential moving average state.

The shared `VERPOConfig` exposes displacement mode, Teacher mode, vocabulary
mode, group-level ZPD, evidence rollout scope, divergence, coefficients, and
Teacher synchronization controls.

## Repository structure

| Path | Role |
| --- | --- |
| `risk_aware_opsd/` | Backend-neutral VERPO loss, ZPD controller, Teacher state, reward, and JSONL contract |
| `verl/verl/trainer/distillation/` | Native veRL VERPO implementation |
| `pipeline/trl/` | JSONL/math-text TRL entry point |
| `pipeline/verl_math/` | Native veRL launcher |
| `configs/verpo/` | Semantic model, protocol, Teacher, arm, and matrix configuration |
| `archive/rlcsd/` | Historical material outside the active VERPO launch path |
| `provenance/` | File-level reproducibility metadata |

The public training interface is the launcher layer. Internal engine scripts
are implementation details and should not be invoked directly.

## Reproduce with TRL

The TRL interface accepts local JSONL/math-text records. Each row contains a
`prompt` and `completion`; `response` is accepted as an alternative completion
field.

```json
{
  "prompt": "local prompt text",
  "completion": "local completion text"
}
```

Install the CPU and development environment, then run a one-step smoke:

```bash
uv sync --extra cpu --extra dev

bash pipeline/trl/run.sh \
  --train-file path/to/private/train.jsonl \
  --output-dir outputs/trl-verpo \
  --max-steps 1
```

This path is intentionally limited to JSONL/math-text. Parquet protocol data
belongs to the native veRL entry point.

## Reproduce with native veRL

Resolve one Section 3 cell without allocating a GPU:

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

Inspect the complete registered configuration matrix before launching it:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_1_7b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --matrix sdpo_five_dataset_teacher_arm \
  --print-command
```

The matrix is a configuration composition of five Section 3 protocols, three
Teacher states, and six Fixed/CTR/FEC Forward- and Reverse-KL arms. It defines
launch cells; it is not an evaluation table.

Formal training requires `SWANLAB_API_KEY` from the host environment. Use
`--print-config` or `--print-command` before any resource-consuming launch.

## Local assets and credentials

Data, model weights, checkpoints, logs, caches, and credentials are operator
inputs and are not committed here. Configure only local paths and host-managed
credentials:

| Variable | Meaning |
| --- | --- |
| `SDPO_DATA_DIR` | Operator-provided local data root |
| `TRAIN_FILE` / `VAL_FILE` | Explicit local file overrides |
| `MODEL_PATH` | Local model snapshot |
| `MODEL_CACHE` / `HF_HOME` | Model cache |
| `SWANLAB_API_KEY` | Host-injected SwanLab credential |
| `OUTPUT_ROOT` | Single-cell output root |
| `MATRIX_OUTPUT_ROOT` | Matrix output root |

This README intentionally omits data distribution details, storage identifiers,
cloud storage, private manifests, and internal asset preparation procedures.
Never place `SWANLAB_API_KEY` in commands, YAML, README examples, manifests, or
logs.

## Configuration

Semantic configuration is composed from model, finetuning, hardware, protocol,
Teacher, and arm overlays. The field reference and available identifiers are
documented in [`configs/verpo/README.md`](configs/verpo/README.md).

Each resolved launch can emit a semantic configuration and backend projections
under its output provenance directory. The `archive/rlcsd/` directory is not
part of the active public launch path.

## Environment

Use the committed `uv.lock` file. CPU/development dependencies are separate
from the GPU training profile:

```bash
uv sync --extra cpu --extra dev
```

For a host with the matching CUDA, veRL, and rollout runtime:

```bash
uv sync --extra gpu
```

Do not commit `.env` files, datasets, model weights, checkpoints, logs, or
generated output directories.
