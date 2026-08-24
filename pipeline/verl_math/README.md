# SDPO Section 3 veRL launcher

`pipeline/verl_math/run.sh` is the only public training entry point. The active
setting is SDPO Section 3 on the registered Biology, Chemistry, Material,
Physics, and Tool Use protocols. Scripts under `engines/` are internal details
and must not be invoked directly.

The launcher resolves the semantic YAML under `configs/verpo/`, checks the
registered model and hardware constraints, bootstraps the Python/veRL
environment, resolves the host-provided data and model locations, enables
SwanLab, and then starts training. Data distribution details are intentionally
outside this repository.

## One-command paper baseline training

Run the following command from the repository root to train the paper-faithful
SDPO and SRPO baselines across all five datasets. The matrix contains 10 cells
and runs them sequentially in the declared order:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_4b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --matrix sdpo_srpo_paper_baselines
```

Each cell receives a separate
`protocol__ema_095__arm` output directory under the matrix output root. The
launcher validates before training, caps every cell at 300 optimizer steps,
saves checkpoints every 50 steps, and preserves rollout and validation output.

Before allocating GPUs, resolve the same matrix without dependency setup,
downloads, credential checks, or training:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_4b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --matrix sdpo_srpo_paper_baselines \
  --print-command
```

Remove `--print-command` after checking the resolved cells to start the complete
matrix. Formal runs use SwanLab online logging; set `SWANLAB_API_KEY` in the
server environment when the deployment does not already provide it. Do not put
credentials in YAML files or shell scripts.

## Single baseline cell

Use `--protocol` and `--arm` to run one dataset/objective pair. This example
starts the SDPO baseline on Biology:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_4b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --protocol sdpo_section3_biology \
  --teacher ema_095 \
  --arm sdpo_jsd_paper
```

For SRPO, change the arm to `srpo_jsd_paper`. The registered protocol names are:

- `sdpo_section3_biology`
- `sdpo_section3_chemistry`
- `sdpo_section3_material`
- `sdpo_section3_physics`
- `sdpo_section3_tooluse`

Add `--print-config` to inspect a single resolved cell without launching it.

## GRPO comparison matrix

To run matched GRPO, SDPO, and SRPO baselines across all five datasets, launch
the 15-cell comparison matrix:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_4b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --matrix grpo_sdpo_srpo_paper_baselines
```

Use `--print-command` first when checking a new host. The `grpo` arm shares the
SRPO prompt, sampling, mini-batch, and learning-rate settings, but disables the
paper Teacher and uses the pure PPO/GRPO objective.

## Qwen3-8B wrong-only branch

The registered `fec_fkl_wrong_only` arm is an isolated Qwen3-8B VERPO-ZPD
branch. It keeps the FEC-FKL method, EMA Teacher, prompts, data, batch, and
training settings unchanged while setting only the evidence rollout scope to
`wrong_only`. The existing `fec_fkl` arm remains the `all` control.

Resolve or start one Biology cell with:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_8b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --protocol sdpo_section3_biology \
  --teacher ema_095 \
  --arm fec_fkl_wrong_only
```

Add `--print-config` for a network-free configuration check. To run the five
registered Section 3 datasets sequentially, use:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_8b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --matrix qwen3_8b_fec_fkl_wrong_only
```

For the matched `all` control, use the same single-cell command with
`--arm fec_fkl`. The two arms resolve to different experiment IDs, configuration
hashes, and output directories; `fec_fkl_wrong_only` cannot be overridden back
to `all` with `--set`.

## Paths and overrides

The launcher accepts host-specific locations through optional environment
variables without changing experiment identity. Data and model assets must be
provided by the operator:

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | Existing local model snapshot. |
| `MODEL_CACHE` or `HF_HOME` | Model cache root. |
| `SDPO_DATA_DIR` | Private local root containing protocol data files. |
| `TRAIN_FILE` / `VAL_FILE` | Explicit training or validation file overrides. |
| `OUTPUT_ROOT` | Output directory for a single-cell launch. |
| `MATRIX_OUTPUT_ROOT` | Matrix root; each cell receives a distinct subdirectory. |
| `SWANLAB_API_KEY` | SwanLab authentication supplied by the server environment. |

For the semantic field reference, baseline definitions, prompt contracts, and
troubleshooting guide, see
[`configs/verpo/README.md`](../../configs/verpo/README.md).
