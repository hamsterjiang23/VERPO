# VERPO-ZPD

Portable VERPO-ZPD training repository with two backends:

- native veRL under `verl/` for the formal SDPO Section 3 five-dataset matrix;
- a dependency-light TRL-compatible JSONL/math-text path under
  `pipeline/trl/run.sh`.

Both backends use the same Fixed/CTR/FEC displacement definitions, Forward- or
Reverse-KL, Top-K support, frozen/snapshot/EMA Teacher state, length-aware
reward, gradient audit, checkpoint retention, and semantic configuration
projection. Large datasets, checkpoints, and rollout artifacts stay outside
Git.

## Native veRL

Resolve a cell without side effects:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_1_7b --finetuning full --hardware a100_8x_80gb \
  --protocol sdpo_section3_biology --teacher frozen --arm fixed_fkl \
  --print-config
```

Resolve all 90 registered cells with `--matrix sdpo_five_dataset_teacher_arm`.
Formal training requires `SWANLAB_API_KEY` from the host environment. The
public launcher verifies the complete SDPO bundle and pinned model before
starting training.

## TRL JSONL smoke

Each JSONL row needs `prompt` and `completion` (or `response`). The TRL path
does not adapt Parquet SDPO data:

```bash
bash pipeline/trl/run.sh \
  --train-file fixtures/math.jsonl --output-dir outputs/trl-smoke
```

The tiny CPU fixture is deterministic and writes a checkpoint plus metrics.

## Boundary and provenance

Archived RLCSD launchers and manifests are under `archive/rlcsd/`; they are not
accepted by the public launcher. Migration provenance records the source
snapshot (`82c1df4`), vendored veRL baseline (`e7e052ab`), file hashes, and
whether each custom file was retained, rewritten, or archived in
`provenance/migration_manifest.json`.
