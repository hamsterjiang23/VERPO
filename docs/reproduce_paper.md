# Reproducing VERPO

## What this release establishes

This is an executable configuration and provenance contract for the rollout-group
evidence protocol. CPU checks and the paper transcription do not establish
reproduction of historical scores. No GPU training was run for this delivery.

The public release was inspected at `cf2b681`; implementation changes are not
committed or pushed automatically. Record the final checkout commit and dirty
status when launching an experiment.

## Evidence contract

For every target, select the first other verifier-correct, format-valid, nonempty
response in the same group, ordered by numeric rollout session ID then stable ID.
Do not use shaped reward to infer correctness. Exclude the target from both positive
and negative pools. The positive Teacher inserts exactly the selected response in
`Correct solution`; each negative Teacher uses `Incorrect candidate solution`.
All branches replay the identical target suffix. Ground truth and offline solutions
are never copied into Teacher evidence or Student/evaluation prompts.

Fixed requires positive evidence. CTR/FEC require both positive and negative
evidence. With missing required evidence, reference and GRPO remain active,
evidence correction is zero and the AM multiplier is one. Thus a group with one
correct rollout cannot give positive evidence to that rollout itself; a group
with one wrong rollout cannot give negative evidence to that wrong rollout.
No cross-batch pool, self evidence, external answer fallback or automatic group
regeneration is enabled by the paper profiles.

`method.evidence_source=rollout_group` is part of the semantic config/hash. The
trainer writes `rollouts/evidence_selection/step_N.jsonl` with selected IDs,
correctness, format, missing-evidence reason and truncation counts. Source texts
are retained in the corresponding rollout output. `q+` denotes replayed Teacher
probabilities, not the selected evidence string. Reference and q0 share the
existing evidence-free Teacher shadow under EMA; the reference loss has its own
normalization and stays independent of the evidence availability mask.

## Registered commands

Run `paper_main` for each of `qwen3_4b`, `qwen3_8b`, and
`llama3_2_1b_instruct` with `full`, `a800_8x_80gb` and the command below:

```bash
bash pipeline/verl_math/run.sh --model qwen3_4b --finetuning full \
  --hardware a800_8x_80gb --matrix paper_main --print-command
```

The matrix contains five tasks × `fec_fkl` (LW) / `fec_advmod_fkl` (AM).
`paper_combined`, `paper_ablations`, and `paper_baselines` are separate matrices.
The matrix config composes before cell overrides; explicit `--set` comes last.
Alternate overrides create a different configuration hash and are exploratory.

Shared settings: 200 **trainer steps**, 32 prompts × 8 rollouts, 32 trajectories
per optimizer minibatch, one PPO epoch, validation every 5, validation before
training, 16 evaluation samples, EMA decay .95, reference coefficient .1,
Top-K 128 per branch, cost alpha .0025 and floor .000025. LW uses evidence
coefficient 1; AM uses evidence coefficient 0 and advantage multiplier coefficient
1; combined uses both 1. Teacher/Student/evaluation thinking switches are false.
Seed and data seed default to 42. This does not promise deterministic replay.

The paper appendix labels the budget as optimizer steps. Here the agreed
compatibility setting is 200 trainer steps; one trainer step normally contains
8 successful optimizer minibatch updates. `verpo/optimizer_updates_max` logs the
Teacher's actual cumulative successful update count (maximum across minibatches
and workers), including restored Teacher state on resume. Skipped updates must not
be counted as successes. Its GPU behavior remains unverified in this delivery.

Save every 50 trainer steps with no checkpoint rotation. Retain every evaluation
and training response. The paper matrix's checkpoint cadence need not coincide
with the every-five-step evaluation points used in the historical paper table.
Do not imply every historical best-evaluation point has a stored checkpoint.

## Runtime overrides and resume

`TRAIN_FILE` / `VAL_FILE` override `SDPO_DATA_DIR`, which overrides the default data
root. Explicit file overrides disable automatic data preparation. They may point
to legacy annotated Parquet files; solution metadata is ignored by the active
Teacher builder. `MODEL_PATH` selects an existing pinned model snapshot.
`MODEL_CACHE` / `HF_HOME` control downloads. `OUTPUT_ROOT` controls a single cell;
`MATRIX_OUTPUT_ROOT` overrides the default matrix root. When setting a custom
matrix root, choose a separate root per backbone/profile.

Default matrix roots include matrix/model/finetuning/hardware; each cell has its
own directory. Nonempty outputs without matching semantic provenance are refused.
After assets are prepared, runtime identity records effective data paths and
SHA256 values, model path/revision and model-config SHA256. Changed identities
cannot resume an existing checkpoint directory.

Before training, provision `SWANLAB_API_KEY` through the host environment, review
the dry-run, then remove `--print-command`. Never invoke the engine script directly.
No paper profile is a resource estimate or a claim that it fits every GPU runtime.

## Scores and selection

Historical Table 2 uses each task's best observed evaluation score, with the test
split also used during training for selection. It is not a held-out final-checkpoint
result. The new summarizer excludes initial step 0 for trained runs, breaks ties
by earlier step, reports fixed-step and late-window statistics separately, and
requires all five tasks for an average. It never chooses a best seed/run implicitly.

```bash
uv run --no-sync python -m scripts.summarize_results \
  --input results/example_run_input.json --output-dir outputs/example_summary
```

The example is a labeled synthetic fixture, not an experiment. Import real
per-step points with the schema shown there, preserve raw prediction paths and
record their provenance. Missing historical identifiers remain null; paper-rounded
scores must not be presented as original source precision. See [results](../results/README.md).
