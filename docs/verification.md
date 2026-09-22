# Verification record — 2026-09-22

Base checkout: `cf2b681686034f3fb7713b43183b45f8187c3b00`.
Implementation branch: `codex/public-repro-rollout-evidence`.
Local worktree at verification: `E:/VERPO-public`. This record describes the
pre-commit CPU checks; Git history records subsequent delivery commits.

## Executed locally

Environment: Windows, Python 3.12.13, Git Bash, and the versions in
`requirements-check.txt`. A pending torch installer was cancelled at the user's
request before installation. `importlib.util.find_spec('torch')` returned absent.

- `.venv/Scripts/python.exe -m pytest tests -q`: **41 passed, 1 skipped** in 33.66 s.
  The skipped module is `test_tensor_contract_optional.py`; its tensor checks
  were not executed. Passing checks cover same-group selection and permutation
  stability, target exclusion, unavailable evidence, every registered config axis
  and matrix through engine dry-run, the three-backbone paper matrix, runtime
  identity protection, CLI help, data conversion and result summarization.
- `.venv/Scripts/python.exe -m compileall -q risk_aware_opsd scripts tests
  verl/verl/trainer/distillation/verpo_protocol.py
  verl/verl/trainer/ppo/v1/trainer_base.py verl/verl/workers/config/actor.py
  verl/verl/workers/engine_workers.py`: passed (syntax only).
- Focused `ruff check` on changed public launcher/data/evidence/result/audit modules
  and root tests: passed. This is not a lint certification of the entire vendored tree.
- `bash -n` run separately on `pipeline/verl_math/run.sh`,
  `pipeline/verl_math/engines/sdpo_section3.sh`, and `pipeline/trl/run.sh`: passed.
  The public Bash wrapper also passed a Qwen3-4B `--print-config` invocation.
- `python -m scripts.prepare_sdpo_data --source-dir data/upstream
  --data-dir data/SDPO/verl`, followed by `--verify-only`: passed.
  All ten pinned upstream SHA256 values and row counts were checked. Twenty
  generated JSONL/Parquet files were verified: 7,947 train + 502 test records.
- `python -m scripts.summarize_results --input results/example_run_input.json
  --output-dir outputs/example_summary`: generated Markdown and machine-readable
  summaries from the **synthetic** fixture, not a model experiment.
- `python -m scripts.check_optional_tensor_tests`: explicitly reported `not_run`
  because torch and other tensor-runtime dependencies are absent.
- `git diff --check`: passed. Citation YAML parsed with 11 authors; the main-table
  transcription contains 24 model/method rows. Local documentation link check: zero missing files.

## Not executed / unavailable

- Synthetic model forward/backward smoke and all tensor/gradient/Top-K numerical tests.
- GPU dependency installation, FSDP/vLLM startup, actual EMA update/resume behavior,
  distributed evidence routing, training and generation audits, and GPU memory fit.
- Historical raw predictions, run IDs, selected steps and source-precision curves
  corresponding to every paper-table cell. These fields remain null in the archive.
- Hosted GitHub Actions execution was not part of these local checks. Its default job also does not install torch.

These gaps limit the evidence: this delivery verifies public CPU contracts and
data preparation, not paper-score reproduction or end-to-end training correctness.
