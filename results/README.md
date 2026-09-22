# Results and provenance

- [paper_v2_table2.json](paper_v2_table2.json): transcription of arXiv v2 Table 2,
  including Base, GRPO, SDPO, SRPO, RLSD, RLCSD, LW and AM rows for three backbones.
- [example_run_input.json](example_run_input.json): synthetic input demonstrating
  the result summarizer, never a training result.
- [Reproduction guide](../docs/reproduce_paper.md): current commands and protocol.

Paper scores are rounded percentages. The recorded Average is the paper's reported
value and must not be reconstructed as though rounded cells were the original
precision. The source table uses per-task evaluation maxima, with the test split
also used for selection; collapsed runs retain their pre-collapse peaks and markers.

Historical run IDs, seeds, selected steps, code/config identities and raw prediction
paths are unavailable in this public transcription. These fields are null, not
inferred from similar-looking local experiments. No checkpoints, raw training-loss
logs, per-step evaluation curves or per-record model responses are included here.
No historical entry is presented as measured with the new rollout-only code.

To generate a report from real runs, provide a `runs` list as in the example,
including model/method/task/seed, run ID, configuration identity, evidence source,
evaluation split/decoding, predictions path and points (`step`, `score`, optional
`optimizer_updates`). Scores for this interface are fractions in [0,1].

```bash
uv run --no-sync python -m scripts.summarize_results \
  --input results/example_run_input.json --output-dir outputs/example_summary
```

Outputs are `report.md` and `metrics.json`. They preserve status and input metadata,
report per-task best (ties earlier, step 0 excluded), fixed step 200, and the mean
and sample standard deviation of observed evaluation points at steps 150–200.
The late-window standard deviation is across time points, not uncertainty across
seeds. Missing tasks/steps produce null, never zero or a partial five-task average.
Runs with different seeds or evidence sources are not pooled. Multiple candidates
for one task/seed/protocol must be selected explicitly by the caller.
