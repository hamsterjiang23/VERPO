# Semantic configuration

Compose `base.yaml` with model, finetuning, hardware, protocol, Teacher and arm.
Matrix overrides precede per-cell overrides; CLI `--set PATH=VALUE` is applied last.
Only the five `sdpo_section3_*` protocol names are supported by the public launcher.

The paper matrices are `paper_main` (LW/AM), `paper_combined`, `paper_ablations`,
and `paper_baselines`. Select each backbone explicitly with `--model`; paper
matrices use 200 trainer steps, validation every 5 and retained checkpoints every
50. General profiles retain historical 300-step defaults and their coefficients.

`method.evidence_source=rollout_group` and `zpd.sibling_selection=correctness` are
required. `zpd.evidence_scope` is `all` or `wrong_only`. Missing required siblings
mask evidence corrections without changing reference or GRPO. The source tag is
included in each configuration hash and persisted provenance.

Method options include divergence, displacement, reference/evidence coefficients,
AM mode/coefficient, Top-K and controller cost. Protocol options include rollout
and validation decoding, batch sizes, seed, trainer budget and checkpoint cadence.
Hardware options include rollout backend, tensor parallelism and memory utilization
(no more than 0.60 in colocated runs). Gradient-audit activation is not supported
by this public launcher and is explicitly rejected.

`--print-config` is a pure YAML inspection. `--print-command` evaluates the actual
shell argument construction without installing/downloading/launching anything.
The native projection is parsed from those same Hydra arguments, not a second
independently-maintained configuration. Unsupported/nonfinite values fail closed.

Run provenance includes semantic JSON/YAML, actual native projection, legacy
synthetic-smoke projection, code commit/dirty state/source hashes, and prepared
runtime data/model identity. Changing explicit data/model inputs cannot silently
resume existing checkpoints. No credentials are included in these files.

See [the reproduction guide](../../docs/reproduce_paper.md) and
[data contract](../../docs/data.md). Historical privileged-context data and RLCSD
archives do not authorize current training or establish historical score provenance.
