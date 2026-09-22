# Implementation notes

## Deviations

- User stopped torch installation. The pending installer was cancelled before torch was installed. Local verification uses stdlib/PyYAML tests and syntax checks only; tensor/gradient tests are supplied but not claimed as executed.

- Work uses an isolated public checkout at `cf2b681`, not the private PGR-Probe working tree. No historical metrics will be attributed to the new rollout-only evidence protocol without matching run provenance.
- Historical artifacts could not be matched to every paper cell. Published scores are explicitly transcribed; unknown run/seed/config/step/prediction fields remain null. No GPU validation or historical reproduction is claimed.
- The old engine's hard-coded historical locks were replaced by composed semantic validation and execution invariants. Known paper matrices and legacy profiles are tested through the same argument construction.

## Discovered edge cases

- A sole correct rollout cannot supply its own positive evidence; a sole wrong rollout cannot supply its own negative evidence. Fixed and contrastive availability therefore differ.
- The old launcher drops semantic parameters, rejects its GRPO enum, and checks historical hard-coded defaults rather than the composed configuration.
- TransferQueue rollout ordinal is the numeric session component, not the output-turn suffix; numeric ordering is tested across batch permutations and session IDs above nine.
- Matrix output paths must distinguish backbones. Resume checks must include actual prepared data hashes/model identity, not only the semantic config hash.
- Data preparation needs PyArrow after runtime setup. Single-row JSONL is also a valid JSON object; both representations are accepted.
- Registered A800 two-card utilization was 0.88; it is now capped at the authorized 0.60 ceiling. Unrecognized and nonfinite overrides fail closed.

## Questions for review

- GPU execution and historical per-record prediction availability remain outside CPU verification. The subsequent user request authorizes committing and pushing this implementation branch.

## Handoff

- Deviations: four, including the explicit no-torch testing constraint.
- Most likely follow-up: provision tensor/GPU verification in an authorized environment.
- Edge cases: six groups documented above; regression tests cover the CPU-observable ones.
- Read `docs/verification.md` for exact passed checks and unexecuted runtime checks.
- Read `docs/reproduce_paper.md` before any formal training or result claims.
