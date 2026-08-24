# Semantic Configuration

The resolver composes one base file with six overlays:

`model`, `finetuning`, `hardware`, `protocol`, `teacher`, and `arm`.

The public native veRL protocols are `sdpo_section3_biology`,
`sdpo_section3_chemistry`, `sdpo_section3_material`, `sdpo_section3_physics`,
and `sdpo_section3_tooluse`. The registered five-dataset matrix expands these
five protocols, three Teacher modes (`frozen`, `snapshot10`, `ema_095`), and
six Fixed/CTR/FEC FKL/RKL arms into 90 unique cells.

`method` is the shared VERPO contract. `zpd` contains only group/sibling
selection and evidence-scope controls. The schema contains no legacy routing
fields.

Use `pipeline/verl_math/run.sh --print-config` before allocating GPUs. The
launcher writes `semantic_config.json`, `native_verl_projection.json`, and
`trl_verpo_projection.json` under each output's `formal/provenance/` directory.
RLCSD material lives under `archive/rlcsd/` and cannot be selected by the public
launcher.
