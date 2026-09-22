# Third-party notices

Project additions are licensed under Apache-2.0; this does not replace licenses
on upstream code, datasets, model weights or the paper.

- **veRL**: vendored under `verl/`; retain its [Apache-2.0 license](verl/LICENSE)
  and file-level ByteDance and other contributor copyright notices.
- **Prompt/grading adaptations**: `verl/verl/trainer/distillation/verpo_protocol.py`
  retains its original MIT attribution and permission notice. Its header records
  source revision `3da75d2209317d7f8fc7a89f3f65b8ad5cc4e4c0`; that historical
  source attribution has not been independently re-audited in this delivery.
- **SDPO data**: downloaded from the pinned `lasgroup/SDPO` revision recorded in
  `configs/sdpo_source_manifest.json`. Science data derive from SciKnowEval and
  Tool Use from ToolAlpaca. Source datasets remain under their upstream terms;
  generated files are not committed or relicensed by this project.
- **Models**: Qwen and Llama weights are fetched separately and retain their model
  licenses/access requirements. No model weight files are included.

Other notices already present inside the vendored tree remain authoritative for
those files. This inventory identifies directly touched components; it is not an
exhaustive license audit of every upstream dependency.
