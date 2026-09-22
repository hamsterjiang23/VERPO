# Native veRL launcher

Use `pipeline/verl_math/run.sh` from the repository root. Engine scripts are internal.

```bash
bash pipeline/verl_math/run.sh --model qwen3_4b --finetuning full \
  --hardware a800_8x_80gb --matrix paper_main --print-command
```

`--print-config` composes semantic YAML only. `--print-command` validates the same
engine argument path used for training. Neither starts dependency installation,
downloads, credential checks, nor GPU allocation. Remove the print flag only on a
prepared Linux GPU host with its SwanLab credential supplied through the environment.

The active evidence source is `rollout_group`. No offline solution annotation is
required. Public data preparation uses the pinned ten-file source manifest;
explicit `TRAIN_FILE`/`VAL_FILE` paths override automatic preparation.

Default matrix paths include model, finetuning and hardware. Different cells have
separate directories; config and effective data/model identity checks prevent
incompatible checkpoint resume. Training logs record actual Teacher optimizer
update counts separately from trainer steps.

- [Reproduction and runtime overrides](../../docs/reproduce_paper.md)
- [Data download, counts and checksums](../../docs/data.md)
- [Semantic configuration](../../configs/verpo/README.md)
- [CPU verification boundary](../../docs/verification.md)
