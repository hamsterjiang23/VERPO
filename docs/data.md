# Public data preparation

Source: [lasgroup/SDPO at 7c457fc](https://github.com/lasgroup/SDPO/tree/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/datasets).
The ten raw file hashes are pinned in [the source manifest](../configs/sdpo_source_manifest.json).
No annotation API, Google Drive credential or generated solution is required.

| Task | Train | Test |
|---|---|---|
| Biology | 450 | 50 |
| Chemistry | 1890 | 210 |
| Material | 841 | 94 |
| Physics | 720 | 80 |
| Tool use | 4046 | 68 |

```bash
uv run --no-sync python -m scripts.prepare_sdpo_data --data-dir data/SDPO/verl
uv run --no-sync python -m scripts.prepare_sdpo_data --data-dir data/SDPO/verl --verify-only
```

For offline use, `--source-dir /path/to/upstream-copy` reads the same pinned
`datasets/.../train.json` and `test.json` paths and verifies identical hashes.
`--jsonl-only` avoids the PyArrow dependency for inspection; native training needs
Parquet. Invalid installed files are replaced only after all new files are staged
and checked. The manifest is installed last and every future use verifies it.

Output files are `<task>.train.jsonl`, `<task>.test.jsonl` and matching `.parquet`
files. `rollout_manifest.json` records counts, SHA256, source revision and protocol.
There are 7,947 train and 502 test records, without annotation acceptance filtering.
The old privileged-context bundle is historical and is not this data release.

Each record has a stable `record_id`, upstream `prompt` messages, `data_source`,
`ability`, `reward_model.style`, verifier-only `reward_model.ground_truth` and
`extra_info` with dataset, split, upstream index and source revision. There is no
required `solution` field. Original system/user text is preserved; the selected
training prompt profile may add its documented Tool Use system message.
Dataset-derived files are ignored by Git and retain their upstream terms.
