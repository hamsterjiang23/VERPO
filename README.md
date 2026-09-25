# VERPO: Verified Evidence Regularized Policy Optimization

[English](README.md) | [简体中文](README.zh-CN.md)

[![arXiv](https://img.shields.io/badge/arXiv-2609.06100-b31b1b.svg)](https://arxiv.org/abs/2609.06100) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2609.06100) · [PDF](https://arxiv.org/pdf/2609.06100) · [Citation](#citation-and-license) · [Reproduction guide](docs/reproduce_paper.md) · [Data](docs/data.md)

This repository provides the implementation of our paper, [VERPO](https://arxiv.org/abs/2609.06100).

**Authors:** Haijiang Li, Chengyu Lv, Yi Zhang, Rui Qian, Zhibing Zhang, Xiangqing Shen, Junjie Yang, Yuchen Zhang, Wenyuan Jiang, Hanqing Hu, Cangqi Zhou

**Contact:** [hj523hj@163.com](mailto:hj523hj@163.com)

## Abstract

Verifiable rewards improve language models through reliable task-level feedback, but methods based on Group Relative Policy Optimization (GRPO) apply a sequence-level advantage uniformly across all tokens. This coarse credit assignment reinforces or penalizes entire responses without identifying which local decisions to preserve, reinforce, or revise. Conversely, evidence-conditioned self-distillation provides denser token-level supervision, yet teacher imitation can transfer stylistic artifacts and miscalibrated confidence that destabilize training when misaligned with task success.

We introduce VERPO, which converts evidence-conditioned guidance into reward-aligned token-level credit assignment while retaining the outcome objective. VERPO decomposes teacher guidance into an evidence-free reference term and signed, evidence-induced corrections at each token. A stopped controller combines selective acceptance, token-wise localization, and cost-aware scaling by balancing alignment with the local GRPO update direction against Fisher movement cost. Furthermore, we introduce Fisher Evidence Contrast (FEC), which attenuates nuisance shifts along an estimated evidence-presence direction through a regularized projection.

Across five scientific reasoning and tool-use tasks, VERPO prevents optimization collapse and consistently achieves the highest multi-task average across model backbones, yielding marked improvements particularly on smaller models over strong baselines. Qualitative diagnostics confirm that token acceptance selectively targets reasoning bottlenecks consistent with local reward alignment and Fisher movement cost.

*Abstract reproduced from [paper v2](https://arxiv.org/abs/2609.06100v2); the experimental claims describe the paper's reported experiments.*

## Overview

<p align="center">
  <img src="assets/overview.png" width="100%" alt="VERPO overview: rollout sampling and rewards, Teacher replay, the ZPD controller, and advantage modulation or weighted evidence correction." />
</p>

**Method overview.** Figure 3 follows four stages: sample rollouts and obtain rewards; construct FEC directions through Teacher replay; estimate token acceptance with the ZPD controller; then apply advantage modulation or weighted evidence correction.

<p align="center">
  <img src="assets/grpo-verpo.png" width="100%" alt="GRPO versus VERPO: sparse sequence rewards compared with evidence-guided local corrections and adaptive token guidance." />
</p>

**Intuition.** Figure 1 contrasts GRPO's sparse sequence rewards with VERPO's evidence-conditioned token guidance.

Figures are rendered directly from the vector artwork in [paper v2](https://arxiv.org/pdf/2609.06100v2). See [asset sources and license](assets/README.md).

<details>
<summary>Usage guide</summary>

- [Quickstart](#quickstart)
- [Paper configurations](#paper-configurations)
- [Repository map](#repository-map)
- [Citation and license](#citation-and-license)

</details>

## Quickstart

Linux and Bash are required for native GPU training. Install
[uv](https://docs.astral.sh/uv/), then create the lightweight environment:

```bash
uv venv --python 3.12
uv pip install -r requirements-check.txt
uv run --no-sync python -m pytest tests -q
```

Inspect a paper configuration without downloads, credentials or GPUs:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_4b --finetuning full --hardware a800_8x_80gb \
  --matrix paper_main --print-command
```

`--print-config` only composes YAML. `--print-command` runs the real engine's
preflight and prints its actual Hydra projection. For a single LW cell:

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_4b --finetuning full --hardware a800_8x_80gb \
  --protocol sdpo_section3_biology --teacher ema_095 --arm fec_fkl \
  --set protocol.total_training_steps=200 --set method.lambda_evi=1.0 \
  --print-command
```

For formal training, provision a compatible GPU host, review the dry-run, then
remove `--print-command`. Metrics are printed to the console and retained in
`formal/train.log`; no experiment-tracking account is required. The launcher
bootstraps the GPU runtime and prepares the pinned public data/model if missing.
Llama access may require Hugging Face model access authorization. No credentials
belong in commands, config files or logs.

## Paper configurations

| Matrix / arm | Meaning |
|---|---|
| `paper_main` | Five tasks × LW/AM; run once for each of the three backbones |
| `paper_combined` | Five tasks × LW+AM |
| `paper_ablations` | Fixed/CTR/FEC FKL and FEC RKL directions |
| `paper_baselines` | Public GRPO/SDPO/SRPO implementations |
| `fec_fkl` | VERPO-LW; paper matrix sets evidence coefficient to 1.0 |
| `fec_advmod_fkl` | VERPO-AM; evidence coefficient 0, advantage coefficient 1 |
| `fec_advmod_fkl_combined` | Both paths enabled |

The paper matrices use **200 trainer steps**, validation every 5, checkpoints
every 50 without pruning, and EMA decay 0.95. This is not 200 optimizer updates.
The historical general-purpose profiles retain their 300-step budget and their
own coefficients. RLSD/RLCSD paper numbers are transcribed in the results archive;
they do not imply runnable baseline matrices in this public release.

## Repository map

- `risk_aware_opsd/`: losses, semantic configuration, data contract and evidence selection.
- `verl/`: vendored native training implementation and tensor tests.
- `pipeline/verl_math/`: native launcher; engine scripts are internal.
- `pipeline/trl/`: synthetic smoke only.
- `configs/`: pinned data manifest and composed training configurations.
- `tests/`: lightweight public-contract tests; optional tensor tests skip without torch.
- `results/`: source-labeled paper transcription and result-input example.

## Citation and license

If you use VERPO in your research, please cite the paper:

**[VERPO: Verified Evidence Regularized Policy Optimization](https://arxiv.org/abs/2609.06100)**,
Haijiang Li et al., arXiv:2609.06100 (2026).
The following entry and [CITATION.cff](CITATION.cff) use the canonical arXiv URL.

```bibtex
@article{li2026verpo,
  title={VERPO: Verified Evidence Regularized Policy Optimization},
  author={Li, Haijiang and Lv, Chengyu and Zhang, Yi and Qian, Rui and Zhang, Zhibing and Shen, Xiangqing and Yang, Junjie and Zhang, Yuchen and Jiang, Wenyuan and Hu, Hanqing and Zhou, Cangqi},
  journal={arXiv preprint arXiv:2609.06100},
  eprint={2609.06100},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  year={2026},
  doi={10.48550/arXiv.2609.06100},
  url={https://arxiv.org/abs/2609.06100}
}
```

Project additions are licensed under [Apache-2.0](LICENSE). Vendored components
retain their original notices; see [third-party notices](THIRD_PARTY_NOTICES.md).
The native backend builds on [veRL](https://github.com/volcengine/verl).
