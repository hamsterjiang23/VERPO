# VERPO

[English](README.md) | [简体中文](README.zh-CN.md)

**Verified Evidence Regularized Policy Optimization**

[![arXiv](https://img.shields.io/badge/arXiv-2609.06100-b31b1b.svg)](https://arxiv.org/abs/2609.06100) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[论文](https://arxiv.org/abs/2609.06100) · [PDF](https://arxiv.org/pdf/2609.06100) · [引用](#引用与许可) · [复现说明](docs/reproduce_paper.md) · [数据准备](docs/data.md) · [结果及来源](results/README.md)

本仓库提供论文 **[VERPO: Verified Evidence Regularized Policy Optimization](https://arxiv.org/abs/2609.06100)** 的实现。

> Haijiang Li, Chengyu Lv, Yi Zhang, Rui Qian, Zhibing Zhang, Xiangqing Shen, Junjie Yang, Yuchen Zhang, Wenyuan Jiang, Hanqing Hu, Cangqi Zhou。
>
> *arXiv:2609.06100*，2026。[DOI: 10.48550/arXiv.2609.06100](https://doi.org/10.48550/arXiv.2609.06100)

VERPO 保留 GRPO 的任务奖励目标，通过带有证据的 Teacher 回放，为逐 token 修正提供指导。
它将监督拆成无证据参考项和带符号的证据修正项，再用停止梯度的控制器，平衡修正方向与奖励方向的一致性以及 Fisher 移动代价。

- **Fisher Evidence Contrast（FEC）**：结合正确证据、错误证据和无证据 Teacher 分支，减弱由证据出现本身引起的分布偏移。
- **两种更新路径**：VERPO-LW 加权证据损失；VERPO-AM 调制 token advantage；另有配置同时启用两条路径。
- **实验设置**：使用 Qwen3-4B、Qwen3-8B 和 Llama-3.2-1B，覆盖生物、化学、材料科学、物理和工具调用五类任务。

## 目录

- [方法概览](#方法概览)
- [当前实现与验证边界](#当前实现与验证边界)
- [最小入口](#最小入口)
- [论文配置](#论文配置)
- [引用与许可](#引用与许可)

## 方法概览

流程：原问题 → 同组 Student rollouts → verifier 判定正确性和格式 → 排除目标自身选取正负证据 → Teacher 回放目标前缀 → FEC 与 token controller → 联合更新。

## 当前实现与验证边界

正式训练入口为原生 veRL。当前版本统一从**同一 prompt group 的其他 rollout** 选择证据，不回退到数据集 solution 或标准答案。标准答案仅供 verifier 使用。
选中的正确回答是证据文本，`q+` 是 Teacher 在该证据条件下回放目标回答前缀得到的分布；`q0` 和 reference 不含证据。EMA 模式沿用 moving Teacher shadow 作为这两个分支，reference 损失不受证据门控关闭。

`pipeline/trl/` 目前只提供微型模型和合成 token 的 **synthetic smoke**，不代表真实预训练模型的 TRL/GRPO 训练支持。
默认 CI 不安装 torch；张量测试在缺少 torch 时明确跳过。CPU 检查不能证明 GPU 训练或论文分数已经复现，详见[验证记录](docs/verification.md)。

## 最小入口

原生 GPU 训练使用 Linux 和 Bash。安装 uv 后：

```bash
uv venv --python 3.12
uv pip install -r requirements-check.txt
uv run --no-sync python -m pytest tests -q

bash pipeline/verl_math/run.sh \
  --model qwen3_4b --finetuning full --hardware a800_8x_80gb \
  --matrix paper_main --print-command
```

`--print-config` 只合成配置；`--print-command` 运行真实引擎参数检查并输出实际 Hydra 配置。
两者均不下载数据或模型，不检查凭据，不分配 GPU。

正式启动前，在服务器环境中配置 `SWANLAB_API_KEY`，审核 dry-run，然后移除 `--print-command`。
启动器会准备 GPU 运行环境、固定版本的数据和模型。Llama 模型可能需要 Hugging Face 访问授权。

## 论文配置

| 标识 | 用途 |
|---|---|
| `paper_main` | 五任务 × LW/AM，三个 backbone 分别运行 |
| `paper_combined` | LW+AM 联合路径 |
| `paper_ablations` | Fixed/CTR/FEC FKL 与 FEC RKL |
| `paper_baselines` | GRPO、SDPO、SRPO |
| `fec_fkl` | LW；论文矩阵指定 λ_evi=1.0 |
| `fec_advmod_fkl` | AM；λ_evi=0，advantage 系数为 1 |

论文矩阵固定 **200 个 trainer step**、每 5 步验证、每 50 步保存并保留全部 checkpoint、EMA 0.95。
200 trainer steps 与 200 optimizer updates 不等价；现有一般用途配置保留其 300-step 预算。
公开版本未提供 RLSD/RLCSD 主表结果对应的可运行矩阵，不能将论文转录视作实现验收。

## 引用与许可

如果在研究中使用 VERPO，请引用以下论文：

**[VERPO: Verified Evidence Regularized Policy Optimization](https://arxiv.org/abs/2609.06100)**，
Haijiang Li 等，arXiv:2609.06100（2026）。
以下 BibTeX 与 [CITATION.cff](CITATION.cff) 均使用 arXiv 的统一入口。

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

主项目新增代码采用 [Apache-2.0](LICENSE)，第三方代码保留原版权和许可声明，见[第三方清单](THIRD_PARTY_NOTICES.md)。
