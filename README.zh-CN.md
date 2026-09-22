# VERPO

[English](README.md) | [简体中文](README.zh-CN.md)

**Verified Evidence Regularized Policy Optimization**

[论文 v2](https://arxiv.org/abs/2609.06100v2) · [复现说明](docs/reproduce_paper.md) · [数据准备](docs/data.md) · [结果及来源](results/README.md)

VERPO 保留 GRPO 的 outcome objective，用停止梯度的 token controller 决定接受多少证据修正。
Fixed、CTR、FEC 定义不同修正方向；LW 加权证据损失，AM 调制 token advantage。

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

## 论文报告结果

以下为论文 v2 表 2 转录，**不是本次 checkout 的测量结果**。
论文使用 test split 记录训练期间评估并选择每个任务的最高分，Average 是五个任务各自最高分的非加权平均。
完整转录保留 collapse 和 reward hacking 标记。

| Backbone | VERPO-LW（%） | VERPO-AM（%） |
|---|---|---|
| Qwen3-4B | 68.57 | 66.71 |
| Qwen3-8B | 71.44 | 70.58 |
| Llama-3.2-1B | 56.57 | 55.19 |

历史 run ID、选中 step、原始预测和未舍入指标尚未逐项对应，因此结果文件中这些字段为 null。
新 rollout-only 运行采用独立证据来源标识，不自动继承历史分数。

## 引用与许可

论文 DOI：`10.48550/arXiv.2609.06100`。完整 BibTeX 见[英文首页](README.md#citation-and-license)，机器可读引用见 [CITATION.cff](CITATION.cff)。
主项目新增代码采用 [Apache-2.0](LICENSE)，第三方代码保留原版权和许可声明，见[第三方清单](THIRD_PARTY_NOTICES.md)。
