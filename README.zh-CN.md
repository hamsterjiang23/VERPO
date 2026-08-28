# VERPO-ZPD

[English](README.md) | [简体中文](README.zh-CN.md)

**VERPO：Verified Evidence-Regularized Policy Optimization（验证证据正则化策略优化）** ——
基于证据感知的策略优化方法，通过 Teacher 引导的、token 级别的分布修正实现。

VERPO 将 GRPO 策略目标与一个带符号修正项结合，该修正项由不同证据条件下的 Teacher
分布推导而来。修正由两级机制控制：prompt 组级别的门控决定该组是否包含有用的奖励
差异，token 级别的 ZPD 控制器则按局部收益与策略移动代价对每个证据方向加权缩放。

本实现围绕一套语义化配置和两种训练后端组织：原生 veRL 提供 Section 3 协议启动器；
TRL 路径提供小型 JSONL / 数学文本接口，用于可移植的 VERPO 实验。

## 目录

- [方法](#方法)
- [变体与控制项](#变体与控制项)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [快速开始：TRL（JSONL）](#快速开始trljsonl)
- [快速开始：原生 veRL](#快速开始原生-verl)
- [语义化配置](#语义化配置)
- [环境变量](#环境变量)
- [数据与凭据](#数据与凭据)
- [引用](#引用)
- [许可证](#许可证)
- [贡献者](#贡献者)
- [致谢](#致谢)

## 方法

```text
prompt
  -> Student rollout
  -> Teacher scoring under evidence conditions
  -> signed Teacher displacement
  -> ZPD benefit/cost controller
  -> GRPO objective + VERPO correction
  -> updated Student policy
```

对于固定的 rollout 前缀，Teacher 分支表示为 token 分布：

- `q0`：无证据 Teacher 分布；
- `qe`：证据条件 Teacher 分布；
- `q+`、`q-`：正 / 负对比 Teacher 分布。

带符号的 Teacher 位移是各主要 VERPO 变体之间唯一的区别：

```text
Fixed:  Delta_t = q_e,t - q_0,t
CTR:    Delta_t = q_+,t - q_-,t
FEC:    任务方向减去干扰方向的 Fisher 投影
```

对于默认的 Forward-KL 控制器，证据修正按如下权重缩放：

```text
w_t = h_t / (h_t + tau * (c_t + rho))
```

其中 `h_t` 是局部证据收益，`c_t` 是局部策略移动代价。FEC 方向在计算带符号修正
之前，先移除证据存在性干扰在 Student 局部 Fisher 意义下的投影。

后端无关的训练器将参考项与证据项组合为：

```text
L_VERPO = lambda_ref * L_ref + lambda_evi * L_evi
```

精确的全词表定义与选取子集的 Top-K 近似实现在 `risk_aware_opsd/verpo_zpd.py` 中。

## 变体与控制项

| 变体 | Teacher 位移 | 用途 |
| --- | --- | --- |
| Fixed | `q_e - q_0` | 证据条件修正 |
| CTR | `q_+ - q_-` | 正 / 负对比修正 |
| FEC | 对比任务方向减去干扰投影 | 去除证据混杂 |
| Forward-KL | 概率空间修正 | 前向散度路径 |
| Reverse-KL | 几何 Teacher 路径与 Student 局部 Fisher 切向 | 反向散度路径 |
| Top-K | 选取子集近似 | 内存友好的词表计算 |

Teacher 状态可配置为：

- `frozen`：固定的初始 Teacher；
- `snapshot10`：按配置的优化器更新间隔同步；
- `ema_095`：以指数滑动平均更新 Teacher 状态。

共享的 `VERPOConfig` 暴露位移模式、Teacher 模式、词表模式、组级别 ZPD、证据 rollout
范围、散度、系数以及 Teacher 同步控制项。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `risk_aware_opsd/` | 后端无关的 VERPO 损失、ZPD 控制器、Teacher 状态、奖励与 JSONL 契约 |
| `scripts/` | 启动、打包与审计工具 |
| `verl/verl/trainer/distillation/` | 原生 veRL 的 VERPO 实现（vendored veRL 树） |
| `pipeline/trl/` | JSONL / 数学文本 TRL 入口 |
| `pipeline/verl_math/` | 原生 veRL 启动器与引擎脚本 |
| `configs/verpo/` | 语义化模型、协议、Teacher、arm 与矩阵配置 |
| `archive/rlcsd/` | 不属于活跃 VERPO 启动路径的历史材料 |
| `provenance/` | 文件级可复现性元数据 |

对外训练接口是启动器层。内部引擎脚本是实现细节，不应直接调用。

## 安装

请使用仓库中已提交的 `uv.lock` 文件。CPU / 开发依赖与 GPU 训练环境相互独立：

```bash
# CPU 开发环境（可移植 TRL 实验、配置解析）
uv sync --extra cpu --extra dev

# 具备匹配 CUDA、veRL 与 rollout 运行时的主机
uv sync --extra gpu
```

原生 veRL GPU 环境另行在 `pipeline/verl_math/requirements.txt` 中锁定。

不要提交 `.env` 文件、数据集、模型权重、checkpoint、日志或生成的输出目录。

## 快速开始：TRL（JSONL）

TRL 接口接受本地 JSONL / 数学文本记录。每行包含 `prompt` 与 `completion`；
`response` 可作为 `completion` 的替代字段。

```json
{
  "prompt": "local prompt text",
  "completion": "local completion text"
}
```

安装 CPU 开发环境后，运行单步冒烟测试：

```bash
uv sync --extra cpu --extra dev

bash pipeline/trl/run.sh \
  --train-file path/to/private/train.jsonl \
  --output-dir outputs/trl-verpo \
  --max-steps 1
```

该路径刻意只支持 JSONL / 数学文本。Parquet 协议数据属于原生 veRL 入口。

## 快速开始：原生 veRL

在不占用 GPU 的情况下解析一个 Section 3 cell：

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_1_7b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --protocol sdpo_section3_biology \
  --teacher frozen \
  --arm fixed_fkl \
  --print-config
```

在正式启动前检查完整注册的配置矩阵：

```bash
bash pipeline/verl_math/run.sh \
  --model qwen3_1_7b \
  --finetuning full \
  --hardware a100_8x_80gb \
  --matrix sdpo_five_dataset_teacher_arm \
  --print-command
```

该矩阵由五个 Section 3 协议、三种 Teacher 状态、六种 Fixed/CTR/FEC
Forward-/Reverse-KL arm 组合配置而成。它定义启动 cell，并不是评估表格。

正式训练需要宿主环境提供 `SWANLAB_API_KEY`。任何消耗资源的启动之前，请先用
`--print-config` 或 `--print-command` 检查。

## 语义化配置

语义化配置由 model、finetuning、hardware、protocol、Teacher 与 arm 六层 overlay
组合而成。字段参考与可用标识符见 [`configs/verpo/README.md`](configs/verpo/README.md)。

每次解析后的启动可以在其输出 provenance 目录下生成语义化配置与各后端投影。
`archive/rlcsd/` 目录不属于活跃的对外启动路径。

## 环境变量

数据、模型权重、checkpoint、日志、缓存与凭据是操作者输入，本仓库不提交这些内容。
只配置本地路径与主机托管的凭据：

| 变量 | 含义 |
| --- | --- |
| `SDPO_DATA_DIR` | 操作者提供的本地数据根目录 |
| `TRAIN_FILE` / `VAL_FILE` | 显式的本地文件覆盖 |
| `MODEL_PATH` | 本地模型快照 |
| `MODEL_CACHE` / `HF_HOME` | 模型缓存 |
| `SWANLAB_API_KEY` | 主机注入的 SwanLab 凭据 |
| `OUTPUT_ROOT` | 单 cell 输出根目录 |
| `MATRIX_OUTPUT_ROOT` | 矩阵输出根目录 |

## 数据与凭据

本 README 有意省略数据分发细节、存储标识、云存储、私有清单与内部资产准备流程。
切勿将 `SWANLAB_API_KEY` 写入命令、YAML、README 示例、清单或日志。

## 引用

论文参考文献将在正式发表后补充于此。在此之前，请引用本仓库：

```bibtex
@misc{verpo-zpd,
  title        = {{VERPO-ZPD}: Verified Evidence-Regularized Policy Optimization},
  author       = {hamsterjiang23 and thomass1003},
  year         = {2026},
  howpublished = {\url{https://github.com/hamsterjiang23/VERPO-ZPD}},
}
```

## 许可证

仓库根目录的项目源代码尚未声明许可证。`verl/` 下的 vendored veRL 树保持其原始的
[Apache License 2.0](verl/LICENSE)。

## 贡献者

| 贡献者 | 角色 | GitHub |
| --- | --- | --- |
| hamsterjiang23 | 维护者 | [@hamsterjiang23](https://github.com/hamsterjiang23) |
| thomass1003 | 贡献者 | [@thomass1003](https://github.com/thomass1003) |

## 致谢

原生后端基于 [veRL](https://github.com/volcengine/verl) —— 由字节跳动 Seed 团队发起、
veRL 社区维护的 RL 训练框架。