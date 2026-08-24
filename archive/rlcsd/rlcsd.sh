#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "Usage: $0 GPUS GLOBAL_BATCH PER_DEVICE_BATCH NUM_EPOCHS GPU_TYPE [forward_kl|reverse_kl] [fixed|ctr|ranked|fec]" >&2
  exit 2
fi

GPUS=$1
GLOBAL_BATCH=$2
PER_DEVICE_BATCH=$3
NUM_EPOCHS=$4
GPU_TYPE=$5
VERPO_DIVERGENCE=${6:-forward_kl}
VERPO_DISPLACEMENT_MODE=${7:-fixed}
TRAINING_OBJECTIVE=${QWEN3_TRAINING_OBJECTIVE:-verpo}
HARDWARE_PROFILE=${QWEN3_HARDWARE_PROFILE:-a100_8x}
MATRIX_PREFLIGHT_ONLY=${MATRIX_PREFLIGHT_ONLY:-0}
PRINT_RESOLVED_CONFIG=${QWEN3_PRINT_RESOLVED_CONFIG:-0}
RLCSD_REUSE_COMPLETED_INITIAL_VALIDATION=${RLCSD_REUSE_COMPLETED_INITIAL_VALIDATION:-0}
CHECKPOINT_CAPACITY_MULTIPLIER=${CHECKPOINT_CAPACITY_MULTIPLIER:-1}
MODELSCOPE_UPLOAD_ENABLED=0
MODELSCOPE_OWNER=${MODELSCOPE_OWNER:-}
MODELSCOPE_REPO_ID=""
MODELSCOPE_REVISION=${MODELSCOPE_REVISION:-master}
MODELSCOPE_PATH_PREFIX=${MODELSCOPE_PATH_PREFIX:-qwen3-1.7b-a800-t4k-dapo-five-arm}
MODELSCOPE_MAX_WORKERS=${MODELSCOPE_MAX_WORKERS:-8}
FORMAL_TOTAL_TRAINING_STEPS=""
BEST_CHECKPOINTS_KEEP_CURRENT=true
ROLLOUT_ENFORCE_EAGER=false
ROLLOUT_ENABLE_PREFIX_CACHING=true
ROLLOUT_MAX_NUM_SEQS=1024
ROLLOUT_ATTENTION_BACKEND=""
ROLLOUT_DISABLE_CASCADE_ATTN=false
ROLLOUT_DISABLE_RADIX_CACHE=false
ROLLOUT_DISABLE_OVERLAP_SCHEDULE=false
ROLLOUT_ENABLE_MEMORY_SAVER=true
ROLLOUT_FREE_CACHE_ENGINE=true
ROLLOUT_BACKEND=vllm
TRAIN_RESPONSE_LENGTH=4096
VAL_RESPONSE_LENGTH=4096
FINETUNING_MODE=${BETA_OPSD_FINETUNING_MODE:-full}
MODEL_LORA_RANK=${BETA_OPSD_LORA_RANK:-0}
MODEL_LORA_ALPHA=${BETA_OPSD_LORA_ALPHA:-128}
MODEL_LORA_TARGET_MODULES=${BETA_OPSD_LORA_TARGET_MODULES:-all-linear}
SEMANTIC_MODEL_LORA_TARGET_MODULES=$MODEL_LORA_TARGET_MODULES
case "$FINETUNING_MODE" in
  full)
    TRAINABLE_PARAMETERS=full
    [[ "$MODEL_LORA_RANK" == 0 ]] || {
      echo "Full finetuning requires BETA_OPSD_LORA_RANK=0" >&2
      exit 2
    }
    ;;
  lora)
    TRAINABLE_PARAMETERS=lora
    [[ "$MODEL_LORA_RANK" =~ ^[1-9][0-9]*$ ]] || {
      echo "LoRA finetuning requires a positive BETA_OPSD_LORA_RANK" >&2
      exit 2
    }
    ;;
  *)
    echo "BETA_OPSD_FINETUNING_MODE must be full or lora" >&2
    exit 2
    ;;
esac
MAX_SEQUENCE_LENGTH=6144
TRAIN_SHUFFLE=true
ACTOR_SHUFFLE=true
DATA_SEED=42
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
WARMUP_STEPS=10
PPO_MINI_BATCH_SIZE=16
REWARD_MANAGER_NAME=dapo
OVERLONG_REWARD_ENABLED=true
GROUP_FILTER_ENABLED=true
ACTOR_PROMPT_SUFFIX='Present your final answer inside \boxed{}, for example \boxed{42}.'
PROMPT_TEMPLATE_VERSION=qwen3_thinking_boxed_v1
VAL_BEFORE_TRAIN=true
FORMAL_VAL_BEFORE_TRAIN=true
LOGGING_FREQ=1
VERPO_TEACHER_MODE=fixed_initial
VERPO_TEACHER_SYNC_INTERVAL=10
VERPO_TEACHER_EMA_DECAY=0.95
ACTOR_USE_KL_LOSS=false
ACTOR_USE_DYNAMIC_BSZ=true
ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ=true
REF_LOG_PROB_USE_DYNAMIC_BSZ=true
EXPERIMENT_PROTOCOL_NAME=t4k_dapo_ranked
VERL_CONFIG_NAME=_generated_ppo_trainer
case "$VERPO_DISPLACEMENT_MODE" in
  fixed) VERPO_DISPLACEMENT_MODE=evidence_vs_none ;;
  ctr) VERPO_DISPLACEMENT_MODE=correct_vs_incorrect ;;
  ranked) VERPO_DISPLACEMENT_MODE=reward_ranked ;;
  fec) ;;
  evidence_vs_none|correct_vs_incorrect|reward_ranked) ;;
  *)
    echo "VERPO displacement mode must be fixed, ctr, ranked, fec, evidence_vs_none, correct_vs_incorrect, or reward_ranked" >&2
    exit 2
    ;;
esac
if [[ "$TRAINING_OBJECTIVE" != verpo && "$TRAINING_OBJECTIVE" != pure_grpo ]]; then
  echo "QWEN3_TRAINING_OBJECTIVE must be verpo or pure_grpo" >&2
  exit 2
fi

case "$HARDWARE_PROFILE" in
  a100_8x_rlcsd_snapshot10)
    EXPECTED_GPUS=8
    EXPECTED_GLOBAL_BATCH=64
    EXPECTED_PER_DEVICE_BATCH=8
    EXPECTED_EPOCHS=30
    EXPECTED_GPU_TYPE=A100
    GPU_NAME_SUBSTRING=A100
    PPO_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_N=8
    VLLM_GPU_MEMORY_UTILIZATION=0.6
    TEACHER_MAX_TOKEN_LEN_PER_GPU=40960
    TRAIN_RESPONSE_LENGTH=16384
    VAL_RESPONSE_LENGTH=38912
    MAX_SEQUENCE_LENGTH=40960
    TRAIN_SHUFFLE=false
    ACTOR_SHUFFLE=false
    DATA_SEED=null
    ROLLOUT_TOP_P=0.95
    ROLLOUT_TOP_K=20
    WARMUP_STEPS=50
    # veRL v1 multiplies this prompt-level value by rollout.n before
    # train_mini_batch; 2 * 8 preserves RLCSD's 16-trajectory mini-batch.
    PPO_MINI_BATCH_SIZE=2
    REWARD_MANAGER_NAME=naive
    OVERLONG_REWARD_ENABLED=false
    GROUP_FILTER_ENABLED=false
    ACTOR_PROMPT_SUFFIX=''
    PROMPT_TEMPLATE_VERSION=rlcsd_public_qwen3_thinking_v1
    VAL_BEFORE_TRAIN=true
    FORMAL_VAL_BEFORE_TRAIN=true
    LOGGING_FREQ=5
    VERPO_TEACHER_MODE=${RLCSD_VERPO_TEACHER_MODE:-snapshot}
    VERPO_TEACHER_SYNC_INTERVAL=${RLCSD_VERPO_TEACHER_SYNC_INTERVAL:-10}
    ACTOR_USE_KL_LOSS=true
    EXPERIMENT_PROTOCOL_NAME=rlcsd_snapshot10
    VERL_CONFIG_NAME=_generated_ppo_trainer
    PROFILE_OUTPUT_SUFFIX="_8xa100_rlcsd_snapshot10"
    PROFILE_REPORT_PREFIX="rlcsd_snapshot10/a100_8x/"
    PROFILE_RUN_SUFFIX="_8xa100_rlcsd_snapshot10"
    BEST_CHECKPOINTS_TO_KEEP=0
    PEAK_CHECKPOINTS_PER_RUN=0
    FORMAL_SAVE_FREQ=50
    FORMAL_TEST_FREQ=50
    ;;
  a100_8x)
    EXPECTED_GPUS=8
    EXPECTED_GLOBAL_BATCH=128
    EXPECTED_PER_DEVICE_BATCH=16
    EXPECTED_EPOCHS=30
    EXPECTED_GPU_TYPE=A100
    GPU_NAME_SUBSTRING=A100
    PPO_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_N=16
    VLLM_GPU_MEMORY_UTILIZATION=0.6
    PROFILE_OUTPUT_SUFFIX=""
    PROFILE_REPORT_PREFIX="t4k_dapo_ranked/"
    PROFILE_RUN_SUFFIX=""
    BEST_CHECKPOINTS_TO_KEEP=0
    PEAK_CHECKPOINTS_PER_RUN=0
    FORMAL_SAVE_FREQ=50
    FORMAL_TEST_FREQ=50
    ;;
  a800_2x_80gb)
    EXPECTED_GPUS=2
    EXPECTED_GLOBAL_BATCH=32
    EXPECTED_PER_DEVICE_BATCH=16
    EXPECTED_EPOCHS=1
    EXPECTED_GPU_TYPE=A800
    GPU_NAME_SUBSTRING=A800
    GPU_MEMORY_MIN_MIB=80000
    GPU_MEMORY_MAX_MIB=83000
    PPO_MICRO_BATCH_SIZE_PER_GPU=2
    ROLLOUT_N=8
    TEACHER_MAX_TOKEN_LEN_PER_GPU=12288
    # Keep the SGLang engine resident instead of using its broken
    # torch-memory-saver pause/resume path (sglang#5920).  Two 80GB cards have
    # enough headroom to keep a larger KV pool while the colocated 1.7B FSDP
    # actor updates; the explicit token cap below remains the hard bound.
    VLLM_GPU_MEMORY_UTILIZATION=0.88
    PROFILE_OUTPUT_SUFFIX="_2xa800_80gb"
    PROFILE_REPORT_PREFIX="t4k_dapo_ranked/a800_2x_80gb/"
    PROFILE_RUN_SUFFIX="_2xa800_80gb"
    BEST_CHECKPOINTS_TO_KEEP=1
    PEAK_CHECKPOINTS_PER_RUN=2
    FORMAL_SAVE_FREQ=100
    FORMAL_TEST_FREQ=50
    FORMAL_TOTAL_TRAINING_STEPS=400
    MODELSCOPE_UPLOAD_ENABLED=1
    # The stable A800 runtime is SGLang/Triton rather than the failed vLLM V1
    # path.  Use CUDA graphs and radix-prefix reuse to maximize decode
    # throughput; the launcher falls back to eager only if the step-1 gate
    # detects a graph/runtime failure.
    ROLLOUT_ENFORCE_EAGER=false
    ROLLOUT_ENABLE_PREFIX_CACHING=true
    # Keep bounded waves for the thinking/4K workload.  vLLM V1 failed with
    # both FA2 and FlashInfer after half of the first 32-request batch, so the
    # A800 profile uses the veRL-pinned SGLang runtime and its Triton attention
    # path instead of another vLLM CUDA-kernel combination.
    # Each server receives 16 prompt groups with eight samples per group, or
    # up to 128 decode sequences.  Run 112 at once as the explicit extreme-
    # throughput profile; this intentionally trades OOM headroom for speed.
    ROLLOUT_MAX_NUM_SEQS=112
    ROLLOUT_BACKEND=sglang
    ROLLOUT_ATTENTION_BACKEND=triton
    ROLLOUT_DISABLE_RADIX_CACHE=false
    # Overlap CPU scheduling with GPU execution.  This is a systems-only
    # optimization and does not change sampling or optimizer semantics.
    ROLLOUT_DISABLE_OVERLAP_SCHEDULE=false
    ROLLOUT_ENABLE_MEMORY_SAVER=false
    # With memory-saver disabled, SGLang's release hooks are no-ops for model
    # tensors but still synchronously flush completed-request/KV allocator
    # state and call empty_cache in the scheduler process before actor
    # backward.  This avoids both the memory-saver pointer bug and retained
    # generation workspaces starving the colocated actor.
    ROLLOUT_FREE_CACHE_ENGINE=true
    ROLLOUT_DISABLE_CASCADE_ATTN=false
    ;;
  a800_2x_rlcsd_snapshot10_lora)
    EXPECTED_GPUS=2
    EXPECTED_GLOBAL_BATCH=32
    EXPECTED_PER_DEVICE_BATCH=16
    EXPECTED_EPOCHS=30
    EXPECTED_GPU_TYPE=A800
    GPU_NAME_SUBSTRING=A800
    GPU_MEMORY_MIN_MIB=80000
    GPU_MEMORY_MAX_MIB=83000
    PPO_MICRO_BATCH_SIZE_PER_GPU=2
    ROLLOUT_N=8
    VLLM_GPU_MEMORY_UTILIZATION=0.88
    TEACHER_MAX_TOKEN_LEN_PER_GPU=40960
    TRAIN_RESPONSE_LENGTH=16384
    VAL_RESPONSE_LENGTH=38912
    MAX_SEQUENCE_LENGTH=40960
    TRAIN_SHUFFLE=false
    ACTOR_SHUFFLE=false
    DATA_SEED=null
    ROLLOUT_TOP_P=0.95
    ROLLOUT_TOP_K=20
    WARMUP_STEPS=50
    # Native veRL v1 multiplies this prompt-level value by rollout.n. 2 * 8
    # therefore preserves RLCSD's effective 16-trajectory mini-batch.
    PPO_MINI_BATCH_SIZE=2
    REWARD_MANAGER_NAME=naive
    OVERLONG_REWARD_ENABLED=false
    GROUP_FILTER_ENABLED=false
    ACTOR_PROMPT_SUFFIX=''
    PROMPT_TEMPLATE_VERSION=rlcsd_public_qwen3_thinking_v1
    VAL_BEFORE_TRAIN=true
    FORMAL_VAL_BEFORE_TRAIN=true
    LOGGING_FREQ=5
    VERPO_TEACHER_MODE=${RLCSD_VERPO_TEACHER_MODE:-snapshot}
    VERPO_TEACHER_SYNC_INTERVAL=${RLCSD_VERPO_TEACHER_SYNC_INTERVAL:-10}
    ACTOR_USE_KL_LOSS=true
    ACTOR_USE_DYNAMIC_BSZ=false
    ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ=false
    REF_LOG_PROB_USE_DYNAMIC_BSZ=false
    EXPERIMENT_PROTOCOL_NAME=rlcsd_a800_lora_batch32_snapshot10
    VERL_CONFIG_NAME=_generated_ppo_trainer
    PROFILE_OUTPUT_SUFFIX="_2xa800_rlcsd_snapshot10_lora64_b32"
    PROFILE_REPORT_PREFIX="rlcsd_snapshot10/a800_2x_lora64_b32/"
    PROFILE_RUN_SUFFIX="_2xa800_rlcsd_snapshot10_lora64_b32"
    # User-authorized 2026-08-08 storage exception for this standalone
    # two-card LoRA profile. Rank every 50-step checkpoint on the registered
    # fixed-validation macro, retain local Top-1, and keep the transient peak
    # at two full checkpoints. Validation predictions and the ranking/deletion
    # manifest remain durable; this exception does not enable ModelScope.
    BEST_CHECKPOINTS_TO_KEEP=1
    BEST_CHECKPOINTS_KEEP_CURRENT=false
    PEAK_CHECKPOINTS_PER_RUN=2
    FORMAL_SAVE_FREQ=50
    FORMAL_TEST_FREQ=50
    MODELSCOPE_UPLOAD_ENABLED=0
    ROLLOUT_BACKEND=sglang
    ROLLOUT_ENFORCE_EAGER=false
    ROLLOUT_ENABLE_PREFIX_CACHING=true
    ROLLOUT_MAX_NUM_SEQS=32
    ROLLOUT_ATTENTION_BACKEND=triton
    ROLLOUT_DISABLE_RADIX_CACHE=false
    ROLLOUT_DISABLE_OVERLAP_SCHEDULE=false
    ROLLOUT_ENABLE_MEMORY_SAVER=false
    ROLLOUT_FREE_CACHE_ENGINE=true
    ROLLOUT_DISABLE_CASCADE_ATTN=false
    ;;
  a100_4x_40gb)
    EXPECTED_GPUS=4
    EXPECTED_GLOBAL_BATCH=64
    EXPECTED_PER_DEVICE_BATCH=16
    EXPECTED_EPOCHS=30
    EXPECTED_GPU_TYPE=A100
    GPU_NAME_SUBSTRING=A100
    GPU_MEMORY_MIN_MIB=40000
    GPU_MEMORY_MAX_MIB=42000
    PPO_MICRO_BATCH_SIZE_PER_GPU=2
    ROLLOUT_N=16
    VLLM_GPU_MEMORY_UTILIZATION=0.55
    PROFILE_OUTPUT_SUFFIX="_4xa100_40gb"
    PROFILE_REPORT_PREFIX="t4k_dapo_ranked/a100_4x_40gb/"
    PROFILE_RUN_SUFFIX="_4xa100_40gb"
    BEST_CHECKPOINTS_TO_KEEP=2
    PEAK_CHECKPOINTS_PER_RUN=3
    FORMAL_SAVE_FREQ=50
    FORMAL_TEST_FREQ=50
    ;;
  *)
    echo "Unknown QWEN3_HARDWARE_PROFILE: $HARDWARE_PROFILE" >&2
    exit 2
    ;;
esac

# The public semantic launcher owns project-level training semantics.  Hardware
# profiles retain only runtime/system defaults; when semantic values are
# present they must become the values validated below and passed to Hydra.
if [[ "${BETA_OPSD_RESOLVED_CONFIG_SOURCE:-}" == verpo_semantic_yaml_v1 ]]; then
  EXPECTED_GPUS=${BETA_OPSD_EXPECTED_GPUS:?missing semantic GPU count}
  EXPECTED_GLOBAL_BATCH=${BETA_OPSD_GLOBAL_PROMPT_BATCH:?missing semantic global batch}
  EXPECTED_PER_DEVICE_BATCH=$((EXPECTED_GLOBAL_BATCH / EXPECTED_GPUS))
  EXPECTED_EPOCHS=${BETA_OPSD_TOTAL_EPOCHS:?missing semantic epoch count}
  EXPECTED_GPU_TYPE=${BETA_OPSD_GPU_NAME_SUBSTRING:?missing semantic GPU type}
  GPU_NAME_SUBSTRING=$EXPECTED_GPU_TYPE
  GPU_MEMORY_MIN_MIB=${BETA_OPSD_GPU_MEMORY_MIN_MIB:?missing semantic GPU memory minimum}
  GPU_MEMORY_MAX_MIB=${BETA_OPSD_GPU_MEMORY_MAX_MIB:?missing semantic GPU memory maximum}
  PPO_MICRO_BATCH_SIZE_PER_GPU=${BETA_OPSD_ACTOR_MICRO_BATCH_PER_GPU:?missing semantic actor micro batch}
  ROLLOUT_N=${BETA_OPSD_ROLLOUT_N:?missing semantic rollout count}
  ROLLOUT_BACKEND=${BETA_OPSD_ROLLOUT_BACKEND:?missing semantic rollout backend}
  ROLLOUT_ENFORCE_EAGER=${BETA_OPSD_ROLLOUT_ENFORCE_EAGER:?missing semantic eager setting}
  ROLLOUT_FREE_CACHE_ENGINE=${BETA_OPSD_FREE_CACHE_ENGINE:?missing semantic cache setting}
  VLLM_GPU_MEMORY_UTILIZATION=${BETA_OPSD_VLLM_GPU_MEMORY_UTILIZATION:?missing semantic rollout memory utilization}
  TRAIN_RESPONSE_LENGTH=${BETA_OPSD_MAX_RESPONSE_LENGTH:?missing semantic response length}
  VAL_RESPONSE_LENGTH=${BETA_OPSD_VAL_RESPONSE_LENGTH:?missing semantic validation response length}
  TEACHER_MAX_TOKEN_LEN_PER_GPU=${BETA_OPSD_TEACHER_MAX_TOKEN_LEN_PER_GPU:?missing semantic Teacher token limit}
  MAX_SEQUENCE_LENGTH=${BETA_OPSD_MAX_MODEL_LEN:?missing semantic model length}
  ROLLOUT_TOP_P=${BETA_OPSD_ROLLOUT_TOP_P:?missing semantic rollout top-p}
  ROLLOUT_TOP_K=${BETA_OPSD_ROLLOUT_TOP_K:?missing semantic rollout top-k}
  WARMUP_STEPS=${BETA_OPSD_WARMUP_STEPS:?missing semantic warmup steps}
  GROUP_FILTER_ENABLED=${BETA_OPSD_FILTER_GROUPS_ENABLED:?missing semantic group-filter setting}
  FORMAL_VAL_BEFORE_TRAIN=${BETA_OPSD_VALIDATION_BEFORE_TRAINING:?missing semantic validation-before-training setting}
  FORMAL_SAVE_FREQ=${BETA_OPSD_SAVE_FREQUENCY:?missing semantic save frequency}
  FORMAL_TEST_FREQ=${BETA_OPSD_VALIDATION_FREQUENCY:?missing semantic validation frequency}
  LOGGING_FREQ=${BETA_OPSD_LOGGING_FREQUENCY:?missing semantic logging frequency}
  if [[ ${BETA_OPSD_TOTAL_TRAINING_STEPS:?missing semantic total steps} == 0 ]]; then
    FORMAL_TOTAL_TRAINING_STEPS=""
  else
    FORMAL_TOTAL_TRAINING_STEPS=$BETA_OPSD_TOTAL_TRAINING_STEPS
  fi
fi
if [[ "$FINETUNING_MODE" == lora && "$ROLLOUT_BACKEND" == sglang && \
      "$MODEL_LORA_TARGET_MODULES" == all-linear ]]; then
  # SGLang 0.5.8 expects an iterable of module names and otherwise iterates the
  # string "all-linear" character by character.  These seven projections are
  # exactly PEFT's all-linear expansion for Qwen3; the output head is excluded.
  MODEL_LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
fi
case "$RLCSD_REUSE_COMPLETED_INITIAL_VALIDATION" in
  0|1) ;;
  *) echo "RLCSD_REUSE_COMPLETED_INITIAL_VALIDATION must be 0 or 1" >&2; exit 2 ;;
esac
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_BATCH_SIZE=${BETA_OPSD_VAL_BATCH:-16}
TRAIN_MAX_SAMPLES=${BETA_OPSD_TRAIN_MAX_SAMPLES:-60000}
MAX_PROMPT_LENGTH=${BETA_OPSD_MAX_PROMPT_LENGTH:-2048}
LEARNING_RATE=${BETA_OPSD_LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${BETA_OPSD_WEIGHT_DECAY:-0.01}
LR_SCHEDULER_TYPE=${BETA_OPSD_LR_SCHEDULER:-constant}
MAX_GRAD_NORM=${BETA_OPSD_ACTOR_GRAD_CLIP:-1.0}
PPO_EPOCHS=${BETA_OPSD_PPO_EPOCHS:-1}
STUDENT_ENABLE_THINKING=${BETA_OPSD_STUDENT_ENABLE_THINKING:-true}
TEACHER_ENABLE_THINKING=${BETA_OPSD_TEACHER_ENABLE_THINKING:-true}
VALIDATION_ENABLE_THINKING=${BETA_OPSD_VALIDATION_ENABLE_THINKING:-true}
ROLLOUT_TEMPERATURE=${BETA_OPSD_ROLLOUT_TEMPERATURE:-1.0}
VALIDATION_N=${BETA_OPSD_VAL_N:-12}
VALIDATION_TEMPERATURE=${BETA_OPSD_VAL_TEMPERATURE:-0.6}
VALIDATION_TOP_P=${BETA_OPSD_VAL_TOP_P:-0.95}
VALIDATION_TOP_K=${BETA_OPSD_VAL_TOP_K:-20}
TENSOR_MODEL_PARALLEL_SIZE=${BETA_OPSD_TENSOR_MODEL_PARALLEL_SIZE:-1}
PROMPT_TEMPLATE_VERSION=${BETA_OPSD_PROMPT_TEMPLATE_VERSION:-$PROMPT_TEMPLATE_VERSION}
if [[ "$HARDWARE_PROFILE" == a800_2x_80gb ]]; then
  EXPECTED_SWANLAB_PROJECT_NAME=qwen3-1.7b-a800-t4k-dapo-five-arm
  SWANLAB_PROJECT_NAME=${SWANLAB_PROJECT_NAME:-$EXPECTED_SWANLAB_PROJECT_NAME}
  if [[ "$SWANLAB_PROJECT_NAME" != "$EXPECTED_SWANLAB_PROJECT_NAME" ]]; then
    echo "A800 five-arm runs must share SwanLab project $EXPECTED_SWANLAB_PROJECT_NAME" >&2
    exit 2
  fi
else
  SWANLAB_PROJECT_NAME=${SWANLAB_PROJECT_NAME:-"qwen3-1.7b-${HARDWARE_PROFILE}-t4k-dapo-matrix"}
fi
TEACHER_MAX_TOKEN_LEN_PER_GPU=${TEACHER_MAX_TOKEN_LEN_PER_GPU:-12288}
GPU_MEMORY_MIN_MIB=${GPU_MEMORY_MIN_MIB:-0}
GPU_MEMORY_MAX_MIB=${GPU_MEMORY_MAX_MIB:-999999}
if [[ "$MATRIX_PREFLIGHT_ONLY" != 0 && "$MATRIX_PREFLIGHT_ONLY" != 1 ]]; then
  echo "MATRIX_PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "$PRINT_RESOLVED_CONFIG" != 0 && "$PRINT_RESOLVED_CONFIG" != 1 ]]; then
  echo "QWEN3_PRINT_RESOLVED_CONFIG must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$CHECKPOINT_CAPACITY_MULTIPLIER" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHECKPOINT_CAPACITY_MULTIPLIER must be a positive integer" >&2
  exit 2
fi

if [[ "$GPUS" != "$EXPECTED_GPUS" || "$GLOBAL_BATCH" != "$EXPECTED_GLOBAL_BATCH" || \
      "$PER_DEVICE_BATCH" != "$EXPECTED_PER_DEVICE_BATCH" || "$NUM_EPOCHS" != "$EXPECTED_EPOCHS" || \
      "$GPU_TYPE" != "$EXPECTED_GPU_TYPE" ]]; then
  echo "Hardware profile $HARDWARE_PROFILE is locked to: $EXPECTED_GPUS $EXPECTED_GLOBAL_BATCH $EXPECTED_PER_DEVICE_BATCH $EXPECTED_EPOCHS $EXPECTED_GPU_TYPE" >&2
  exit 2
fi
if (( GLOBAL_BATCH != GPUS * PER_DEVICE_BATCH )); then
  echo "GLOBAL_BATCH must equal GPUS * PER_DEVICE_BATCH" >&2
  exit 2
fi
if [[ "$VERPO_DIVERGENCE" != "forward_kl" && "$VERPO_DIVERGENCE" != "reverse_kl" ]]; then
  echo "VERPO divergence must be forward_kl or reverse_kl" >&2
  exit 2
fi
case "$VERPO_DIVERGENCE:$VERPO_DISPLACEMENT_MODE" in
  forward_kl:evidence_vs_none) VERPO_ARM=fkl_fixed ;;
  reverse_kl:evidence_vs_none) VERPO_ARM=rkl_fixed ;;
  forward_kl:correct_vs_incorrect) VERPO_ARM=fkl_ctr ;;
  reverse_kl:correct_vs_incorrect) VERPO_ARM=rkl_ctr ;;
  forward_kl:reward_ranked) VERPO_ARM=fkl_ranked ;;
  reverse_kl:reward_ranked) VERPO_ARM=rkl_ranked ;;
  forward_kl:fec) VERPO_ARM=fkl_fec ;;
  reverse_kl:fec) VERPO_ARM=rkl_fec ;;
  *)
    echo "Unsupported VERPO divergence/displacement combination" >&2
    exit 2
    ;;
esac
if [[ "$TRAINING_OBJECTIVE" == pure_grpo ]]; then
  VERPO_ARM=pure_grpo
  ACTOR_LOSS_MODE=ppo
  VERPO_ENABLED=false
else
  ACTOR_LOSS_MODE=verpo_zpd
  VERPO_ENABLED=true
fi

DEFAULT_VERPO_EXPERIMENT_ID=$VERPO_ARM
if [[ "$VERPO_ARM" == rkl_fec ]]; then
  DEFAULT_VERPO_EXPERIMENT_ID=a_rkl_fec_lr1_le4
fi
VERPO_EXPERIMENT_ID=${VERPO_EXPERIMENT_ID:-$DEFAULT_VERPO_EXPERIMENT_ID}
VERPO_LAMBDA_REF=${VERPO_LAMBDA_REF:-0.1}
VERPO_LAMBDA_EVI=${VERPO_LAMBDA_EVI:-0.5}
VERPO_TAU=${VERPO_TAU:-1.0}
VERPO_RHO=${VERPO_RHO:-${BETA_OPSD_VERPO_RHO:-0.0001}}
VERPO_COST_FLOOR=${VERPO_COST_FLOOR:-0.001}
VERPO_COST_BETA=${VERPO_COST_BETA:-1.0}
VERPO_TEMPERATURE=${VERPO_TEMPERATURE:-${BETA_OPSD_VERPO_TEMPERATURE:-1.0}}
VERPO_VOCAB_CHUNK_SIZE=${VERPO_VOCAB_CHUNK_SIZE:-${BETA_OPSD_VERPO_VOCAB_CHUNK_SIZE:-4096}}
VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS=${VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS:-1}
VERPO_PROJECTION_EPSILON=${VERPO_PROJECTION_EPSILON:-1e-8}
VERPO_VOCAB_MODE=${VERPO_VOCAB_MODE:-${BETA_OPSD_VERPO_VOCAB_MODE:-topk_truncated}}
VERPO_TOP_K=${VERPO_TOP_K:-${BETA_OPSD_VERPO_TOP_K:-128}}
VERPO_EVIDENCE_ROLLOUT_SCOPE=${VERPO_EVIDENCE_ROLLOUT_SCOPE:-all}
VERPO_GROUP_ZPD_ENABLED=${VERPO_GROUP_ZPD_ENABLED:-false}
VERPO_GROUP_ZPD_EPSILON=${VERPO_GROUP_ZPD_EPSILON:-${BETA_OPSD_VERPO_GROUP_ZPD_EPSILON:-0.0}}
VERPO_GROUP_ZPD_MODE=${VERPO_GROUP_ZPD_MODE:-${BETA_OPSD_VERPO_GROUP_ZPD_MODE:-reward_ranked}}
VERPO_ALLOW_NEGATIVE_BENEFIT=${VERPO_ALLOW_NEGATIVE_BENEFIT:-false}
VERPO_SMOKE_ALLOW_UNBOXED_CONTRASTIVE=${VERPO_SMOKE_ALLOW_UNBOXED_CONTRASTIVE:-${BETA_OPSD_VERPO_SMOKE_ALLOW_UNBOXED_CONTRASTIVE:-false}}
VERPO_GRADIENT_AUDIT_ENABLED=${VERPO_GRADIENT_AUDIT_ENABLED:-${BETA_OPSD_VERPO_GRADIENT_AUDIT_ENABLED:-false}}
VERPO_GRADIENT_AUDIT_MAX_STEPS=${VERPO_GRADIENT_AUDIT_MAX_STEPS:-${BETA_OPSD_VERPO_GRADIENT_AUDIT_MAX_STEPS:-0}}
VERPO_GRADIENT_AUDIT_MAX_PARAMETER_ELEMENTS=${VERPO_GRADIENT_AUDIT_MAX_PARAMETER_ELEMENTS:-${BETA_OPSD_VERPO_GRADIENT_AUDIT_MAX_PARAMETER_ELEMENTS:-1000000}}
VERPO_GRADIENT_AUDIT_MAX_PARAMETER_TENSORS=${VERPO_GRADIENT_AUDIT_MAX_PARAMETER_TENSORS:-${BETA_OPSD_VERPO_GRADIENT_AUDIT_MAX_PARAMETER_TENSORS:-16}}
MODEL_ID=${BETA_OPSD_MODEL_ID:-qwen3_1_7b}
if ! [[ "$MODEL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Semantic model id contains unsafe path characters: $MODEL_ID" >&2
  exit 2
fi
VERPO_MODEL_REPO=${VERPO_MODEL_REPO:-Qwen/Qwen3-1.7B}
VERPO_MODEL_REVISION=${VERPO_MODEL_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
if [[ "${BETA_OPSD_RESOLVED_CONFIG_SOURCE:-}" == verpo_semantic_yaml_v1 ]]; then
  if [[ "$VERPO_MODEL_REPO" != "${BETA_OPSD_MODEL_REPO:?missing semantic model repository}" ]]; then
    echo "Semantic model repository mismatch: $VERPO_MODEL_REPO != $BETA_OPSD_MODEL_REPO" >&2
    exit 2
  fi
  if [[ "$VERPO_MODEL_REVISION" != "${BETA_OPSD_MODEL_REVISION:?missing semantic model revision}" ]]; then
    echo "Semantic model revision mismatch: $VERPO_MODEL_REVISION != $BETA_OPSD_MODEL_REVISION" >&2
    exit 2
  fi
fi
case "$VERPO_ALLOW_NEGATIVE_BENEFIT" in
  true|false) ;;
  *) echo "VERPO_ALLOW_NEGATIVE_BENEFIT must be true or false" >&2; exit 2 ;;
esac
if [[ "$VERPO_ALLOW_NEGATIVE_BENEFIT" == true ]]; then
  PROFILE_OUTPUT_SUFFIX="${PROFILE_OUTPUT_SUFFIX}_signed_benefit"
  PROFILE_RUN_SUFFIX="${PROFILE_RUN_SUFFIX}-signed-benefit"
fi
if [[ -z "${VERPO_SIBLING_SELECTION_MODE:-}" ]]; then
  if [[ "$VERPO_DISPLACEMENT_MODE" == reward_ranked ]]; then
    VERPO_SIBLING_SELECTION_MODE=reward_ranked
  else
    VERPO_SIBLING_SELECTION_MODE=correctness
  fi
fi
case "$VERPO_VOCAB_MODE" in
  full) VERPO_VOCAB_IDENTITY_SUFFIX="" ;;
  topk_truncated)
    [[ "$VERPO_TOP_K" =~ ^[1-9][0-9]*$ ]] || { echo "VERPO_TOP_K must be positive" >&2; exit 2; }
    VERPO_VOCAB_IDENTITY_SUFFIX="_topk_truncated_k${VERPO_TOP_K}"
    ;;
  *) echo "VERPO_VOCAB_MODE must be full or topk_truncated" >&2; exit 2 ;;
esac
if [[ "$VERPO_VOCAB_MODE" == "topk_truncated" ]]; then
  PROFILE_OUTPUT_SUFFIX="${PROFILE_OUTPUT_SUFFIX}${VERPO_VOCAB_IDENTITY_SUFFIX}"
  PROFILE_RUN_SUFFIX="${PROFILE_RUN_SUFFIX}${VERPO_VOCAB_IDENTITY_SUFFIX//_/-}"
fi
VERPO_SCOPE_IDENTITY_EMBEDDED=0
case "$VERPO_EXPERIMENT_ID" in
  "$VERPO_ARM") ;;
  a_ranked)
    [[ "$VERPO_ARM" == fkl_ranked ]] || {
      echo "a_ranked requires forward_kl + reward_ranked" >&2
      exit 2
    }
    ;;
  a_fec)
    [[ "$VERPO_ARM" == fkl_fec ]] || {
      echo "a_fec requires forward_kl + fec" >&2
      exit 2
    }
    ;;
  a_fec_lr1_le4)
    [[ "$VERPO_ARM" == fkl_fec ]] || {
      echo "a_fec_lr1_le4 requires forward_kl + fec" >&2
      exit 2
    }
    VERPO_LAMBDA_REF=1.0
    VERPO_LAMBDA_EVI=4.0
    ;;
  a_rkl_fec_lr1_le4)
    [[ "$VERPO_ARM" == rkl_fec ]] || {
      echo "a_rkl_fec_lr1_le4 requires reverse_kl + fec" >&2
      exit 2
    }
    VERPO_LAMBDA_REF=1.0
    VERPO_LAMBDA_EVI=4.0
    ;;
  a800_pure_grpo_t4k_dapo)
    [[ "$TRAINING_OBJECTIVE" == pure_grpo && "$VERPO_ARM" == pure_grpo ]] || {
      echo "$VERPO_EXPERIMENT_ID requires QWEN3_TRAINING_OBJECTIVE=pure_grpo" >&2
      exit 2
    }
    VERPO_LAMBDA_REF=0.0
    VERPO_LAMBDA_EVI=0.0
    ;;
  a800_fkl_fixed_t4k_dapo_lr05_le10_all|a800_fkl_ctr_t4k_dapo_lr05_le10_all|\
  a800_fkl_fec_t4k_dapo_lr05_le10_all|a800_rkl_fec_t4k_dapo_lr05_le10_all)
    [[ "$TRAINING_OBJECTIVE" == verpo ]] || {
      echo "$VERPO_EXPERIMENT_ID requires QWEN3_TRAINING_OBJECTIVE=verpo" >&2
      exit 2
    }
    EXPECTED_A800_ARM=${VERPO_EXPERIMENT_ID#a800_}
    EXPECTED_A800_ARM=${EXPECTED_A800_ARM%%_t4k_dapo_*}
    [[ "$VERPO_ARM" == "$EXPECTED_A800_ARM" ]] || {
      echo "$VERPO_EXPERIMENT_ID requires VERPO_ARM=$EXPECTED_A800_ARM" >&2
      exit 2
    }
    VERPO_LAMBDA_REF=0.5
    VERPO_LAMBDA_EVI=1.0
    ;;
  a_rkl_fec_lr1_le1_wrong_only)
    [[ "$VERPO_ARM" == rkl_fec ]] || {
      echo "a_rkl_fec_lr1_le1_wrong_only requires reverse_kl + fec" >&2
      exit 2
    }
    VERPO_LAMBDA_REF=1.0
    VERPO_LAMBDA_EVI=1.0
    VERPO_EVIDENCE_ROLLOUT_SCOPE=all
    VERPO_SCOPE_IDENTITY_EMBEDDED=1
    BEST_CHECKPOINTS_TO_KEEP=0
    PEAK_CHECKPOINTS_PER_RUN=0
    ;;
  a_rkl_fec_lr1_le4_wrong_only)
    [[ "$VERPO_ARM" == rkl_fec ]] || {
      echo "a_rkl_fec_lr1_le4_wrong_only requires reverse_kl + fec" >&2
      exit 2
    }
    VERPO_LAMBDA_REF=1.0
    VERPO_LAMBDA_EVI=4.0
    VERPO_EVIDENCE_ROLLOUT_SCOPE=all
    VERPO_SCOPE_IDENTITY_EMBEDDED=1
    BEST_CHECKPOINTS_TO_KEEP=0
    PEAK_CHECKPOINTS_PER_RUN=0
    ;;
  *)
    echo "Unknown or incompatible VERPO_EXPERIMENT_ID: $VERPO_EXPERIMENT_ID" >&2
    exit 2
    ;;
esac
case "$VERPO_EVIDENCE_ROLLOUT_SCOPE" in
  all) ;;
  *)
    echo "VERPO_EVIDENCE_ROLLOUT_SCOPE must be all" >&2
    exit 2
    ;;
esac
case "$VERPO_EXPERIMENT_ID" in
  a_ranked|a_fec|a_fec_lr1_le4|a_rkl_fec_lr1_le4|a_rkl_fec_lr1_le1_wrong_only|a_rkl_fec_lr1_le4_wrong_only)
    [[ "$HARDWARE_PROFILE" == a100_4x_40gb ]] || {
      echo "$VERPO_EXPERIMENT_ID requires QWEN3_HARDWARE_PROFILE=a100_4x_40gb" >&2
      exit 2
    }
    ;;
  a800_pure_grpo_t4k_dapo|a800_fkl_fixed_t4k_dapo_lr05_le10_all|\
  a800_fkl_ctr_t4k_dapo_lr05_le10_all|a800_fkl_fec_t4k_dapo_lr05_le10_all|\
  a800_rkl_fec_t4k_dapo_lr05_le10_all)
    [[ "$HARDWARE_PROFILE" == a800_2x_80gb ]] || {
      echo "$VERPO_EXPERIMENT_ID requires QWEN3_HARDWARE_PROFILE=a800_2x_80gb" >&2
      exit 2
    }
    ;;
esac

if [[ "$PRINT_RESOLVED_CONFIG" == 1 ]]; then
  if [[ "$VERPO_TEACHER_MODE" == fixed_initial ]]; then
    VERPO_TEACHER_SOURCE=frozen_reference
  else
    VERPO_TEACHER_SOURCE=actor_shadow
  fi
  printf '%s\n' \
    "profile=$HARDWARE_PROFILE" \
    "protocol=$EXPERIMENT_PROTOCOL_NAME" \
    "verl_config_name=$VERL_CONFIG_NAME" \
    "gpus=$GPUS" \
    "gpu_type=$GPU_TYPE" \
    "global_prompt_batch_size=$GLOBAL_BATCH" \
    "per_device_prompt_batch_size=$PER_DEVICE_BATCH" \
    "group_size=$ROLLOUT_N" \
    "trajectories_per_rollout_batch=$((GLOBAL_BATCH * ROLLOUT_N))" \
    "num_epochs=$EXPECTED_EPOCHS" \
    "model_id=$MODEL_ID" \
    "model_repo=$VERPO_MODEL_REPO" \
    "model_revision=$VERPO_MODEL_REVISION" \
    "finetuning_mode=$FINETUNING_MODE" \
    "trainable_parameters=$TRAINABLE_PARAMETERS" \
    "model_lora_rank=$MODEL_LORA_RANK" \
    "model_lora_alpha=$MODEL_LORA_ALPHA" \
    "model_lora_target_modules_semantic=$SEMANTIC_MODEL_LORA_TARGET_MODULES" \
    "model_lora_target_modules=$MODEL_LORA_TARGET_MODULES" \
    "model_lora_merge=false" \
    "learning_rate=$LEARNING_RATE" \
    "weight_decay=$WEIGHT_DECAY" \
    "warmup_steps=$WARMUP_STEPS" \
    "lr_scheduler_type=$LR_SCHEDULER_TYPE" \
    "max_grad_norm=$MAX_GRAD_NORM" \
    "max_train_samples=$TRAIN_MAX_SAMPLES" \
    "max_prompt_length=$MAX_PROMPT_LENGTH" \
    "max_completion_length=$TRAIN_RESPONSE_LENGTH" \
    "validation_max_completion_length=$VAL_RESPONSE_LENGTH" \
    "actor_max_token_len_per_gpu=$MAX_SEQUENCE_LENGTH" \
    "teacher_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU" \
    "rlcsd_public_ppo_mini_batch_size=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))" \
    "native_v1_ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
    "ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    "actor_use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ" \
    "rollout_log_prob_use_dynamic_bsz=$ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ" \
    "ref_log_prob_use_dynamic_bsz=$REF_LOG_PROB_USE_DYNAMIC_BSZ" \
    "optimizer_updates_per_rollout_batch=$((GLOBAL_BATCH / PPO_MINI_BATCH_SIZE))" \
    "shuffle=$TRAIN_SHUFFLE" \
    "actor_minibatch_shuffle=$ACTOR_SHUFFLE" \
    "data_seed=$DATA_SEED" \
    "actor_data_loader_seed=42" \
    "reward_manager=$REWARD_MANAGER_NAME" \
    "overlong_reward_enabled=$OVERLONG_REWARD_ENABLED" \
    "dynamic_group_filter_enabled=$GROUP_FILTER_ENABLED" \
    "rollout_temperature=$ROLLOUT_TEMPERATURE" \
    "rollout_top_p=$ROLLOUT_TOP_P" \
    "rollout_top_k=$ROLLOUT_TOP_K" \
    "validation_n=$VALIDATION_N" \
    "validation_temperature=$VALIDATION_TEMPERATURE" \
    "validation_top_p=$VALIDATION_TOP_P" \
    "validation_top_k=$VALIDATION_TOP_K" \
    "student_enable_thinking=$STUDENT_ENABLE_THINKING" \
    "teacher_enable_thinking=$TEACHER_ENABLE_THINKING" \
    "validation_enable_thinking=$VALIDATION_ENABLE_THINKING" \
    "validation_before_training=$FORMAL_VAL_BEFORE_TRAIN" \
    "reuse_completed_initial_validation=$RLCSD_REUSE_COMPLETED_INITIAL_VALIDATION" \
    "rollout_gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION" \
    "save_steps=$FORMAL_SAVE_FREQ" \
    "evaluation_steps=$FORMAL_TEST_FREQ" \
    "logging_steps=$LOGGING_FREQ" \
    "actor_use_kl_loss=$ACTOR_USE_KL_LOSS" \
    "actor_kl_loss_coef=0.0" \
    "teacher_mode=$VERPO_TEACHER_MODE" \
    "teacher_sync_interval=$VERPO_TEACHER_SYNC_INTERVAL" \
    "teacher_ema_decay=$VERPO_TEACHER_EMA_DECAY" \
    "teacher_source=$VERPO_TEACHER_SOURCE" \
    "rollout_backend=$ROLLOUT_BACKEND" \
    "verpo_divergence=$VERPO_DIVERGENCE" \
    "verpo_displacement_mode=$VERPO_DISPLACEMENT_MODE" \
    "lambda_ref=$VERPO_LAMBDA_REF" \
    "lambda_evi=$VERPO_LAMBDA_EVI" \
    "best_checkpoints_to_keep=$BEST_CHECKPOINTS_TO_KEEP" \
    "best_checkpoints_keep_current=$BEST_CHECKPOINTS_KEEP_CURRENT" \
    "modelscope_upload_enabled=$MODELSCOPE_UPLOAD_ENABLED" \
    "group_zpd_epsilon=$VERPO_GROUP_ZPD_EPSILON" \
    "group_zpd_mode=$VERPO_GROUP_ZPD_MODE" \
    "sibling_selection_mode=$VERPO_SIBLING_SELECTION_MODE" \
    "verpo_rho=$VERPO_RHO" \
    "verpo_temperature=$VERPO_TEMPERATURE" \
    "verpo_vocab_mode=$VERPO_VOCAB_MODE" \
    "verpo_top_k=$VERPO_TOP_K" \
    "verpo_vocab_chunk_size=$VERPO_VOCAB_CHUNK_SIZE" \
    "allow_negative_benefit=$VERPO_ALLOW_NEGATIVE_BENEFIT"
  exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
VENV_DIR=${VENV_DIR:-"$ROOT/.venvs/qwen3-1.7b-verpo-zpd"}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
PYTHON_BIN="$VENV_DIR/bin/python"
export PATH="$VENV_DIR/bin:$PATH"
DATA_ROOT=${DATA_ROOT:-"$ROOT/data/rlcsd"}
MODEL_CACHE=${MODEL_CACHE:-"${HF_HOME:-$ROOT/.cache/huggingface}"}
MODEL_PREPARE_ARGS=(
  --model-repo "$VERPO_MODEL_REPO"
  --model-revision "$VERPO_MODEL_REVISION"
)
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_PREPARE_ARGS+=(--model-path "$MODEL_PATH")
fi
DEFAULT_OUTPUT_ROOT="$ROOT/outputs/risk_aware_opsd/${MODEL_ID}_verpo_zpd_t4k_dapo_ranked_${VERPO_EXPERIMENT_ID}${PROFILE_OUTPUT_SUFFIX}"
if [[ "$HARDWARE_PROFILE" == a100_8x_rlcsd_snapshot10 ]]; then
  DEFAULT_OUTPUT_ROOT="$ROOT/outputs/risk_aware_opsd/${MODEL_ID}_verpo_zpd_rlcsd_snapshot10_${VERPO_EXPERIMENT_ID}${PROFILE_OUTPUT_SUFFIX}"
elif [[ "$HARDWARE_PROFILE" == a800_2x_rlcsd_snapshot10_lora ]]; then
  DEFAULT_OUTPUT_ROOT="$ROOT/outputs/risk_aware_opsd/${MODEL_ID}_verpo_zpd_rlcsd_a800_lora_b32_${VERPO_EXPERIMENT_ID}${PROFILE_OUTPUT_SUFFIX}"
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}
RUN_ROOT="$OUTPUT_ROOT/formal"
SMOKE_ROOT=${SMOKE_ROOT:-"$OUTPUT_ROOT/smoke_$(date -u +%Y%m%dT%H%M%SZ)"}
MANIFEST="$RUN_ROOT/provenance/assets.json"
DEFAULT_REPORT_DIR="$ROOT/reports/${MODEL_ID}/${PROFILE_REPORT_PREFIX}$VERPO_EXPERIMENT_ID"
REPORT_DIR=${REPORT_DIR:-$DEFAULT_REPORT_DIR}

export PYTHONPATH="$ROOT:$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
if [[ "$ROLLOUT_BACKEND" == vllm ]]; then
  export VLLM_USE_V1=${VLLM_USE_V1:-1}
  if [[ -n "$ROLLOUT_ATTENTION_BACKEND" ]]; then
    export VLLM_ATTENTION_BACKEND="$ROLLOUT_ATTENTION_BACKEND"
  fi
else
  unset VLLM_USE_V1 VLLM_ATTENTION_BACKEND
fi
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}

# Slurm images may export ROCm visibility variables even on NVIDIA-only nodes.
# veRL deliberately rejects mixed CUDA/HIP/ROCR visibility, so remove the
# irrelevant variables before Ray inherits the launch environment.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

mkdir -p "$RUN_ROOT/provenance" "$SMOKE_ROOT" "$REPORT_DIR"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#GPU_NAMES[@]} -ne "$EXPECTED_GPUS" ]]; then
  echo "Expected exactly $EXPECTED_GPUS visible GPUs, found ${#GPU_NAMES[@]}" >&2
  exit 1
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "$name" != *"$GPU_NAME_SUBSTRING"* ]]; then
    echo "Expected $GPU_NAME_SUBSTRING GPUs, found: $name" >&2
    exit 1
  fi
done
mapfile -t GPU_MEMORY_MIB < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
for memory in "${GPU_MEMORY_MIB[@]}"; do
  memory=${memory//[[:space:]]/}
  if ! [[ "$memory" =~ ^[0-9]+$ ]] || (( memory < GPU_MEMORY_MIN_MIB || memory > GPU_MEMORY_MAX_MIB )); then
    echo "Expected GPU memory in [${GPU_MEMORY_MIN_MIB}, ${GPU_MEMORY_MAX_MIB}] MiB, found: $memory" >&2
    exit 1
  fi
done
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$RUN_ROOT/provenance/gpus.csv"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY is required for the formal run" >&2
  exit 1
fi
if [[ "$MODELSCOPE_UPLOAD_ENABLED" == 1 && -z "${MODELSCOPE_TOKEN:-}" ]]; then
  echo "MODELSCOPE_TOKEN is required for the A800 formal run" >&2
  exit 1
fi
if [[ "$MODELSCOPE_UPLOAD_ENABLED" == 1 && -z "$MODELSCOPE_OWNER" ]]; then
  echo "MODELSCOPE_OWNER is required for the A800 formal run" >&2
  exit 1
fi
if [[ "$MODELSCOPE_UPLOAD_ENABLED" == 1 ]]; then
  if [[ "$MODELSCOPE_OWNER" == */* || "$MODELSCOPE_OWNER" == "." || "$MODELSCOPE_OWNER" == ".." ]]; then
    echo "MODELSCOPE_OWNER must be one ModelScope namespace component" >&2
    exit 1
  fi
  MODELSCOPE_REPO_ID="$MODELSCOPE_OWNER/$VERPO_EXPERIMENT_ID"
fi
if ! [[ "$MODELSCOPE_MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODELSCOPE_MAX_WORKERS must be a positive integer" >&2
  exit 1
fi

VENV_DIR="$VENV_DIR" BOOTSTRAP_PYTHON="$BOOTSTRAP_PYTHON" ROLLOUT_BACKEND="$ROLLOUT_BACKEND" \
  bash "$SCRIPT_DIR/setup_env.sh" 2>&1 | tee "$RUN_ROOT/provenance/environment_setup.log"

if [[ "$MODELSCOPE_UPLOAD_ENABLED" == 1 ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/ensure_modelscope_private_repo.py" \
    --repo-id "$MODELSCOPE_REPO_ID" \
    --audit-path "$RUN_ROOT/modelscope_uploads/repository_result.json"
fi

NCCL_LIBRARY_DIR=$(
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import pathlib

distribution = importlib.metadata.distribution("nvidia-nccl-cu12")
candidates = [
    pathlib.Path(distribution.locate_file(entry)).resolve().parent
    for entry in distribution.files or ()
    if pathlib.PurePosixPath(str(entry)).name == "libnccl.so.2"
]
if not candidates:
    raise SystemExit("nvidia-nccl-cu12 is installed but libnccl.so.2 was not found")
print(candidates[0])
PY
)
case ":${LD_LIBRARY_PATH:-}:" in
  *":$NCCL_LIBRARY_DIR:"*) ;;
  *) export LD_LIBRARY_PATH="$NCCL_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
"$PYTHON_BIN" - <<'PY'
import ctypes

ctypes.CDLL("libnccl.so.2")
PY
printf 'ROCR_VISIBLE_DEVICES=<unset>\nHIP_VISIBLE_DEVICES=<unset>\nNCCL_LIBRARY_DIR=%s\n' \
  "$NCCL_LIBRARY_DIR" > "$RUN_ROOT/provenance/nvidia_runtime_env.txt"
printf 'ROLLOUT_BACKEND=%s\nROLLOUT_ATTENTION_BACKEND=%s\n' \
  "$ROLLOUT_BACKEND" "$ROLLOUT_ATTENTION_BACKEND" \
  >> "$RUN_ROOT/provenance/nvidia_runtime_env.txt"

ATTN_IMPLEMENTATION=flash_attention_2
USE_REMOVE_PADDING=true
FLASH_ATTN_VERSION=$(
  "$PYTHON_BIN" -c 'import importlib.metadata; print(importlib.metadata.version("flash-attn"))'
)
"$PYTHON_BIN" - <<'PY'
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
output = flash_attn_func(q, q, q, causal=True)
output.float().sum().backward()
torch.cuda.synchronize()
if output.shape != q.shape or not torch.isfinite(output).all():
    raise SystemExit("FlashAttention CUDA forward/backward smoke failed")
PY
printf 'ATTN_IMPLEMENTATION=%s\nUSE_REMOVE_PADDING=%s\nFLASH_ATTN_VERSION=%s\nFLASH_ATTN_CUDA_SMOKE=passed\n' \
  "$ATTN_IMPLEMENTATION" "$USE_REMOVE_PADDING" "$FLASH_ATTN_VERSION" \
  >> "$RUN_ROOT/provenance/nvidia_runtime_env.txt"

cp "$VENV_DIR/.qwen3_1_7b_verpo_zpd_requirements.sha256" \
  "$RUN_ROOT/provenance/requirements.source.sha256"
"$PYTHON_BIN" -m pip freeze --all > "$RUN_ROOT/provenance/requirements.resolved.txt"
"$PYTHON_BIN" -m pip check > "$RUN_ROOT/provenance/dependency_check.txt"

"$PYTHON_BIN" "$ROOT/scripts/prepare_verpo_rlcsd.py" \
  --data-root "$DATA_ROOT" \
  --model-cache "$MODEL_CACHE" \
  "${MODEL_PREPARE_ARGS[@]}" \
  --output-root "$SMOKE_ROOT" \
  --manifest "$MANIFEST" \
  --rollout-backend "$ROLLOUT_BACKEND" \
  --hardware-profile "$HARDWARE_PROFILE" \
  --experiment-id "$VERPO_EXPERIMENT_ID" \
  --verpo-arm "$VERPO_ARM" \
  --lambda-ref "$VERPO_LAMBDA_REF" \
  --lambda-evi "$VERPO_LAMBDA_EVI" \
  --tau "$VERPO_TAU" \
  --cost-floor "$VERPO_COST_FLOOR" \
  --cost-beta "$VERPO_COST_BETA" \
  --contrastive-num-negative-hints "$VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS" \
  --projection-epsilon "$VERPO_PROJECTION_EPSILON" \
  --evidence-rollout-scope "$VERPO_EVIDENCE_ROLLOUT_SCOPE" \
  --allow-negative-benefit "$VERPO_ALLOW_NEGATIVE_BENEFIT" \
  --mode smoke

if [[ "$MATRIX_PREFLIGHT_ONLY" == 1 ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/prepare_verpo_rlcsd.py" \
    --data-root "$DATA_ROOT" \
    --model-cache "$MODEL_CACHE" \
    "${MODEL_PREPARE_ARGS[@]}" \
    --output-root "$RUN_ROOT" \
    --manifest "$MANIFEST" \
    --rollout-backend "$ROLLOUT_BACKEND" \
    --hardware-profile "$HARDWARE_PROFILE" \
    --experiment-id "$VERPO_EXPERIMENT_ID" \
    --verpo-arm "$VERPO_ARM" \
    --lambda-ref "$VERPO_LAMBDA_REF" \
    --lambda-evi "$VERPO_LAMBDA_EVI" \
    --tau "$VERPO_TAU" \
    --cost-floor "$VERPO_COST_FLOOR" \
    --cost-beta "$VERPO_COST_BETA" \
    --contrastive-num-negative-hints "$VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS" \
    --projection-epsilon "$VERPO_PROJECTION_EPSILON" \
    --evidence-rollout-scope "$VERPO_EVIDENCE_ROLLOUT_SCOPE" \
    --allow-negative-benefit "$VERPO_ALLOW_NEGATIVE_BENEFIT" \
    --checkpoint-capacity-multiplier "$CHECKPOINT_CAPACITY_MULTIPLIER" \
    --peak-checkpoints-per-run "$PEAK_CHECKPOINTS_PER_RUN" \
    --mode formal
  echo "Matrix preflight passed for $CHECKPOINT_CAPACITY_MULTIPLIER arm(s)."
  exit 0
fi

PREPARED_MODEL_PATH=$(
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["model_path"])' "$MANIFEST"
)
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_PATH_MATCH=$(
    "$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve() == pathlib.Path(sys.argv[2]).resolve())' \
      "$MODEL_PATH" "$PREPARED_MODEL_PATH"
  )
  if [[ "$MODEL_PATH_MATCH" != True ]]; then
    echo "Prepared model path differs from semantic MODEL_PATH: $MODEL_PATH != $PREPARED_MODEL_PATH" >&2
    exit 2
  fi
fi
MODEL_PATH=$PREPARED_MODEL_PATH
TRAIN_FILE="$DATA_ROOT/deepmath_filtered_level5_7/train.parquet"
VAL_FILE="$DATA_ROOT/amc23+aime24+aime25/val.parquet"
REWARD_FILE="$ROOT/risk_aware_opsd/rlcsd_verl_reward.py"
SMOKE_REWARD_FILE="$ROOT/risk_aware_opsd/rlcsd_verl_smoke_reward.py"

ROLLOUT_ENGINE_ARGS=()
if [[ "$ROLLOUT_BACKEND" == vllm ]]; then
  ROLLOUT_ENGINE_ARGS+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn="$ROLLOUT_DISABLE_CASCADE_ATTN"
  )
else
  ROLLOUT_ENGINE_ARGS+=(
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="$ROLLOUT_ATTENTION_BACKEND"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache="$ROLLOUT_DISABLE_RADIX_CACHE"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule="$ROLLOUT_DISABLE_OVERLAP_SCHEDULE"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver="$ROLLOUT_ENABLE_MEMORY_SAVER"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=2048
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=2048
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=458752
  )
fi

COMMON_ARGS=(
  --config-name "$VERL_CONFIG_NAME"
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.rollout_correction.rollout_is=token
  algorithm.rollout_correction.rollout_is_threshold=2.0
  algorithm.rollout_correction.rollout_is_batch_normalize=false
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.train_batch_size="$GLOBAL_BATCH"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.train_max_samples="$TRAIN_MAX_SAMPLES"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$TRAIN_RESPONSE_LENGTH"
  data.filter_overlong_prompts=true
  data.truncation=error
  data.shuffle="$TRAIN_SHUFFLE"
  data.validation_shuffle=false
  data.seed="$DATA_SEED"
  +data.apply_chat_template_kwargs.enable_thinking="$STUDENT_ENABLE_THINKING"
  +data.val_apply_chat_template_kwargs.enable_thinking="$VALIDATION_ENABLE_THINKING"
  "+data.actor_prompt_suffix='$ACTOR_PROMPT_SUFFIX'"
  +data.prompt_template_version="$PROMPT_TEMPLATE_VERSION"
  +data.thinking_system_prompt=false
  reward.custom_reward_function.path="$REWARD_FILE"
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name="$REWARD_MANAGER_NAME"
  +reward.reward_kwargs.max_resp_len="$TRAIN_RESPONSE_LENGTH"
  +reward.reward_kwargs.overlong_buffer_cfg.enable="$OVERLONG_REWARD_ENABLED"
  +reward.reward_kwargs.overlong_buffer_cfg.len=512
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward.reward_kwargs.overlong_buffer_cfg.log=true
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank="$MODEL_LORA_RANK"
  actor_rollout_ref.model.lora_alpha="$MODEL_LORA_ALPHA"
  actor_rollout_ref.model.target_modules="$MODEL_LORA_TARGET_MODULES"
  actor_rollout_ref.model.lora.merge=false
  actor_rollout_ref.nccl_timeout=7200
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPLEMENTATION"
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING"
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.loss_mode="$ACTOR_LOSS_MODE"
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.actor.optim.weight_decay="$WEIGHT_DECAY"
  actor_rollout_ref.actor.optim.lr_warmup_steps="$WARMUP_STEPS"
  actor_rollout_ref.actor.optim.lr_scheduler_type="$LR_SCHEDULER_TYPE"
  actor_rollout_ref.actor.grad_clip="$MAX_GRAD_NORM"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.actor.ppo_epochs="$PPO_EPOCHS"
  actor_rollout_ref.actor.shuffle="$ACTOR_SHUFFLE"
  actor_rollout_ref.actor.data_loader_seed=42
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.use_dynamic_bsz="$ACTOR_USE_DYNAMIC_BSZ"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_SEQUENCE_LENGTH"
  actor_rollout_ref.actor.use_kl_loss="$ACTOR_USE_KL_LOSS"
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.use_fused_kernels=false
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.fsdp_config.full_determinism=false
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.verpo.enabled="$VERPO_ENABLED"
  actor_rollout_ref.actor.verpo.teacher_mode="$VERPO_TEACHER_MODE"
  actor_rollout_ref.actor.verpo.teacher_sync_interval="$VERPO_TEACHER_SYNC_INTERVAL"
  actor_rollout_ref.actor.verpo.teacher_ema_decay="$VERPO_TEACHER_EMA_DECAY"
  actor_rollout_ref.actor.verpo.divergence="$VERPO_DIVERGENCE"
  actor_rollout_ref.actor.verpo.displacement_mode="$VERPO_DISPLACEMENT_MODE"
  actor_rollout_ref.actor.verpo.group_zpd_enabled="$VERPO_GROUP_ZPD_ENABLED"
  actor_rollout_ref.actor.verpo.group_zpd_epsilon="$VERPO_GROUP_ZPD_EPSILON"
  actor_rollout_ref.actor.verpo.group_zpd_mode="$VERPO_GROUP_ZPD_MODE"
  actor_rollout_ref.actor.verpo.sibling_selection_mode="$VERPO_SIBLING_SELECTION_MODE"
  actor_rollout_ref.actor.verpo.evidence_rollout_scope="$VERPO_EVIDENCE_ROLLOUT_SCOPE"
  actor_rollout_ref.actor.verpo.contrastive_num_negative_hints="$VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS"
  actor_rollout_ref.actor.verpo.smoke_allow_unboxed_contrastive="$VERPO_SMOKE_ALLOW_UNBOXED_CONTRASTIVE"
  actor_rollout_ref.actor.verpo.projection_epsilon="$VERPO_PROJECTION_EPSILON"
  actor_rollout_ref.actor.verpo.lambda_ref="$VERPO_LAMBDA_REF"
  actor_rollout_ref.actor.verpo.lambda_evi="$VERPO_LAMBDA_EVI"
  actor_rollout_ref.actor.verpo.tau="$VERPO_TAU"
  actor_rollout_ref.actor.verpo.rho="$VERPO_RHO"
  actor_rollout_ref.actor.verpo.cost_floor="$VERPO_COST_FLOOR"
  actor_rollout_ref.actor.verpo.cost_beta="$VERPO_COST_BETA"
  actor_rollout_ref.actor.verpo.allow_negative_benefit="$VERPO_ALLOW_NEGATIVE_BENEFIT"
  actor_rollout_ref.actor.verpo.temperature="$VERPO_TEMPERATURE"
  actor_rollout_ref.actor.verpo.vocab_chunk_size="$VERPO_VOCAB_CHUNK_SIZE"
  actor_rollout_ref.actor.verpo.vocab_mode="$VERPO_VOCAB_MODE"
  actor_rollout_ref.actor.verpo.top_k="$VERPO_TOP_K"
  actor_rollout_ref.actor.verpo.teacher_max_token_len_per_gpu="$TEACHER_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.verpo.gradient_audit_enabled="$VERPO_GRADIENT_AUDIT_ENABLED"
  actor_rollout_ref.actor.verpo.gradient_audit_max_steps="$VERPO_GRADIENT_AUDIT_MAX_STEPS"
  actor_rollout_ref.actor.verpo.gradient_audit_max_parameter_elements="$VERPO_GRADIENT_AUDIT_MAX_PARAMETER_ELEMENTS"
  actor_rollout_ref.actor.verpo.gradient_audit_max_parameter_tensors="$VERPO_GRADIENT_AUDIT_MAX_PARAMETER_TENSORS"
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE"
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE"
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P"
  actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K"
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER"
  actor_rollout_ref.rollout.enable_prefix_caching="$ROLLOUT_ENABLE_PREFIX_CACHING"
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE"
  actor_rollout_ref.rollout.layered_summon=true
  actor_rollout_ref.rollout.seed=42
  actor_rollout_ref.rollout.full_determinism=false
  actor_rollout_ref.rollout.calculate_log_probs=true
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_SEQUENCE_LENGTH"
  actor_rollout_ref.rollout.max_model_len="$MAX_SEQUENCE_LENGTH"
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_SEQUENCE_LENGTH"
  "${ROLLOUT_ENGINE_ARGS[@]}"
  +actor_rollout_ref.rollout.custom.val_response_length="$VAL_RESPONSE_LENGTH"
  +actor_rollout_ref.rollout.custom.hidden_probe.enabled=false
  +actor_rollout_ref.rollout.custom.privileged_text_mode="${BETA_OPSD_PRIVILEGED_TEXT_MODE:-solution_answer}"
  +actor_rollout_ref.rollout.custom.teacher_chat_template_kwargs.enable_thinking="$TEACHER_ENABLE_THINKING"
  +actor_rollout_ref.rollout.custom.thinking_system_prompt=false
  +actor_rollout_ref.rollout.custom.teacher_wrapper_variant="${BETA_OPSD_TEACHER_WRAPPER_VARIANT:-neutral}"
  actor_rollout_ref.rollout.val_kwargs.n="$VALIDATION_N"
  actor_rollout_ref.rollout.val_kwargs.do_sample=true
  actor_rollout_ref.rollout.val_kwargs.temperature="$VALIDATION_TEMPERATURE"
  actor_rollout_ref.rollout.val_kwargs.top_p="$VALIDATION_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.top_k="$VALIDATION_TOP_K"
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$REF_LOG_PROB_USE_DYNAMIC_BSZ"
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_SEQUENCE_LENGTH"
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync
  trainer.critic_warmup=0
  trainer.n_gpus_per_node="$GPUS"
  trainer.nnodes=1
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
  +trainer.logging_freq="$LOGGING_FREQ"
  trainer.max_actor_ckpt_to_keep=null
  trainer.max_critic_ckpt_to_keep=null
  +algorithm.filter_groups.enable="$GROUP_FILTER_ENABLED"
  +algorithm.filter_groups.metric=seq_final_reward
  +algorithm.filter_groups.min_reward_range=1.0e-6
  +algorithm.filter_groups.max_num_gen_batches=4
  trainer.project_name="$SWANLAB_PROJECT_NAME"
)
if [[ "$ACTOR_USE_DYNAMIC_BSZ" == false ]]; then
  COMMON_ARGS+=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
  )
fi

"$PYTHON_BIN" -m pytest -q \
  "$ROOT/verl/tests/trainer/ppo/test_verpo_zpd_math_on_cpu.py" \
  "$ROOT/verl/tests/trainer/ppo/test_rlcsd_verpo_protocol_on_cpu.py" \
  "$ROOT/verl/tests/trainer/ppo/test_rlcsd_verl_rewards_on_cpu.py" \
  "$ROOT/verl/tests/experimental/reward_loop/test_naive_reward_context_on_cpu.py"

if [[ -f "$SMOKE_ROOT/checkpoints/latest_checkpointed_iteration.txt" ]] && \
   [[ "$(tr -d '[:space:]' < "$SMOKE_ROOT/checkpoints/latest_checkpointed_iteration.txt")" == 2 ]] && \
   [[ -d "$SMOKE_ROOT/checkpoints/global_step_2/actor" ]]; then
  echo "Reusing completed smoke global_step_2 from $SMOKE_ROOT"
  "$PYTHON_BIN" "$ROOT/scripts/check_verpo_rlcsd_smoke.py" --smoke-root "$SMOKE_ROOT" --objective "$TRAINING_OBJECTIVE"
else
if [[ -f "$SMOKE_ROOT/checkpoints/latest_checkpointed_iteration.txt" ]] && \
   [[ "$(tr -d '[:space:]' < "$SMOKE_ROOT/checkpoints/latest_checkpointed_iteration.txt")" == 1 ]] && \
   [[ -d "$SMOKE_ROOT/checkpoints/global_step_1/actor" ]]; then
  echo "Reusing completed smoke global_step_1 from $SMOKE_ROOT"
else
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  "${COMMON_ARGS[@]}" \
  reward.custom_reward_function.path="$SMOKE_REWARD_FILE" \
  algorithm.filter_groups.enable=false \
  data.train_max_samples="$GLOBAL_BATCH" \
  data.max_response_length=512 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.verpo.teacher_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.verpo.smoke_allow_unboxed_contrastive=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.custom.val_response_length=512 \
  trainer.val_before_train=false \
  trainer.total_training_steps=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.logger='["console"]' \
  trainer.experiment_name="${MODEL_ID}_verpo_zpd_${EXPERIMENT_PROTOCOL_NAME}_${VERPO_EXPERIMENT_ID}${PROFILE_RUN_SUFFIX}_smoke" \
  trainer.default_local_dir="$SMOKE_ROOT/checkpoints" \
  trainer.rollout_data_dir="$SMOKE_ROOT/rollouts" \
  trainer.validation_data_dir="$SMOKE_ROOT/validation" \
  hydra.run.dir="$SMOKE_ROOT/hydra" \
  2>&1 | tee "$SMOKE_ROOT/train.log"
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  "${COMMON_ARGS[@]}" \
  reward.custom_reward_function.path="$SMOKE_REWARD_FILE" \
  algorithm.filter_groups.enable=false \
  data.train_max_samples="$GLOBAL_BATCH" \
  data.max_response_length=512 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.verpo.teacher_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.verpo.smoke_allow_unboxed_contrastive=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.custom.val_response_length=512 \
  trainer.val_before_train=false \
  trainer.total_training_steps=2 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.logger='["console"]' \
  trainer.experiment_name="${MODEL_ID}_verpo_zpd_${EXPERIMENT_PROTOCOL_NAME}_${VERPO_EXPERIMENT_ID}${PROFILE_RUN_SUFFIX}_smoke" \
  trainer.default_local_dir="$SMOKE_ROOT/checkpoints" \
  trainer.rollout_data_dir="$SMOKE_ROOT/rollouts" \
  trainer.validation_data_dir="$SMOKE_ROOT/validation" \
  hydra.run.dir="$SMOKE_ROOT/hydra" \
  2>&1 | tee -a "$SMOKE_ROOT/train.log"

"$PYTHON_BIN" "$ROOT/scripts/check_verpo_rlcsd_smoke.py" --smoke-root "$SMOKE_ROOT" --objective "$TRAINING_OBJECTIVE"
fi

"$PYTHON_BIN" "$ROOT/scripts/prepare_verpo_rlcsd.py" \
  --data-root "$DATA_ROOT" \
  --model-cache "$MODEL_CACHE" \
  "${MODEL_PREPARE_ARGS[@]}" \
  --output-root "$RUN_ROOT" \
  --manifest "$MANIFEST" \
  --rollout-backend "$ROLLOUT_BACKEND" \
  --hardware-profile "$HARDWARE_PROFILE" \
  --experiment-id "$VERPO_EXPERIMENT_ID" \
  --verpo-arm "$VERPO_ARM" \
  --lambda-ref "$VERPO_LAMBDA_REF" \
  --lambda-evi "$VERPO_LAMBDA_EVI" \
  --tau "$VERPO_TAU" \
  --cost-floor "$VERPO_COST_FLOOR" \
  --cost-beta "$VERPO_COST_BETA" \
  --contrastive-num-negative-hints "$VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS" \
  --projection-epsilon "$VERPO_PROJECTION_EPSILON" \
  --evidence-rollout-scope "$VERPO_EVIDENCE_ROLLOUT_SCOPE" \
  --allow-negative-benefit "$VERPO_ALLOW_NEGATIVE_BENEFIT" \
  --peak-checkpoints-per-run "$PEAK_CHECKPOINTS_PER_RUN" \
  --mode formal

FORMAL_RUNTIME_VAL_BEFORE_TRAIN=$FORMAL_VAL_BEFORE_TRAIN
FORMAL_TEE_ARGS=()
if [[ "$RLCSD_REUSE_COMPLETED_INITIAL_VALIDATION" == 1 ]]; then
  [[ "$HARDWARE_PROFILE" == a800_2x_rlcsd_snapshot10_lora ]] || {
    echo "Initial-validation reuse is registered only for a800_2x_rlcsd_snapshot10_lora" >&2
    exit 2
  }
  [[ "$FORMAL_VAL_BEFORE_TRAIN" == true ]] || {
    echo "Cannot reuse initial validation when the registered protocol disables it" >&2
    exit 2
  }
  "$PYTHON_BIN" "$ROOT/scripts/check_rlcsd_initial_validation.py" \
    --validation-root "$RUN_ROOT/validation" \
    --audit-output "$RUN_ROOT/provenance/initial_validation_reuse.json"
  FORMAL_RUNTIME_VAL_BEFORE_TRAIN=false
  FORMAL_TEE_ARGS=(-a)
fi

FORMAL_ARGS=(
  "${COMMON_ARGS[@]}"
  trainer.val_before_train="$FORMAL_RUNTIME_VAL_BEFORE_TRAIN"
  trainer.total_epochs="$EXPECTED_EPOCHS"
  trainer.save_freq="$FORMAL_SAVE_FREQ"
  trainer.test_freq="$FORMAL_TEST_FREQ"
  trainer.logger='["console","swanlab"]'
  trainer.experiment_name="${MODEL_ID}_verpo_zpd_${EXPERIMENT_PROTOCOL_NAME}_${VERPO_EXPERIMENT_ID}${PROFILE_RUN_SUFFIX}_formal"
  trainer.default_local_dir="$RUN_ROOT/checkpoints"
  trainer.rollout_data_dir="$RUN_ROOT/rollouts"
  trainer.validation_data_dir="$RUN_ROOT/validation"
  hydra.run.dir="$RUN_ROOT/hydra"
)
if [[ -n "$FORMAL_TOTAL_TRAINING_STEPS" ]]; then
  FORMAL_ARGS+=(trainer.total_training_steps="$FORMAL_TOTAL_TRAINING_STEPS")
fi
if (( BEST_CHECKPOINTS_TO_KEEP > 0 )); then
  FORMAL_ARGS+=(
    +trainer.best_checkpoint_retention.keep="$BEST_CHECKPOINTS_TO_KEEP"
    +trainer.best_checkpoint_retention.keep_current="$BEST_CHECKPOINTS_KEEP_CURRENT"
  )
fi
if [[ "$MODELSCOPE_UPLOAD_ENABLED" == 1 ]]; then
  FORMAL_ARGS+=(
    +trainer.best_checkpoint_retention.keep_current=false
    +trainer.modelscope_upload.enabled=true
    +trainer.modelscope_upload.async_enabled=true
    +trainer.modelscope_upload.terminal_drain=true
    +trainer.modelscope_upload.queue_workers=1
    +trainer.modelscope_upload.repo_id="$MODELSCOPE_REPO_ID"
    +trainer.modelscope_upload.experiment_id="$VERPO_EXPERIMENT_ID"
    +trainer.modelscope_upload.token_env=MODELSCOPE_TOKEN
    +trainer.modelscope_upload.path_prefix="$MODELSCOPE_PATH_PREFIX"
    +trainer.modelscope_upload.revision="$MODELSCOPE_REVISION"
    +trainer.modelscope_upload.max_workers="$MODELSCOPE_MAX_WORKERS"
    +trainer.modelscope_upload.audit_dir="$RUN_ROOT/modelscope_uploads"
  )
fi
printf '%q ' "$PYTHON_BIN" -m verl.trainer.main_ppo "${FORMAL_ARGS[@]}" > "$RUN_ROOT/resolved_command.txt"
printf '\n' >> "$RUN_ROOT/resolved_command.txt"

"$PYTHON_BIN" -m verl.trainer.main_ppo "${FORMAL_ARGS[@]}" 2>&1 | tee "${FORMAL_TEE_ARGS[@]}" "$RUN_ROOT/train.log"

"$PYTHON_BIN" "$ROOT/scripts/finalize_verpo_rlcsd.py" \
  --run-root "$RUN_ROOT" \
  --report-dir "$REPORT_DIR" \
  --repo-root "$ROOT" \
  --arm "$VERPO_ARM" \
  --experiment-id "$VERPO_EXPERIMENT_ID" \
  --divergence "$VERPO_DIVERGENCE" \
  --displacement-mode "$VERPO_DISPLACEMENT_MODE" \
  --lambda-ref "$VERPO_LAMBDA_REF" \
  --lambda-evi "$VERPO_LAMBDA_EVI" \
  --tau "$VERPO_TAU" \
  --cost-floor "$VERPO_COST_FLOOR" \
  --cost-beta "$VERPO_COST_BETA" \
  --contrastive-num-negative-hints "$VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS" \
  --projection-epsilon "$VERPO_PROJECTION_EPSILON" \
  --evidence-rollout-scope "$VERPO_EVIDENCE_ROLLOUT_SCOPE" \
  --allow-negative-benefit "$VERPO_ALLOW_NEGATIVE_BENEFIT" \
  --hardware-profile "$HARDWARE_PROFILE" \
  --modelscope-model-id "$MODELSCOPE_REPO_ID"
