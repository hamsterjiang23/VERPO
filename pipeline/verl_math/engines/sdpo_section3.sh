#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 GPUS GLOBAL_BATCH PER_DEVICE_BATCH NUM_EPOCHS GPU_TYPE DIVERGENCE DISPLACEMENT" >&2
  exit 2
fi

GPUS=$1
GLOBAL_BATCH=$2
PER_DEVICE_BATCH=$3
NUM_EPOCHS=$4
GPU_TYPE=$5
VERPO_DIVERGENCE=$6
VERPO_DISPLACEMENT_MODE=$7

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

TRAINING_OBJECTIVE=${QWEN3_TRAINING_OBJECTIVE:-verpo}
EXPERIMENT_ID=${VERPO_EXPERIMENT_ID:-sdpo_section3}
MODEL_ID=${VERPO_MODEL_ID:-qwen3_1_7b}
MODEL_REPO=${VERPO_MODEL_REPO:-Qwen/Qwen3-1.7B}
MODEL_REVISION=${VERPO_MODEL_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
MODEL_TRUST_REMOTE_CODE=${VERPO_MODEL_TRUST_REMOTE_CODE:-true}
CHAT_TEMPLATE_THINKING_CONTROL=${VERPO_CHAT_TEMPLATE_THINKING_CONTROL:-enable_thinking}
PROMPT_TEMPLATE_VERSION=${VERPO_PROMPT_TEMPLATE_VERSION:-sdpo_official_qwen3_no_thinking_v1}
MODEL_PATH=${MODEL_PATH:-$MODEL_REPO}
MODEL_CACHE=${MODEL_CACHE:-"${HF_HOME:-$ROOT/.cache/huggingface}"}
TRAIN_FILE=${TRAIN_FILE:-}
VAL_FILE=${VAL_FILE:-}
DEFAULT_SDPO_DATA_DIR="$ROOT/data/SDPO/verl"
if [[ -n "$TRAIN_FILE" ]]; then
  DEFAULT_SDPO_DATA_DIR=$(dirname "$TRAIN_FILE")
fi
SDPO_DATA_DIR=${SDPO_DATA_DIR:-$DEFAULT_SDPO_DATA_DIR}
SDPO_AUTO_DOWNLOAD=${SDPO_AUTO_DOWNLOAD:-true}
SDPO_DATA_VERSION=${SDPO_DATA_VERSION:-sdpo_rollout_group_v1}
VENV_DIR=${VENV_DIR:-"$ROOT/.venvs/qwen3-1.7b-verpo-zpd"}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
PYTHON_BIN="$VENV_DIR/bin/python"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT/outputs/sdpo_section3"}
RUN_ID=${RUN_ID:-sdpo_section3}
EXPECTED_GPUS=${VERPO_EXPECTED_GPUS:-$GPUS}
EXPECTED_GPU_NAME=${VERPO_GPU_NAME_SUBSTRING:-$GPU_TYPE}
GPU_MEMORY_MIN_MIB=${VERPO_GPU_MEMORY_MIN_MIB:-80000}
GPU_MEMORY_MAX_MIB=${VERPO_GPU_MEMORY_MAX_MIB:-83000}
ROLLOUT_BACKEND=${VERPO_ROLLOUT_BACKEND:-vllm}

ROLLOUT_N=${VERPO_ROLLOUT_N:-8}
MINI_BATCH_TRAJECTORIES=${VERPO_MINI_BATCH_TRAJECTORIES:-32}
PPO_MINI_BATCH_SIZE=$((MINI_BATCH_TRAJECTORIES / ROLLOUT_N))
MAX_PROMPT_LENGTH=${VERPO_MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${VERPO_MAX_RESPONSE_LENGTH:-8192}
VAL_RESPONSE_LENGTH=${VERPO_VAL_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${VERPO_MAX_MODEL_LEN:-18944}
TEACHER_MAX_TOKEN_LEN_PER_GPU=${VERPO_TEACHER_MAX_TOKEN_LEN_PER_GPU:-18944}
VAL_BATCH=${VERPO_VAL_BATCH:-1}
VAL_N=${VERPO_VAL_N:-16}
VAL_TEMPERATURE=${VERPO_VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VERPO_VAL_TOP_P:-0.95}
VAL_TOP_K=${VERPO_VAL_TOP_K:--1}
ROLLOUT_TEMPERATURE=${VERPO_ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${VERPO_ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${VERPO_ROLLOUT_TOP_K:--1}
ROLLOUT_IMPORTANCE_CLIP=${VERPO_ROLLOUT_IMPORTANCE_CLIP:-2.0}
MAX_REPROMPT_LENGTH=${VERPO_MAX_REPROMPT_LENGTH:-10240}
NORMALIZE_ADVANTAGE_BY_STD=${VERPO_NORMALIZE_ADVANTAGE_BY_STD:-false}
CLIP_RATIO_HIGH=${VERPO_CLIP_RATIO_HIGH:-0.28}
VLLM_GPU_MEMORY_UTILIZATION=${VERPO_VLLM_GPU_MEMORY_UTILIZATION:-0.55}
ROLLOUT_ENFORCE_EAGER=${VERPO_ROLLOUT_ENFORCE_EAGER:-false}
ROLLOUT_FREE_CACHE_ENGINE=${VERPO_FREE_CACHE_ENGINE:-true}
ACTOR_MICRO_BATCH_PER_GPU=${VERPO_ACTOR_MICRO_BATCH_PER_GPU:-1}
TENSOR_MODEL_PARALLEL_SIZE=${VERPO_TENSOR_MODEL_PARALLEL_SIZE:-1}
LEARNING_RATE=${VERPO_LEARNING_RATE:-1.0e-5}
WEIGHT_DECAY=${VERPO_WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${VERPO_WARMUP_STEPS:-10}
LR_SCHEDULER=${VERPO_LR_SCHEDULER:-constant}
GRAD_CLIP=${VERPO_ACTOR_GRAD_CLIP:-1.0}
PPO_EPOCHS=${VERPO_PPO_EPOCHS:-1}
SAVE_FREQ=${VERPO_SAVE_FREQUENCY:-0}
SAVE_TRAINING_ROLLOUTS=${VERPO_SAVE_TRAINING_ROLLOUTS:-true}
TEST_FREQ=${VERPO_VALIDATION_FREQUENCY:-5}
TOTAL_TRAINING_STEPS=${VERPO_TOTAL_TRAINING_STEPS:-0}
LOGGING_FREQ=${VERPO_LOGGING_FREQUENCY:-1}
VAL_BEFORE_TRAIN=${VERPO_VALIDATION_BEFORE_TRAINING:-false}
TRAIN_SHUFFLE=${VERPO_TRAIN_SHUFFLE:-true}
ACTOR_SHUFFLE=${VERPO_ACTOR_SHUFFLE:-false}
DATA_SEED=${VERPO_DATA_SEED:-42}
TRAIN_MAX_SAMPLES=${VERPO_TRAIN_MAX_SAMPLES:-0}
STUDENT_ENABLE_THINKING=${VERPO_STUDENT_ENABLE_THINKING:-false}
TEACHER_ENABLE_THINKING=${VERPO_TEACHER_ENABLE_THINKING:-false}
VALIDATION_ENABLE_THINKING=${VERPO_VALIDATION_ENABLE_THINKING:-false}
REWARD_MODE=${VERPO_REWARD_MODE:-}
PROJECT_NAME=${VERPO_PROJECT_NAME:-verpo}
PAPER_BASELINE_ENABLED=${PAPER_BASELINE_ENABLED:-false}
PAPER_BASELINE_OBJECTIVE=${PAPER_BASELINE_OBJECTIVE:-sdpo}
PAPER_BASELINE_DIVERGENCE=${PAPER_BASELINE_DIVERGENCE:-jsd}
PAPER_BASELINE_JSD_ALPHA=${PAPER_BASELINE_JSD_ALPHA:-0.5}
PAPER_BASELINE_TOP_K=${PAPER_BASELINE_TOP_K:-100}
PAPER_BASELINE_ADD_TAIL=${PAPER_BASELINE_ADD_TAIL:-true}
PAPER_BASELINE_SUCCESS_THRESHOLD=${PAPER_BASELINE_SUCCESS_THRESHOLD:-0.5}
PAPER_BASELINE_EMA_DECAY=${PAPER_BASELINE_EMA_DECAY:-0.95}
PAPER_BASELINE_TEMPERATURE=${PAPER_BASELINE_TEMPERATURE:-1.0}
PAPER_BASELINE_IS_CLIP=${PAPER_BASELINE_IS_CLIP:-2.0}
PAPER_BASELINE_ENTROPY_BETA=${PAPER_BASELINE_ENTROPY_BETA:-1.0}
PAPER_BASELINE_ENTROPY_SUPPORT=${PAPER_BASELINE_ENTROPY_SUPPORT:-full_vocab}
PAPER_BASELINE_SIBLING_SELECTION=${PAPER_BASELINE_SIBLING_SELECTION:-first_correct}
PAPER_BASELINE_REMOVE_THINKING=${PAPER_BASELINE_REMOVE_THINKING:-true}
PAPER_BASELINE_PROMPT_PROFILE=${PAPER_BASELINE_PROMPT_PROFILE:-}

VERPO_ENABLED=true
ACTOR_LOSS_MODE=verpo_zpd
case "$TRAINING_OBJECTIVE" in
  verpo) ;;
  pure_grpo)
    VERPO_ENABLED=false
    ACTOR_LOSS_MODE=ppo
    ;;
  sdpo)
    VERPO_ENABLED=false
    PAPER_BASELINE_ENABLED=true
    PAPER_BASELINE_OBJECTIVE=sdpo
    ACTOR_LOSS_MODE=sdpo_jsd
    ;;
  srpo)
    VERPO_ENABLED=false
    PAPER_BASELINE_ENABLED=true
    PAPER_BASELINE_OBJECTIVE=srpo
    ACTOR_LOSS_MODE=srpo_jsd
    ;;
  *)
    echo "QWEN3_TRAINING_OBJECTIVE must be verpo, pure_grpo, sdpo, or srpo" >&2
    exit 2
    ;;
esac
DATA_PAPER_PROMPT_PROFILE=""
if [[ "$PAPER_BASELINE_ENABLED" == true || "$PAPER_BASELINE_PROMPT_PROFILE" == sdpo_official_v2 || "$PAPER_BASELINE_PROMPT_PROFILE" == srpo_v1 ]]; then
  DATA_PAPER_PROMPT_PROFILE=$PAPER_BASELINE_PROMPT_PROFILE
fi

case "$VERPO_DISPLACEMENT_MODE" in
  fixed) VERPO_DISPLACEMENT_MODE=evidence_vs_none ;;
  ctr) VERPO_DISPLACEMENT_MODE=correct_vs_incorrect ;;
  ranked) VERPO_DISPLACEMENT_MODE=reward_ranked ;;
  fec|evidence_vs_none|correct_vs_incorrect|reward_ranked) ;;
  *) echo "Unsupported VERPO displacement: $VERPO_DISPLACEMENT_MODE" >&2; exit 2 ;;
esac

require_equal() {
  local name=$1 actual=$2 expected=$3
  if [[ "$actual" != "$expected" ]]; then
    echo "SDPO Section 3 lock mismatch: $name=$actual, expected $expected" >&2
    exit 2
  fi
}

require_numeric_equal() {
  local name=$1 actual=$2 expected=$3 actual_normalized expected_normalized
  if ! printf -v actual_normalized '%.17g' "$actual" \
    || ! printf -v expected_normalized '%.17g' "$expected"; then
    echo "SDPO Section 3 numeric lock is not a number: $name=$actual, expected $expected" >&2
    exit 2
  fi
  require_equal "$name" "$actual_normalized" "$expected_normalized"
}

case "$SDPO_AUTO_DOWNLOAD" in
  true|false) ;;
  *) echo "SDPO_AUTO_DOWNLOAD must be true or false" >&2; exit 2 ;;
esac

# Semantic values are validated by the public resolver. Validate execution
# invariants here too; historical hard-coded defaults are not the contract.
require_equal GPUS "$GPUS" "$EXPECTED_GPUS"
require_equal PER_DEVICE_BATCH "$PER_DEVICE_BATCH" "$((GLOBAL_BATCH / GPUS))"
(( TOTAL_TRAINING_STEPS > 0 && MINI_BATCH_TRAJECTORIES > 0 && ROLLOUT_N > 0 )) || { echo "Invalid training budget" >&2; exit 2; }
(( MINI_BATCH_TRAJECTORIES % ROLLOUT_N == 0 )) || { echo "Invalid minibatch/rollout ratio" >&2; exit 2; }
case "$CHAT_TEMPLATE_THINKING_CONTROL" in enable_thinking|none) ;; *) exit 2 ;; esac
if [[ "$TRAINING_OBJECTIVE" == verpo ]]; then
  require_equal evidence_source "${VERPO_EVIDENCE_SOURCE:-}" rollout_group
  require_equal sibling_selection "${VERPO_SIBLING_SELECTION_MODE:-}" correctness
fi

case "$REWARD_MODE" in
  sdpo_sciknoweval_binary) EXPECTED_VALIDATION_DATA_SOURCE=sciknoweval ;;
  sdpo_tooluse_binary) EXPECTED_VALIDATION_DATA_SOURCE=tooluse ;;
  *) echo "Unsupported SDPO Section 3 reward mode: $REWARD_MODE" >&2; exit 2 ;;
esac

PRINT_ONLY=${QWEN3_PRINT_RESOLVED_CONFIG:-0}
if [[ "$PRINT_ONLY" != 1 ]]; then
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "SDPO bootstrap environment is missing: $PYTHON_BIN" >&2
  echo "Launch through pipeline/verl_math/run.sh so dependencies are installed automatically" >&2
  exit 2
fi

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 2; }
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
require_equal detected_gpu_count "${#GPU_ROWS[@]}" "$GPUS"
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"$EXPECTED_GPU_NAME"* ]] || {
    echo "Expected $EXPECTED_GPU_NAME GPU, got: $row" >&2
    exit 2
  }
  memory=${row##*, }
  (( memory >= GPU_MEMORY_MIN_MIB && memory <= GPU_MEMORY_MAX_MIB )) || {
    echo "Expected GPU memory in [$GPU_MEMORY_MIN_MIB,$GPU_MEMORY_MAX_MIB] MiB, got: $row" >&2
    exit 2
  }
done

case "$ROLLOUT_BACKEND" in
  vllm|sglang) ;;
  *) echo "SDPO Section 3 requires rollout backend vllm or sglang" >&2; exit 2 ;;
esac
export ROLLOUT_BACKEND
# shellcheck source=setup_env.sh
source "$SCRIPT_DIR/setup_env.sh"
export PYTHONPATH="$ROOT/verl:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$SDPO_AUTO_DOWNLOAD" == true ]]; then
  "$PYTHON_BIN" -m scripts.prepare_sdpo_data --data-dir "$SDPO_DATA_DIR"
fi

[[ -f "$TRAIN_FILE" ]] || { echo "Missing annotated SDPO train parquet: $TRAIN_FILE" >&2; exit 2; }
[[ -f "$VAL_FILE" ]] || { echo "Missing processed SDPO test parquet: $VAL_FILE" >&2; exit 2; }


MODEL_ARGS=(
  --repo-id "$MODEL_REPO"
  --revision "$MODEL_REVISION"
  --cache-dir "$MODEL_CACHE"
)
if [[ "$MODEL_PATH" != "$MODEL_REPO" ]]; then
  MODEL_ARGS+=(--model-path "$MODEL_PATH")
fi
MODEL_PATH=$("$PYTHON_BIN" "$ROOT/scripts/prepare_sdpo_model.py" "${MODEL_ARGS[@]}")
[[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || {
  echo "Automatic model preparation did not produce a valid snapshot: $MODEL_PATH" >&2
  exit 2
}

fi  # environment/data/model preparation; dry-run never enters this block
RUN_ROOT="$OUTPUT_ROOT/formal"
if [[ "$PRINT_ONLY" != 1 ]]; then
  "$PYTHON_BIN" -m scripts.check_runtime_identity --run-root "$RUN_ROOT" \
    --train "$TRAIN_FILE" --validation "$VAL_FILE" --model "$MODEL_PATH" --revision "$MODEL_REVISION"
  mkdir -p "$RUN_ROOT/provenance" "$RUN_ROOT/checkpoints" "$RUN_ROOT/rollouts" "$RUN_ROOT/validation" "$RUN_ROOT/hydra"
fi
REWARD_FILE="$ROOT/risk_aware_opsd/sdpo_verl_reward.py"
LOGGER='["console"]'
CHAT_TEMPLATE_ARGS=()
if [[ "$CHAT_TEMPLATE_THINKING_CONTROL" == enable_thinking ]]; then
  CHAT_TEMPLATE_ARGS+=(
    +data.apply_chat_template_kwargs.enable_thinking="$STUDENT_ENABLE_THINKING"
    +data.val_apply_chat_template_kwargs.enable_thinking="$VALIDATION_ENABLE_THINKING"
    +actor_rollout_ref.rollout.custom.teacher_chat_template_kwargs.enable_thinking="$TEACHER_ENABLE_THINKING"
  )
fi

ARGS=(
  --config-name _generated_ppo_trainer
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo="$NORMALIZE_ADVANTAGE_BY_STD"
  algorithm.use_kl_in_reward=false
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.rollout_correction.rollout_is=token
  algorithm.rollout_correction.rollout_is_threshold="$ROLLOUT_IMPORTANCE_CLIP"
  algorithm.rollout_correction.rollout_is_batch_normalize=false
  +algorithm.filter_groups.enable="${VERPO_FILTER_GROUPS_ENABLED:-false}"
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.train_batch_size="$GLOBAL_BATCH"
  data.val_batch_size="$VAL_BATCH"
  data.train_max_samples="$TRAIN_MAX_SAMPLES"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=true
  data.truncation=error
  data.shuffle="$TRAIN_SHUFFLE"
  data.validation_shuffle=false
  data.seed="$DATA_SEED"
  data.trust_remote_code="$MODEL_TRUST_REMOTE_CODE"
  +data.actor_prompt_suffix=''
  +data.prompt_template_version="$PROMPT_TEMPLATE_VERSION"
  +data.paper_prompt_profile="$DATA_PAPER_PROMPT_PROFILE"
  +data.thinking_system_prompt=false
  reward.custom_reward_function.path="$REWARD_FILE"
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name=naive
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.trust_remote_code="$MODEL_TRUST_REMOTE_CODE"
  actor_rollout_ref.model.lora_rank="${VERPO_LORA_RANK:-0}"
  actor_rollout_ref.model.lora_alpha="${VERPO_LORA_ALPHA:-128}"
  actor_rollout_ref.model.target_modules="${VERPO_TARGET_MODULES:-all-linear}"
  actor_rollout_ref.model.lora.merge=false
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.loss_mode="$ACTOR_LOSS_MODE"
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.actor.optim.weight_decay="$WEIGHT_DECAY"
  actor_rollout_ref.actor.optim.lr_warmup_steps="$WARMUP_STEPS"
  actor_rollout_ref.actor.optim.lr_scheduler_type="$LR_SCHEDULER"
  actor_rollout_ref.actor.grad_clip="$GRAD_CLIP"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH_PER_GPU"
  actor_rollout_ref.actor.ppo_epochs="$PPO_EPOCHS"
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH"
  actor_rollout_ref.actor.shuffle="$ACTOR_SHUFFLE"
  actor_rollout_ref.actor.data_loader_seed="$DATA_SEED"
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.fsdp_config.full_determinism=false
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  ++actor_rollout_ref.actor.verpo.evidence_source="${VERPO_EVIDENCE_SOURCE:-rollout_group}"
  actor_rollout_ref.actor.verpo.enabled="$VERPO_ENABLED"
  actor_rollout_ref.actor.verpo.teacher_mode="${VERPO_TEACHER_MODE:-ema}"
  actor_rollout_ref.actor.verpo.teacher_sync_interval="${VERPO_TEACHER_SYNC_INTERVAL:-10}"
  actor_rollout_ref.actor.verpo.teacher_ema_decay="${VERPO_TEACHER_EMA_DECAY:-0.95}"
  actor_rollout_ref.actor.verpo.divergence="$VERPO_DIVERGENCE"
  actor_rollout_ref.actor.verpo.displacement_mode="$VERPO_DISPLACEMENT_MODE"
  actor_rollout_ref.actor.verpo.lambda_ref="${VERPO_LAMBDA_REF:-0.1}"
  actor_rollout_ref.actor.verpo.lambda_evi="${VERPO_LAMBDA_EVI:-1.0}"
  actor_rollout_ref.actor.verpo.advantage_modulation="${VERPO_ADVANTAGE_MODULATION:-none}"
  actor_rollout_ref.actor.verpo.advantage_modulation_lambda="${VERPO_ADVANTAGE_MODULATION_LAMBDA:-0.0}"
  actor_rollout_ref.actor.verpo.cost_alpha="${VERPO_COST_ALPHA:-0.0025}"
  actor_rollout_ref.actor.verpo.cost_epsilon="${VERPO_COST_EPSILON:-0.000025}"
  # Legacy parameterization (intentionally not active):
  # actor_rollout_ref.actor.verpo.tau="${VERPO_TAU:-1.0}"
  # actor_rollout_ref.actor.verpo.rho="${VERPO_RHO:-0.0001}"
  # actor_rollout_ref.actor.verpo.cost_floor="${VERPO_COST_FLOOR:-0.001}"
  # actor_rollout_ref.actor.verpo.cost_beta="${VERPO_COST_BETA:-1.0}"
  actor_rollout_ref.actor.verpo.allow_negative_benefit="${VERPO_ALLOW_NEGATIVE_BENEFIT:-false}"
  actor_rollout_ref.actor.verpo.temperature="$ROLLOUT_TEMPERATURE"
  actor_rollout_ref.actor.verpo.vocab_mode="${VERPO_VOCAB_MODE:-topk_truncated}"
  actor_rollout_ref.actor.verpo.top_k="${VERPO_TOP_K:-128}"
  actor_rollout_ref.actor.verpo.vocab_chunk_size="${VERPO_VOCAB_CHUNK_SIZE:-4096}"
  actor_rollout_ref.actor.verpo.teacher_max_token_len_per_gpu="$TEACHER_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.verpo.teacher_max_reprompt_len="$MAX_REPROMPT_LENGTH"
  actor_rollout_ref.actor.verpo.group_zpd_enabled="${VERPO_GROUP_ZPD_ENABLED:-false}"
  actor_rollout_ref.actor.verpo.group_zpd_epsilon="${VERPO_GROUP_ZPD_EPSILON:-0.0}"
  actor_rollout_ref.actor.verpo.group_zpd_mode="${VERPO_GROUP_ZPD_MODE:-reward_ranked}"
  actor_rollout_ref.actor.verpo.sibling_selection_mode="${VERPO_SIBLING_SELECTION_MODE:-correctness}"
  actor_rollout_ref.actor.verpo.evidence_rollout_scope="${VERPO_EVIDENCE_ROLLOUT_SCOPE:-all}"
  actor_rollout_ref.actor.verpo.contrastive_num_negative_hints="${VERPO_CONTRASTIVE_NUM_NEGATIVE_HINTS:-1}"
  actor_rollout_ref.actor.verpo.projection_epsilon="${VERPO_PROJECTION_EPSILON:-1.0e-8}"
  ++actor_rollout_ref.actor.paper_baseline._target_=verl.workers.config.PaperBaselineConfig
  ++actor_rollout_ref.actor.paper_baseline.enabled="$PAPER_BASELINE_ENABLED"
  ++actor_rollout_ref.actor.paper_baseline.objective="$PAPER_BASELINE_OBJECTIVE"
  ++actor_rollout_ref.actor.paper_baseline.divergence="$PAPER_BASELINE_DIVERGENCE"
  ++actor_rollout_ref.actor.paper_baseline.jsd_alpha="$PAPER_BASELINE_JSD_ALPHA"
  ++actor_rollout_ref.actor.paper_baseline.top_k="$PAPER_BASELINE_TOP_K"
  ++actor_rollout_ref.actor.paper_baseline.add_tail_bucket="$PAPER_BASELINE_ADD_TAIL"
  ++actor_rollout_ref.actor.paper_baseline.success_reward_threshold="$PAPER_BASELINE_SUCCESS_THRESHOLD"
  ++actor_rollout_ref.actor.paper_baseline.teacher_ema_decay="$PAPER_BASELINE_EMA_DECAY"
  ++actor_rollout_ref.actor.paper_baseline.temperature="$PAPER_BASELINE_TEMPERATURE"
  ++actor_rollout_ref.actor.paper_baseline.is_clip="$PAPER_BASELINE_IS_CLIP"
  ++actor_rollout_ref.actor.paper_baseline.entropy_beta="$PAPER_BASELINE_ENTROPY_BETA"
  ++actor_rollout_ref.actor.paper_baseline.entropy_support="$PAPER_BASELINE_ENTROPY_SUPPORT"
  ++actor_rollout_ref.actor.paper_baseline.sibling_selection="$PAPER_BASELINE_SIBLING_SELECTION"
  ++actor_rollout_ref.actor.paper_baseline.remove_thinking_from_demonstration="$PAPER_BASELINE_REMOVE_THINKING"
  ++actor_rollout_ref.actor.paper_baseline.teacher_max_token_len_per_gpu="$TEACHER_MAX_TOKEN_LEN_PER_GPU"
  ++actor_rollout_ref.actor.paper_baseline.max_reprompt_len="$MAX_REPROMPT_LENGTH"
  ++actor_rollout_ref.actor.paper_baseline.prompt_profile="$PAPER_BASELINE_PROMPT_PROFILE"
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE"
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE"
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P"
  actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K"
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER"
  actor_rollout_ref.rollout.enable_prefix_caching=true
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE"
  actor_rollout_ref.rollout.layered_summon=true
  actor_rollout_ref.rollout.seed="${VERPO_SEED:-42}"
  actor_rollout_ref.rollout.calculate_log_probs=true
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH_PER_GPU"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN"
  +actor_rollout_ref.rollout.custom.val_response_length="$VAL_RESPONSE_LENGTH"
  +actor_rollout_ref.rollout.custom.thinking_system_prompt=false
  +actor_rollout_ref.rollout.custom.teacher_wrapper_variant=neutral
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N"
  actor_rollout_ref.rollout.val_kwargs.do_sample=true
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE"
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K"
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH_PER_GPU"
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync
  trainer.critic_warmup=0
  trainer.n_gpus_per_node="$GPUS"
  trainer.nnodes=1
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
  +trainer.logging_freq="$LOGGING_FREQ"
  trainer.total_epochs="$NUM_EPOCHS"
  trainer.save_freq="$SAVE_FREQ"
  trainer.max_actor_ckpt_to_keep=null
  trainer.max_critic_ckpt_to_keep=null
  trainer.test_freq="$TEST_FREQ"
  trainer.logger="$LOGGER"
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$RUN_ID"
  trainer.default_local_dir="$RUN_ROOT/checkpoints"
  +trainer.expected_validation_data_sources="[$EXPECTED_VALIDATION_DATA_SOURCE]"
  hydra.run.dir="$RUN_ROOT/hydra"
)

if (( ${#CHAT_TEMPLATE_ARGS[@]} > 0 )); then
  ARGS+=("${CHAT_TEMPLATE_ARGS[@]}")
fi

if [[ "$SAVE_TRAINING_ROLLOUTS" == true ]]; then
  ARGS+=(
    trainer.rollout_data_dir="$RUN_ROOT/rollouts"
    trainer.validation_data_dir="$RUN_ROOT/validation"
  )
else
  ARGS+=(
    trainer.rollout_data_dir=null
    trainer.validation_data_dir=null
  )
fi

if (( TOTAL_TRAINING_STEPS > 0 )); then
  ARGS+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi

if [[ "$PRINT_ONLY" == 1 ]]; then
  printf '%s\n' '# hydra_arguments' "${ARGS[@]}"
  exit 0
fi

printf '%q ' "$PYTHON_BIN" -m verl.trainer.main_ppo "${ARGS[@]}" > "$RUN_ROOT/provenance/resolved_command.sh"
printf '\n' >> "$RUN_ROOT/provenance/resolved_command.sh"
"$PYTHON_BIN" -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$RUN_ROOT/train.log"
