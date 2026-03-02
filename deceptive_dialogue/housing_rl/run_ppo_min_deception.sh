#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional conda activation. Script still works if the environment is already active.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  ACTIVE_ENV_NAME="${CONDA_DEFAULT_ENV:-}"

  TARGET_CONDA_ENV="${CONDA_ENV_NAME:-}"
  if [[ -z "$TARGET_CONDA_ENV" ]]; then
    # Reuse an already-active non-base env; otherwise default to openrlhf.
    if [[ -n "$ACTIVE_ENV_NAME" && "$ACTIVE_ENV_NAME" != "base" ]]; then
      TARGET_CONDA_ENV="$ACTIVE_ENV_NAME"
    else
      TARGET_CONDA_ENV="openrlhf"
    fi
  fi

  if ! conda activate "$TARGET_CONDA_ENV" >/dev/null 2>&1; then
    echo "Error: failed to activate conda env '$TARGET_CONDA_ENV'." >&2
    echo "Tip: run 'conda env list' and then rerun with CONDA_ENV_NAME=<env> bash run_ppo_min_deception.sh" >&2
    exit 1
  fi
fi

# Prefer binaries from the active conda env if one is active.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi

# Use the caller-selected runtime Python if provided, otherwise default to python3.
RUNTIME_PYTHON="${RUNTIME_PYTHON:-python3}"
if ! command -v "$RUNTIME_PYTHON" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    RUNTIME_PYTHON="python"
  fi
fi

for cmd in "$RUNTIME_PYTHON" ray vllm; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found in PATH." >&2
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi not found. This PPO script requires an NVIDIA GPU environment." >&2
  exit 1
fi

if ! "$RUNTIME_PYTHON" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Error: torch.cuda.is_available() is False. GPU training will not work in this environment." >&2
  echo "Debug: RUNTIME_PYTHON=$RUNTIME_PYTHON" >&2
  command -v "$RUNTIME_PYTHON" >&2 || true
  "$RUNTIME_PYTHON" - <<'PY' >&2 || true
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
except Exception as e:
    print("torch import/check failed:", repr(e))
PY
  exit 1
fi

echo "Using Python: $(command -v "$RUNTIME_PYTHON")"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Using conda env: $(basename "$CONDA_PREFIX")"
fi

if [[ -n "${OPENRLHF_DIR:-}" ]]; then
  OPENRLHF_DIR_CANDIDATE="$OPENRLHF_DIR"
else
  # Common layouts:
  # 1) alongside the parent repo:   ~/deception/OpenRLHF
  # 2) side-by-side in home:        ~/OpenRLHF
  for candidate in "$SCRIPT_DIR/../../OpenRLHF" "$HOME/OpenRLHF"; do
    if [[ -d "$candidate" ]]; then
      OPENRLHF_DIR_CANDIDATE="$candidate"
      break
    fi
  done
fi
OPENRLHF_DIR="$(cd "${OPENRLHF_DIR_CANDIDATE:-}" 2>/dev/null && pwd || true)"
if [[ -z "$OPENRLHF_DIR" || ! -d "$OPENRLHF_DIR" ]]; then
  echo "Error: OpenRLHF repo not found. Set OPENRLHF_DIR to your OpenRLHF checkout path." >&2
  echo "Tried: ${OPENRLHF_DIR_CANDIDATE:-<none>} and common defaults ($SCRIPT_DIR/../../OpenRLHF, $HOME/OpenRLHF)" >&2
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/data/in/ppo_data" ]]; then
  echo "Error: dataset directory missing at $SCRIPT_DIR/data/in/ppo_data" >&2
  echo "Run: bash generate_ppo_dataset.sh" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/data/in/ppo_data/train.jsonl" ]]; then
  echo "Error: train.jsonl not found in $SCRIPT_DIR/data/in/ppo_data" >&2
  echo "Run: bash generate_ppo_dataset.sh" >&2
  exit 1
fi

# User-configurable settings via environment variables.
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"
TEMP_DIR="${TEMP_DIR:-}"
SAVE_PATH="${SAVE_PATH:-$SCRIPT_DIR/checkpoints/housing/llama-8b-49k-ppo-belief-misalignment}"
WANDB_TOKEN="${WANDB:-}"
SERVER_GPU="${SERVER_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8270}"
RAY_PORT="${RAY_PORT:-6382}"
RAY_AGENT_PORT="${RAY_AGENT_PORT:-52366}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8000}"
REWARD_SERVER_MAX_MODEL_LEN="${REWARD_SERVER_MAX_MODEL_LEN:-4096}"
REWARD_SERVER_GPU_MEMORY_UTILIZATION="${REWARD_SERVER_GPU_MEMORY_UTILIZATION:-0.6}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-meta-llama/Meta-Llama-3-8B}"
REWARD_SERVER_MODEL="${REWARD_SERVER_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
REWARD_SCRIPT="${REWARD_SCRIPT:-$SCRIPT_DIR/reward_scripts/reward_func_dictionary.py}"
PROMPT_DATA_DIR="${PROMPT_DATA_DIR:-$SCRIPT_DIR/data/in/ppo_data}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-512}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-256}"
ZERO_STAGE="${ZERO_STAGE:-3}"
COLOCATE_ALL_MODELS="${COLOCATE_ALL_MODELS:-0}"
PACKING_SAMPLES="${PACKING_SAMPLES:-0}"
USE_BF16="${USE_BF16:-1}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"
REF_REWARD_OFFLOAD="${REF_REWARD_OFFLOAD:-1}"
INIT_KL_COEF="${INIT_KL_COEF:-}"
VLLM_ENABLE_SLEEP="${VLLM_ENABLE_SLEEP:-0}"
DEEPSPEED_ENABLE_SLEEP="${DEEPSPEED_ENABLE_SLEEP:-0}"
SINGLE_GPU_MODE="${SINGLE_GPU_MODE:-}"
if [[ ! -f "$REWARD_SCRIPT" ]]; then
  echo "Error: reward script not found at $REWARD_SCRIPT" >&2
  exit 1
fi

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
if [[ "$NUM_TRAIN_GPUS" -lt 1 ]]; then
  echo "Error: TRAIN_GPUS must contain at least one GPU id (e.g. TRAIN_GPUS=0,1)." >&2
  exit 1
fi

if [[ -z "$SINGLE_GPU_MODE" ]]; then
  if [[ "$NUM_TRAIN_GPUS" -eq 1 ]]; then
    SINGLE_GPU_MODE=1
  else
    SINGLE_GPU_MODE=0
  fi
fi

if [[ "$SINGLE_GPU_MODE" == "1" ]]; then
  # Single-GPU PPO only works if the actor/reference/critic/vLLM share the
  # device instead of requesting dedicated GPUs from Ray.
  COLOCATE_ALL_MODELS=1
  PACKING_SAMPLES=0
  REF_REWARD_OFFLOAD=1
  VLLM_ENABLE_SLEEP=1
  DEEPSPEED_ENABLE_SLEEP=1
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"

  if [[ -z "$INIT_KL_COEF" ]]; then
    INIT_KL_COEF=0
  fi
  if [[ -z "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    VLLM_GPU_MEMORY_UTILIZATION=0.2
  fi
else
  if [[ -z "$INIT_KL_COEF" ]]; then
    INIT_KL_COEF=0.01
  fi
  if [[ -z "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
    VLLM_GPU_MEMORY_UTILIZATION=0.4
  fi
fi

if [[ "$SINGLE_GPU_MODE" == "1" ]]; then
  echo "Single GPU mode: enabled"
fi

mkdir -p "$SAVE_PATH"
mkdir -p "$SCRIPT_DIR/data/in/ppo_data"

DOWNLOAD_ARGS=()
if [[ -n "$DOWNLOAD_DIR" ]]; then
  mkdir -p "$DOWNLOAD_DIR"
  DOWNLOAD_ARGS+=(--download_dir="$DOWNLOAD_DIR")
fi

RAY_TEMP_ARGS=()
if [[ -n "$TEMP_DIR" ]]; then
  mkdir -p "$TEMP_DIR"
  RAY_TEMP_ARGS+=(--temp-dir="$TEMP_DIR")
fi

WANDB_ARGS=()
if [[ -n "$WANDB_TOKEN" && "$WANDB_TOKEN" != "..." ]]; then
  WANDB_ARGS+=(--use_wandb "$WANDB_TOKEN")
fi

RUNTIME_ENV_JSON="{\"working_dir\": \"$OPENRLHF_DIR\"}"

PPO_ARGS=(
  --ref_num_nodes 1
  --ref_num_gpus_per_node 1
  --critic_num_nodes 1
  --critic_num_gpus_per_node 1
  --actor_num_nodes 1
  --actor_num_gpus_per_node 1
  --vllm_num_engines "$VLLM_NUM_ENGINES"
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --pretrain "$PRETRAIN_MODEL"
  --remote_rm_url "$REWARD_SCRIPT"
  --save_path "$SAVE_PATH"
  --micro_train_batch_size "$MICRO_TRAIN_BATCH_SIZE"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --micro_rollout_batch_size "$MICRO_ROLLOUT_BATCH_SIZE"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --max_samples "$MAX_SAMPLES"
  --max_epochs "$MAX_EPOCHS"
  --prompt_max_len "$PROMPT_MAX_LEN"
  --generate_max_len "$GENERATE_MAX_LEN"
  --zero_stage "$ZERO_STAGE"
  --actor_learning_rate 5e-7
  --critic_learning_rate 9e-6
  --init_kl_coef "$INIT_KL_COEF"
  --prompt_data "json@$PROMPT_DATA_DIR"
  --input_key in_text
  --normalize_reward
)

if [[ "$COLOCATE_ALL_MODELS" == "1" ]]; then
  PPO_ARGS+=(--colocate_all_models)
fi

if [[ "$REF_REWARD_OFFLOAD" == "1" ]]; then
  PPO_ARGS+=(--ref_reward_offload)
fi

if [[ "$PACKING_SAMPLES" == "1" ]]; then
  PPO_ARGS+=(--packing_samples)
fi

if [[ "$USE_BF16" == "1" ]]; then
  PPO_ARGS+=(--bf16)
fi

if [[ "$USE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  PPO_ARGS+=(--gradient_checkpointing)
fi

if [[ "$VLLM_ENABLE_SLEEP" == "1" ]]; then
  PPO_ARGS+=(--vllm_enable_sleep)
fi

if [[ "$DEEPSPEED_ENABLE_SLEEP" == "1" ]]; then
  PPO_ARGS+=(--deepspeed_enable_sleep)
fi

echo "Starting vLLM reward server on GPU $SERVER_GPU (port $REWARD_SERVER_PORT)..."
echo "[reward] which python: $(command -v $RUNTIME_PYTHON)"
echo "[reward] python -V: $($RUNTIME_PYTHON -V 2>&1)"
echo "[reward] which vllm: $(command -v vllm || true)"
echo "[reward] whoami HF: $($RUNTIME_PYTHON - <<'PY'
from huggingface_hub import whoami, HfFolder
print("token_present:", bool(HfFolder.get_token()))
try:
    print("whoami:", whoami())
except Exception as e:
    print("whoami_error:", repr(e))
PY
)"
echo "[reward] env tokens present:"
env | grep -E 'HF_TOKEN|HUGGINGFACE|HF_HOME|TRANSFORMERS_CACHE|XDG_CACHE_HOME' | sed 's/=.*/=***hidden***/' || true
CUDA_VISIBLE_DEVICES="$SERVER_GPU" nohup vllm serve "$REWARD_SERVER_MODEL" \
  --port="$REWARD_SERVER_PORT" \
  --max-model-len "$REWARD_SERVER_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$REWARD_SERVER_GPU_MEMORY_UTILIZATION" \
  "${DOWNLOAD_ARGS[@]}" \
  > "$SCRIPT_DIR/llamainstruct_reward_server.out" 2>&1 &

echo "Starting Ray head on GPUs $TRAIN_GPUS..."
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" ray start --head \
  --node-ip-address 0.0.0.0 \
  --dashboard-port="$RAY_DASHBOARD_PORT" \
  --port="$RAY_PORT" \
  --dashboard-agent-listen-port="$RAY_AGENT_PORT" \
  --num-gpus "$NUM_TRAIN_GPUS" \
  "${RAY_TEMP_ARGS[@]}"

echo "Submitting PPO job to Ray..."
nohup ray job submit --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- "$RUNTIME_PYTHON" -m openrlhf.cli.train_ppo_ray \
  "${PPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  > "$SCRIPT_DIR/ppo49k3.out" 2>&1 &

echo "PPO launch complete."
echo "Reward server log: $SCRIPT_DIR/llamainstruct_reward_server.out"
echo "Ray job submit log: $SCRIPT_DIR/ppo49k3.out"
