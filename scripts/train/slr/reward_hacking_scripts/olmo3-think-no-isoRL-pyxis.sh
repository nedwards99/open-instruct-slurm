#!/bin/bash
# Pyxis/enroot translation of olmo3-think-no-isoRL.sh (same directory).
# Only the container-invocation mechanics changed (Apptainer -> Pyxis); the
# GRPO_ARGS and the SLURM_PROCID role-branching logic inside the srun bash -c
# block are untouched, since that part is runtime-agnostic.
#
# OLMo-3 7B Think RL (GRPO) on Slurm (8 nodes):
#   Task 0 = judge (code API + LLM judge vLLM)
#   Task 1 = Ray head (gradient updates via grpo_fast.py)
#   Tasks 2-7 = Ray workers (48 vLLM inference engines)
#SBATCH --job-name=RLVR-Olmo-no-IsoRL-base-judge
#SBATCH --partition=all
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j/output.out
#SBATCH --error=logs/%x_%j/error.err
#SBATCH --qos=normal
#SBATCH --open-mode=append

# --- 1. Configuration ---
JOB_NAME="RLVR-Olmo-no-IsoRL-base-judge"
BASE_DIR="/path/to/your/open-instruct"   # your fork's clone on cluster storage
CONTAINER_IMAGE="helffml/open_instruct_dev:slr"   # Docker Hub is enroot's default registry, no prefix needed
OUTPUT_DIR="$BASE_DIR/output/$JOB_NAME"
RAY_PORT=6379
LLM_JUDGE_MODEL="Qwen/Qwen3-32B"
LLM_JUDGE_PORT=8000
LLM_JUDGE_NUM_ENGINES=8
CODE_API_PORT=1234

export HOME="$BASE_DIR"
export JOB_NAME="$JOB_NAME"

# --- 2. Resolve node IPs ---
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '2p')
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
JUDGE_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
JUDGE_IP=$(srun --nodes=1 --ntasks=1 -w "$JUDGE_NODE" hostname --ip-address)
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
HOSTED_VLLM_API_BASE="http://${JUDGE_IP}:${LLM_JUDGE_PORT}/v1"
CODE_API_URL="http://${JUDGE_IP}:${CODE_API_PORT}/test_program"

echo "=========================================="
echo "Job: $JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Nodes: $SLURM_NODELIST"
echo "Head: $HEAD_NODE ($HEAD_IP:$RAY_PORT)"
echo "Judge: $JUDGE_NODE ($JUDGE_IP) — LLM: $LLM_JUDGE_PORT, Code: $CODE_API_PORT"
echo "Ray dashboard: ssh -L 8265:$HEAD_IP:8265 <login-node>"
echo "=========================================="

# --- 3. Directories ---
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/.cache/nltk_data" "$BASE_DIR/.cache/open_instruct_dataset_cache" "$OUTPUT_DIR" "$OUTPUT_DIR/rollouts"

# --- 3a. Load secrets (API keys, tokens) from file if it exists. Not in the repo — create your own with your keys.
if [ -f "$BASE_DIR/secrets.env" ]; then
  source "$BASE_DIR/secrets.env"
fi

# --- 4. Container environment ---
# No separate "pre-pull to a shared file" step here, unlike Apptainer's SIF
# workflow — Pyxis/enroot pulls (and caches per-node) directly via
# --container-image below. Env vars are exported into THIS shell; Pyxis
# propagates the submitting shell's exported environment into the container
# by default (this is the documented default, but worth a quick sanity check
# — e.g. `srun --container-image=... bash -c 'echo $RAY_ADDRESS'` — the first
# time you use this on your cluster, since exact behavior can be site-tuned).
export TMPDIR=/tmp
export BASE_DIR_IN_CONTAINER=/stage
export UV_CACHE_DIR=/stage/.cache/uv
export HF_HOME=/stage/.cache/huggingface
export TRITON_CACHE_DIR=/tmp/.cache/triton
export HF_HUB_OFFLINE=True
export NLTK_DATA=/stage/.cache/nltk_data
export TOKENIZERS_PARALLELISM=FALSE
export WANDB_ENTITY=helff
export WANDB_PROJECT=Reward-Shortcut
export RAY_ADDRESS
export RAY_PORT
export RAY_HEAD_PROCID=1
export RAY_DEDUP_LOGS=0
export HOSTED_VLLM_API_BASE
export CODE_API_URL
export CODE_API_PORT
export LLM_JUDGE_MODEL
export LLM_JUDGE_PORT
export LLM_JUDGE_NUM_ENGINES
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=ERROR
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=
export CURL_CA_BUNDLE=
# HF_TOKEN and WANDB_API_KEY come from secrets.env sourced above — exported
# there, not redeclared here.

# --- 5. One srun, N tasks: task 0 = head (Ray + grpo_fast.py), others = workers. Use SLURM_PROCID (hostname can differ in container). ---
GRPO_ARGS="--exp_name $JOB_NAME \
  --queue_dashboard_port 8765 \
  --beta 0.0 \
  --num_samples_per_prompt_rollout 8 \
  --num_unique_prompts_rollout 64 \
  --num_mini_batches 1 \
  --num_epochs 1 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --output_dir $OUTPUT_DIR \
  --rollouts_save_path $OUTPUT_DIR/rollouts \
  --save_traces \
  --dataset_local_cache_dir /stage/.cache/open_instruct_dataset_cache \
  --kl_estimator 2 \
  --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 AIML-TUDA/SLR-Bench:v1-All 5.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 AIML-TUDA/SLR-Bench:v1-All 4 \
  --dataset_mixer_eval_list_splits train \
  --max_prompt_token_length 5000 \
  --response_length 25000 \
  --pack_length 35840 \
  --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
  --chat_template_name olmo_thinker \
  --non_stop_penalty False \
  --apply_language_consistency_penalty False \
  --mask_truncated_completions False \
  --temperature 1.0 \
  --ground_truths_key ground_truth \
  --sft_messages_key prompt \
  --total_episodes 10000000 \
  --deepspeed_stage 3 \
  --num_learners_per_node 8 \
  --vllm_num_engines 48 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.85 \
  --vllm_sync_backend nccl \
  --lr_scheduler_type constant \
  --apply_verifiable_reward true \
  --slr_reward base \
  --slr_reward_function partial \
  --slr_parsing simple \
  --llm_judge_model hosted_vllm/$LLM_JUDGE_MODEL \
  --llm_judge_timeout 1200 \
  --llm_judge_max_tokens 2048 \
  --llm_judge_max_context_length 32768 \
  --llm_judge_temperature 0.7 \
  --clip_higher 0.272 \
  --code_api_url $CODE_API_URL \
  --code_pass_rate_reward_threshold 0.99 \
  --code_max_execution_time 6 \
  --seed 1 \
  --local_eval_every -1 \
  --eval_receive_timeout 600 \
  --save_freq 50 \
  --try_launch_beaker_eval_jobs_on_weka False \
  --gradient_checkpointing \
  --with_tracking \
  --checkpoint_state_freq 100 \
  --checkpoint_state_dir $OUTPUT_DIR/checkpoints \
  --backend_timeout 1200 \
  --inflight_updates true \
  --async_steps 8 \
  --advantage_normalization_type centered \
  --truncated_importance_sampling_ratio_cap 2.0 \
  --push_to_hub false"

# Do not pass SLURM_PROCID=... (script's value is unset; each srun task has its own in the environment). Container inherits it.
#
# Pyxis translation of the Apptainer invocation:
#   apptainer exec --nv --writable-tmpfs "${APPTAINER_ENV[@]}" "$SIF_FILE" bash -c '...'
# becomes:
#   srun --container-image=... --container-mounts=... --container-workdir=... bash -c '...'
# GPU passthrough (Apptainer's --nv) and writable rootfs (Apptainer's
# --writable-tmpfs) don't need explicit flags under Pyxis/enroot: GPUs
# allocated via --gpus-per-node are passed through automatically, and enroot
# containers are writable by default (unlike a read-only .sif).
srun --nodes=8 --ntasks=8 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$BASE_DIR:/stage" \
  --container-workdir="/stage" \
  --container-name="oi_${SLURM_JOB_ID}" \
  bash -c '
    cd /stage
    if [ "${SLURM_PROCID:-0}" = "1" ]; then
      # --- Ray head + training ---
      source scripts/train/slr/ray_setup.sh
      /usr/bin/sleep 20  # extra wait for workers to join
      uv run python -c "import nltk; nltk.download(\"punkt_tab\", quiet=True); nltk.download(\"punkt\", quiet=True)"
      uv run python open_instruct/grpo_fast.py '"$GRPO_ARGS"' || true
      uv run ray stop --force 2>/dev/null || true

    elif [ "${SLURM_PROCID:-0}" = "0" ]; then
      # --- Judge node: code API + LLM judge ---
      source scripts/train/slr/code_api_setup.sh
      source scripts/train/slr/judge_setup.sh

    else
      # --- Ray worker (blocks until head exits) ---
      source scripts/train/slr/ray_setup.sh
    fi
  '
