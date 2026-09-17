#!/bin/bash
# Pyxis/enroot translation of single_gpu_on_beaker.sh, retargeted at SLR-Bench.
#
# single_gpu_on_beaker.sh's Beaker-specific plumbing (mason.py, `beaker account
# whoami`, --cluster/--budget/--workspace) is just Ai2's wrapper for "build
# image, submit to our cluster" -- it never reaches grpo_fast.py. The only
# part that matters is: source a ray-node-setup script, then run
# `python open_instruct/grpo_fast.py <args>` with --single_gpu_mode. This
# script reproduces exactly that on 1 Slurm node / 1 GPU via Pyxis, with:
#   - Ray head started inline (no multi-node coordination needed for 1 node)
#   - dataset/model/reward args swapped for SLR-Bench instead of the r1-style
#     GSM8K debug config
#   - SLR_PROMPT_VARIANT / SLR_PROMPT_PARAPHRASE_IDX / SLR_PROMPT_POSITION
#     exported so slr_bench_prepare_v1 (dataset_transformation.py) picks them up
#
# This is a smoke test: confirms the container, swi-prolog verifier, dataset
# prep, and GRPO loop all work together on your actual cluster, in ~10-15 min
# on a small model, before attempting the real 7B/multi-node run.
#SBATCH --job-name=slr-single-gpu-smoke
#SBATCH --partition=p_csunivie_gres,p_datamining
#SBATCH --account=datamining
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j/output.out
#SBATCH --error=logs/%x_%j/error.err
#SBATCH --open-mode=append

# --- 1. Configuration ---
JOB_NAME="slr-single-gpu-smoke"
BASE_DIR="/path/to/your/open-instruct"   # your fork's clone on cluster storage
CONTAINER_IMAGE="helffml/open_instruct_dev:slr"   # Docker Hub is enroot's default registry, no prefix needed
OUTPUT_DIR="$BASE_DIR/output/$JOB_NAME"

export HOME="$BASE_DIR"
export JOB_NAME="$JOB_NAME"

echo "=========================================="
echo "Job: $JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# --- 2. Directories ---
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/.cache/nltk_data" "$BASE_DIR/.cache/open_instruct_dataset_cache" "$OUTPUT_DIR" "$OUTPUT_DIR/rollouts"

# --- 2a. Load secrets (HF_TOKEN, WANDB_API_KEY) from file if it exists. Not in the repo.
if [ -f "$BASE_DIR/secrets.env" ]; then
  source "$BASE_DIR/secrets.env"
fi

# --- 3. Container environment (propagated into the container by Pyxis by default) ---
export TMPDIR=/tmp
export UV_CACHE_DIR=/stage/.cache/uv
export HF_HOME=/stage/.cache/huggingface
export TRITON_CACHE_DIR=/tmp/.cache/triton
export HF_HUB_OFFLINE=True
export NLTK_DATA=/stage/.cache/nltk_data
export TOKENIZERS_PARALLELISM=FALSE
export WANDB_ENTITY=  # fill in your wandb entity, or drop --with_tracking below
export WANDB_PROJECT=Reward-Shortcut
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=ERROR
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=
export CURL_CA_BUNDLE=

# --- 3a. Prompt-variant sweep knobs (read by slr_bench_prepare_v1 in
# dataset_transformation.py). Unset/"neutral" = unmodified baseline.
export SLR_PROMPT_VARIANT="${SLR_PROMPT_VARIANT:-neutral}"
export SLR_PROMPT_PARAPHRASE_IDX="${SLR_PROMPT_PARAPHRASE_IDX:-0}"
export SLR_PROMPT_POSITION="${SLR_PROMPT_POSITION:-prepend}"

# --- 4. GRPO args ---
# Small model (Qwen3-0.6B) + small SLR-Bench tier (v1-Basic, 250 rows) so this
# finishes in minutes on 1 GPU. Swap --model_name_or_path and the dataset
# mixer for the real run once this passes.
GRPO_ARGS="--exp_name $JOB_NAME \
  --dataset_mixer_list AIML-TUDA/SLR-Bench:v1-Basic 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list AIML-TUDA/SLR-Bench:v1-Basic 8 \
  --dataset_mixer_eval_list_splits test \
  --max_prompt_token_length 1024 \
  --response_length 1024 \
  --pack_length 2048 \
  --per_device_train_batch_size 1 \
  --num_unique_prompts_rollout 8 \
  --num_samples_per_prompt_rollout 4 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --output_dir $OUTPUT_DIR \
  --rollouts_save_path $OUTPUT_DIR/rollouts \
  --dataset_local_cache_dir /stage/.cache/open_instruct_dataset_cache \
  --apply_verifiable_reward true \
  --ground_truths_key ground_truth \
  --sft_messages_key prompt \
  --slr_reward base \
  --slr_reward_function partial \
  --slr_parsing simple \
  --temperature 0.7 \
  --inflight_updates True \
  --learning_rate 3e-7 \
  --total_episodes 200 \
  --deepspeed_stage 2 \
  --with_tracking \
  --num_epochs 1 \
  --num_learners_per_node 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_sync_backend gloo \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_enforce_eager \
  --beta 0.0 \
  --load_ref_policy true \
  --seed 3 \
  --local_eval_every 1 \
  --save_traces \
  --gradient_checkpointing \
  --push_to_hub false \
  --single_gpu_mode"

# --- 5. Launch ---
# No multi-node Ray coordination needed (1 node): start a local head inline,
# replacing configs/beaker_configs/ray_node_setup.sh's Beaker-hostname lookup.
srun --nodes=1 --ntasks=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$BASE_DIR:/stage,/var/spool/slurmd:/var/spool/slurmd" \
  --container-workdir="/stage" \
  --container-name="oi_${SLURM_JOB_ID}" \
  bash -c '
    cd /stage
    ray stop --force 2>/dev/null || true
    mkdir -p "$HOME/.triton/autotune"
    ray start --head --port=6379 --dashboard-host=0.0.0.0
    uv run python -c "import nltk; nltk.download(\"punkt_tab\", quiet=True); nltk.download(\"punkt\", quiet=True)"
    uv run python open_instruct/grpo_fast.py '"$GRPO_ARGS"'
    uv run ray stop --force 2>/dev/null || true
  '
