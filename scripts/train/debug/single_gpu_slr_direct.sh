#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run this FROM INSIDE an already-active interactive container shell, e.g.:
#
#   srun --mem=64G --cpus-per-task=32 --gres=gpu:1 --time=0-02:00:00 --nodes=1 \
#     --partition=p_csunivie_gres,p_datamining --account=datamining \
#     --container-image="helffml/open_instruct_dev:slr" \
#     --container-mounts="/var/spool/slurmd:/var/spool/slurmd" \
#     --container-workdir="/mnt/nlp-data/home/users/nicholase99cs/open-instruct-slurm" \
#     --container-name="open-instruct-slr" \
#     --pty bash
#
#   bash scripts/train/debug/single_gpu_slr_direct.sh
#
# --mem=64G is not decoration: the first use of this image on a node makes
# pyxis convert it to a squashfs, and that conversion was OOM-killed at 16G.
#
# It does NOT itself invoke srun/sbatch/container flags -- it assumes the GPU,
# container, and working directory are already set up by that shell. This is
# the "just run it and watch the output live" smoke test; once this passes,
# scripts/train/debug/single_gpu_slr_pyxis.sh is the equivalent submitted as
# an unattended sbatch job.
#
# ---------------------------------------------------------------------------
# Why no `uv run` here, unlike the 8-node reward_hacking_scripts
# ---------------------------------------------------------------------------
# Those scripts bind-mount the repo over /stage, which hides the image's own
# venv at /stage/.venv, so `uv run` builds a fresh one (torch, vllm,
# flash-attn: tens of GB, needs PyPI) inside the repo. We keep the image's
# venv -- already first on PATH -- and put this checkout first on PYTHONPATH
# instead, so edits under open_instruct/ take effect with nothing to install.
#
# The one thing the image lacks is `openenv-core`, added to this fork's
# pyproject.toml after the image was built and imported at module scope via
# open_instruct/environments/*. That is the "openenv not available" error.
# scripts/train/debug/stage_python_extras.sh drops it (and nothing else) into
# .cache/pyextra; run that once, from a machine with internet.
# ---------------------------------------------------------------------------

# Repo root, so the script works no matter where it is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Caches
#
# Model and dataset caches go on shared storage so they survive the job: /tmp
# is node-local and wiped, which means re-downloading the model every single
# run -- mildly annoying at 0.6B, hours of wasted allocation at 7B. Compile
# and scratch caches stay node-local, where their latency actually matters.
# ---------------------------------------------------------------------------

CACHE_ROOT="${OI_CACHE_ROOT:-${REPO_ROOT}/.cache}"

# $HOME inside the container is /srv/home/..., which is mounted READ-ONLY, and
# a surprising number of libraries write under ~/ at import time: flashinfer
# creates a JIT workspace at $HOME/.cache/flashinfer (FLASHINFER_WORKSPACE_BASE
# defaults to Path.home()), vLLM writes usage stats to $HOME/.config, NLTK
# downloads to $HOME/nltk_data. Each one is a hard crash, not a warning.
# Pointing HOME at writable shared storage fixes the whole class at once --
# the paper's 8-node scripts do the same thing with `export HOME="$BASE_DIR"`.
export HOME="${CACHE_ROOT}/home"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export NLTK_DATA="${CACHE_ROOT}/nltk_data"
export DATASET_LOCAL_CACHE_DIR="${CACHE_ROOT}/open_instruct_dataset_cache"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export XDG_CACHE_HOME="/tmp/${USER}/cache"

# Packages the image predates, dependency-free so they cannot shadow the
# image's numpy/pydantic/torch stack. See stage_python_extras.sh.
OI_PYEXTRA="${OI_PYEXTRA:-${CACHE_ROOT}/pyextra}"
export PYTHONPATH="${REPO_ROOT}:${OI_PYEXTRA}${PYTHONPATH:+:${PYTHONPATH}}"

# Secrets: load from a gitignored file, never hardcode here.
# Create $REPO_ROOT/secrets.env yourself with:
#   export HF_TOKEN=...
#   export WANDB_API_KEY=...
if [ -f secrets.env ]; then
    source secrets.env
fi

# Deliberately NOT setting HF_HUB_OFFLINE=1 -- unlike the paper authors' own
# cluster, we don't yet know whether this cluster's compute nodes have
# internet access. Leaving it unset lets the first real run tell us: if the
# HF downloads below hang or error on a network failure, that's our answer,
# and we switch to pre-downloading from a node that does have access.

mkdir -p \
    "${HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${NLTK_DATA}" \
    "${DATASET_LOCAL_CACHE_DIR}" \
    "${UV_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${XDG_CACHE_HOME}"

export TOKENIZERS_PARALLELISM=FALSE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE=0
# vLLM v1 sends engine-core outputs over a socket encoded with msgspec, which
# cannot encode torch.dtype; without this the engine thread dies with
# "Object of type <class 'torch.dtype'> is not serializable" right after the
# API server comes up. "Insecure" refers to the pickle fallback, which here is
# local IPC between our own processes on one node. Every other launcher in
# this repo sets it too -- mason.py:103 for Beaker jobs, the other debug
# scripts, and the paper's apptainer env block.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# The login node exports SSL_CERT_FILE=/usr/lib/ssl/cert.pem and pyxis passes
# your environment into the container, where that path does not exist. httpx
# reads SSL_CERT_FILE when it builds an SSL context, so vLLM's OpenAI client
# dies on it *after* the engine has already loaded the model. Repoint it at a
# bundle that exists in here -- the paper's apptainer scripts hardcode
# /etc/ssl/certs/ca-certificates.crt for the same reason; certifi is the
# fallback because it ships with the venv and is therefore always present.
if [ ! -f "${SSL_CERT_FILE:-}" ]; then
    if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
        export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    else
        export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
    fi
fi
# Same trap, different consumers (requests, curl): blank them rather than let
# them point at host-only paths.
unset REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
# And once more for the runtime dir: the login shell exports
# XDG_RUNTIME_DIR=/run/user/<uid>, which does not exist in the container.
[ -d "${XDG_RUNTIME_DIR:-}" ] || unset XDG_RUNTIME_DIR
# Don't phone home with usage stats: these nodes do have internet, and the
# reporter writes to $HOME/.config on the way out.
export VLLM_NO_USAGE_STATS=1

# ---------------------------------------------------------------------------
# Prompt-variant sweep knobs, read by slr_bench_prepare_v1 via
# slr/prompt_variants.py. SLR_PROMPT_* is what the model TRAINS on;
# SLR_EVAL_PROMPT_* is what it is EVALUATED on, and defaults to neutral --
# inoculation shows up as behaviour on the unmodified prompt, so the eval set
# must not carry the training instruction.
#
#   SLR_PROMPT_VARIANT=permission bash scripts/train/debug/single_gpu_slr_direct.sh
#
# Families: neutral, scope_narrow, scope_broad, permission, goal_redefinition.
# ---------------------------------------------------------------------------

export SLR_PROMPT_VARIANT="${SLR_PROMPT_VARIANT:-neutral}"
export SLR_PROMPT_PARAPHRASE_IDX="${SLR_PROMPT_PARAPHRASE_IDX:-0}"
export SLR_PROMPT_POSITION="${SLR_PROMPT_POSITION:-prepend}"
export SLR_EVAL_PROMPT_VARIANT="${SLR_EVAL_PROMPT_VARIANT:-neutral}"

# ---------------------------------------------------------------------------
# Data — SLR-Bench
#
# v1-Basic is 3053 train / 50 validation / 250 test rows, the cheap tier for
# smoke tests; the real runs use v1-All.
# ---------------------------------------------------------------------------

# SLR-Bench tier. The prompt budget has to follow it: prompts over the limit
# are DROPPED, not truncated, so too small a budget silently shrinks the
# dataset -- and because shortcutting rises with task complexity, it drops
# exactly the hardest items, damping the effect under study. Measured longest
# prompts per tier: Basic ~580 tok, Easy ~950, Medium ~2840, Hard ~10300.
# Watch the "initial=N, final=N, filtered=0 (0.0%)" line in the log to confirm.
DATASET_CONFIG="${DATASET_CONFIG:-v1-All}"
case "${DATASET_CONFIG}" in
    v1-Basic)  tier_prompt_length=1024 ;;
    v1-Easy)   tier_prompt_length=1536 ;;
    v1-Medium) tier_prompt_length=3584 ;;
    v1-Hard)   tier_prompt_length=12288 ;;
    *)         tier_prompt_length=5000 ;;  # v1-All: the paper's setting
esac
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:-${tier_prompt_length}}"

dataset_mixer_train="AIML-TUDA/SLR-Bench:${DATASET_CONFIG} ${TRAIN_MIX_WEIGHT:-1.0}"
dataset_mixer_eval="AIML-TUDA/SLR-Bench:${DATASET_CONFIG} ${EVAL_MIX_COUNT:-32}"

# ---------------------------------------------------------------------------
# Model — Hugging Face
# ---------------------------------------------------------------------------

# Defaults are the EXPERIMENT configuration, not a smoke test: Qwen3-4B on
# v1-All (all difficulty tiers, as the paper trains on), the paper's optimizer
# settings, vLLM tuned for throughput, neutral-prompt eval every 25 steps.
# A cheap plumbing check is one override away:
#
#   MODEL=Qwen/Qwen3-0.6B DATASET_CONFIG=v1-Basic TOTAL_EPISODES=64 \
#     VLLM_ENFORCE_EAGER=1 bash scripts/train/debug/single_gpu_slr_direct.sh
#
# Everything below is overridable from the environment:
#
#   MODEL=Qwen/Qwen3-1.7B RESPONSE_LENGTH=4096 TOTAL_EPISODES=2000 \
#     bash scripts/train/debug/single_gpu_slr_direct.sh
#
# Note Qwen3 models open a <think> block by default and the verifier only reads
# what follows </think>; at 2048 nearly every rollout truncates mid-thought,
# which both wastes the sample and feeds the Prolog judge malformed rules that
# burn its 5-second timeout. Budget generously for real runs.
MODEL="${MODEL:-Qwen/Qwen3-4B}"
# 8192, not 2048: at 2048 a measured 95% of rollouts ended in `length` rather
# than `stop`, so the model never closed its <think> block, and the verifier
# ended up scraping a rule out of the scratchpad instead of grading an answer.
# A high cap is cheap for well-behaved rollouts (generation stops at EOS) and
# only costs time on the ones that would have been garbage anyway.
RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}"
PACK_LENGTH="${PACK_LENGTH:-$((MAX_PROMPT_TOKEN_LENGTH + RESPONSE_LENGTH))}"
TOTAL_EPISODES="${TOTAL_EPISODES:-200}"
# Every step is thorough but expensive: each eval rollout is scored by both
# Prolog judges, and degenerate rules cost the full timeout twice.
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-25}"

# Batch shape and engine placement: smoke-test sized by default, override to
# scale up. The paper runs 64 prompts x 8 samples across 8 learners and 40-48
# vLLM engines; the 8 samples per prompt is a GRPO requirement (the advantage
# is computed within each prompt's group) so that one is a default, not a
# smoke-test value.
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-8}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}"
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-1}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"
# NOTE: this is a fraction of the WHOLE card, so it means different absolute
# sizes per GPU type: 0.25 is ~35GB on a 141GB H200 and ~20GB on an 80GB H100.
# 0.42 on an H200 gave vLLM 59GB and OOM'd the trainer at the first optimizer
# step, so raise it only on smaller cards.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"

DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-2}"
# Last-resort memory lever: push Adam state (~12x params in fp32) to CPU RAM.
# This node has ~2TB, so it fits easily; the cost is host<->device traffic every
# optimizer step. Needs stage 3. Use when pack_length has grown past what one
# card holds, rather than shrinking the prompt/response budget further.
DEEPSPEED_OFFLOAD_OPTIMIZER="${DEEPSPEED_OFFLOAD_OPTIMIZER:-false}"

# --single_gpu_mode packs the learner and the vLLM engine onto one device by
# giving each a GPU fraction. It is wrong once either is scaled up, so it is
# added only when this really is a single-GPU run.
single_gpu_args=()
if [ "${NUM_LEARNERS_PER_NODE}" = "1" ] && [ "${VLLM_NUM_ENGINES}" = "1" ]; then
    single_gpu_args+=(--single_gpu_mode)
    # gloo, because learner and engine share one device as separate processes.
    VLLM_SYNC_BACKEND="${VLLM_SYNC_BACKEND:-gloo}"
else
    # Real multi-GPU: nccl, as the paper's scripts use. gloo would make every
    # weight sync crawl.
    VLLM_SYNC_BACKEND="${VLLM_SYNC_BACKEND:-nccl}"
fi

# Eager mode skips CUDA graph capture: faster to start, slower to generate.
# Right for a smoke test, wrong for a long run -- set VLLM_ENFORCE_EAGER=0.
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
eager_args=()
if [ "${VLLM_ENFORCE_EAGER}" = "1" ]; then
    eager_args+=(--vllm_enforce_eager)
fi

# The reference policy is a second copy of the model and, with beta 0.0, it
# contributes nothing to the loss. Kept on by default to match the paper;
# set LOAD_REF_POLICY=false to reclaim the memory when scaling the model up.
LOAD_REF_POLICY="${LOAD_REF_POLICY:-true}"

model_name_or_path="${MODEL}"

# Unset means "use the tokenizer's own chat template", which is right for Qwen.
# OLMo needs an explicit one -- the paper's scripts pass olmo_thinker -- so set
# CHAT_TEMPLATE_NAME=olmo_thinker when switching to Olmo-3-7B-Think-DPO.
chat_template_args=()
if [ -n "${CHAT_TEMPLATE_NAME:-}" ]; then
    chat_template_args+=(--chat_template_name "${CHAT_TEMPLATE_NAME}")
fi

# Name the run after what distinguishes it, so a sweep is legible in wandb and
# on disk: slr-permission-Qwen3-1.7B, slr-neutral-Qwen3-1.7B, and so on.
# Every factor that distinguishes a run belongs in the name: variant, model,
# tier and seed. Seed especially -- checkpoint_state_dir hangs off OUTPUT_DIR,
# so two seeds of the same condition would otherwise overwrite each other's
# resumable state.
SEED="${SEED:-3}"
exp_name="slr-${SLR_PROMPT_VARIANT}-$(basename "${MODEL}")-${DATASET_CONFIG}-s${SEED}"
EXP_NAME="${EXP_NAME:-${exp_name}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/${EXP_NAME}}"

# Mid-run saves matter for more than safety: your retreat plan evaluates the
# trained model on the neutral prompt, and comparing the hacking gap OVER
# training needs intermediate checkpoints, not just the final weights.
# checkpoint_state_* is the resumable optimizer state, which is what lets a
# multi-day run survive a Slurm time limit.
SAVE_FREQ="${SAVE_FREQ:-25}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:-50}"
# Generation runs this many steps ahead of training. The paper uses 8; 1 keeps
# the loop synchronous and easier to read while debugging.
ASYNC_STEPS="${ASYNC_STEPS:-1}"

mkdir -p "${OUTPUT_DIR}/rollouts"

# Weights & Biases. Enabled automatically when a key is present, because the
# failure mode otherwise is nasty: wandb falls back to an interactive login
# prompt, which in an sbatch job means the run hangs until the time limit
# rather than failing. If tracking is forced on without a key we go offline
# instead (sync later with `wandb sync`).
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-slr-inoculation}"
WITH_TRACKING="${WITH_TRACKING:-auto}"
if [ "${WITH_TRACKING}" = "auto" ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        WITH_TRACKING=1
    else
        WITH_TRACKING=0
    fi
fi
tracking_args=()
if [ "${WITH_TRACKING}" = "1" ]; then
    tracking_args+=(--with_tracking --wandb_project_name "${WANDB_PROJECT_NAME}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        tracking_args+=(--wandb_entity "${WANDB_ENTITY}")
    fi
    if [ -z "${WANDB_API_KEY:-}" ]; then
        export WANDB_MODE="${WANDB_MODE:-offline}"
    fi
fi

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Direct GRPO launch (SLR-Bench single-GPU smoke test)"
echo "============================================================"
echo "Repo root:       ${REPO_ROOT}"
echo "Python:          $(command -v python)"
echo "Experiment:      ${EXP_NAME}"
echo "Output dir:      ${OUTPUT_DIR}"
if [ "${WITH_TRACKING}" = "1" ]; then
    echo "Tracking:        wandb project ${WANDB_PROJECT_NAME} (mode ${WANDB_MODE:-online})"
else
    echo "Tracking:        off (set WANDB_API_KEY in secrets.env to enable)"
fi
echo "Model:           ${model_name_or_path}"
echo "Dataset:         SLR-Bench ${DATASET_CONFIG}"
echo "vLLM memory:     ${VLLM_GPU_MEMORY_UTILIZATION} of the card"
echo "Budget:          prompt ${MAX_PROMPT_TOKEN_LENGTH} + response ${RESPONSE_LENGTH} = pack ${PACK_LENGTH}"
echo "HOME (writable): ${HOME}"
echo "SSL_CERT_FILE:   ${SSL_CERT_FILE}"
echo "HF_HOME:         ${HF_HOME}"
echo "Dataset cache:   ${DATASET_LOCAL_CACHE_DIR}"
echo "Python extras:   ${OI_PYEXTRA}"
echo "Train prompt:    ${SLR_PROMPT_VARIANT} (paraphrase ${SLR_PROMPT_PARAPHRASE_IDX}, ${SLR_PROMPT_POSITION})"
echo "Eval prompt:     ${SLR_EVAL_PROMPT_VARIANT}"
echo "============================================================"

python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
swipl --version

if ! python -c "import openenv" 2>/dev/null; then
    echo "ERROR: cannot import openenv. Run this once from a machine with internet:" >&2
    echo "  bash scripts/train/debug/stage_python_extras.sh" >&2
    exit 1
fi
python -c "import openenv; print('openenv loaded from:', openenv.__file__)"
python -c "import open_instruct; print('open_instruct loaded from:', open_instruct.__file__)"

# Before anything imports ground_truth_utils: open_instruct/IFEvalG downloads
# punkt at import time, and inside the container $HOME is read-only, so
# without NLTK_DATA pointing somewhere writable that import dies.
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

# Exercise the whole reward path (swipl, parser, both judges) on canned
# answers before spending GPU time. SKIP_VERIFIER_PREFLIGHT=1 to skip.
if [ "${SKIP_VERIFIER_PREFLIGHT:-0}" != "1" ]; then
    python scripts/train/debug/slr_verifier_preflight.py --parsing simple --reward-function partial
fi

# ---------------------------------------------------------------------------
# Ray — one node, so a local head is all that's needed (no multi-node
# coordination like the 8-node reward_hacking_scripts). RAY_ADDRESS is
# exported so grpo_fast.py attaches to that head rather than starting a
# second cluster of its own.
# ---------------------------------------------------------------------------

RAY_PORT="${RAY_PORT:-6379}"
ray stop --force 2>/dev/null || true
ray start --head --port="${RAY_PORT}" --dashboard-host=0.0.0.0
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

# ---------------------------------------------------------------------------
# Training
#
# Steps per run = TOTAL_EPISODES / (NUM_UNIQUE_PROMPTS_ROLLOUT x
# NUM_SAMPLES_PER_PROMPT_ROLLOUT). At the defaults that is 200 / 64 = 3 steps,
# which is a plumbing check; the paper's hacking gap only separates around
# step 250, so a real run wants TOTAL_EPISODES in the tens of thousands.
#
# The settings below match olmo3-think-no-isoRL.sh wherever the choice affects
# WHAT is learned rather than how fast: temperature 1.0, lr 1e-6 held constant,
# 8 samples per prompt, clip_higher 0.272, centered advantages, importance
# sampling cap 2.0, beta 0. Keeping these aligned is what makes results
# comparable to the paper. Batch shape and engine counts are deliberately NOT
# copied -- those describe their 64 GPUs, not yours.
# ---------------------------------------------------------------------------

python open_instruct/grpo_fast.py \
    --exp_name "${EXP_NAME}" \
    --dataset_mixer_list ${dataset_mixer_train} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ${dataset_mixer_eval} \
    --dataset_mixer_eval_list_splits test \
    --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
    --response_length "${RESPONSE_LENGTH}" \
    --pack_length "${PACK_LENGTH}" \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
    --num_mini_batches 1 \
    --model_name_or_path "${model_name_or_path}" \
    ${chat_template_args[@]+"${chat_template_args[@]}"} \
    --output_dir "${OUTPUT_DIR}" \
    --rollouts_save_path "${OUTPUT_DIR}/rollouts" \
    --save_freq "${SAVE_FREQ}" \
    --checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}" \
    --checkpoint_state_dir "${OUTPUT_DIR}/checkpoints" \
    --async_steps "${ASYNC_STEPS}" \
    --backend_timeout 1200 \
    --eval_receive_timeout 600 \
    --dataset_local_cache_dir "${DATASET_LOCAL_CACHE_DIR}" \
    --apply_verifiable_reward true \
    --ground_truths_key ground_truth \
    --sft_messages_key prompt \
    --slr_reward base \
    --slr_reward_function partial \
    --slr_parsing simple \
    --temperature 1.0 \
    --inflight_updates True \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --clip_higher 0.272 \
    --advantage_normalization_type centered \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --kl_estimator 2 \
    --total_episodes "${TOTAL_EPISODES}" \
    --deepspeed_stage "${DEEPSPEED_STAGE}" \
    --deepspeed_offload_optimizer "${DEEPSPEED_OFFLOAD_OPTIMIZER}" \
    --num_epochs 1 \
    --num_learners_per_node "${NUM_LEARNERS_PER_NODE}" \
    --vllm_num_engines "${VLLM_NUM_ENGINES}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend "${VLLM_SYNC_BACKEND}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    ${eager_args[@]+"${eager_args[@]}"} \
    --beta 0.0 \
    --load_ref_policy "${LOAD_REF_POLICY}" \
    --seed "${SEED}" \
    --local_eval_every "${LOCAL_EVAL_EVERY}" \
    --save_traces \
    --gradient_checkpointing \
    --push_to_hub false \
    ${tracking_args[@]+"${tracking_args[@]}"} \
    ${single_gpu_args[@]+"${single_gpu_args[@]}"}

ray stop --force 2>/dev/null || true
