#!/bin/bash
# Unattended sbatch version of scripts/train/debug/single_gpu_slr_direct.sh.
#
#   sbatch scripts/train/debug/single_gpu_slr_pyxis.sh
#   SLR_PROMPT_VARIANT=permission sbatch --export=ALL,SLR_PROMPT_VARIANT \
#       scripts/train/debug/single_gpu_slr_pyxis.sh
#
# It deliberately holds no training arguments of its own: it allocates the
# node, starts the container, and runs the direct script inside it, so the two
# cannot drift apart. Everything about the run -- model, dataset, GRPO
# hyperparameters, prompt variant -- lives in the direct script.
#
# --mem=64G is not decoration: the first use of this image on a node makes
# pyxis convert it to a squashfs, and that conversion was OOM-killed at 16G.
#SBATCH --job-name=slr-single-gpu-smoke
#SBATCH --partition=p_csunivie_gres,p_datamining
#SBATCH --account=datamining
# dgx1 is the V100 node (compute capability 7.0): vLLM cannot run bf16
# there and dies with "Bfloat16 is only supported on GPUs with compute
# capability of at least 8.0". Every other node is Ampere or newer.
#SBATCH --exclude=dgx1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Flat files, not logs/%x_%j/output.out: Slurm opens these before the script
# body runs and does not create intermediate directories, so a per-job
# subdirectory means the job dies at submission with nowhere to write.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/nlp-data/home/users/nicholase99cs/open-instruct-slurm}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-helffml/open_instruct_dev:slr}"  # Docker Hub is enroot's default registry

# A STABLE container name, not a per-job one: pyxis reuses an existing
# container of this name on the node, so the 4-5 minute registry pull and
# squashfs build happen once per node instead of once per job. The tradeoff is
# that the container filesystem then persists between jobs -- fine here, since
# everything that matters (checkout, caches, outputs) lives on /mnt/nlp-data
# and nothing is installed into the container. To force a rebuild, run
# `enroot remove pyxis_${CONTAINER_NAME}` on the node in question.
CONTAINER_NAME="${CONTAINER_NAME:-open-instruct-slr}"

mkdir -p "${BASE_DIR}/logs"

echo "=========================================="
echo "Job: ${SLURM_JOB_NAME:-slr-single-gpu-smoke} (ID: ${SLURM_JOB_ID:-none})"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Repo: ${BASE_DIR}"
echo "=========================================="

# The repo is NOT bind-mounted over /stage: that would hide the image's own
# venv at /stage/.venv, which is exactly the environment we want to run in.
# The direct script points PYTHONPATH at this checkout instead.
srun --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="/var/spool/slurmd:/var/spool/slurmd" \
  --container-workdir="${BASE_DIR}" \
  --container-name="${CONTAINER_NAME}" \
  bash "${BASE_DIR}/scripts/train/debug/single_gpu_slr_direct.sh"
