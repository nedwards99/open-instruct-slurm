FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles" \
    LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    make \
    sudo \
    nginx \
    swi-prolog \
    && apt-get autoremove -y \
    && mkdir -p /etc/nginx/conf.d \
    && rm -rf /var/lib/apt/lists/*
# swi-prolog is required at runtime: open_instruct/slr/slr_verifier.py shells
# out to `swipl` to score SLR-Bench predictions. Without it, every
# SLRBenchVerifier reward call fails.

# This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# NOTE: upstream also installs Mellanox MFT/DOCA-OFED drivers, the Google
# Cloud CLI, and the Beaker CLI here. All three are Ai2-infrastructure-specific
# (their InfiniBand NICs, GCS storage, and Beaker job submission respectively)
# and irrelevant for a plain Slurm+Pyxis setup that calls open_instruct/grpo_fast.py
# directly. Removed them — the Beaker CLI download in particular has a 10s
# --max-time hitting beaker.org, which is what was timing out on this network.

COPY --from=ghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/

WORKDIR /stage/

ENV UV_CACHE_DIR=/root/.cache/uv \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UV_COMPILE_BYTECODE=0

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv run --frozen python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Separate COPY commands required: Docker copies directory *contents*, not the directory itself
COPY configs configs
COPY scripts scripts
COPY mason.py mason.py
COPY open_instruct open_instruct
COPY oe-eval-interna[l] oe-eval-internal/

ARG GIT_COMMIT="" \
    GIT_BRANCH=""

ENV GIT_COMMIT=${GIT_COMMIT} \
    GIT_BRANCH=${GIT_BRANCH} \
    PATH=/stage/.venv/bin:$PATH
