#!/bin/bash
# Stage the Python packages this checkout needs but the container image
# predates, into a directory that gets appended to PYTHONPATH.
#
#   bash scripts/train/debug/stage_python_extras.sh
#
# Run it once, from anywhere with internet access (the login node works, and
# so do the compute nodes); the result lives on shared storage, so jobs just
# read it.
#
# ---------------------------------------------------------------------------
# Why not `uv sync` / `pip install`?
# ---------------------------------------------------------------------------
# The image ships a complete, working venv at /stage/.venv: 266 packages,
# CUDA-matched torch, vllm, flash-attn, tens of GB. The only thing wrong with
# it is that this fork's pyproject.toml has since grown `openenv-core`, which
# open_instruct/environments/* imports at module scope and grpo_fast.py
# therefore needs at startup. Rebuilding that venv to add a few hundred KB of
# pure Python would be absurd, and `uv run` inside the repo does exactly that.
#
# Packages are unpacked WITHOUT their dependency closure, deliberately.
# Anything on PYTHONPATH shadows the image's site-packages, so installing the
# full closure would put a second numpy/pydantic/fastapi in front of the ones
# torch and vLLM were built against. The list below is instead the closure of
# openenv-core MINUS what the image already has, so nothing here shadows
# anything -- each package is simply absent from the image.
#
# Versions come from this fork's uv.lock. To re-derive the list after changing
# the lock or the image, take the uv.lock dependency closure of openenv-core
# and subtract the image's site-packages inventory.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OI_PYEXTRA="${OI_PYEXTRA:-${REPO_ROOT}/.cache/pyextra}"

PACKAGES=(
    "openenv-core==0.2.1"
    "fastmcp==2.11.3"
    "authlib==1.6.8"
    "cyclopts==4.5.3"
    "docutils==0.22.4"
    "exceptiongroup==1.3.1"
    "isodate==0.7.2"
    "jsonschema-path==0.3.4"
    "lazy-object-proxy==1.12.0"
    "more-itertools==10.8.0"
    "openapi-core==0.22.0"
    "openapi-pydantic==0.5.1"
    "openapi-schema-validator==0.6.3"
    "openapi-spec-validator==0.7.2"
    "pathable==0.4.4"
    "pyperclip==1.11.0"
    "rfc3339-validator==0.1.4"
    "rich-rst==1.3.2"
    "tomli==2.4.0"
    "tomli-w==1.2.0"
)

mkdir -p "${OI_PYEXTRA}"
echo "Staging ${#PACKAGES[@]} packages into ${OI_PYEXTRA}"

python3 - "${OI_PYEXTRA}" "${PACKAGES[@]}" <<'PY'
import io
import json
import sys
import urllib.request
import zipfile

target, *specs = sys.argv[1:]

for spec in specs:
    name, _, version = spec.partition("==")
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as response:
        release = json.load(response)

    # Prefer a pure-Python wheel; fall back to one built for the image's
    # interpreter and platform (CPython 3.12 on manylinux x86_64). A wheel
    # tagged for another interpreter -- PyPy, say -- must not be picked just
    # because its payload happens to be pure Python.
    def pick(predicate):
        return next((f for f in release["urls"] if f["packagetype"] == "bdist_wheel" and predicate(f["filename"])), None)

    wheel = (
        pick(lambda n: n.endswith("-py3-none-any.whl"))
        or pick(lambda n: n.endswith("-py2.py3-none-any.whl"))
        or pick(lambda n: "cp312" in n and "manylinux" in n and "x86_64" in n)
    )
    if wheel is None:
        raise SystemExit(f"{spec}: no wheel on PyPI for CPython 3.12 / manylinux x86_64")
    with urllib.request.urlopen(wheel["url"], timeout=120) as response:
        payload = response.read()
    zipfile.ZipFile(io.BytesIO(payload)).extractall(target)
    print(f"  {wheel['filename']}")
PY

echo
echo "Done. The launch scripts put this on PYTHONPATH automatically."
