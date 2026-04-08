#!/bin/bash
echo $1

virtual_env_dir=${1:-".venv"}
requirements_file=${2:-"requirements.txt"}

# ---------------------------------------------------------------------------
# FASRC / SLURM-friendly setup
#
# IMPORTANT: run this inside an interactive GPU allocation, e.g.
#     salloc -p gpu --gres=gpu:1 -c 8 --mem=32G -t 02:00:00
# Do NOT run on a login node — the torch/k2/espnet install is too heavy.
# ---------------------------------------------------------------------------

# Point package caches at netscratch so we don't blow the home-dir quota.
# Override by exporting these before calling the script.
FASRC_CACHE_BASE=${FASRC_CACHE_BASE:-/n/netscratch/iqss_sponsored/Lab/zshi/.cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FASRC_CACHE_BASE/uv}
export PIXI_HOME=${PIXI_HOME:-$FASRC_CACHE_BASE/pixi}
export HF_HOME=${HF_HOME:-$FASRC_CACHE_BASE/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
mkdir -p "$UV_CACHE_DIR" "$PIXI_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE"

# Try to load FASRC Lmod modules for system-level deps. Harmless elsewhere.
if command -v module >/dev/null 2>&1; then
    module load ffmpeg     2>/dev/null || true
    # phonemizer needs espeak-ng at runtime
    module load espeak-ng  2>/dev/null || true
    # CUDA: torch/k2 wheels bundle their own runtime, but loading a recent
    # cuda module ensures the driver/toolkit are visible for any from-source
    # builds. Adjust the version to whatever is current on FASRC.
    module load cuda       2>/dev/null || true
fi

# Check pixi
if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi not found. Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    # pixi installer drops the binary in $PIXI_HOME/bin (or ~/.pixi/bin) but
    # does NOT update the current shell's PATH — add it explicitly.
    export PATH="$PIXI_HOME/bin:$HOME/.pixi/bin:$PATH"
else
    echo "pixi is already installed"
fi

# Check uv
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
    # Same story for uv: installer writes to ~/.local/bin/env but only
    # touches future shells. Source it now (or fall back to PATH).
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed"
fi

# Only fall back to pixi-installed ffmpeg if the module load above didn't
# provide it. On FASRC the module is preferred.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found via module — installing via pixi..."
    pixi global install ffmpeg
fi

# If .venv doesn't exist, create it
if [ ! -d "$virtual_env_dir" ]; then
    echo "Creating $virtual_env_dir..."
    uv venv -p 3.10 $virtual_env_dir
else
    echo "$virtual_env_dir already exists"
fi

# Activate the virtual environment
echo "Activating $virtual_env_dir..."
. $virtual_env_dir/bin/activate


uv pip install -r $requirements_file

# Install icefall after the main requirements to avoid conflicts with espnet.
if [ ! -d "icefall" ]; then
    echo "Installing icefall..."
    git clone https://github.com/k2-fsa/icefall
fi
cd icefall
rm -rf .git
uv pip install -r requirements.txt
uv pip install -e .
cd ..

echo "Setup complete."