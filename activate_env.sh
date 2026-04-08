# PhonBenchDev environment activation — FASRC-friendly.
#
# This file MUST be sourced, not executed:
#     . ./activate_env.sh        # or:  source ./activate_env.sh
#
# Running it as `./activate_env.sh` will appear to do nothing because the
# venv activation and exported variables would die with the subshell.
#
# Safe to source from any directory; the script resolves its own location.

# ---------------------------------------------------------------------------
# Resolve the project root (the directory this script lives in)
# ---------------------------------------------------------------------------
# ${BASH_SOURCE[0]} only works under bash; fall back to $0 for zsh/sh.
_ENV_SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
PROJECT_ROOT="$(cd "$(dirname "$_ENV_SCRIPT_SOURCE")" && pwd)"
export PROJECT_ROOT
unset _ENV_SCRIPT_SOURCE

# ---------------------------------------------------------------------------
# Cache directories — keep wheel/model/dataset caches off the home quota
# ---------------------------------------------------------------------------
# Override any of these by exporting them BEFORE sourcing this file.
export FASRC_CACHE_BASE="${FASRC_CACHE_BASE:-/n/netscratch/iqss_sponsored/Lab/zshi/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FASRC_CACHE_BASE/uv}"
export PIXI_HOME="${PIXI_HOME:-$FASRC_CACHE_BASE/pixi}"
export HF_HOME="${HF_HOME:-$FASRC_CACHE_BASE/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$UV_CACHE_DIR" "$PIXI_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE"

# ---------------------------------------------------------------------------
# FASRC Lmod modules (harmless no-op on non-FASRC systems)
# ---------------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
    module load ffmpeg     2>/dev/null || true
    module load espeak-ng  2>/dev/null || true
    module load cuda       2>/dev/null || true
fi

# Make pixi-installed binaries (e.g. ffmpeg) and uv visible in PATH
export PATH="$PIXI_HOME/bin:$HOME/.pixi/bin:$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# Activate the project virtualenv
# ---------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: $PROJECT_ROOT/.venv not found. Run setup_uv.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------
echo "PhonBenchDev environment activated"
echo "  PROJECT_ROOT      = $PROJECT_ROOT"
echo "  python            = $(command -v python)"
echo "  HF_HOME           = $HF_HOME"
echo "  UV_CACHE_DIR      = $UV_CACHE_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
    _GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    [ -n "$_GPU_NAME" ] && echo "  GPU               = $_GPU_NAME"
    unset _GPU_NAME
else
    echo "  GPU               = (no nvidia-smi — likely a login/CPU node)"
fi
