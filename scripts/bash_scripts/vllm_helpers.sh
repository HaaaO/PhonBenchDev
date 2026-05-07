#!/usr/bin/env bash

# Shared helpers for vLLM-Omni benchmark scripts.
# Configure with:
#   QWEN25_VLLM_MODEL=Qwen/Qwen2.5-Omni-3B
#   QWEN25_VLLM_PORT=8091
#   QWEN25_VLLM_ARGS="--stage-configs-path /path/to/stage.yaml"
#   VLLM_EXECUTABLE=/path/to/vllm_omni.sif
#   QWEN25_VLLM_BIN=/path/to/qwen25/vllm
#   QWEN3_VLLM_BIN=/path/to/qwen3/vllm

QWEN25_VLLM_MODEL="${QWEN25_VLLM_MODEL:-Qwen/Qwen2.5-Omni-3B}"
QWEN25_VLLM_HOST="${QWEN25_VLLM_HOST:-127.0.0.1}"
QWEN25_VLLM_PID="${QWEN25_VLLM_PID:-}"
QWEN25_VLLM_PORT="${QWEN25_VLLM_PORT:-}"
QWEN25_VLLM_LOG="${QWEN25_VLLM_LOG:-}"
VLLM_BIN="${VLLM_BIN:-vllm}"   # override to the server-env's vllm binary
QWEN25_VLLM_BIN="${QWEN25_VLLM_BIN:-$VLLM_BIN}"

# Default to single-GPU stage config (all stages on device 0).
# Override QWEN25_VLLM_ARGS to use a different layout (e.g. multi-GPU).
_VLLM_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN25_VLLM_ARGS="${QWEN25_VLLM_ARGS:---stage-configs-path ${_VLLM_HELPERS_DIR}/qwen25_omni_single_gpu.yaml}"

# Triton JIT-compiles a CUDA driver wrapper at startup and needs Python.h.
# The system Python lacks dev headers; borrow them from a FASRC python module.
# We load the module just long enough to read its include path, then unload so
# the active venv's python keeps priority on PATH. Only C_INCLUDE_PATH persists.
_VLLM_PY_MODULE="${VLLM_PY_HEADERS_MODULE:-python/3.12.11-fasrc02}"
if [[ -z "${_VLLM_PYTHON_H_READY:-}" ]] && command -v module >/dev/null 2>&1; then
    if module load "$_VLLM_PY_MODULE" 2>/dev/null; then
        _vllm_inc="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))' 2>/dev/null)"
        module unload "$_VLLM_PY_MODULE" 2>/dev/null || true
        if [[ -n "$_vllm_inc" && -f "$_vllm_inc/Python.h" ]]; then
            export C_INCLUDE_PATH="${_vllm_inc}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
            export _VLLM_PYTHON_H_READY=1
        else
            echo "vllm_helpers: $_VLLM_PY_MODULE did not yield Python.h; Triton may fail" >&2
        fi
    else
        echo "vllm_helpers: failed to module-load $_VLLM_PY_MODULE; Triton may fail" >&2
    fi
fi

get_free_vllm_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

_qwen25_vllm_is_ready() {
    local port=$1
    curl -fsS "http://${QWEN25_VLLM_HOST}:${port}/health" >/dev/null 2>&1 \
        || curl -fsS "http://${QWEN25_VLLM_HOST}:${port}/v1/models" >/dev/null 2>&1
}

wait_for_vllm_health() {
    local port=$1
    local timeout_seconds=${2:-900}
    local deadline=$((SECONDS + timeout_seconds))

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to wait for vLLM readiness" >&2
        return 1
    fi

    echo "Waiting for Qwen2.5 vLLM server on ${QWEN25_VLLM_HOST}:${port}" >&2
    while (( SECONDS < deadline )); do
        if _qwen25_vllm_is_ready "$port"; then
            echo "Qwen2.5 vLLM server is ready on port ${port}" >&2
            return 0
        fi
        if [[ -n "${QWEN25_VLLM_PID:-}" ]] && ! kill -0 "$QWEN25_VLLM_PID" 2>/dev/null; then
            echo "Qwen2.5 vLLM server exited before becoming ready" >&2
            [[ -n "${QWEN25_VLLM_LOG:-}" ]] && tail -n 80 "$QWEN25_VLLM_LOG" >&2
            return 1
        fi
        sleep 5
    done

    echo "Timed out waiting for Qwen2.5 vLLM server on port ${port}" >&2
    [[ -n "${QWEN25_VLLM_LOG:-}" ]] && tail -n 80 "$QWEN25_VLLM_LOG" >&2
    return 1
}

start_qwen25_vllm() {
    local model=${1:-$QWEN25_VLLM_MODEL}
    local port=${QWEN25_VLLM_PORT:-}
    local log_dir=${QWEN25_VLLM_LOG_DIR:-/n/iqss_sponsored/Lab/zshi/slurm_logs}
    local vllm_bin=${QWEN25_VLLM_BIN:-$VLLM_BIN}
    local extra_args=()

    if [[ -z "$port" ]]; then
        port=$(get_free_vllm_port)
    elif _qwen25_vllm_is_ready "$port"; then
        echo "Using existing Qwen2.5 vLLM server on port ${port}" >&2
        QWEN25_VLLM_PORT="$port"
        QWEN25_VLLM_PID=""
        return 0
    fi

    mkdir -p "$log_dir"
    QWEN25_VLLM_PORT="$port"
    QWEN25_VLLM_LOG="${QWEN25_VLLM_LOG:-${log_dir}/qwen25_vllm_${SLURM_JOB_ID:-manual}_${port}.log}"
    if [[ -n "${QWEN25_VLLM_ARGS:-}" ]]; then
        read -r -a extra_args <<< "$QWEN25_VLLM_ARGS"
    fi

    echo "Starting Qwen2.5 vLLM server: model=${model}, port=${port}" >&2
    echo "Qwen2.5 vLLM binary: ${vllm_bin}" >&2
    echo "vLLM log: ${QWEN25_VLLM_LOG}" >&2

    if [[ -x "$vllm_bin" ]] || command -v "$vllm_bin" >/dev/null 2>&1; then
        "$vllm_bin" serve "$model" \
            --omni \
            --host "$QWEN25_VLLM_HOST" \
            --port "$port" \
            "${extra_args[@]}" \
            > "$QWEN25_VLLM_LOG" 2>&1 &
    elif [[ -n "${VLLM_EXECUTABLE:-}" ]] && [[ -f "$VLLM_EXECUTABLE" ]] && command -v apptainer >/dev/null 2>&1; then
        apptainer exec --cleanenv --nv \
            --env HF_HOME="${HF_HOME:-}" \
            -B "${VLLM_BIND:-$PWD}" \
            "$VLLM_EXECUTABLE" \
            vllm serve "$model" \
                --omni \
                --host "$QWEN25_VLLM_HOST" \
                --port "$port" \
                "${extra_args[@]}" \
            > "$QWEN25_VLLM_LOG" 2>&1 &
    else
        echo "Unable to start vLLM-Omni. Install a vllm executable or set VLLM_EXECUTABLE=/path/to/vllm_omni.sif with apptainer available." >&2
        return 1
    fi

    QWEN25_VLLM_PID=$!
    wait_for_vllm_health "$port" "${QWEN25_VLLM_TIMEOUT:-900}"
}

stop_qwen25_vllm() {
    if [[ -n "${QWEN25_VLLM_PID:-}" ]] && kill -0 "$QWEN25_VLLM_PID" 2>/dev/null; then
        echo "Stopping Qwen2.5 vLLM server pid=${QWEN25_VLLM_PID}" >&2
        kill "$QWEN25_VLLM_PID" 2>/dev/null || true
        wait "$QWEN25_VLLM_PID" 2>/dev/null || true
    fi
    QWEN25_VLLM_PID=""
}

# ----- Qwen3-Omni-30B-A3B (Instruct/Thinking) -------------------------------
# Qwen3-Omni needs vLLM-Omni's model registry. The vllm-omni CLI only switches
# to that path when --omni is present; plain vLLM falls through to the generic
# Transformers multimodal wrapper and crashes during processor profiling.
QWEN3_VLLM_MODEL="${QWEN3_VLLM_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
QWEN3_VLLM_HOST="${QWEN3_VLLM_HOST:-127.0.0.1}"
QWEN3_VLLM_PID="${QWEN3_VLLM_PID:-}"
QWEN3_VLLM_PORT="${QWEN3_VLLM_PORT:-}"
QWEN3_VLLM_LOG="${QWEN3_VLLM_LOG:-}"
QWEN3_VLLM_STAGE_CONFIG="${QWEN3_VLLM_STAGE_CONFIG:-${_VLLM_HELPERS_DIR}/qwen3_omni_thinker_single_gpu.yaml}"
# Extra serve flags appended after --stage-configs-path. Engine-level settings
# for vLLM-Omni are in QWEN3_VLLM_STAGE_CONFIG.
QWEN3_VLLM_ARGS="${QWEN3_VLLM_ARGS:-}"
QWEN3_VLLM_BIN="${QWEN3_VLLM_BIN:-$VLLM_BIN}"

_vllm_resolve_bin() {
    local vllm_bin=${1:-$VLLM_BIN}

    if [[ -x "$vllm_bin" ]]; then
        printf '%s\n' "$vllm_bin"
    else
        command -v "$vllm_bin" 2>/dev/null || true
    fi
}

_vllm_python_for_bin() {
    local resolved_bin=${1:-}
    local bin_dir=""

    if [[ -n "$resolved_bin" ]]; then
        bin_dir="$(cd "$(dirname "$resolved_bin")" && pwd)"
        if [[ -x "${bin_dir}/python" ]]; then
            printf '%s\n' "${bin_dir}/python"
            return 0
        fi
        if [[ -x "${bin_dir}/python3" ]]; then
            printf '%s\n' "${bin_dir}/python3"
            return 0
        fi
    fi

    command -v python3 2>/dev/null || command -v python 2>/dev/null || true
}

check_qwen3_vllm_moe_ops() {
    local vllm_bin=${1:-${QWEN3_VLLM_BIN:-$VLLM_BIN}}
    local resolved_bin
    local python_bin

    resolved_bin="$(_vllm_resolve_bin "$vllm_bin")"
    python_bin="$(_vllm_python_for_bin "$resolved_bin")"
    if [[ -z "$python_bin" ]]; then
        echo "Unable to find Python for vLLM preflight check" >&2
        return 1
    fi

    "$python_bin" -c '
import sys
import torch

try:
    import vllm._moe_C  # noqa: F401
except Exception as exc:
    print("vLLM MoE CUDA extension failed to import.", file=sys.stderr)
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)

missing = [
    name for name in ("topk_softmax", "moe_align_block_size", "moe_sum")
    if not hasattr(torch.ops._moe_C, name)
]
if missing:
    print(
        "vLLM MoE CUDA extension is missing required ops: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    sys.exit(1)
' 2>&1
}

_qwen3_vllm_is_ready() {
    local port=$1
    curl -fsS "http://${QWEN3_VLLM_HOST}:${port}/health" >/dev/null 2>&1 \
        || curl -fsS "http://${QWEN3_VLLM_HOST}:${port}/v1/models" >/dev/null 2>&1
}

wait_for_qwen3_vllm_health() {
    local port=$1
    local timeout_seconds=${2:-1800}
    local deadline=$((SECONDS + timeout_seconds))

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to wait for vLLM readiness" >&2
        return 1
    fi

    echo "Waiting for Qwen3 vLLM server on ${QWEN3_VLLM_HOST}:${port}" >&2
    while (( SECONDS < deadline )); do
        if _qwen3_vllm_is_ready "$port"; then
            echo "Qwen3 vLLM server is ready on port ${port}" >&2
            return 0
        fi
        if [[ -n "${QWEN3_VLLM_PID:-}" ]] && ! kill -0 "$QWEN3_VLLM_PID" 2>/dev/null; then
            echo "Qwen3 vLLM server exited before becoming ready" >&2
            [[ -n "${QWEN3_VLLM_LOG:-}" ]] && tail -n 80 "$QWEN3_VLLM_LOG" >&2
            return 1
        fi
        sleep 5
    done

    echo "Timed out waiting for Qwen3 vLLM server on port ${port}" >&2
    [[ -n "${QWEN3_VLLM_LOG:-}" ]] && tail -n 80 "$QWEN3_VLLM_LOG" >&2
    return 1
}

start_qwen3_vllm() {
    local model=${1:-$QWEN3_VLLM_MODEL}
    local port=${QWEN3_VLLM_PORT:-}
    local log_dir=${QWEN3_VLLM_LOG_DIR:-/n/iqss_sponsored/Lab/zshi/slurm_logs}
    local vllm_bin=${QWEN3_VLLM_BIN:-$VLLM_BIN}
    local extra_args=()

    if [[ -z "$port" ]]; then
        port=$(get_free_vllm_port)
    elif _qwen3_vllm_is_ready "$port"; then
        echo "Using existing Qwen3 vLLM server on port ${port}" >&2
        QWEN3_VLLM_PORT="$port"
        QWEN3_VLLM_PID=""
        return 0
    fi

    mkdir -p "$log_dir"
    QWEN3_VLLM_PORT="$port"
    if [[ ! -f "$QWEN3_VLLM_STAGE_CONFIG" ]]; then
        echo "Qwen3 vLLM-Omni stage config not found: ${QWEN3_VLLM_STAGE_CONFIG}" >&2
        return 1
    fi
    if [[ -x "$vllm_bin" ]] || command -v "$vllm_bin" >/dev/null 2>&1; then
        local moe_preflight_output
        if ! moe_preflight_output="$(check_qwen3_vllm_moe_ops "$vllm_bin")"; then
            echo "$moe_preflight_output" >&2
            echo "Qwen3 vLLM-Omni preflight failed before loading weights." >&2
            echo "This model uses Qwen3-MoE layers, so vLLM must have a working _moe_C extension." >&2
            echo "On FasRC, this commonly means rebuilding vLLM on the cluster OS or running it in a container with a compatible glibc/CUDA stack." >&2
            return 1
        fi
    fi
    # Tag the log with the model's last path segment so Instruct/Thinking
    # don't clobber each other when started in the same job.
    local model_tag="${model##*/}"
    QWEN3_VLLM_LOG="${log_dir}/qwen3_vllm_${model_tag}_${SLURM_JOB_ID:-manual}_${port}.log"
    if [[ -n "${QWEN3_VLLM_ARGS:-}" ]]; then
        read -r -a extra_args <<< "$QWEN3_VLLM_ARGS"
    fi

    echo "Starting Qwen3 vLLM server: model=${model}, port=${port}" >&2
    echo "Qwen3 vLLM binary: ${vllm_bin}" >&2
    echo "Qwen3 vLLM-Omni stage config: ${QWEN3_VLLM_STAGE_CONFIG}" >&2
    echo "vLLM log: ${QWEN3_VLLM_LOG}" >&2

    if [[ -x "$vllm_bin" ]] || command -v "$vllm_bin" >/dev/null 2>&1; then
        "$vllm_bin" serve "$model" \
            --omni \
            --host "$QWEN3_VLLM_HOST" \
            --port "$port" \
            --stage-configs-path "$QWEN3_VLLM_STAGE_CONFIG" \
            "${extra_args[@]}" \
            > "$QWEN3_VLLM_LOG" 2>&1 &
    elif [[ -n "${VLLM_EXECUTABLE:-}" ]] && [[ -f "$VLLM_EXECUTABLE" ]] && command -v apptainer >/dev/null 2>&1; then
        apptainer exec --cleanenv --nv \
            --env HF_HOME="${HF_HOME:-}" \
            -B "${VLLM_BIND:-$PWD}" \
            "$VLLM_EXECUTABLE" \
            vllm serve "$model" \
                --omni \
                --host "$QWEN3_VLLM_HOST" \
                --port "$port" \
                --stage-configs-path "$QWEN3_VLLM_STAGE_CONFIG" \
                "${extra_args[@]}" \
            > "$QWEN3_VLLM_LOG" 2>&1 &
    else
        echo "Unable to start Qwen3 vLLM-Omni. Install a vllm executable or set VLLM_EXECUTABLE=/path/to/vllm_omni.sif with apptainer available." >&2
        return 1
    fi

    QWEN3_VLLM_PID=$!
    wait_for_qwen3_vllm_health "$port" "${QWEN3_VLLM_TIMEOUT:-1800}"
}

stop_qwen3_vllm() {
    if [[ -n "${QWEN3_VLLM_PID:-}" ]] && kill -0 "$QWEN3_VLLM_PID" 2>/dev/null; then
        echo "Stopping Qwen3 vLLM server pid=${QWEN3_VLLM_PID}" >&2
        kill "$QWEN3_VLLM_PID" 2>/dev/null || true
        wait "$QWEN3_VLLM_PID" 2>/dev/null || true
    fi
    QWEN3_VLLM_PID=""
    QWEN3_VLLM_PORT=""
    QWEN3_VLLM_LOG=""
}
