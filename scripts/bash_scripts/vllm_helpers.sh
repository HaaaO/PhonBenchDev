#!/usr/bin/env bash

# Shared helpers for vLLM-Omni benchmark scripts.
# Configure with:
#   QWEN25_VLLM_MODEL=Qwen/Qwen2.5-Omni-3B
#   QWEN25_VLLM_PORT=8091
#   QWEN25_VLLM_ARGS="--stage-configs-path /path/to/stage.yaml"
#   VLLM_EXECUTABLE=/path/to/vllm_omni.sif

QWEN25_VLLM_MODEL="${QWEN25_VLLM_MODEL:-Qwen/Qwen2.5-Omni-3B}"
QWEN25_VLLM_HOST="${QWEN25_VLLM_HOST:-127.0.0.1}"
QWEN25_VLLM_PID="${QWEN25_VLLM_PID:-}"
QWEN25_VLLM_PORT="${QWEN25_VLLM_PORT:-}"
QWEN25_VLLM_LOG="${QWEN25_VLLM_LOG:-}"

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
    echo "vLLM log: ${QWEN25_VLLM_LOG}" >&2

    if command -v vllm >/dev/null 2>&1; then
        vllm serve "$model" \
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
