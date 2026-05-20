#!/usr/bin/env bash
#SBATCH -J cmu_kids_qweninstruct
#SBATCH -p gpu
# Qwen3-Omni-30B at bf16 needs ~60 GB VRAM; request an 80 GB-class GPU.
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -o /n/iqss_sponsored/Lab/zshi/slurm_logs/%x_%j.out
#SBATCH -e /n/iqss_sponsored/Lab/zshi/slurm_logs/%x_%j.out

# Run Qwen/Qwen3-Omni-30B-A3B-Instruct on cmu_kids_final_kaldi,
# then score PER + inventory.
# Submit:  sbatch /n/iqss_sponsored/Lab/zshi/PhonBenchDev/scripts/bash_scripts/cmu_kids_qwen.sh

set -u
mkdir -p /n/iqss_sponsored/Lab/zshi/slurm_logs

export HF_HOME="${HF_HOME:-/n/iqss_sponsored/Lab/zshi/.cache/huggingface}"

cd /n/iqss_sponsored/Lab/zshi/PhonBenchDev
# shellcheck disable=SC1091
source .venv/bin/activate
export QWEN25_VLLM_BIN=/n/iqss_sponsored/Lab/zshi/vllm-omni-env/.venv/bin/vllm
export QWEN3_VLLM_BIN=/n/iqss_sponsored/Lab/zshi/vllm-omni-env-src/.venv/bin/vllm
# shellcheck disable=SC1091
source scripts/bash_scripts/vllm_helpers.sh

DATA_DIR=/n/iqss_sponsored/Lab/zshi/prism-evalsets
DATASET=cmu_kids_final_kaldi
TAG=$(date +%Y%m%d_%H%M%S)

# ===== Inference ==============================================================

# 1. POWSM (attention + CTC)
# python src/main.py \
#     experiment=inference/transcribe_powsm \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_powsm_${TAG}

# 2. POWSM-CTC (CTC only)
# python src/main.py \
#     experiment=inference/transcribe_powsm_ctc \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_powsm_ctc_${TAG}

# 3. W2V2P-LV60
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft \
#     task_name=inf_${DATASET}_lv60_${TAG}

# 4. W2V2P-XLSR53
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft \
#     task_name=inf_${DATASET}_xlsr53_${TAG}

# 5. MultiIPA (ctag)
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns \
#     task_name=inf_${DATASET}_ctag_${TAG}

# 6. ZIPA-CTC
# python src/main.py \
#     experiment=inference/transcribe_zipactc \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k \
#     task_name=inf_${DATASET}_zipactc_${TAG}

# 7. ZIPA-CTC-NS
# python src/main.py \
#     experiment=inference/transcribe_zipactc \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k \
#     task_name=inf_${DATASET}_zipactc_ns_${TAG}

# 8a. Gemini 2.5 Flash (default in transcribe_gemini.yaml)
# python src/main.py \
#     experiment=inference/transcribe_gemini \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_gemini_${TAG}

# 8b. Gemini 3.0 Flash (override model_name on the CLI; verify the exact id
#     against https://ai.google.dev/gemini-api/docs/models if the API 404s)
# python src/main.py \
#     experiment=inference/transcribe_gemini \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.inference_runner.client_config.model_name=gemini-3-flash-preview \
#     task_name=inf_${DATASET}_gemini3_${TAG}

# 8c. GPT-audio-1.5
# python src/main.py \
#     experiment=inference/transcribe_gptaudio \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_gptaudio_${TAG}

# 8c2. GPT-Realtime-2 via OpenAI Realtime WebSocket (buffer baseline)
# python src/main.py \
#     experiment=inference/transcribe_gptrealtime2 \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_gptrealtime2_${TAG}

# 8c3. GPT-Realtime-2 via response.create input + forced tool call
# python src/main.py \
#     experiment=inference/transcribe_gptrealtime2_response_input_tool \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_gptrealtime2_response_input_tool_${TAG}

# 8d. Gemini 2.5 Flash + canonical IPA prompt
# python src/main.py \
#     experiment=inference/transcribe_gemini \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     task_name=inf_${DATASET}_gemini_canonical_${TAG}

# 8e. GPT-audio-1.5 + canonical IPA prompt
# python src/main.py \
#     experiment=inference/transcribe_gptaudio \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     task_name=inf_${DATASET}_gptaudio_canonical_${TAG}

# 8e2. GPT-Realtime-2 + canonical IPA prompt
# python src/main.py \
#     experiment=inference/transcribe_gptrealtime2 \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     task_name=inf_${DATASET}_gptrealtime2_canonical_${TAG}

# 8f. Qwen2.5-Omni-3B via vLLM-Omni (plain + canonical IPA prompts)
start_qwen25_vllm || { echo "Aborting: vLLM-Omni server failed to start" >&2; exit 1; }
trap stop_qwen25_vllm EXIT
python src/main.py \
    experiment=inference/transcribe_qwen25omni3b \
    data=powsmeval \
    data.dataset_name=${DATASET} \
    data.data_dir=$DATA_DIR/$DATASET \
    data.portable_wavscp=True \
    inference.port=${QWEN25_VLLM_PORT} \
    inference.num_workers=1 \
    task_name=inf_${DATASET}_qwen25omni3b_${TAG}
# python src/main.py \
#     experiment=inference/transcribe_qwen25omni3b \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     inference.port=${QWEN25_VLLM_PORT} \
#     task_name=inf_${DATASET}_qwen25omni3b_canonical_${TAG}
# vLLM-Omni shutdown is slow (~25 min). When Qwen3 blocks below are enabled
# we must pay it now to free the GPU; when they are commented out, the
# post-scoring stop_qwen25_vllm at the bottom of this script runs instead.

# 8g. Qwen3-Omni-30B-A3B-Instruct via vLLM-Omni
# Sequential with 8h: a single 80 GB GPU can only hold one 30 B Qwen3-Omni
# model at a time. We start Instruct, run inference, then stop it before
# launching Thinking.
# stop_qwen25_vllm
# trap - EXIT
# echo
# echo "=== Qwen3 Instruct inference ($(date)) ==="
# echo "model: Qwen/Qwen3-Omni-30B-A3B-Instruct"
# echo "dataset: ${DATA_DIR}/${DATASET}"
# start_qwen3_vllm "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
#     || { echo "Aborting: Qwen3-Instruct vLLM failed to start" >&2; exit 1; }
# trap stop_qwen3_vllm EXIT
# python src/main.py \
#     experiment=inference/transcribe_qweninstruct \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.port=${QWEN3_VLLM_PORT} \
#     inference.num_workers=1 \
#     task_name=inf_${DATASET}_qweninstruct_${TAG}
# stop_qwen3_vllm
# trap - EXIT

# 8h. Qwen3-Omni-30B-A3B-Thinking via vLLM-Omni
# start_qwen3_vllm "Qwen/Qwen3-Omni-30B-A3B-Thinking" \
#     || { echo "Aborting: Qwen3-Thinking vLLM failed to start" >&2; exit 1; }
# trap stop_qwen3_vllm EXIT
# python src/main.py \
#     experiment=inference/transcribe_qwenthinking \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.port=${QWEN3_VLLM_PORT} \
#     inference.num_workers=1 \
#     task_name=inf_${DATASET}_qwenthinking_${TAG}
# stop_qwen3_vllm
# trap - EXIT

# 9. BabAR (BabyHuBERT + MLP phoneme head, TinyVox-trained)
# python src/main.py \
#     experiment=inference/transcribe_babar \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_babar_${TAG}

# 10. HuPER (WavLM phone recognizer, ARPAbet -> IPA)
# python src/main.py \
#     experiment=inference/transcribe_huper \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_huper_${TAG}

# 11. HuPER Corrector (audio + canonical IPA -> realized IPA)
# python src/main.py \
#     experiment=inference/transcribe_huper_corrector \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     inference.inference_runner.canonical_file=$DATA_DIR/$DATASET/text.canonical \
#     task_name=inf_${DATASET}_huper_corrector_${TAG}

# 12a. Azure Pronunciation Assessment scripted
#      (audio + word-level canonical script -> IPA phones)
# Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in the environment.
# python src/main.py \
#     experiment=inference/transcribe_azure_pronunciation \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     data.require_word_canonical=True \
#     inference.inference_runner.use_reference_text=True \
#     task_name=inf_${DATASET}_azure_scripted_${TAG}

# 12b. Azure Pronunciation Assessment unscripted
#      (audio only, no target word-level canonical script)
# python src/main.py \
#     experiment=inference/transcribe_azure_pronunciation \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR/$DATASET \
#     data.portable_wavscp=True \
#     data.require_word_canonical=False \
#     inference.inference_runner.use_reference_text=False \
#     task_name=inf_${DATASET}_azure_unscripted_${TAG}

# ===== Scoring ================================================================
echo
echo "=== Scoring ($(date)) ==="

MODELS=(qweninstruct)
RUNS_ROOT=/n/iqss_sponsored/Lab/zshi/PhonBenchDev/exp/runs

for mv in "${MODELS[@]}"; do
    pattern="${RUNS_ROOT}/inf_${DATASET}_${mv}_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]"

    # Score every matching non-Old Hydra run dir, newest first.
    mapfile -t run_dirs < <(find $pattern -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
                | awk '$2 !~ "/exp/runs/Old/"' \
                | sort -nr | awk '{print $2}')
    if [[ ${#run_dirs[@]} -eq 0 ]]; then
        echo "SKIP $mv: no non-Old hydra run dir matching $pattern"
        continue
    fi

    for run_dir in "${run_dirs[@]}"; do
        if [[ -s "$run_dir/inventory_results.txt" ]]; then
            echo "SKIP $mv: already evaluated ($run_dir/inventory_results.txt)"
            continue
        fi

        # Merge per-worker transcription.<i>.jsonl -> transcription.json
        python scripts/jsonl2json.py --dirname "$run_dir"

        pred="$run_dir/transcription.json"
        if [[ ! -s "$pred" ]]; then
            echo "SKIP $mv: empty/missing $pred"
            continue
        fi
        if python - "$pred" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
if not data:
    raise SystemExit(0)
if all(str(key).startswith("__error__") for key in data):
    raise SystemExit(0)
raise SystemExit(1)
PY
        then
            echo "SKIP $mv: transcription contains only worker setup errors"
            continue
        fi

        echo "--- $mv: $run_dir ---"
        python -m src.metrics.phone_recognition \
            --evaluation_name "$mv" \
            --prediction_file "$pred" \
            --output_file "$run_dir/inventory_results.csv" \
            --gt_field target \
            --pred_field processed_transcript \
            --key_field utt_id \
            --language_field lang_sym \
            --canonical_file "$DATA_DIR/$DATASET/text.canonical"
        echo "    results: $run_dir/inventory_results.csv"
    done
done

stop_qwen25_vllm
trap - EXIT

# stop_qwen3_vllm
# trap - EXIT

echo
echo "=== DONE: $(date) ==="
echo "outputs under: /n/iqss_sponsored/Lab/zshi/PhonBenchDev/exp/runs/inf_${DATASET}_*_${TAG}"
