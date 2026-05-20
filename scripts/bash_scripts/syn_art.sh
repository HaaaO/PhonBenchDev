#!/usr/bin/env bash
#SBATCH -J syn_art_all_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 1:00:00
#SBATCH -o /n/iqss_sponsored/Lab/zshi/slurm_logs/%x_%j.out
#SBATCH -e /n/iqss_sponsored/Lab/zshi/slurm_logs/%x_%j.out

# Run PRiSM phoneme models on synthetic_articulation (Kaldi-style eval set), then score PER + inventory.
# Submit:  sbatch /n/iqss_sponsored/Lab/zshi/PhonBenchDev/scripts/bash_scripts/syn_art.sh

set -u
mkdir -p /n/iqss_sponsored/Lab/zshi/slurm_logs

export HF_HOME="${HF_HOME:-/n/iqss_sponsored/Lab/zshi/.cache/huggingface}"

cd /n/iqss_sponsored/Lab/zshi/PhonBenchDev
# shellcheck disable=SC1091
source .venv/bin/activate
export VLLM_BIN=/n/iqss_sponsored/Lab/zshi/vllm-omni-env/.venv/bin/vllm
# shellcheck disable=SC1091
source scripts/bash_scripts/vllm_helpers.sh

DATA_DIR=/n/iqss_sponsored/Lab/zshi/prism-evalsets
DATASET=synthetic_articulation
TAG=$(date +%Y%m%d_%H%M%S)

# ===== Inference ==============================================================

# 1. POWSM (attention + CTC)
# python src/main.py \
#     experiment=inference/transcribe_powsm \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_powsm_${TAG}

# 2. POWSM-CTC (CTC only)
# python src/main.py \
#     experiment=inference/transcribe_powsm_ctc \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_powsm_ctc_${TAG}

# # 3. W2V2P-LV60
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft \
#     task_name=inf_${DATASET}_lv60_${TAG}

# # 4. W2V2P-XLSR53
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft \
#     task_name=inf_${DATASET}_xlsr53_${TAG}

# # 5. MultiIPA (ctag)
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns \
#     task_name=inf_${DATASET}_ctag_${TAG}

# # 6. ZIPA-CTC
# python src/main.py \
#     experiment=inference/transcribe_zipactc \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k \
#     task_name=inf_${DATASET}_zipactc_${TAG}

# 7. ZIPA-CTC-NS
# python src/main.py \
#     experiment=inference/transcribe_zipactc \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k \
#     task_name=inf_${DATASET}_zipactc_ns_${TAG}

# 8a. Gemini 2.5 Flash (default in transcribe_gemini.yaml)
python src/main.py \
    experiment=inference/transcribe_gemini \
    data=powsmeval \
    data.dataset_name=${DATASET} \
    data.data_dir=$DATA_DIR \
    data.portable_wavscp=True \
    task_name=inf_${DATASET}_gemini_${TAG}

# 8b. Gemini 3.0 Flash (override model_name on the CLI; verify the exact id
#     against https://ai.google.dev/gemini-api/docs/models if the API 404s)
# python src/main.py \
#     experiment=inference/transcribe_gemini \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.client_config.model_name=gemini-3-flash-preview \
#     task_name=inf_${DATASET}_gemini3_${TAG}

# 8c. GPT-audio-1.5
# python src/main.py \
#     experiment=inference/transcribe_gptaudio \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_gptaudio_${TAG}

# 8c2. GPT-Realtime-2 via OpenAI Realtime WebSocket
# python src/main.py \
#     experiment=inference/transcribe_gptrealtime2 \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_gptrealtime2_${TAG}

# 8d. Gemini 2.5 Flash + canonical IPA prompt
# python src/main.py \
#     experiment=inference/transcribe_gemini \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     task_name=inf_${DATASET}_gemini_canonical_${TAG}

# 8e. GPT-audio-1.5 + canonical IPA prompt
# python src/main.py \
#     experiment=inference/transcribe_gptaudio \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     task_name=inf_${DATASET}_gptaudio_canonical_${TAG}

# 8e2. GPT-Realtime-2 + canonical IPA prompt
# python src/main.py \
#     experiment=inference/transcribe_gptrealtime2 \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     task_name=inf_${DATASET}_gptrealtime2_canonical_${TAG}

# 8f. Qwen2.5-Omni-3B via vLLM-Omni (plain + canonical IPA prompts)
# start_qwen25_vllm || { echo "Aborting: vLLM-Omni server failed to start" >&2; exit 1; }
# trap stop_qwen25_vllm EXIT
# python src/main.py \
#     experiment=inference/transcribe_qwen25omni3b \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.port=${QWEN25_VLLM_PORT} \
#     inference.num_workers=1 \
#     task_name=inf_${DATASET}_qwen25omni3b_${TAG}
# python src/main.py \
#     experiment=inference/transcribe_qwen25omni3b \
#     prompt=transcribe_ipa_canonical \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     data.require_canonical=True \
#     inference.port=${QWEN25_VLLM_PORT} \
#     task_name=inf_${DATASET}_qwen25omni3b_canonical_${TAG}


# 9. BabAR (BabyHuBERT + MLP phoneme head, TinyVox-trained)
# python src/main.py \
#     experiment=inference/transcribe_babar \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_babar_${TAG}

# # 10. HuPER (WavLM phone recognizer, ARPAbet -> IPA)
# python src/main.py \
#     experiment=inference/transcribe_huper \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_huper_${TAG}

# # 11. HuPER Corrector (audio + canonical IPA -> realized IPA)
# python src/main.py \
#     experiment=inference/transcribe_huper_corrector \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.canonical_file=$DATA_DIR/$DATASET/text.canonical \
#     task_name=inf_${DATASET}_huper_corrector_${TAG}

# ===== Scoring ================================================================
echo
echo "=== Scoring ($(date)) ==="

MODELS=(powsm powsm_ctc lv60 xlsr53 ctag zipactc zipactc_ns gemini gemini3 gptaudio gptrealtime2 gemini_canonical gptaudio_canonical gptrealtime2_canonical qwen25omni3b qwen25omni3b_canonical babar huper huper_corrector)

for mv in "${MODELS[@]}"; do
    task_name="inf_${DATASET}_${mv}_${TAG}"
    out_base="/n/iqss_sponsored/Lab/zshi/PhonBenchDev/exp/runs/${task_name}"

    # Hydra writes to <out_base>/<timestamp>/ — pick the newest.
    run_dir=$(find "$out_base" -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
                | sort -nr | awk 'NR==1 {print $2}')
    if [[ -z "$run_dir" ]]; then
        echo "SKIP $mv: no hydra run dir under $out_base"
        continue
    fi
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

    echo "--- $mv ---"
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

# stop_qwen25_vllm
# trap - EXIT

echo
echo "=== DONE: $(date) ==="
echo "outputs under: /n/iqss_sponsored/Lab/zshi/PhonBenchDev/exp/runs/inf_${DATASET}_*_${TAG}"
