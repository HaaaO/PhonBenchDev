#!/usr/bin/env bash
#SBATCH -J cmu_kids_all_eval
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /n/iqss_sponsored/Lab/zshi/slurm_logs/%x_%j.out
#SBATCH -e /n/iqss_sponsored/Lab/zshi/slurm_logs/%x_%j.out

# Run PRiSM models on cmu_kids (Kaldi-style eval set), then score PER + inventory.
# Submit:  sbatch /n/iqss_sponsored/Lab/zshi/PhonBenchDev/scripts/bash_scripts/cmu_kids.sh

set -u
mkdir -p /n/iqss_sponsored/Lab/zshi/slurm_logs

export HF_HOME="${HF_HOME:-/n/iqss_sponsored/Lab/zshi/.cache/huggingface}"

cd /n/iqss_sponsored/Lab/zshi/PhonBenchDev
# shellcheck disable=SC1091
source .venv/bin/activate

DATA_DIR=/n/iqss_sponsored/Lab/zshi/prism-evalsets
DATASET=cmu_kids
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

# 3. W2V2P-LV60
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft \
#     task_name=inf_${DATASET}_lv60_${TAG}

# 4. W2V2P-XLSR53
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft \
#     task_name=inf_${DATASET}_xlsr53_${TAG}

# 5. MultiIPA (ctag)
# python src/main.py \
#     experiment=inference/transcribe_w2v2ph \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns \
#     task_name=inf_${DATASET}_ctag_${TAG}

# 6. ZIPA-CTC
python src/main.py \
    experiment=inference/transcribe_zipactc \
    data=powsmeval \
    data.dataset_name=${DATASET} \
    data.data_dir=$DATA_DIR \
    data.portable_wavscp=True \
    inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k \
    task_name=inf_${DATASET}_zipactc_${TAG}

# 7. ZIPA-CTC-NS
python src/main.py \
    experiment=inference/transcribe_zipactc \
    data=powsmeval \
    data.dataset_name=${DATASET} \
    data.data_dir=$DATA_DIR \
    data.portable_wavscp=True \
    inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k \
    task_name=inf_${DATASET}_zipactc_ns_${TAG}

# 8. Gemini
# python src/main.py \
#     experiment=inference/transcribe_gemini \
#     data=powsmeval \
#     data.dataset_name=${DATASET} \
#     data.data_dir=$DATA_DIR \
#     data.portable_wavscp=True \
#     task_name=inf_${DATASET}_gemini_${TAG}

# ===== Scoring ================================================================
echo
echo "=== Scoring ($(date)) ==="

MODELS=(powsm powsm_ctc lv60 xlsr53 ctag zipactc zipactc_ns gemini)

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
        --language_field lang_sym
    echo "    results: $run_dir/inventory_results.csv"
done

echo
echo "=== DONE: $(date) ==="
echo "outputs under: /n/iqss_sponsored/Lab/zshi/PhonBenchDev/exp/runs/inf_${DATASET}_*_${TAG}"
