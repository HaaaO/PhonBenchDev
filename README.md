______________________________________________________________________

<div align="center">

# PhonBench

### Benchmarking Speech Foundation Models for Phone-Level Recognition of Children's Speech

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<br>
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

**PhonBench** is a new project built on top of the [PRiSM](https://github.com/changelinglab/prism) codebase, extending it to benchmark speech foundation models for phone-level recognition of children's speech.

The accompanying paper is titled **"PhonBench: Benchmarking Speech Foundation Models for Phone-Level Recognition of Children's Speech"**.

## 🚀 Quickstart

```bash
# clone the PhonBench project
git clone <phonbench-repo-url>
cd PhonBenchDev

# create environment with your favourite package manager 
# and install dependencies from requirements.txt
# We provide "setup_uv.sh" for doing these and activating environment
. ./setup_uv.sh

```

This codebase pulls datasets from the huggingface collection [https://huggingface.co/collections/changelinglab/prism](https://huggingface.co/collections/changelinglab/prism).

### Activating the environment on FASRC (Harvard SLURM)

After the one-time `setup_uv.sh` install, activate the environment in any new shell session with:

```bash
# 1. Get on a GPU compute node (skip if you're already on one)
salloc -p gpu --gres=gpu:1 -c 8 --mem=32G -t 03:00:00

# 2. cd into the project
cd /n/netscratch/iqss_sponsored/Lab/zshi/PhonBenchDev

# 3. Source the activation helper (must be `.` or `source`, NOT ./)
. ./activate_env.sh
```

`activate_env.sh` sources the `.venv`, points HuggingFace / uv / pixi caches at netscratch (so they don't fill up the home-dir quota), loads the FASRC `ffmpeg`, `espeak-ng`, and `cuda` modules, and sets `PROJECT_ROOT` for the Hydra configs. It must be **sourced**, not executed — running it as `./activate_env.sh` will appear to do nothing because the activation would die with the subshell.


## How to run

Train model with default configuration

```bash
# train on CPU
python src/main.py trainer=cpu

# train on GPU
python src/main.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# For probing experiments using hidden representations
python src/main.py experiment=probing/geolocation_vaani_powsm

# For inference experiments
python src/main.py experiment=inference/vaani_powsmpr
```

You can override any parameter from command line like this

```bash
python src/main.py trainer.max_epochs=20 data.batch_size=64
```

## Running all models on the Harvard dataset

The Harvard `towre_words` data lives at `/n/netscratch/iqss_sponsored/Lab/zshi/harvard` with a Kaldi-style index at `harvard/index.yaml`. Each command below runs one model variant on that dataset using the generic `powsmeval` data config; only the experiment, model-specific args, and `task_name` change.

### CTC / encoder–decoder models

```bash
# 1. PoWSM
python src/main.py experiment=inference/transcribe_powsm \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  task_name=inf_towre_words_powsm

# 2. PoWSM-CTC
python src/main.py experiment=inference/transcribe_powsm_ctc \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  task_name=inf_towre_words_powsm_ctc

# 3a. wav2vec2-phoneme — ctag
python src/main.py experiment=inference/transcribe_w2v2ph \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns \
  task_name=inf_towre_words_ctag

# 3b. wav2vec2-phoneme — lv60
python src/main.py experiment=inference/transcribe_w2v2ph \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft \
  task_name=inf_towre_words_lv60

# 3c. wav2vec2-phoneme — xlsr53
python src/main.py experiment=inference/transcribe_w2v2ph \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft \
  task_name=inf_towre_words_xlsr53

# 4a. ZipA-CTC
python src/main.py experiment=inference/transcribe_zipactc \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k \
  task_name=inf_towre_words_zipactc

# 4b. ZipA-CTC (noisy student)
python src/main.py experiment=inference/transcribe_zipactc \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k \
  task_name=inf_towre_words_zipactc_ns
```

### LLM-based models (require extra services)

Qwen3-Omni runs need a vLLM server first (see `scripts/start_vllm.sh`); pass its port via `inference.port=`. Gemini needs `GEMINI_API_KEY` exported and forces `num_workers=1`.

```bash
# 5. Qwen3-Omni Instruct (vLLM)
python src/main.py experiment=inference/transcribe_qweninstruct \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  inference.port=8000 \
  task_name=inf_towre_words_qweni

# 6. Qwen3-Omni Thinking (vLLM)
python src/main.py experiment=inference/transcribe_qwenthinking \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True inference.num_workers=4 \
  inference.port=8000 \
  task_name=inf_towre_words_qwent

# 7. Gemini 2.5 Flash
GEMINI_API_KEY=... python src/main.py experiment=inference/transcribe_gemini \
  data=powsmeval data.dataset_name=towre_words \
  data.dataset_config_path=/n/netscratch/iqss_sponsored/Lab/zshi/harvard/index.yaml \
  data.data_dir=/n/netscratch/iqss_sponsored/Lab/zshi/harvard \
  data.portable_wavscp=True \
  task_name=inf_towre_words_gemini
```

Notes:
- These launch as plain `python src/main.py` on the current node — wrap each in `sbatch` for FASRC GPU scheduling.
- `inference.num_workers=4` is conservative; the configs default higher (`powsm:10`, `w2v2ph/zipactc:15`). Bump up if you have the GPU memory.
- The model list mirrors `scripts/run.sh:68-82`.

## More Documentation

- **[Features & Capabilities](docs/features.md)** - Look at this to train on multi-gpu, run hyper-param searches etc.
- **[Running Inference](docs/running_inference.md)** - Guide for running phone recognition inference with pre-trained models
- **[Tokenization Workflow](docs/tokenization.md)** - How to build vocabularies and use tokenizers for IPA transcripts
- **[Contributing Guide](CONTRIBUTING.md)** - Project structure, workflow, and best practices for contributors

## Citation

If you use this work in your research, please cite the **PhonBench** paper: *"PhonBench: Benchmarking Speech Foundation Models for Phone-Level Recognition of Children's Speech"* (citation details forthcoming).

PhonBench reuses the codebase from the PRiSM project — please also cite and link to the original PRiSM paper: [PRiSM: Benchmarking Phone Realization in Speech Models](https://arxiv.org/abs/2601.14046).

## ❤️ Acknowledgement

This repository structure is based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).
