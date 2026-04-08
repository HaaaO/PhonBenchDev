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
