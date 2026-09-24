# RB-LoRA
<img width="5496" height="2358" alt="overview" src="https://github.com/user-attachments/assets/2cf311c8-c81e-4732-a88a-460aeb4b7d40" />


**RB-LoRA** is the official PyTorch implementation of  
**“RB-LoRA: Rank-Balanced Aggregation for Low-Rank Adaptation with Federated Fine-Tuning”**  
(*to appear in Findings of EACL 2026*).

📄 **Project Page:** https://seonha01.github.io/rb-lora/

This repository provides a clean and reproducible framework for federated fine-tuning
with **heterogeneous LoRA ranks**, and implements several aggregation baselines
alongside the proposed **rank-balanced aggregation** method.

---

## Overview

In federated LoRA, clients often adopt **different adapter ranks** due to heterogeneous
device and data constraints.  
Naïve aggregation strategies (e.g., zero-padding, replication, or stacking) can introduce
bias or inefficiency when reconciling such heterogeneous updates.

RB-LoRA addresses this issue by:
- decomposing LoRA updates into **rank-wise components**, and
- aggregating them using **analytically derived, rank-balanced weights**.

The framework supports both **language models and vision transformers**, and is designed
to be easily extensible to new aggregation strategies.

---

## Installation

We recommend using a dedicated conda environment.

```bash
conda env create -f environment.yml
conda activate rblora
````

---

## Quick Start

All commands should be executed from the repository root (`rb_lora/`).

### 1. Prepare Federated Data

The following command generates a federated split with
10 clients and a non-IID setting (example configuration):

```bash
python prepare_data.py 10 2
```

This will create a `./data/` directory containing client-local datasets.

---

### 2. Federated Training with RB-LoRA

```bash
python train.py \
    --method "RB-LoRA" \
    --global_model "meta-llama/Llama-3.1-8B" \
    --data_path "./data" \
    --output_dir "./outputs/rb_lora" \
    --num_clients 10 \
    --num_communication_rounds 3 \
    --local_num_epochs 1 \
    --local_batch_size 32 \
    --train_on_inputs \
    --group_by_length
```

The aggregated global LoRA adapters and evaluation results will be saved under
`./outputs/rb_lora/`.

---

## Implemented Aggregation Methods

The following aggregation strategies are implemented in a unified interface:

* **RB-LoRA**
  Rank-balanced aggregation with rank-wise decomposition and SVD-based projection
  (proposed method).

* **Uniform HETLoRA**
  Uniform averaging after zero-padding all adapters to the maximum rank.

* **Weighted HETLoRA**
  Dataset-size–weighted averaging with zero-padding.

* **FLoRA**
  Stacking (concatenation) of LoRA adapters along the rank dimension, followed by
  rank reduction.

All methods can be selected via the `--method` argument.

---

## Directory Structure

All paths below are relative to the repository root (`rb_lora/`):

```
rb_lora/
├── aggregation.py        # Aggregation logic
├── train.py              # Federated training loop
├── client.py             # Client-side training
├── prepare_data.py       # Federated data preparation
├── configs/              # YAML configs for experiments
├── templates/            # Prompt templates
├── data/                 # Federated datasets (generated)
├── outputs/              # Training outputs and checkpoints
└── scripts/              # Example training scripts
```

---

## Reproducing Paper Results

The configurations used in the paper are provided under `configs/`.
For detailed experiment commands and ablation settings, please refer to:

```
COMMANDS.md
```

---

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{ha2026rb,
  title={RB-LoRA: Rank-Balanced Aggregation for Low-Rank Adaptation with Federated Fine-Tuning},
  author={Ha, Sihyeon and Oh, Yongjeong and Jeon, Yo-Seb},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2026},
  pages={1737--1746},
  year={2026}
}
```

---

## License

This project is released under the MIT License.
