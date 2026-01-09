# RB-LoRA

Rank-Based LoRA Aggregation for Federated Learning with Heterogeneous LoRA Ranks.

## Quick Start

All commands should be run from the `rb_lora/` directory.

### 1. Prepare Data

```bash
python prepare_data.py 10 2
```

### 2. Train

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

## Available Methods

- **RB-LoRA**: SVD-based rank reduction
- **Uniform HETLoRA**: Uniform averaging with zero-padding
- **Weighted HETLoRA**: Dataset-size weighted averaging with zero-padding
- **FLoRA**: Stacking (concatenation) along rank dimension

## Paths

All paths are relative to the `rb_lora/` directory:
- Data: `./data/`
- Output: `./outputs/`
- Templates: `templates/`
- Configs: `configs/`

For detailed usage, see `COMMANDS.md`.
