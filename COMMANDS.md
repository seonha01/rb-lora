# Commands

Run all commands from the `rb_lora/` directory.

## Data Preparation

```bash
python prepare_data.py <num_clients> [diff_quantity] [seed]
```

## Training

### RB-LoRA

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

### Baselines

```bash
python train.py --method "Uniform HETLoRA" --global_model "meta-llama/Llama-3.1-8B" --data_path "./data" --output_dir "./outputs/uniform_hetlora" --num_clients 10 --num_communication_rounds 3 --reduction_method "truncate"

python train.py --method "Weighted HETLoRA" --global_model "meta-llama/Llama-3.1-8B" --data_path "./data" --output_dir "./outputs/weighted_hetlora" --num_clients 10 --num_communication_rounds 3 --reduction_method "truncate"

python train.py --method "FLoRA" --global_model "meta-llama/Llama-3.1-8B" --data_path "./data" --output_dir "./outputs/flora" --num_clients 10 --num_communication_rounds 3
```
