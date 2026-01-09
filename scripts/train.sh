#!/bin/bash
# RB-LoRA: Training Script
# Run this script from rb_lora/ directory

# Get script directory (rb_lora/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || exit 1  # Go to rb_lora/

export TORCH_COMPILE_DISABLE=1

# Configuration
MODEL="meta-llama/Llama-3.1-8B"
DATA_PATH="./data"
OUTPUT_DIR="./outputs"
NUM_CLIENTS=10
NUM_ROUNDS=50
METHOD="RB-LoRA"  # Options: "RB-LoRA", "Uniform HETLoRA", "Weighted HETLoRA", "FLoRA"

# Training hyperparameters
LOCAL_BATCH_SIZE=32
LOCAL_MICRO_BATCH_SIZE=16
LOCAL_NUM_EPOCHS=1
LOCAL_LEARNING_RATE=2e-5
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES='["q_proj","v_proj"]'

# Run training (from rb_lora/ directory)
python train.py \
    --global_model "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_communication_rounds $NUM_ROUNDS \
    --num_clients $NUM_CLIENTS \
    --client_selection_frac 1.0 \
    --method "$METHOD" \
    --local_batch_size $LOCAL_BATCH_SIZE \
    --local_micro_batch_size $LOCAL_MICRO_BATCH_SIZE \
    --local_num_epochs $LOCAL_NUM_EPOCHS \
    --local_learning_rate $LOCAL_LEARNING_RATE \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --train_on_inputs \
    --group_by_length \
    --reduction_method "svd"

echo "Training completed!"

