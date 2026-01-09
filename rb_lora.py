"""RB-LoRA: Rank-Based LoRA Aggregation"""

import os
import torch
from typing import Dict


def distribute_lora_weights_with_svd(
    global_params: Dict[str, torch.Tensor],
    client_lora_rank_dict: Dict[str, int],
    epoch: int,
    output_dir: str,
    alpha: float,
) -> None:
    """RB-LoRA: Distribute global LoRA to clients using SVD-based rank reduction."""
    weighted_A, weighted_B = {}, {}
    for key, W in global_params.items():
        if 'lora_A' in key:
            layer = key.split('lora_A')[0].rstrip('.')
            weighted_A[layer] = W
        elif 'lora_B' in key:
            layer = key.split('lora_B')[0].rstrip('.')
            weighted_B[layer] = W

    for client_key, r in client_lora_rank_dict.items():
        client_id = int(client_key.split('_')[-1])
        client_sd = {}

        for layer, A_full in weighted_A.items():
            B_full = weighted_B.get(layer)
            if B_full is None:
                continue

            max_r = min(B_full.shape[1], A_full.shape[0])
            if not (0 < r <= max_r):
                raise ValueError(
                    f"{client_key}: rank {r} out of bounds for layer {layer} (max {max_r})"
                )

            BA = B_full @ A_full
            U, S, Vt = torch.svd(BA)
            lora_B = U[:, :r] @ torch.diag(S[:r])
            lora_A = Vt[:r, :]

            merge_rate = alpha / r
            client_sd[f"{layer}.lora_A.local.weight"] = lora_A
            client_sd[f"{layer}.lora_B.local.weight"] = lora_B / merge_rate

        single_dir = os.path.join(output_dir, str(epoch), f"local_output_{client_id}")
        os.makedirs(single_dir, exist_ok=True)
        save_path = os.path.join(single_dir, "pytorch_model_after_reduction.bin")
        torch.save(client_sd, save_path)
        print(f"Saved Client{client_id} adapter (rank={r}) to {save_path}")


def distribute_lora_weights_truncate(
    global_params: Dict[str, torch.Tensor],
    client_lora_rank_dict: Dict[str, int],
    epoch: int,
    output_dir: str,
    alpha: float,
) -> None:
    """Truncation-based rank reduction (baseline)."""
    weighted_A, weighted_B = {}, {}
    for key, W in global_params.items():
        if 'lora_A' in key:
            layer = key.split('lora_A')[0].rstrip('.')
            weighted_A[layer] = W
        elif 'lora_B' in key:
            layer = key.split('lora_B')[0].rstrip('.')
            weighted_B[layer] = W

    for client_key, r in client_lora_rank_dict.items():
        client_id = int(client_key.split('_')[-1])
        client_sd = {}

        for layer, A_full in weighted_A.items():
            B_full = weighted_B.get(layer)
            if B_full is None:
                continue

            max_r = min(B_full.shape[1], A_full.shape[0])
            if not (0 < r <= max_r):
                raise ValueError(
                    f"{client_key}: rank {r} out of bounds for layer {layer} (max {max_r})"
                )

            lora_B = B_full[:, :r]
            lora_A = A_full[:r, :]

            merge_rate = alpha / r
            client_sd[f"{layer}.lora_A.local.weight"] = lora_A
            client_sd[f"{layer}.lora_B.local.weight"] = lora_B / merge_rate

        single_dir = os.path.join(output_dir, str(epoch), f"local_output_{client_id}")
        os.makedirs(single_dir, exist_ok=True)
        save_path = os.path.join(single_dir, "pytorch_model_after_reduction.bin")
        torch.save(client_sd, save_path)
        print(f"Saved Client{client_id} adapter (rank={r}, truncate) to {save_path}")

