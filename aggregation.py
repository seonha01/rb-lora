"""LoRA Aggregation Methods for Federated Learning with Heterogeneous Ranks"""

import os
import torch
from torch.nn.functional import pad, normalize
from typing import Dict, Set, List


def zero_pad_lora_tensor(tensor: torch.Tensor, target_rank: int, is_lora_A: bool) -> torch.Tensor:
    """Zero-pad LoRA tensor to target rank."""
    if is_lora_A:
        cur_rank = tensor.shape[0]
        if cur_rank >= target_rank:
            return tensor
        return pad(tensor, (0, 0, 0, target_rank - cur_rank), value=0.0)
    else:
        cur_rank = tensor.shape[1]
        if cur_rank >= target_rank:
            return tensor
        return pad(tensor, (0, target_rank - cur_rank), value=0.0)


def aggregate_uniform_hetlora(
    model,
    selected_clients_set: Set[int],
    output_dir: str,
    local_dataset_len_dict: Dict[int, int],
    epoch: int,
) -> Dict[str, torch.Tensor]:
    """Uniform HETLoRA: Uniform averaging with zero-padding."""
    state_dicts = []
    for cid in selected_clients_set:
        path = os.path.join(output_dir, str(epoch), f"local_output_{cid}", "pytorch_model.bin")
        state_dicts.append(torch.load(path))

    device = next(iter(state_dicts[0].values())).device

    weights = torch.full(
        (len(selected_clients_set),),
        1.0 / len(selected_clients_set),
        dtype=torch.float32,
        device=device
    )

    max_ranks = {}
    all_keys = state_dicts[0].keys()
    for sd in state_dicts:
        for key in sd:
            if "lora_A" in key:
                max_ranks[key] = max(max_ranks.get(key, 0), sd[key].shape[0])  # A: (r, d)
            elif "lora_B" in key:
                max_ranks[key] = max(max_ranks.get(key, 0), sd[key].shape[1])

    aggregated = {}
    for key in all_keys:
        agg = None
        for i, sd in enumerate(state_dicts):
            if key not in sd:
                continue
            w = weights[i]
            param = sd[key].to(device)
            if "lora_A" in key:
                t = zero_pad_lora_tensor(param, max_ranks[key], is_lora_A=True)
            elif "lora_B" in key:
                t = zero_pad_lora_tensor(param, max_ranks[key], is_lora_A=False)
            else:
                t = param
            agg = w * t if agg is None else agg + w * t
        if agg is not None:
            aggregated[key] = agg

    return aggregated


def aggregate_weighted_hetlora(
    model,
    selected_clients_set: Set[int],
    output_dir: str,
    local_dataset_len_dict: Dict[int, int],
    epoch: int,
) -> Dict[str, torch.Tensor]:
    """Weighted HETLoRA: Dataset-size weighted averaging with zero-padding."""
    weights = normalize(
        torch.tensor([local_dataset_len_dict[cid] for cid in selected_clients_set], dtype=torch.float32),
        p=1, dim=0
    )

    state_dicts = []
    for cid in selected_clients_set:
        path = os.path.join(output_dir, str(epoch), f"local_output_{cid}", "pytorch_model.bin")
        state_dicts.append(torch.load(path))

    max_ranks = {}
    all_keys = state_dicts[0].keys()
    for sd in state_dicts:
        for key in sd:
            if "lora_A" in key:
                max_ranks[key] = max(max_ranks.get(key, 0), sd[key].shape[0])  # A: (r, d)
            elif "lora_B" in key:
                max_ranks[key] = max(max_ranks.get(key, 0), sd[key].shape[1])

    aggregated = {}
    for key in all_keys:
        agg = None
        for i, sd in enumerate(state_dicts):
            if key not in sd:
                continue
            w = weights[i]
            if "lora_A" in key:
                t = zero_pad_lora_tensor(sd[key], max_ranks[key], is_lora_A=True)
            elif "lora_B" in key:
                t = zero_pad_lora_tensor(sd[key], max_ranks[key], is_lora_A=False)
            else:
                t = sd[key]
            agg = w * t if agg is None else agg + w * t
        if agg is not None:
            aggregated[key] = agg

    return aggregated


def aggregate_flora(
    model,
    selected_clients_set: Set[int],
    output_dir: str,
    epoch: int,
) -> Dict[str, torch.Tensor]:
    """FLoRA: Stacking (concatenation) along rank dimension."""
    state_dicts = []
    for cid in selected_clients_set:
        path = os.path.join(output_dir, str(epoch), f"local_output_{cid}", "pytorch_model.bin")
        state_dicts.append(torch.load(path))

    concat_dict = {}

    for key in state_dicts[0].keys():
        if "lora_A" in key:
            tensors = [sd[key] for sd in state_dicts if key in sd]
            concat_dict[key] = torch.cat(tensors, dim=0)
        elif "lora_B" in key:
            tensors = [sd[key] for sd in state_dicts if key in sd]
            concat_dict[key] = torch.cat(tensors, dim=1)
        else:
            base = None
            for sd in state_dicts:
                if base is None:
                    base = sd[key].clone()
                else:
                    base += sd[key]
            concat_dict[key] = base / len(state_dicts)

    return concat_dict


def compute_column_weights_from_ranks(
    client_dataset_size_dict: Dict[str, int],
    client_lora_rank_dict: Dict[str, int]
) -> List[float]:
    """Compute column weights for weighted column-wise aggregation."""
    total_size = sum(client_dataset_size_dict.values())
    max_rank = max(client_lora_rank_dict.values())
    denom = [0.0] * max_rank

    for client, rank in client_lora_rank_dict.items():
        size = client_dataset_size_dict.get(client, 0)
        for j in range(rank):
            denom[j] += size

    weights = []
    for j in range(max_rank):
        if denom[j] == 0:
            weights.append(0.0)
        else:
            weights.append(total_size / denom[j])

    return weights

