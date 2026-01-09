"""Method Registry: Mapping between paper method names and implementation functions."""

from typing import Dict, Callable, Optional
from aggregation import (
    aggregate_uniform_hetlora,
    aggregate_weighted_hetlora,
    aggregate_flora,
    compute_column_weights_from_ranks,
)
import rb_lora as rb_lora_module
distribute_lora_weights_with_svd = rb_lora_module.distribute_lora_weights_with_svd
distribute_lora_weights_truncate = rb_lora_module.distribute_lora_weights_truncate

METHOD_REGISTRY: Dict[str, Dict[str, any]] = {
    "Uniform HETLoRA": {
        "aggregation_fn": aggregate_uniform_hetlora,
        "description": "Uniform averaging with zero-padding (baseline)",
        "needs_column_weights": False,
        "needs_reduction": True,
    },
    "Weighted HETLoRA": {
        "aggregation_fn": aggregate_weighted_hetlora,
        "description": "Dataset-size weighted averaging with zero-padding (baseline)",
        "needs_column_weights": False,
        "needs_reduction": True,
    },
    "FLoRA": {
        "aggregation_fn": aggregate_flora,
        "description": "Stacking (concatenation) along rank dimension (baseline)",
        "needs_column_weights": False,
        "needs_reduction": False,  # FLoRA already has full rank
    },
    "RB-LoRA": {
        "aggregation_fn": None,  # RB-LoRA uses same aggregation as Weighted HETLoRA
        "description": "Rank-Based LoRA aggregation with SVD-based rank reduction",
        "needs_column_weights": False,
        "needs_reduction": True,
        "reduction_method": "svd",  # Must use SVD for RB-LoRA
    },
}

LEGACY_NAME_MAP: Dict[str, str] = {
    "naive": "Uniform HETLoRA",
    "fedavg": "Weighted HETLoRA",
    "concat": "FLoRA",
    "column_client_weighted": "Weighted HETLoRA",
}

def get_method_info(method_name: str) -> Dict[str, any]:
    """Get method information from registry."""
    if method_name in LEGACY_NAME_MAP:
        method_name = LEGACY_NAME_MAP[method_name]
    
    if method_name not in METHOD_REGISTRY:
        available = ", ".join(METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown method: {method_name}. Available methods: {available}"
        )
    
    return METHOD_REGISTRY[method_name]

def get_aggregation_function(method_name: str) -> Callable:
    """Get the aggregation function for a given method name."""
    info = get_method_info(method_name)
    
    if method_name == "RB-LoRA":
        return METHOD_REGISTRY["Weighted HETLoRA"]["aggregation_fn"]
    
    fn = info["aggregation_fn"]
    if fn is None:
        raise ValueError(f"Method {method_name} does not have an aggregation function")
    
    return fn

def list_available_methods() -> list:
    return list(METHOD_REGISTRY.keys())

