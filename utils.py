"""Utility Functions for Federated Learning"""

import numpy as np
from typing import Set


def client_selection(
    num_clients: int, 
    client_selection_frac: float, 
    client_selection_strategy: str = "random",
    other_info: int = None
) -> Set[int]:
    """Select clients for a communication round."""
    if other_info is not None:
        np.random.seed(other_info)
    
    if client_selection_strategy == "random":
        num_selected = max(int(client_selection_frac * num_clients), 1)
        selected_clients_set = set(
            np.random.choice(np.arange(num_clients), num_selected, replace=False)
        )
        return selected_clients_set
    else:
        raise ValueError(f"Unknown client selection strategy: {client_selection_strategy}")

