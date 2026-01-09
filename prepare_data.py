"""Data Preparation Script for Federated Learning"""

import sys
import pandas as pd
import numpy as np
import random
import os
import json


def prepare_federated_data(num_clients: int, diff_quantity: int = 2, seed: int = 42):
    """Prepare federated datasets for clients."""
    np.random.seed(seed)
    random.seed(seed)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    dolly_path = os.path.join(project_root, "new-databricks-dolly-15k.json")
    alpaca_path = os.path.join(project_root, "alpaca_data.json")
    
    if not os.path.exists(dolly_path):
        dolly_path = os.path.join(current_dir, "new-databricks-dolly-15k.json")
    if not os.path.exists(alpaca_path):
        alpaca_path = os.path.join(current_dir, "alpaca_data.json")
    
    df = pd.read_json(dolly_path, orient='records')
    alpaca_df = pd.read_json(alpaca_path, orient='records')
    alpaca_df.rename(columns={'output': 'response', "input": "context"}, inplace=True)
    alpaca_df["category"] = "alpaca"
    alpaca_df = alpaca_df.sample(n=42000, random_state=seed)
    
    sorted_df = df.sort_values(by=['category'])
    grouped = sorted_df.groupby('category')
    sampled_df = grouped.apply(lambda x: x.sample(n=10))
    sampled_df = sampled_df.reset_index(level=0, drop=True)
    remaining_df = sorted_df.drop(index=sampled_df.index)
    
    alpaca_sorted_df = alpaca_df.sort_values(by=['category'])
    alpaca_grouped = alpaca_sorted_df.groupby('category')
    alpaca_sampled_df = alpaca_grouped.apply(lambda x: x.sample(n=80))
    alpaca_sampled_df = alpaca_sampled_df.reset_index(level=0, drop=True)
    alpaca_remaining_df = alpaca_sorted_df.drop(index=alpaca_sampled_df.index)
    
    sampled_df = pd.concat([sampled_df, alpaca_sampled_df], ignore_index=True, sort=False)
    remaining_df = pd.concat([remaining_df, alpaca_remaining_df], ignore_index=True, sort=False)
    
    sampled_df = sampled_df.reset_index().drop('index', axis=1)
    remaining_df = remaining_df.reset_index().drop('index', axis=1)
    
    data_path = os.path.join(current_dir, "data", str(num_clients))
    os.makedirs(data_path, exist_ok=True)
    
    remaining_df_dic = remaining_df.to_dict(orient='records')
    with open(os.path.join(data_path, "global_training.json"), 'w') as outfile:
        json.dump(remaining_df_dic, outfile)
    
    sampled_df_dic = sampled_df.to_dict(orient='records')
    with open(os.path.join(data_path, "global_test.json"), 'w') as outfile:
        json.dump(sampled_df_dic, outfile)
    
    if diff_quantity == 2:
        client_data_sizes = [500, 500, 1000, 1000, 1500, 1500, 1900, 1900, 2500, 2500]
        assert len(client_data_sizes) == num_clients, "client_data_sizes length must match num_clients"
        assert sum(client_data_sizes) <= len(remaining_df), "Not enough data for manual allocation"
        
        remaining_df_shuffled = remaining_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        start = 0
        idx_partition = []
        for size in client_data_sizes:
            end = start + size
            indices = remaining_df_shuffled.iloc[start:end].index.tolist()
            idx_partition.append(indices)
            start = end
    
    elif diff_quantity == 0:
        min_size = 0
        min_require_size = 40
        alpha = 0.5
        
        N = len(remaining_df)
        category_uniques = remaining_df['category'].unique().tolist()
        
        while min_size < min_require_size:
            idx_partition = [[] for _ in range(num_clients)]
            for k in range(len(category_uniques)):
                category_rows_k = remaining_df.loc[remaining_df['category'] == category_uniques[k]]
                category_rows_k_index = category_rows_k.index.values
                np.random.shuffle(category_rows_k_index)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_partition)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(category_rows_k_index)).astype(int)[:-1]
                idx_partition = [idx_j + idx.tolist() for idx_j, idx in
                               zip(idx_partition, np.split(category_rows_k_index, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_partition])
    
    else:
        num_shards_per_clients = 2
        remaining_df_index = remaining_df.index.values
        shards = np.array_split(remaining_df_index, int(num_shards_per_clients * num_clients))
        random.shuffle(shards)
        shards = [shards[i:i + num_shards_per_clients] for i in range(0, len(shards), num_shards_per_clients)]
        idx_partition = [np.concatenate(shards[n]).tolist() for n in range(num_clients)]
    
    for client_id, idx in enumerate(idx_partition):
        print(f"\nGenerating local training dataset for Client_{client_id}")
        sub_remaining_df = remaining_df.loc[idx]
        sub_remaining_df = sub_remaining_df.reset_index().drop('index', axis=1)
        sub_remaining_df_dic = sub_remaining_df.to_dict(orient='records')
        
        with open(os.path.join(data_path, f"local_training_{client_id}.json"), 'w') as outfile:
            json.dump(sub_remaining_df_dic, outfile)
    
    print("\n[Summary] Client-wise dataset sizes:")
    for client_id, idx in enumerate(idx_partition):
        print(f"Client {client_id}: {len(idx)} samples")
    
    print(f"\nData saved to: {data_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py <num_clients> [diff_quantity] [seed]")
        print("  diff_quantity: 0=Dirichlet, 1=IID, 2=Manual (default: 2)")
        print("  seed: Random seed (default: 42)")
        sys.exit(1)
    
    num_clients = int(sys.argv[1])
    diff_quantity = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    prepare_federated_data(num_clients, diff_quantity, seed)

