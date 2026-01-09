"""RB-LoRA: Federated Training Script"""

import os
import sys
import json
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
import datasets
from datasets.utils.logging import set_verbosity_error

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from method_registry import get_aggregation_function, get_method_info, list_available_methods
from aggregation import compute_column_weights_from_ranks
import rb_lora as rb_lora_module
from client import FederatedClient
from prompter import Prompter
from utils import client_selection

distribute_lora_weights_with_svd = rb_lora_module.distribute_lora_weights_with_svd
distribute_lora_weights_truncate = rb_lora_module.distribute_lora_weights_truncate

set_verbosity_error()


def fl_finetune(
    global_model: str = '',
    data_path: str = './data',
    dev_data_path: str = './mmlu_test_1444.jsonl',
    output_dir: str = './outputs/',
    client_selection_strategy: str = 'random',
    client_selection_frac: float = 1.0,
    num_communication_rounds: int = 50,
    num_clients: int = 10,
    local_batch_size: int = 64,
    local_micro_batch_size: int = 8,
    local_num_epochs: int = 10,
    local_learning_rate: float = 3e-5,
    local_val_set_size: int = 0,
    cutoff_len: int = 512,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    prompt_template_name: str = "alpaca",
    hf_token_json: str = './HF_key.json',
    method: str = 'RB-LoRA',
    client_lora_rank_dict: Dict[str, int] = None,
    client_dataset_size_dict: Dict[str, int] = None,
    reduction_method: str = 'svd',
):
    """Federated fine-tuning with heterogeneous LoRA ranks."""
    if method not in list_available_methods():
        available = ", ".join(list_available_methods())
        raise ValueError(f"Unknown method: {method}. Available: {available}")
    
    method_info = get_method_info(method)
    aggregation_fn = get_aggregation_function(method)
    
    if method == "RB-LoRA":
        reduction_method = "svd"
    
    if client_lora_rank_dict is None:
        client_lora_rank_dict = {
            f"client_{i}": [4, 4, 16, 16, 64, 64, 128, 128, 256, 256][i] 
            for i in range(num_clients)
        }
    
    if client_dataset_size_dict is None:
        client_dataset_size_dict = {
            f"client_{i}": [500, 500, 1000, 1000, 1500, 1500, 1900, 1900, 2500, 2500][i]
            for i in range(num_clients)
        }
    
    hf_token = None
    if hf_token_json and os.path.exists(hf_token_json):
        with open(hf_token_json, 'r') as f:
            keys = json.load(f)
            hf_token = keys.get("hf_token", None)
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("\n" + "="*60)
        print(f"RB-LoRA: Federated Training with Heterogeneous LoRA Ranks")
        print("="*60)
        print(f"Method: {method} ({method_info['description']})")
        print(f"Global Model: {global_model}")
        print(f"Output Directory: {output_dir}")
        print(f"Communication Rounds: {num_communication_rounds}")
        print(f"Number of Clients: {num_clients}")
        print(f"Client Selection: {client_selection_strategy} (frac={client_selection_frac})")
        print(f"Reduction Method: {reduction_method}")
        print("="*60 + "\n")
    
    assert global_model, "Please specify --global_model"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(current_dir, data_path))
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(current_dir, output_dir))
    if hf_token_json and not os.path.isabs(hf_token_json):
        hf_token_json = os.path.abspath(os.path.join(current_dir, hf_token_json))
    
    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), f"Data path does not exist: {data_path}"
    
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    if global_model == 'gpt2':
        base_model = GPT2LMHeadModel.from_pretrained(
            global_model, load_in_8bit=False, dtype=torch.float32, device_map=device_map
        )
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            dtype=torch.float32,
            device_map=device_map,
            token=hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=hf_token)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point.get("context", ""),
            data_point.get("response", ""),
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point.get("context", "")
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt
    
    output_dir = os.path.join(output_dir, str(num_clients))
    os.makedirs(output_dir, exist_ok=True)
    previously_selected_clients_set = set()
    local_dataset_len_dict = {}
    global_lora_sd = None
    
    print("\nStarting federated training...\n")
    
    for epoch in tqdm(range(num_communication_rounds), desc="FL Rounds"):
        print(f"\n{'='*60}")
        print(f"Communication Round {epoch + 1}/{num_communication_rounds}")
        print(f"{'='*60}")
        
        print("\n[1/4] Selecting clients...")
        selected_clients_set = client_selection(
            num_clients, client_selection_frac, client_selection_strategy, other_info=epoch
        )
        print(f"Selected clients: {sorted(selected_clients_set)}")
        
        base_model = prepare_model_for_kbit_training(base_model)
        
        print(f"\n[2/4] Local training on {len(selected_clients_set)} clients...")
        for client_id in selected_clients_set:
            rank = client_lora_rank_dict[f"client_{client_id}"]
            print(f"  Client {client_id}: rank={rank}")
            
            config = LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, config)
            
            if epoch > 0:
                ckpt_path = os.path.join(
                    output_dir,
                    str(epoch - 1),
                    f"local_output_{client_id}",
                    "pytorch_model_after_reduction.bin"
                )
                if os.path.isfile(ckpt_path):
                    client_adapter = torch.load(ckpt_path)
                    set_peft_model_state_dict(model, client_adapter, "default")
            
            if not ddp and torch.cuda.device_count() > 1:
                model.is_parallelizable = True
                model.model_parallel = True
            
            client = FederatedClient(client_id, model, data_path, output_dir)
            client.prepare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp
            )
            client.initiate_local_training()
            client.train()
            
            # Save trained adapter
            model, local_dataset_len_dict, previously_selected_clients_set, _ = (
                client.terminate_local_training(epoch, local_dataset_len_dict, previously_selected_clients_set)
            )
            del client
        
        print(f"\n[3/4] Aggregating adapters using {method}...")
        if method == "FLoRA":
            global_params = aggregation_fn(model, selected_clients_set, output_dir, epoch)
        else:
            global_params = aggregation_fn(
                model, selected_clients_set, output_dir, local_dataset_len_dict, epoch
            )
        
        model = get_peft_model(base_model, LoraConfig(
            r=max(client_lora_rank_dict.values()) if method != "FLoRA" else sum(client_lora_rank_dict.values()),
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ))
        set_peft_model_state_dict(model, global_params, "default")
        torch.save(global_params, os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        
        if method_info.get("needs_reduction", True):
            print(f"\n[4/4] Reducing ranks using {reduction_method}...")
            if reduction_method == "svd":
                distribute_lora_weights_with_svd(
                    global_params, client_lora_rank_dict, epoch, output_dir, lora_alpha
                )
            else:  # truncate
                distribute_lora_weights_truncate(
                    global_params, client_lora_rank_dict, epoch, output_dir, lora_alpha
                )
        else:
            print(f"\n[4/4] No rank reduction needed for {method}")
        
        print(f"\nRound {epoch + 1} completed.\n")
    
    print("\n" + "="*60)
    print("Federated training completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import fire
    fire.Fire(fl_finetune)

