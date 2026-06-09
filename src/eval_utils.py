import os
import torch

from src.custom_tokenizer import CustomTokenizer
from src.config import (
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn, LANGUAGE
)
from src.model import GPT

def load_model_and_tokenizer(device, checkpoint_path="sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt"):
    print(f"Using device: {device}")
    if LANGUAGE in ["hindi", "hinglish"]:
        tokenizer = CustomTokenizer()
        tokenizer.load()
    else:
        raise ValueError("This benchmark suite is designed for Hindi/Hinglish custom tokenizer.")

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        hidden_dim_ffn=hidden_dim_ffn
    )

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Defaulting to latest.")
        if os.path.exists("checkpoints"):
            available = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pt")])
            if not available:
                print("No checkpoints found. Returning None.")
                return None, None
            checkpoint_path = os.path.join("checkpoints", available[-1])
        else:
            print("checkpoints directory not found. Returning None.")
            return None, None

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        elif k.startswith("module."):
            new_state_dict[k[len("module.") :]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer

def arabic_to_devanagari(text):
    if not isinstance(text, str):
        text = str(text)
    mapping = str.maketrans('0123456789', '०१२३४५६७८९')
    return text.translate(mapping)
