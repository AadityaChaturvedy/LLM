import os
import torch
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
try:
    from peft import PeftModel
except ImportError:
    print("Please install peft: pip install peft")
    exit(1)

from src.model import GPT
from src.config import (
    context_length, CHECKPOINT_PATH
)

# Paths
BASE_CHECKPOINT = CHECKPOINT_PATH
LORA_ADAPTER_DIR = "sft_checkpoints_lora/lora_epoch_4"  # Default to best/last epoch
FUSED_CHECKPOINT = "sft_checkpoints_lora/ckpt_lora_fused.pt"

def main():
    print(f"Loading Base Checkpoint from {BASE_CHECKPOINT}...")
    checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[k_clean] = v
        
    ckpt_vocab_size, ckpt_embedding_dim = new_state_dict["embedding.token_embedding.weight"].shape
    ckpt_d_model = new_state_dict["blocks.0.mha.wq.weight"].shape[1]
    
    # Infer hidden_dim_ffn dynamically, supporting both standard FFN and MoE FFN
    if "blocks.0.ffn.w_down.weight" in new_state_dict:
        ckpt_hidden_ffn = new_state_dict["blocks.0.ffn.w_down.weight"].shape[1]
    else:
        ffn_w_down_keys = [k for k in new_state_dict.keys() if k.startswith("blocks.0.ffn.") and k.endswith(".w_down.weight")]
        if ffn_w_down_keys:
            ckpt_hidden_ffn = new_state_dict[ffn_w_down_keys[0]].shape[1]
        else:
            raise KeyError("Could not find any FFN weight (like w_down.weight) in blocks.0 to determine hidden_dim_ffn.")
            
    ckpt_num_layers = sum(1 for k in new_state_dict.keys() if k.endswith(".mha.wq.weight"))
    ckpt_num_heads = 16 

    # Detect GQA parameters from checkpoint shapes
    use_gqa = False
    num_kv_heads = ckpt_num_heads
    if "blocks.0.mha.wk.weight" in new_state_dict:
        wk_out_features = new_state_dict["blocks.0.mha.wk.weight"].shape[0]
        d_k = ckpt_d_model // ckpt_num_heads
        num_kv_heads = wk_out_features // d_k
        # Force GQA if wk has fewer KV heads or if q_scale is present in the checkpoint
        use_gqa = (num_kv_heads < ckpt_num_heads) or ("blocks.0.mha.q_scale" in new_state_dict)

    # Reconstruct base model
    model = GPT(
        vocab_size=ckpt_vocab_size,
        embedding_dim=ckpt_embedding_dim,
        context_length=context_length,
        num_layers=ckpt_num_layers,
        num_heads=ckpt_num_heads,
        num_kv_heads=num_kv_heads,
        d_model=ckpt_d_model,
        hidden_dim_ffn=ckpt_hidden_ffn,
        use_gqa=use_gqa
    )
    model.load_state_dict(new_state_dict)

    print(f"Loading LoRA Adapters from {LORA_ADAPTER_DIR}...")
    if not os.path.exists(LORA_ADAPTER_DIR):
        print(f"Error: LoRA directory {LORA_ADAPTER_DIR} not found. Run finetune_lora.py first.")
        return

    # Wrap model with PEFT and load adapter weights
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR)
    
    print("Fusing LoRA weights into base model (merge_and_unload)...")
    model = model.merge_and_unload()
    
    print(f"Saving Fused Model to {FUSED_CHECKPOINT}...")
    torch.save({"model": model.state_dict()}, FUSED_CHECKPOINT)
    print("Done! You can now run benchmark_suite.py on the fused checkpoint.")

if __name__ == "__main__":
    main()
