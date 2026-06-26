import os
import time
import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext
from tqdm import tqdm

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from src.custom_tokenizer import CustomTokenizer
from src.config import (
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn,
    LANGUAGE, MoE, moe_aux_loss_weight, moe_aux_loss_warmup_steps,
    CHECKPOINT_PATH
)
from src.model import GPT

SFT_CHECKPOINT_DIR = "sft_checkpoints_lora"
BASE_CHECKPOINT = CHECKPOINT_PATH
EPOCHS = 4
BATCH_SIZE = 2
ACCUMULATION_STEPS = 32
LR = 2e-4
MAX_SAMPLES = 100 # Set to 100 for testing, change to 150000 for full run

class InstructDataset(Dataset):
    def __init__(self, tokenizer, max_len=512, max_samples=MAX_SAMPLES):
        self.data = []
        eos_id = tokenizer.tokenizer.token_to_id.get("</s>", 0)
        
        # Load multiple datasets to match Airavata mixture
        configs = ["Indic_ShareLlama", "Wiki_Conv", "Anudesh", "FLAN-v2"]
        datasets_list = []
        for conf in configs:
            try:
                ds = load_dataset("ai4bharat/indic-align", conf, split="train")
                datasets_list.append(ds)
            except Exception as e:
                print(f"Skipping {conf}: {e}")
                
        if not datasets_list:
            raise ValueError("No datasets loaded.")
            
        print(f"Processing up to {max_samples} samples from mixed datasets...")
        
        is_master = int(os.environ.get("RANK", 0)) <= 0
        
        count = 0
        for ds, conf in zip(datasets_list, configs):
            for i, sample in enumerate(ds):
                if count >= max_samples:
                    break
                    
                if i == 0 and is_master:
                    print(f"[{conf}] First sample keys: {list(sample.keys())}")
                    if "hin_Deva" in sample:
                        print(f"[{conf}] hin_Deva type: {type(sample['hin_Deva'])}")
                        if isinstance(sample['hin_Deva'], list) and len(sample['hin_Deva']) > 0:
                            print(f"[{conf}] hin_Deva[0] type: {type(sample['hin_Deva'][0])}")
                    
                instruction = None
                response = None
                
                if "hin_Deva" in sample:
                    conv = sample["hin_Deva"]
                    if isinstance(conv, list) and len(conv) > 0:
                        turn = conv[0]
                        # Format 1: List of dicts [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                        if isinstance(turn, dict):
                            for j in range(len(conv) - 1):
                                role = str(conv[j].get("role", conv[j].get("from", ""))).lower()
                                if role in ["user", "human", "prompter", "question"]:
                                    instruction = conv[j].get("content", conv[j].get("value", ""))
                                    response = conv[j+1].get("content", conv[j+1].get("value", ""))
                                    break
                            # Fallback
                            if not instruction and len(conv) >= 2:
                                instruction = conv[0].get("content", conv[0].get("value", ""))
                                response = conv[1].get("content", conv[1].get("value", ""))
                        
                        # Format 2: List of lists [["prompt", "response"]]
                        elif isinstance(turn, list) and len(turn) >= 2:
                            instruction = turn[0]
                            response = turn[1]
                        
                if not instruction or not response:
                    continue
                    
                prompt_str = f"प्रश्न: {instruction.strip()}\nउत्तर: "
                
                enc_prompt = tokenizer.encode(prompt_str)
                p_ids = enc_prompt.ids if hasattr(enc_prompt, 'ids') else enc_prompt
                
                enc_ans = tokenizer.encode(response.strip())
                a_ids = enc_ans.ids if hasattr(enc_ans, 'ids') else enc_ans
                
                seq = p_ids + a_ids + [eos_id]
                
                if len(seq) <= max_len:
                    x = seq[:-1]
                    y = seq[1:]
                    
                    L = len(p_ids)
                    y_masked = [-100] * (L - 1) + y[L - 1:]
                    
                    self.data.append({
                        "x": x,
                        "y": y_masked
                    })
                    count += 1
            if count >= max_samples:
                break
                
        print(f"Retained {len(self.data)} samples that fit within {max_len} tokens.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_id=0):
    max_len = max(len(item["x"]) for item in batch)
    xs = []
    ys = []
    for item in batch:
        x = item["x"]
        y = item["y"]
        pad_len = max_len - len(x)
        xs.append(x + [pad_id] * pad_len)
        ys.append(y + [-100] * pad_len)
    
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(SFT_CHECKPOINT_DIR, exist_ok=True)
    
    def log_line(msg):
        if is_master_process:
            print(msg)

    log_line(f"Using device: {device} (World Size: {ddp_world_size})")

    if LANGUAGE in ["hindi", "hinglish"]:
        tokenizer = CustomTokenizer()
        tokenizer.load()
    else:
        raise ValueError("Must use Hindi/Hinglish tokenizer.")
        
    pad_id = tokenizer.tokenizer.token_to_id.get("<pad>", 0)

    if ddp and not is_master_process:
        dist.barrier()
        
    instruct_dataset = InstructDataset(tokenizer, max_len=context_length, max_samples=MAX_SAMPLES)
    
    if ddp and is_master_process:
        dist.barrier()
    
    sampler = DistributedSampler(instruct_dataset) if ddp else None
    dataloader = DataLoader(
        instruct_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        shuffle=(sampler is None),
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id)
    )

    if not os.path.exists(BASE_CHECKPOINT):
        log_line(f"Base checkpoint {BASE_CHECKPOINT} not found! Check path.")
        if ddp: dist.destroy_process_group()
        return

    log_line(f"Loading base weights from {BASE_CHECKPOINT}...")
    checkpoint = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=True)
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
    
    if not PEFT_AVAILABLE:
        log_line("Error: 'peft' library is not installed. Please run: pip install peft")
        if ddp: dist.destroy_process_group()
        sys.exit(1)
        
    log_line("Applying LoRA parameters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["wq", "wk", "wv", "out_linear", "w_gate", "w_up", "w_down"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    
    if is_master_process:
        model.print_trainable_parameters()
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if "cuda" in device and amp_dtype == torch.bfloat16:
        model.to(device, dtype=torch.bfloat16)
    else:
        model.to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    try:
        import bitsandbytes as bnb
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=LR)
    except ImportError:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=LR)

    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    model.train()
    optimizer_steps = math.ceil(len(dataloader) * EPOCHS / ACCUMULATION_STEPS)
    
    WARMUP_STEPS = 100
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, optimizer_steps - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    global_step = 0
    
    log_line(f"Starting LoRA Fine-Tuning for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        if ddp: sampler.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", disable=not is_master_process)
        for step, (xb, yb) in enumerate(pbar):
            xb, yb = xb.to(device), yb.to(device)
            
            is_last_micro_step = ((step + 1) % ACCUMULATION_STEPS == 0) or (step + 1 == len(dataloader))
            
            if ddp and not is_last_micro_step:
                ctx = model.no_sync()
            else:
                ctx = nullcontext()

            with ctx:
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=amp_dtype):
                    if MoE:
                        logits, aux_loss = model(xb, return_aux_loss=True)
                    else:
                        logits = model(xb)
                        aux_loss = logits.new_zeros(())
                    B, T, C = logits.shape
                    lm_loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T), ignore_index=-100)
                    aux_warmup = min(1.0, (global_step + 1) / max(1, moe_aux_loss_warmup_steps))
                    aux_weight = moe_aux_loss_weight * aux_warmup
                    loss = (lm_loss + aux_weight * aux_loss) / ACCUMULATION_STEPS
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if is_last_micro_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_master_process and global_step % 10 == 0:
                    pbar.set_postfix({"Loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}"})

        if is_master_process:
            raw_model = model.module if hasattr(model, "module") else model
            # Save only the LoRA adapters
            cp_path = os.path.join(SFT_CHECKPOINT_DIR, f"lora_epoch_{epoch+1}")
            raw_model.save_pretrained(cp_path)
            log_line(f"Saved LoRA weights to {cp_path}")

    if ddp:
        dist.destroy_process_group()
    log_line("LoRA Fine-Tuning Complete!")

if __name__ == "__main__":
    main()
