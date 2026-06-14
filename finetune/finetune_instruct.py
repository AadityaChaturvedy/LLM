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

from src.custom_tokenizer import CustomTokenizer
from src.config import (
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn,
    LANGUAGE
)
from src.model import GPT

def arabic_to_devanagari(text):
    if not isinstance(text, str): return text
    return text.translate(str.maketrans('0123456789', '०१२३४५६७८९'))

SFT_CHECKPOINT_DIR = "sft_checkpoints_instruct"
BASE_CHECKPOINT = "checkpoints/ckpt_step_120000.pt"
EPOCHS = 2
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
LR = 2e-5
MAX_SAMPLES = 50000

class InstructDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=512, max_samples=MAX_SAMPLES):
        self.data = []
        eos_id = tokenizer.tokenizer.token_to_id.get("</s>", 0)
        
        print(f"Processing up to {max_samples} samples...")
        count = 0
        for sample in hf_dataset:
            if count >= max_samples:
                break
                
            # Extract from the hin_Deva column
            instruction = None
            response = None
            
            # Format 1: IndoWordNet style
            if sample.get("language") == "hi" and "interactions" in sample:
                conv = sample["interactions"]
                if isinstance(conv, list) and len(conv) >= 2:
                    instruction = conv[0]
                    response = conv[1]
                    
            # Format 2: OpenAssistant_T / Indic_ShareLlama style
            elif "hin_Deva" in sample:
                conv = sample["hin_Deva"]
                if isinstance(conv, list) and len(conv) > 0:
                    turn = conv[0]
                    if isinstance(turn, list) and len(turn) >= 2:
                        instruction = turn[0]
                        response = turn[1]
                    
            if not instruction or not response:
                continue
                
            instruction = arabic_to_devanagari(instruction.strip())
            response = arabic_to_devanagari(response.strip())
                
            prompt_str = f"प्रश्न: {instruction}\nउत्तर: "
            
            enc_prompt = tokenizer.encode(prompt_str)
            p_ids = enc_prompt.ids if hasattr(enc_prompt, 'ids') else enc_prompt
            
            enc_ans = tokenizer.encode(response.strip())
            a_ids = enc_ans.ids if hasattr(enc_ans, 'ids') else enc_ans
            
            seq = p_ids + a_ids + [eos_id]
            
            if len(seq) <= max_len:
                x = seq[:-1]
                y = seq[1:]
                
                L = len(p_ids)
                # We mask the prompt so loss only computes over the response
                y_masked = [-100] * (L - 1) + y[L - 1:]
                
                self.data.append({
                    "x": x,
                    "y": y_masked
                })
                count += 1
                
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

    # 1. Load Tokenizer & Dataset
    if LANGUAGE in ["hindi", "hinglish"]:
        tokenizer = CustomTokenizer()
        tokenizer.load()
    else:
        raise ValueError("Must use Hindi/Hinglish tokenizer.")
        
    pad_id = tokenizer.tokenizer.token_to_id.get("<pad>", 0)

    if is_master_process:
        log_line("Loading ai4bharat/indic-align dataset...")
    
    if ddp and not is_master_process:
        dist.barrier()
        
    # Using 'OpenAssistant_T' for high-quality conversational data
    hf_dataset = load_dataset("ai4bharat/indic-align", "OpenAssistant_T", split="train")
    instruct_dataset = InstructDataset(hf_dataset, tokenizer, max_len=context_length, max_samples=MAX_SAMPLES)
    
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

    # 2. Load Base Model
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
    ckpt_hidden_ffn = new_state_dict["blocks.0.ffn.w_down.weight"].shape[1]
    ckpt_num_layers = sum(1 for k in new_state_dict.keys() if k.endswith(".mha.wq.weight"))
    ckpt_num_heads = 16 

    # Detect GQA parameters from checkpoint shapes
    use_gqa = False
    num_kv_heads = ckpt_num_heads
    if "blocks.0.mha.wk.weight" in new_state_dict:
        wk_out_features = new_state_dict["blocks.0.mha.wk.weight"].shape[0]
        d_k = ckpt_d_model // ckpt_num_heads
        num_kv_heads = wk_out_features // d_k
        use_gqa = num_kv_heads < ckpt_num_heads

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
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(device)

    if "cuda" in device:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # 3. Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR)
    except ImportError:
        optimizer = optim.AdamW(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    # 4. Training Loop
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
    accum_loss = 0.0
    
    log_line(f"Starting Instruction Fine-Tuning for {EPOCHS} epochs...")

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
                    logits = model(xb)
                    B, T, C = logits.shape
                    loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T), ignore_index=-100) / ACCUMULATION_STEPS
                
                accum_loss += loss.item()
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if is_last_micro_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_master_process and global_step % 10 == 0:
                    pbar.set_postfix({"Loss": f"{accum_loss:.4f}"})
                
                accum_loss = 0.0

        if is_master_process:
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            cp_path = os.path.join(SFT_CHECKPOINT_DIR, f"ckpt_instruct_epoch_{epoch+1}.pt")
            torch.save({"model": raw_model.state_dict()}, cp_path)
            log_line(f"Saved {cp_path}")

    if ddp:
        dist.destroy_process_group()
    log_line("Instruction Fine-Tuning Complete!")

if __name__ == "__main__":
    main()
