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

from src.custom_tokenizer import CustomTokenizer
from src.config import (
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn,
    LANGUAGE, MoE, moe_aux_loss_weight, moe_aux_loss_warmup_steps,
    CHECKPOINT_PATH
)
from src.model import GPT

SFT_CHECKPOINT_DIR = "sft_checkpoints"
BASE_CHECKPOINT = CHECKPOINT_PATH
EPOCHS = 3
BATCH_SIZE = 2
ACCUMULATION_STEPS = 16
LR = 1e-5

class QADataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=512):
        self.data = []
        eos_id = tokenizer.tokenizer.token_to_id.get("</s>", 0)
        
        print(f"Processing {len(hf_dataset)} samples...")
        for sample in hf_dataset:
            context = sample["context"]
            question = sample["question"]
            if "answers" in sample:
                if len(sample["answers"]["text"]) == 0:
                    continue
                answer = sample["answers"]["text"][0]
            else:
                answer = sample["answer_text"]
            
            prompt_str = f"सन्दर्भ: {context}\nप्रश्न: {question}\nउत्तर: "
            
            enc_prompt = tokenizer.encode(prompt_str)
            p_ids = enc_prompt.ids if hasattr(enc_prompt, 'ids') else enc_prompt
            
            enc_ans = tokenizer.encode(answer)
            a_ids = enc_ans.ids if hasattr(enc_ans, 'ids') else enc_ans
            
            seq = p_ids + a_ids + [eos_id]
            
            if len(seq) <= max_len:
                x = seq[:-1]
                y = seq[1:]
                
                L = len(p_ids)
                # We want loss only on answer tokens + eos
                # Note: y is seq[1:]. The prompt is seq[:L].
                # So y[:L-1] corresponds to predicting the prompt tokens seq[1:L].
                # We mask out the first L-1 elements so the model learns to predict the FIRST answer token
                # given the LAST prompt token. Masking L elements accidentally masked the first answer token!
                y_masked = [-100] * (L - 1) + y[L - 1:]
                
                self.data.append({
                    "x": x,
                    "y": y_masked
                })
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
        log_line("Loading 50k Large Hindi SQuAD dataset...")
        
    if ddp and not is_master_process:
        dist.barrier()
        
    hf_dataset = load_dataset("json", data_files="data/hindi_squad_large.jsonl", split="train")
    
    if ddp and is_master_process:
        dist.barrier()
        
    qa_dataset = QADataset(hf_dataset, tokenizer, max_len=context_length)
    
    sampler = DistributedSampler(qa_dataset) if ddp else None
    dataloader = DataLoader(
        qa_dataset, 
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
    
    # Clean keys first (remove DDP / compile prefixes)
    new_state_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[k_clean] = v
        
    # Dynamically infer dimensions from the cleaned checkpoint
    emb_key = "embedding.token_embedding.weight" if "embedding.token_embedding.weight" in new_state_dict else "embedding.weight"
    ckpt_vocab_size, ckpt_embedding_dim = new_state_dict[emb_key].shape
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
    ckpt_num_heads = 16 # Hardcoding to 16 since d_model=1024 for the 252M architecture (1024 // 64 = 16)
 
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
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if "cuda" in device and amp_dtype == torch.bfloat16:
        model.to(device, dtype=torch.bfloat16)
    else:
        model.to(device)

    if "cuda" in device and os.environ.get("COMPILE", "0") == "1":
        log_line("Compiling model (COMPILE=1)...")
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
    
    WARMUP_STEPS = 50
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, optimizer_steps - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    global_step = 0
    accum_loss = 0.0
    
    log_line(f"Starting SFT for {EPOCHS} epochs...")

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
                    # Explicit ignore_index=-100 is safer.
                    # Note on DDP: With DDP + no_sync(), loss from different micro-steps isn't synchronized
                    # until the last step. Effective loss scale can drift between ranks if batch sizes are uneven at tail.
                    lm_loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T), ignore_index=-100)
                    aux_warmup = min(1.0, (global_step + 1) / max(1, moe_aux_loss_warmup_steps))
                    aux_weight = moe_aux_loss_weight * aux_warmup
                    loss = (lm_loss + aux_weight * aux_loss) / ACCUMULATION_STEPS
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

        # Save checkpoint end of epoch
        if is_master_process:
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            cp_path = os.path.join(SFT_CHECKPOINT_DIR, f"ckpt_sft_epoch_{epoch+1}.pt")
            torch.save({"model": raw_model.state_dict()}, cp_path)
            log_line(f"Saved {cp_path}")

    if ddp:
        dist.destroy_process_group()
    log_line("SFT Complete!")

if __name__ == "__main__":
    main()
