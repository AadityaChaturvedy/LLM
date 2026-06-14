import os
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.config import context_length
from src.eval_utils import load_model_and_tokenizer, arabic_to_devanagari

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt"

@torch.no_grad()
def evaluate_copa(model, tokenizer, limit=None):
    print("\n--- Loading ai4bharat/indic_glue COPA-Hi ---")
    try:
        ds = load_dataset("ai4bharat/indic_glue", "copa.hi", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    all_rows = []
    for row in ds:
        all_rows.append(row)
            
    import random
    random.seed(42)
    random.shuffle(all_rows)
    
    if limit is not None:
        all_rows = all_rows[:limit]
            
    print(f"Loaded {len(all_rows)} questions.")
    
    correct = 0
    total = 0
    
    pbar = tqdm(total=len(all_rows), desc="Evaluating COPA")

    for row in all_rows:
        premise = row['premise']
        choice1 = row['choice1']
        choice2 = row['choice2']
        q_type = row['question']
        true_label = row['label'] # 0 for choice1, 1 for choice2
        
        if q_type == 'cause':
            prompt = f"स्थिति: {premise}\nकारण: "
        else:
            prompt = f"स्थिति: {premise}\nप्रभाव: "
            
        prompt = arabic_to_devanagari(prompt)
        enc_prompt = tokenizer.encode(prompt)
        prompt_ids = enc_prompt.ids if hasattr(enc_prompt, 'ids') else enc_prompt
        
        if len(prompt_ids) == 0:
            pbar.update(1)
            continue
            
        best_loss = float('inf')
        best_idx = None
        
        for idx, opt_text in enumerate([choice1, choice2]):
            enc_opt = tokenizer.encode(" " + opt_text)
            opt_ids = enc_opt.ids if hasattr(enc_opt, 'ids') else enc_opt
            
            if len(opt_ids) == 0:
                continue
                
            full_ids = prompt_ids + opt_ids
            
            if len(full_ids) > context_length:
                full_ids = full_ids[-context_length:]
                prompt_len_in_full = len(full_ids) - len(opt_ids)
            else:
                prompt_len_in_full = len(prompt_ids)
                
            if prompt_len_in_full <= 0:
                continue
                
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=DEVICE)
            y = torch.tensor([full_ids[1:]], dtype=torch.long, device=DEVICE)
            
            logits, _ = model(x)
            
            start_idx = prompt_len_in_full - 1
            max_end = len(full_ids) - 1
            end_idx = min(start_idx + len(opt_ids), max_end)
            
            if end_idx <= start_idx:
                continue
            
            loss = F.cross_entropy(
                logits[0, start_idx:end_idx],
                y[0, start_idx:end_idx],
                reduction='sum'
            ).item() / max(1, end_idx - start_idx)
            
            if loss < best_loss:
                best_loss = loss
                best_idx = idx

        if best_idx is None:
            pbar.update(1)
            continue
                
        if best_idx == true_label:
            correct += 1
            
        total += 1
        pbar.update(1)
        
        if total % 10 == 0:
            pbar.set_description(f"COPA Acc: {(correct/total)*100:.1f}%")
            
    pbar.close()
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n========================================================")
        print(f"                 COPA HINDI RESULTS                     ")
        print(f"========================================================")
        print(f"Questions Evaluated : {total}")
        print(f"Correct Answers     : {correct}")
        print(f"Final Accuracy      : {accuracy:.2f}%")
        print(f"========================================================\n")
    else:
        print("No valid questions evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate (default: all)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to the model checkpoint")
    args = parser.parse_args()
    
    print("Initializing COPA Evaluator...")
    model, tokenizer = load_model_and_tokenizer(DEVICE, args.checkpoint)
    if model:
        evaluate_copa(model, tokenizer, limit=args.limit)
