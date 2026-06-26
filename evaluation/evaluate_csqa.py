import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.config import context_length, CHECKPOINT_PATH
from src.eval_utils import load_model_and_tokenizer, arabic_to_devanagari

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def evaluate_csqa(model, tokenizer, limit=None):
    print("\n--- Loading ai4bharat/indic_glue CSQA-Hi ---")
    try:
        ds = load_dataset("ai4bharat/indic_glue", "csqa.hi", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    all_rows = []
    for row in ds:
        q = row['question']
        opts = row['options']
        a = row['answer']
        if '<MASK>' in q and len(opts) == 4:
            all_rows.append((q, opts, a))
            
    import random
    random.seed(42)
    random.shuffle(all_rows)
    
    if limit is not None:
        all_rows = all_rows[:limit]
            
    print(f"Loaded {len(all_rows)} questions.")
    
    correct = 0
    total = 0
    
    pbar = tqdm(total=len(all_rows), desc="Evaluating CSQA")
    
    for q_text, opts, true_ans in all_rows:
        best_loss = float('inf')
        best_option = None
        
        for opt_text in opts:
            before_mask = q_text.split('<MASK>')[0]
            before_mask = arabic_to_devanagari(before_mask)
            
            enc_prefix = tokenizer.encode(before_mask)
            prefix_ids = enc_prefix.ids if hasattr(enc_prefix, 'ids') else enc_prefix
            
            opt_text_deva = arabic_to_devanagari(opt_text)
            enc_opt = tokenizer.encode(" " + opt_text_deva)
            opt_ids = enc_opt.ids if hasattr(enc_opt, 'ids') else enc_opt
            
            if len(opt_ids) == 0:
                continue
                
            full_ids = prefix_ids + opt_ids
            
            if len(full_ids) > context_length:
                full_ids = full_ids[-context_length:]
                prompt_len_in_full = len(full_ids) - len(opt_ids)
            else:
                prompt_len_in_full = len(prefix_ids)
                
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
                best_option = opt_text
                
        if best_option is None:
            pbar.update(1)
            continue
                
        if best_option == true_ans:
            correct += 1
            
        total += 1
        pbar.update(1)
        
        if total % 10 == 0:
            pbar.set_description(f"CSQA Acc: {(correct/total)*100:.1f}%")
            
    pbar.close()
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n========================================================")
        print(f"                 CSQA HINDI RESULTS                     ")
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
    
    print("Initializing CSQA Evaluator...")
    model, tokenizer = load_model_and_tokenizer(DEVICE, args.checkpoint)
    if model:
        evaluate_csqa(model, tokenizer, limit=args.limit)
