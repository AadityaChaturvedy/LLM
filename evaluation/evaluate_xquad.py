import os
import argparse
import torch
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
import re
import string

from src.config import context_length
from src.eval_utils import load_model_and_tokenizer, arabic_to_devanagari

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt"

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        exclude.update(['।', '॥'])
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

@torch.no_grad()
def evaluate_xquad(model, tokenizer, limit=None):
    print("\n--- Loading google/xtreme XQuAD-Hi ---")
    try:
        dataset = load_dataset("google/xtreme", "XQuAD.hi", split="validation")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    all_rows = []
    for row in dataset:
        all_rows.append(row)
            
    import random
    random.seed(42)
    random.shuffle(all_rows)
    
    if limit is not None:
        all_rows = all_rows[:limit]
            
    print(f"Loaded {len(all_rows)} questions.")
    
    exact_match = 0
    total_f1 = 0.0
    total = 0
    
    pbar = tqdm(total=len(all_rows), desc="Evaluating XQuAD")

    for sample in all_rows:
        context = sample["context"]
        question = sample["question"]
        true_answers = sample["answers"]["text"]

        prompt = f"सन्दर्भ: {context}\nप्रश्न: {question}\nउत्तर: "

        enc = tokenizer.encode(prompt)
        ids = enc.ids if hasattr(enc, 'ids') else enc
        
        # Give it 64 tokens of room to generate the answer
        if len(ids) >= context_length - 64:
            pbar.update(1)
            continue
            
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        
        generated_ids = []
        
        # Use autocast to run the model in half-precision (bfloat16/float16) for a massive speedup
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        eos_id = tokenizer.tokenizer.token_to_id.get("</s>", 2) if hasattr(tokenizer, "tokenizer") else 2
        
        with torch.autocast(device_type=DEVICE if DEVICE == 'cuda' else 'cpu', dtype=autocast_dtype):
            for _ in range(64):
                logits = model(x)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                if next_token.item() == eos_id:
                    break
                    
                token_str = tokenizer.decode([next_token.item()])
                if '\n' in token_str and len(generated_ids) > 3:
                    break
                    
                generated_ids.append(next_token.item())
                x = torch.cat((x, next_token), dim=1)

        gen_answer = tokenizer.decode(generated_ids).strip()
        
        norm_gen = normalize_answer(gen_answer)
        
        best_em = 0
        best_f1 = 0.0
        
        for true_answer in true_answers:
            norm_true = normalize_answer(true_answer)

            em = 1 if norm_gen == norm_true else 0
                
            gen_tokens = norm_gen.split()
            true_tokens = norm_true.split()
            
            common = Counter(gen_tokens) & Counter(true_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                f1 = 0.0
            else:
                prec = 1.0 * num_same / len(gen_tokens)
                rec = 1.0 * num_same / len(true_tokens)
                f1 = 2 * (prec * rec) / (prec + rec)
                
            best_em = max(best_em, em)
            best_f1 = max(best_f1, f1)
            
        exact_match += best_em
        total_f1 += best_f1
        total += 1
        pbar.update(1)
        
        if total % 10 == 0:
            pbar.set_description(f"XQuAD EM: {(exact_match/total)*100:.1f}% | F1: {(total_f1/total)*100:.1f}")
            
    pbar.close()
    
    if total > 0:
        em_pct = (exact_match / total) * 100
        f1_pct = (total_f1 / total) * 100
        print(f"\n========================================================")
        print(f"                 XQuAD HINDI RESULTS                    ")
        print(f"========================================================")
        print(f"Questions Evaluated : {total}")
        print(f"Exact Match (EM)    : {em_pct:.2f}%")
        print(f"F1 Score            : {f1_pct:.2f}%")
        print(f"========================================================\n")
    else:
        print("No valid questions evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate (default: all)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to the model checkpoint")
    args = parser.parse_args()
    
    print("Initializing XQuAD Evaluator...")
    model, tokenizer = load_model_and_tokenizer(DEVICE, args.checkpoint)
    if model:
        evaluate_xquad(model, tokenizer, limit=args.limit)
