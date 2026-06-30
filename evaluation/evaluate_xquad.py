import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import argparse
import random
import re
import string
from collections import Counter
from contextlib import nullcontext

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import context_length, CHECKPOINT_PATH
from src.eval_utils import load_model_and_tokenizer, get_device


def normalize_answer(s):
    """Lower, strip articles/punctuation, collapse whitespace."""
    s = re.sub(r'\b(a|an|the)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in set(string.punctuation) | {'।', '॥'})
    return ' '.join(s.split())


@torch.no_grad()
def evaluate_xquad(model, tokenizer, device, limit=None):
    print("\n--- Loading google/xtreme XQuAD-Hi ---")
    try:
        ds = load_dataset("google/xtreme", "XQuAD.hi", split="validation")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    all_rows = list(ds)
    random.seed(42)
    random.shuffle(all_rows)
    if limit is not None:
        all_rows = all_rows[:limit]

    print(f"Loaded {len(all_rows)} questions.")

    exact_match = 0
    total_f1 = 0.0
    total = 0
    eos_id = tokenizer.tokenizer.token_to_id.get("</s>", 2) if hasattr(tokenizer, "tokenizer") else 2

    # ponytail: autocast only helps on CUDA; no-op elsewhere
    if device == 'cuda':
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.autocast('cuda', dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    pbar = tqdm(total=len(all_rows), desc="Evaluating XQuAD")

    for sample in all_rows:
        prompt = f"सन्दर्भ: {sample['context']}\nप्रश्न: {sample['question']}\nउत्तर: "
        enc = tokenizer.encode(prompt)
        ids = enc.ids if hasattr(enc, 'ids') else enc

        if len(ids) >= context_length - 64:
            pbar.update(1)
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        generated_ids = []

        with amp_ctx:
            for _ in range(64):
                logits, _ = model(x)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                if next_token.item() == eos_id:
                    break
                tok_str = tokenizer.decode([next_token.item()])
                if '\n' in tok_str and len(generated_ids) > 3:
                    break
                generated_ids.append(next_token.item())
                x = torch.cat((x, next_token), dim=1)

        gen_answer = tokenizer.decode(generated_ids).strip()
        norm_gen = normalize_answer(gen_answer)

        best_em, best_f1 = 0, 0.0
        for true_answer in sample["answers"]["text"]:
            norm_true = normalize_answer(true_answer)
            best_em = max(best_em, int(norm_gen == norm_true))

            gen_tokens = norm_gen.split()
            true_tokens = norm_true.split()
            common = Counter(gen_tokens) & Counter(true_tokens)
            num_same = sum(common.values())
            if num_same and gen_tokens:
                prec = num_same / len(gen_tokens)
                rec = num_same / len(true_tokens)
                best_f1 = max(best_f1, 2 * prec * rec / (prec + rec))

        exact_match += best_em
        total_f1 += best_f1
        total += 1
        pbar.update(1)
        if total % 10 == 0:
            pbar.set_description(f"XQuAD EM: {(exact_match/total)*100:.1f}% | F1: {(total_f1/total)*100:.1f}")

    pbar.close()
    if total > 0:
        print(f"\n========================================================")
        print(f"{'  XQuAD HINDI RESULTS  ':^56}")
        print(f"========================================================")
        print(f"Questions Evaluated : {total}")
        print(f"Exact Match (EM)    : {(exact_match/total)*100:.2f}%")
        print(f"F1 Score            : {(total_f1/total)*100:.2f}%")
        print(f"========================================================\n")
    else:
        print("No valid questions evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    if model:
        evaluate_xquad(model, tokenizer, device, limit=args.limit)
