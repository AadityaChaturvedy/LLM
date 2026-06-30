import os
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.custom_tokenizer import CustomTokenizer
from src.config import (
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn, LANGUAGE,
    CHECKPOINT_PATH
)
from src.model import GPT


def get_device():
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(device, checkpoint_path=CHECKPOINT_PATH):
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
    return text.translate(str.maketrans('0123456789', '०१२३४५६७८९'))


def devanagari_to_arabic(text):
    if not isinstance(text, str):
        text = str(text)
    return text.translate(str.maketrans('०१२३४५६७८९', '0123456789'))


# ---------------------------------------------------------------------------
# Shared scoring kernel for multiple-choice benchmarks
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_options(model, tokenizer, prompt, options, device):
    """Score each option by average cross-entropy loss.

    Options should include a leading space if needed (e.g. ' word' not 'word').
    Returns the index of the lowest-loss option, or None.
    """
    enc_prompt = tokenizer.encode(prompt)
    prompt_ids = enc_prompt.ids if hasattr(enc_prompt, 'ids') else enc_prompt
    if not prompt_ids:
        return None

    best_loss = float('inf')
    best_idx = None

    for idx, opt_text in enumerate(options):
        enc_opt = tokenizer.encode(opt_text)
        opt_ids = enc_opt.ids if hasattr(enc_opt, 'ids') else enc_opt
        if not opt_ids:
            continue

        full_ids = prompt_ids + opt_ids
        if len(full_ids) > context_length:
            full_ids = full_ids[-context_length:]
            prompt_len = len(full_ids) - len(opt_ids)
        else:
            prompt_len = len(prompt_ids)

        if prompt_len <= 0:
            continue

        x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
        logits, _ = model(x)

        start = prompt_len - 1
        end = min(start + len(opt_ids), len(full_ids) - 1)
        if end <= start:
            continue

        loss = F.cross_entropy(
            logits[0, start:end], y[0, start:end], reduction='mean'
        ).item()

        if loss < best_loss:
            best_loss = loss
            best_idx = idx

    return best_idx


def run_scoring_benchmark(model, tokenizer, device, name, rows, limit=None, shuffle=True):
    """Run a multiple-choice scoring benchmark.

    Args:
        rows: list of (prompt_str, options_list, answer_idx) tuples.
        limit: max questions to evaluate (after shuffling).
        shuffle: if True, shuffle with seed 42 before limiting.
    """
    if shuffle:
        random.seed(42)
        random.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    print(f"Loaded {len(rows)} questions.")
    correct = total = 0
    pbar = tqdm(total=len(rows), desc=f"Evaluating {name}")

    for prompt, options, answer_idx in rows:
        pred = score_options(model, tokenizer, prompt, options, device)
        if pred is None:
            pbar.update(1)
            continue
        if pred == answer_idx:
            correct += 1
        total += 1
        pbar.update(1)
        if total % 10 == 0:
            pbar.set_description(f"{name} Acc: {(correct/total)*100:.1f}%")

    pbar.close()
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n========================================================")
        print(f"{'  ' + name + ' RESULTS  ':^56}")
        print(f"========================================================")
        print(f"Questions Evaluated : {total}")
        print(f"Correct Answers     : {correct}")
        print(f"Final Accuracy      : {accuracy:.2f}%")
        print(f"========================================================\n")
    else:
        print("No valid questions evaluated.")
