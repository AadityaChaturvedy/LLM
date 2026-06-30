import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import argparse
from datasets import load_dataset
from src.config import CHECKPOINT_PATH
from src.eval_utils import (
    load_model_and_tokenizer, arabic_to_devanagari,
    get_device, run_scoring_benchmark
)

LETTERS = ["क", "ख", "ग", "घ"]
ANS_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def load_mmlu(limit=None):
    """Load MMLU Hindi (streaming). Limit applied during download, not after shuffle."""
    print("\n--- Downloading/Loading FreedomIntelligence/MMLU_Hindi ---")
    ds = load_dataset("FreedomIntelligence/MMLU_Hindi", "default", split="test", streaming=True)
    iterator = iter(ds)
    try:
        next(iterator)  # skip broken first row (used as schema keys)
    except StopIteration:
        return []

    rows = []
    for row in iterator:
        if limit is not None and len(rows) >= limit:
            break
        vals = list(row.values())
        q, opts = str(vals[0]), [str(v) for v in vals[1:5]]
        ans = str(vals[5]).strip().upper()
        if ans not in ANS_MAP:
            continue

        prompt = f"प्रश्न: {q}\n"
        for i, opt in enumerate(opts):
            prompt += f"{LETTERS[i]}) {opt}\n"
        prompt += "उत्तर: "
        prompt = arabic_to_devanagari(prompt)
        # ponytail: no leading space — answer letter follows "उत्तर: " directly
        rows.append((prompt, LETTERS, ANS_MAP[ans]))

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Number of questions to evaluate (default: 500)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    if model:
        # ponytail: MMLU streams in order, limit applied during loading — no shuffle
        run_scoring_benchmark(model, tokenizer, device, "MMLU HINDI",
                              load_mmlu(args.limit), shuffle=False)
