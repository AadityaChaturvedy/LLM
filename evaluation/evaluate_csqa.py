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


def load_csqa():
    ds = load_dataset("ai4bharat/indic_glue", "csqa.hi", split="test")
    rows = []
    for row in ds:
        q, opts, a = row['question'], row['options'], row['answer']
        if '<MASK>' not in q or len(opts) != 4:
            continue
        prompt = arabic_to_devanagari(q.split('<MASK>')[0])
        try:
            answer_idx = opts.index(a)
        except ValueError:
            continue
        rows.append((prompt, [" " + arabic_to_devanagari(o) for o in opts], answer_idx))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    if model:
        run_scoring_benchmark(model, tokenizer, device, "CSQA HINDI", load_csqa(), args.limit)
