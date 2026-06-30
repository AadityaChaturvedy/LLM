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

TITLE_MAP = {'titleA': 0, 'titleB': 1, 'titleC': 2, 'titleD': 3}


def load_wstp():
    ds = load_dataset("ai4bharat/indic_glue", "wstp.hi", split="test")
    rows = []
    for row in ds:
        ans = row['correctTitle']
        if ans not in TITLE_MAP:
            continue
        prompt = arabic_to_devanagari(f"पाठ: {row['sectionText']}\nशीर्षक: ")
        opts = [" " + arabic_to_devanagari(row[k]) for k in ('titleA', 'titleB', 'titleC', 'titleD')]
        rows.append((prompt, opts, TITLE_MAP[ans]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    if model:
        run_scoring_benchmark(model, tokenizer, device, "WSTP HINDI", load_wstp(), args.limit)
