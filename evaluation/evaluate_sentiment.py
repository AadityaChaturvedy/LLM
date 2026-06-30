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

# idx matches label: 0=negative, 1=neutral, 2=positive
SENTIMENT_OPTIONS = [" नकारात्मक", " तटस्थ", " सकारात्मक"]


def load_sentiment():
    ds = load_dataset("ai4bharat/indic_glue", "iitp-mr.hi", split="test")
    return [
        (arabic_to_devanagari(f"समीक्षा: {row['text']}\nभावना: "),
         SENTIMENT_OPTIONS, row['label'])
        for row in ds
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    if model:
        run_scoring_benchmark(model, tokenizer, device, "SENTIMENT HINDI",
                              load_sentiment(), args.limit)
