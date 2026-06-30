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

LABEL_MAP = {
    'Informative': 'सूचनात्मक', 'Descriptive': 'वर्णनात्मक',
    'Narrative': 'कथात्मक', 'Dialogue': 'संवाद',
    'Other': 'अन्य', 'Argumentative': 'तर्कपूर्ण'
}


def load_dm():
    ds = load_dataset("ai4bharat/indic_glue", "md.hi", split="test")
    raw = [(row['sentence'], row['discourse_mode']) for row in ds]
    labels = sorted(set(dm for _, dm in raw))
    hindi_opts = [" " + LABEL_MAP.get(l, l) for l in labels]
    rows = []
    for text, ans in raw:
        prompt = arabic_to_devanagari(f"वाक्य: {text}\nमोड: ")
        rows.append((prompt, hindi_opts, labels.index(ans)))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    if model:
        run_scoring_benchmark(model, tokenizer, device, "DM HINDI", load_dm(), args.limit)
