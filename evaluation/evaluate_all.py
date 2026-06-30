import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import argparse
from src.config import CHECKPOINT_PATH
from src.eval_utils import load_model_and_tokenizer, get_device, run_scoring_benchmark

from evaluate_csqa import load_csqa
from evaluate_dm import load_dm
from evaluate_sentiment import load_sentiment
from evaluate_wstp import load_wstp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all Hindi evaluation benchmarks.")
    parser.add_argument("--limit", type=int, default=None, help="Max questions per benchmark")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    parser.add_argument("--mmlu", action="store_true", help="Also run MMLU Hindi")
    parser.add_argument("--xquad", action="store_true", help="Also run XQuAD Hindi")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device, args.checkpoint)
    if not model:
        sys.exit(1)

    # Always run: Sentiment, CSQA, WSTP, DM
    run_scoring_benchmark(model, tokenizer, device, "SENTIMENT HINDI", load_sentiment(), args.limit)
    run_scoring_benchmark(model, tokenizer, device, "CSQA HINDI", load_csqa(), args.limit)
    run_scoring_benchmark(model, tokenizer, device, "WSTP HINDI", load_wstp(), args.limit)
    run_scoring_benchmark(model, tokenizer, device, "DM HINDI", load_dm(), args.limit)

    # Optional
    if args.mmlu:
        from evaluate_mmlu import load_mmlu
        run_scoring_benchmark(model, tokenizer, device, "MMLU HINDI",
                              load_mmlu(args.limit), shuffle=False)
    if args.xquad:
        from evaluate_xquad import evaluate_xquad
        evaluate_xquad(model, tokenizer, device, limit=args.limit)
