import argparse
import json
import os
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tokenizer_benchmark.benchmark_datasets import load_named_dataset
from tokenizer_benchmark.metrics import character_coverage, safe_decode
from tokenizer_benchmark.tokenizer_wrappers import build_tokenizers


def analyze_line(wrapper, text):
    ids = wrapper.encode(text)
    decoded = safe_decode(wrapper, ids)
    return {
        "tokenizer": wrapper.name,
        "token_count": len(ids),
        "unk_count": ids.count(wrapper.unk_id) if wrapper.unk_id is not None else 0,
        "char_coverage": character_coverage(text, decoded) * 100,
        "decoded": decoded,
    }


def main():
    parser = argparse.ArgumentParser(description="Find tokenizer failure and regression examples.")
    parser.add_argument("--dataset", default="sample")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizers", default="custom,tiktoken")
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--output", default="tokenizer_benchmark/results/error_examples.json")
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    tokenizer_names = [name.strip() for name in args.tokenizers.split(",") if name.strip()]
    wrappers, failures = build_tokenizers(tokenizer_names)
    lines = load_named_dataset(args.dataset, limit=args.limit, seed=args.seed, local_path=args.local_path)

    examples = []
    for text in lines:
        per_tokenizer = [analyze_line(wrapper, text) for wrapper in wrappers]
        custom = next((item for item in per_tokenizer if item["tokenizer"] == "custom"), None)
        best_other = min(
            [item for item in per_tokenizer if item["tokenizer"] != "custom"],
            key=lambda item: item["token_count"],
            default=None,
        )
        token_gap = 0
        if custom and best_other:
            token_gap = custom["token_count"] - best_other["token_count"]
        examples.append(
            {
                "text": text,
                "token_gap_custom_minus_best_other": token_gap,
                "min_coverage": min(item["char_coverage"] for item in per_tokenizer),
                "max_unk_count": max(item["unk_count"] for item in per_tokenizer),
                "tokenizers": per_tokenizer,
            }
        )

    poor_compression = sorted(
        examples, key=lambda item: item["token_gap_custom_minus_best_other"], reverse=True
    )[: args.top_k]
    poor_coverage = sorted(examples, key=lambda item: item["min_coverage"])[: args.top_k]
    unknowns = sorted(examples, key=lambda item: item["max_unk_count"], reverse=True)[: args.top_k]

    payload = {
        "dataset": args.dataset,
        "tokenizers": tokenizer_names,
        "failures": failures,
        "poor_compression": poor_compression,
        "poor_coverage": poor_coverage,
        "unknowns": unknowns,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {args.output}")


if __name__ == "__main__":
    main()
