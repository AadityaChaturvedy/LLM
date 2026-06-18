import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tokenizer_benchmark.benchmark_datasets import load_named_dataset, save_dataset_snapshot
from tokenizer_benchmark.metrics import evaluate_tokenizer_on_lines
from tokenizer_benchmark.tokenizer_wrappers import build_tokenizers


DEFAULT_DATASETS = ["sample", "hinglish"]
DEFAULT_TOKENIZERS = ["custom", "tiktoken"]
RESULT_FIELDS = [
    "dataset",
    "tokenizer",
    "vocab_size",
    "num_lines",
    "total_words",
    "total_chars",
    "total_bytes",
    "total_tokens",
    "fertility",
    "chars_per_token",
    "bytes_per_token",
    "unk_rate",
    "empty_rate",
    "continuation_rate",
    "char_coverage",
    "speed_lines_per_sec",
    "speed_chars_per_sec",
    "speed_mb_per_sec",
]


def read_config(path):
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install pyyaml or run without --config.") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_csv_arg(value, default):
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})


def format_float(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_markdown(path, rows):
    headers = [
        "Dataset",
        "Tokenizer",
        "Vocab",
        "Fertility",
        "Chars/Tok",
        "UNK%",
        "Coverage%",
        "Lines/s",
        "MB/s",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            values = [
                row["dataset"],
                row["tokenizer"],
                row.get("vocab_size") or "",
                format_float(row["fertility"]),
                format_float(row["chars_per_token"]),
                format_float(row["unk_rate"]),
                format_float(row["char_coverage"]),
                format_float(row["speed_lines_per_sec"]),
                format_float(row["speed_mb_per_sec"]),
            ]
            handle.write("| " + " | ".join(str(value) for value in values) + " |\n")


def add_relative_gains(rows, reference="custom"):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["dataset"], {})[row["tokenizer"]] = row

    for row in rows:
        ref = grouped.get(row["dataset"], {}).get(reference)
        if not ref or row["tokenizer"] == reference:
            row["fertility_vs_reference"] = 1.0
            row["chars_per_token_vs_reference"] = 1.0
            continue
        row["fertility_vs_reference"] = (
            row["fertility"] / ref["fertility"] if ref["fertility"] else None
        )
        row["chars_per_token_vs_reference"] = (
            row["chars_per_token"] / ref["chars_per_token"] if ref["chars_per_token"] else None
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run tokenizer benchmarks on fixed datasets.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names.")
    parser.add_argument("--tokenizers", default=None, help="Comma-separated tokenizer names.")
    parser.add_argument("--limit", type=int, default=None, help="Max lines per dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-path", default=None, help="Path for dataset name 'local'.")
    parser.add_argument("--output-dir", default="tokenizer_benchmark/results")
    parser.add_argument("--repeat-speed", type=int, default=3)
    parser.add_argument("--strict", action="store_true", help="Fail if any tokenizer cannot load.")
    parser.add_argument("--snapshot-datasets", action="store_true", help="Save exact evaluated lines.")
    args = parser.parse_args()

    config = read_config(args.config)
    datasets = parse_csv_arg(args.datasets, config.get("datasets", DEFAULT_DATASETS))
    tokenizers = parse_csv_arg(args.tokenizers, config.get("tokenizers", DEFAULT_TOKENIZERS))
    limit = args.limit if args.limit is not None else config.get("limit")
    output_dir = args.output_dir or config.get("output_dir", "tokenizer_benchmark/results")
    os.makedirs(output_dir, exist_ok=True)

    wrappers, failures = build_tokenizers(
        tokenizers,
        extra_hf=config.get("extra_hf_tokenizers", {}),
        sentencepiece=config.get("sentencepiece_tokenizers", {}),
        strict=args.strict,
    )

    rows = []
    dataset_snapshots = {}
    for dataset_name in datasets:
        print(f"[INFO] Loading dataset: {dataset_name}")
        lines = load_named_dataset(dataset_name, limit=limit, seed=args.seed, local_path=args.local_path)
        if args.snapshot_datasets:
            dataset_snapshots[dataset_name] = save_dataset_snapshot(dataset_name, lines, output_dir)

        for wrapper in wrappers:
            print(f"[INFO] Benchmarking {wrapper.name} on {dataset_name} ({len(lines)} lines)")
            row = evaluate_tokenizer_on_lines(wrapper, lines, repeat_speed=args.repeat_speed)
            row["dataset"] = dataset_name
            rows.append(row)

    rows = add_relative_gains(rows, reference="custom")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "datasets": datasets,
        "tokenizers_requested": tokenizers,
        "tokenizers_loaded": [wrapper.name for wrapper in wrappers],
        "tokenizer_failures": failures,
        "limit": limit,
        "seed": args.seed,
        "dataset_snapshots": dataset_snapshots,
    }

    json_path = os.path.join(output_dir, "main_results.json")
    csv_path = os.path.join(output_dir, "main_results.csv")
    md_path = os.path.join(output_dir, "tables.md")
    write_json(json_path, {"metadata": metadata, "results": rows})
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"[DONE] Wrote {json_path}")
    print(f"[DONE] Wrote {csv_path}")
    print(f"[DONE] Wrote {md_path}")


if __name__ == "__main__":
    main()
