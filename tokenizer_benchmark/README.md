# Tokenizer Benchmarking Suite

This folder is a standalone research harness for evaluating the custom Hindi/Hinglish tokenizer without changing the main LLM code. It imports the trained tokenizer from `src/`, then runs benchmark, ablation, reporting, and error-analysis scripts from this folder only.

## Files

- `run_benchmark.py`: Main benchmark runner. Produces JSON, CSV, and Markdown tables.
- `tokenizer_wrappers.py`: Shared wrappers for custom, HuggingFace, tiktoken, and SentencePiece tokenizers.
- `benchmark_datasets.py`: Public and local dataset loaders.
- `metrics.py`: Compression, fertility, OOV, character coverage, continuation rate, and speed metrics.
- `ablation.py`: Tests tokenizer design choices without editing the main tokenizer.
- `error_analysis.py`: Finds examples with poor compression, low round-trip coverage, or unknown tokens.
- `report.py`: Converts JSON results into paper-ready Markdown and LaTeX tables.
- `config.yaml`: Default benchmark config.
- `results/`: Generated result files.
- `evaluate.py` and `evaluate_baselines.py`: Older simple scripts retained for quick smoke tests.

## Setup

```bash
pip install -r tokenizer_benchmark/requirements.txt
```

Make sure the custom tokenizer has already been trained and saved under the paths expected by `src/config.py`.

## Quick Run

This uses local sample text and does not require HuggingFace downloads:

```bash
python tokenizer_benchmark/run_benchmark.py \
  --datasets sample,hinglish \
  --tokenizers custom,tiktoken \
  --output-dir tokenizer_benchmark/results
```

## Research Benchmark

Run public datasets and common baselines:

```bash
python tokenizer_benchmark/run_benchmark.py \
  --datasets flores,xlsum,xquad,csqa,hinglish \
  --tokenizers custom,muril,sarvam,krutrim,llama,qwen,mistral,gemma,tiktoken \
  --limit 1000 \
  --snapshot-datasets \
  --output-dir tokenizer_benchmark/results
```

Some baselines are gated or require network access. The runner skips unavailable tokenizers by default and records failures in `main_results.json`. Add `--strict` if you want the script to fail instead.

## Ablations

```bash
python tokenizer_benchmark/ablation.py \
  --dataset flores \
  --limit 1000 \
  --output tokenizer_benchmark/results/ablation_results.json
```

Current ablations:

- normal custom tokenizer
- no acronym transliteration
- no Devanagari cluster pre-tokenization
- no encode cache

## Error Analysis

```bash
python tokenizer_benchmark/error_analysis.py \
  --dataset flores \
  --tokenizers custom,tiktoken,muril \
  --limit 1000 \
  --output tokenizer_benchmark/results/error_examples.json
```

## Paper Tables

```bash
python tokenizer_benchmark/report.py \
  --input tokenizer_benchmark/results/main_results.json \
  --output-dir tokenizer_benchmark/results
```

This writes:

- `paper_table.md`
- `paper_table.tex`

## Metrics

- `fertility`: tokens per whitespace word. Lower is better.
- `chars_per_token`: Unicode characters per token. Higher is better.
- `bytes_per_token`: UTF-8 bytes per token. Higher is better.
- `unk_rate`: percent of tokens equal to the tokenizer's unknown ID.
- `empty_rate`: percent of lines that produce zero tokens.
- `continuation_rate`: percent of words that split into more than one token when encoded in isolation.
- `char_coverage`: round-trip character coverage after decoding.
- `speed_lines_per_sec`, `speed_chars_per_sec`, `speed_mb_per_sec`: encoding throughput.

Only compare results produced by the same script on the same dataset snapshot.
