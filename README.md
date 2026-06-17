# Hindi/Hinglish Transformer LLM

> Hindi/Hinglish Transformer LLM is an effort to build a competitive Hindi-first language model stack entirely from scratch, including tokenizer, architecture, training pipeline, and evaluation suite, without relying on pretrained English-centric models.

## Highlights

- Custom Hindi/Hinglish tokenizer trained from scratch
- 252M parameter model trained on 7.96B tokens
- Faster than MuRIL, LLaMA-3, Qwen and Mistral tokenizers in benchmarks
- 39.55% CSQA-Hi accuracy
- Full training + finetuning pipeline in PyTorch
- No pretrained backbone used

## Documentation

For deep dives into specific components, see:
- [Architecture & Training](docs/architecture.md): GQA, RoPE, Pre-RMSNorm, SwiGLU, and our PyTorch optimization pipeline.
- [Tokenizer & Benchmarks](docs/tokenizer.md): Our custom Devanagari-aware BPE tokenizer and its compression/speed benchmarks.
- [Scaling History](docs/benchmarks.md): Checkpoints and evolution from 18M to 1.05B parameters.
- [Evaluation](docs/evaluation.md): Deep dive into our 252M model's performance on CSQA, WSTP, DM, and other tasks. *(Note: The model was optimized primarily for Hindi language modeling and tokenizer research rather than benchmark-specific supervised performance.)*

## Repository Structure

```
.
├── benchmark/
├── docs/
├── evaluation/
├── finetune/
├── src/
├── generate.py
├── generate_qa.py
├── plot_loss.py
├── train.py
└── requirements.txt
```

## Setup & Usage

```bash
pip install -r requirements.txt
```

Training (toggle `LANGUAGE`, `TRAIN_TOKENIZER`, `TRAIN_LLM` in `src/config.py` first):

```bash
python train.py                              # single GPU
torchrun --nproc_per_node=4 train.py          # multi-GPU DDP
```

Inference:

```bash
python generate.py        # free-form generation, loads from sft_checkpoints_instruct/
python generate_qa.py     # extractive QA, loads from sft_checkpoints/
```

## Reproducibility

- **Training logs:** Self-reported in docs; `log/` directory is gitignored.
- **Checkpoints:** `checkpoints/` and `sft_checkpoints/` are gitignored. Checkpoints must be retrained from scratch.
- **Evaluation scripts:** Provided in `evaluation/` and `benchmark/` to reproduce scores against your own checkpoints.
- **Hardware:** Developed and evaluated on standard GPU nodes (specific node specs omitted from public repo).
- **Random seeds:** Set locally but not strictly enforced across all data loaders.

## Notes & Limitations

- **QA hallucination:** The model can name plausible but wrong entities instead of the ones stated in the passage (e.g., naming Aurangzeb instead of Shah Jahan).
- **MLQA SFT regression:** MLQA SFT made both perplexity and F1 worse than the base model; only Instruct SFT improves on the base.
- **Config drift:** The configuration currently committed in `src/config.py` does not match the evaluated 252M checkpoint.

## Citation & Acknowledgements

If you use this codebase, tokenizer, or model architecture in your own research or projects, please provide adequate acknowledgement and cite this repository:

```bibtex
@software{chaturvedy_hindi_llm_2026,
  author       = {Aaditya Chaturvedy},
  title        = {Hindi/Hinglish Transformer LLM},
  year         = 2026,
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/AadityaChaturvedy/LLM}}
}
```