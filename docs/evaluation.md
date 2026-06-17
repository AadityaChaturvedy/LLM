# Evaluation

`evaluation/` (per-task scripts: CSQA, WSTP, DM, Sentiment, COPA, XQuAD, MMLU) and `benchmark/benchmark_suite.py` (unified suite: perplexity on Hindi Wikipedia, XQuAD-Hi EM/F1, a Hindi few-shot arithmetic probe, tokenizer efficiency vs. LLaMA-3, generation latency, and MMLU-style log-likelihood scoring). All numbers below are for the 252M checkpoint and its three fine-tuned variants.

**CSQA-Hi accuracy**, placed against published baselines:

| Model | Type | Params | CSQA-Hi |
|---|---|---|---|
| Random Baseline | — | — | 25.00 |
| XLM-R | Encoder (fine-tuned) | 270M | 30.62 |
| **Ours (finetune_qa)** | Decoder LLM (fine-tuned) | 252M | **36.84** |
| IndicBERT-Large | Encoder (fine-tuned) | — | 37.01 |
| **Ours (finetune_lora)** | Decoder LLM (LoRA) | 252M | **38.34** |
| HindiLLM-Small | Decoder LLM | 124M | 38.53 |
| **Ours (finetune_instruct)** | Decoder LLM (fine-tuned) | 252M | **38.82** |
| mBERT | Encoder (fine-tuned) | 178M | 39.00 |
| **Ours (Base)** | Decoder LLM | 252M | **39.55** |
| IndicBERT-Base | Encoder (fine-tuned) | 35M | 41.55 |
| HindiLLM-Medium | Decoder LLM | 354M | 44.71 |
| GPT-3.5 Turbo (zero-shot) | Decoder LLM | ~175B | 44.56 |
| GPT-3.5 Turbo (few-shot) | Decoder LLM | ~175B | 50.84 |

Worth being upfront about: the base (untuned) checkpoint scores slightly higher than all three of our own fine-tuned variants on this task — fine-tuning didn't help CSQA-Hi specifically.

**WSTP / DM accuracy:**

| Model | WSTP (%) | DM (%) |
|---|---|---|
| Random Baseline | ~25.00 | ~16.67 |
| XLM-R | 76.92 | 79.94 |
| IndicBERT-Large | 77.80 | N/A |
| HindiLLM-Small | N/A | 78.68 |
| mBERT | 80.12 | 71.20 |
| IndicBERT-Base | 74.02 | 78.44 |
| HindiLLM-Medium | 77.19 | 80.48 |
| GPT-3.5 Turbo (zero-shot) | 76.75 | 50.91 |
| GPT-3.5 Turbo (few-shot) | 74.25 | 48.89 |
| **Ours (finetune_qa)** | 47.49 | 37.91 |
| **Ours (Base)** | 45.68 | 37.61 |
| **Ours (finetune_instruct)** | 47.51 | 38.01 |
| **Ours (LoRA)** | 47.28 | 38.01 |

The encoder baselines (XLM-R, mBERT, IndicBERT) and larger decoder models clearly lead on these two tasks. Our model is well above random but trails the established baselines. *Note: The model was optimized primarily for Hindi language modeling and tokenizer research rather than benchmark-specific supervised performance.*

**Sentiment Analysis / COPA:**

| Model | Sentiment (%) | COPA (%) |
|---|---|---|
| Random Baseline | 33.33 | 50.00 |
| **Ours (finetune_qa)** | 38.39 | 50.56 |
| **Ours (Base)** | 42.26 | 50.56 |
| **Ours (finetune_instruct)** | 40.32 | 50.11 |
| **Ours (LoRA)** | 48.39 | 51.00 |

COPA results sit close to the 50% random-chance floor across all variants — treat this task's results as inconclusive at this model scale rather than a real capability signal.

**XQuAD-Hi (extractive QA) across training stages:**

| Stage | PPL (↓) | XQuAD F1 (↑) |
|---|---|---|
| Base pretrain | 8.76 | 4.3% |
| MLQA SFT | 12.02 | 1.1% |
| Instruct SFT | 8.91 | 6.0% |

The MLQA SFT stage is a notable regression (both worse perplexity and worse F1 than the base model) before Instruct SFT recovers and improves on it — worth digging into if MLQA SFT is meant to be part of the final pipeline.

## Example Generations

Free-form generation from the base checkpoint (`generate.py`), greedy continuation of a story prompt:

> **Prompt:** एक बार की बात है, एक घने जंगल में एक शेर रहता था। वह बहुत
> **Continuation:** एक बार की बात है, एक घने जंगल में एक शेर रहता था। वह बहुत ही बुद्धिमान और समझदार था... (continues into a coherent but loosely-connected narrative about the lion and a tiger)

Extractive QA via `generate_qa.py`, given a short reference passage:

> **Context:** भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो)... इसका मुख्यालय बेंगलुरु में है।
> **Question:** इसरो का मुख्यालय कहाँ स्थित है?
> **Model answer:** बेंगलुरू — **correct**

> **Context:** ताजमहल... इसका निर्माण मुग़ल सम्राट शाहजहाँ ने अपनी पत्नी मुमताज़ महल की याद में करवाया था।
> **Question:** ताजमहल का निर्माण किसने करवाया था?
> **Model answer:** औरंगज़ेब — **incorrect** (the passage explicitly says Shah Jahan; the model named his son instead)

Including both outcomes deliberately — the model can extract an explicit fact correctly but can also confidently substitute a plausible-sounding wrong entity, which is the kind of failure mode worth flagging rather than hiding.
