import os
import time
import math
import re
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import string
from collections import Counter

import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.config import (
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn,
    TOKENIZER_VOCAB_PATH, LANGUAGE, CHECKPOINT_PATH as CONFIG_CHECKPOINT_PATH
)
from src.eval_utils import load_model_and_tokenizer, arabic_to_devanagari, devanagari_to_arabic

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = CONFIG_CHECKPOINT_PATH
NUM_PPL_SAMPLES = 100
NUM_XQUAD_SAMPLES = 50
LLAMA_TOKENIZER = "meta-llama/Meta-Llama-3-8B"


@torch.no_grad()
def benchmark_perplexity(model, tokenizer):
    print("\n--- 1. Perplexity (PPL) on Wikipedia-Hindi ---")
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load Wikipedia-Hindi: {e}")
        return None

    total_loss = 0.0
    total_tokens = 0
    stride = context_length // 2

    print(f"Calculating perplexity over {NUM_PPL_SAMPLES} documents with sliding window (stride={stride})...")
    for i, doc in enumerate(dataset):
        if i >= NUM_PPL_SAMPLES:
            break
        text = doc["text"]
        enc = tokenizer.encode(text)
        ids = enc.ids if hasattr(enc, 'ids') else enc
        
        if len(ids) <= 1:
            continue
            
        # Sliding window with stride
        for chunk_start in range(0, len(ids) - 1, stride):
            chunk_ids = ids[chunk_start : chunk_start + context_length + 1]
            if len(chunk_ids) < 2:
                continue

            x = torch.tensor([chunk_ids[:-1]], dtype=torch.long, device=DEVICE)
            y = torch.tensor([chunk_ids[1:]], dtype=torch.long, device=DEVICE)

            logits, _ = model(x)
            B, T, C = logits.shape
            
            # If sliding window overlaps, we typically only score the non-overlapping part 
            # to avoid double-counting loss. For simplicity/efficiency, we just sum over the whole chunk here, 
            # but ideally you mask the loss for tokens already scored in the previous window.
            # A simpler correct approach: score only the last (T - stride) tokens, unless chunk_start == 0
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T), reduction='none')
            
            if chunk_start > 0:
                 tokens_to_score = min(stride, len(chunk_ids) - 1)
                 loss = loss[-tokens_to_score:]
                 
            total_loss += loss.sum().item()
            total_tokens += loss.numel()

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf')
        
    print(f"Average Loss: {avg_loss:.4f} | Perplexity: {ppl:.4f}")
    return ppl

def normalize_answer(s):
    """Lower text and remove punctuation, articles/Hindi stop words, and extra whitespace."""
    def remove_articles(text):
        # English articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Hindi common stop words (articles/prepositions equivalent)
        text = re.sub(r'\b(का|की|के|को|में|से|पर|है|हैं|था|थी|थे|और|या)\b', ' ', text)
        return text
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        # Include Hindi danda
        exclude.add('।')
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

@torch.no_grad()
def benchmark_xquad(model, tokenizer):
    print("\n--- 2. Extraction & Reading Comprehension (XQuAD-Hi) ---")
    try:
        dataset = load_dataset("google/xtreme", "XQuAD.hi", split="validation")
    except Exception as e:
        print(f"Failed to load XQuAD-Hi: {e}")
        return None, None

    exact_match = 0
    total_f1 = 0.0
    samples_evaluated = 0
    samples_skipped = 0
    i = 0

    while samples_evaluated < NUM_XQUAD_SAMPLES and i < len(dataset):
        sample = dataset[i]
        i += 1
        context = sample["context"]
        question = sample["question"]
        true_answer = sample["answers"]["text"][0]

        prompt = f"सन्दर्भ: {context}\nप्रश्न: {question}\nउत्तर: "
        prompt = arabic_to_devanagari(prompt)

        enc = tokenizer.encode(prompt)
        ids = enc.ids if hasattr(enc, 'ids') else enc
        
        if len(ids) >= context_length - 20:
            samples_skipped += 1
            continue
            
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        
        generated_ids = []
        for _ in range(20):
            logits, _ = model(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_str = tokenizer.decode([next_token.item()])
            if '\n' in token_str or '</s>' in token_str:
                break
            generated_ids.append(next_token.item())
            x = torch.cat((x, next_token), dim=1)

        gen_answer = tokenizer.decode(generated_ids).strip()
        
        norm_gen = normalize_answer(gen_answer)
        norm_true = normalize_answer(true_answer)

        # Standard Exact Match
        if norm_gen == norm_true:
            exact_match += 1
            
        # Standard F1 using Counter
        gen_tokens = norm_gen.split()
        true_tokens = norm_true.split()
        
        common = Counter(gen_tokens) & Counter(true_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1 = 0.0
        else:
            prec = 1.0 * num_same / len(gen_tokens)
            rec = 1.0 * num_same / len(true_tokens)
            f1 = 2 * (prec * rec) / (prec + rec)
            
        total_f1 += f1
        samples_evaluated += 1

    if samples_evaluated == 0:
        return 0, 0

    if samples_skipped > 0:
        print(f"[INFO] Skipped {samples_skipped} samples because context exceeded max length.")

    em_score = (exact_match / samples_evaluated) * 100
    f1_score = (total_f1 / samples_evaluated) * 100
    print(f"Evaluated {samples_evaluated} samples. EM: {em_score:.2f}% | F1: {f1_score:.2f}%")
    return em_score, f1_score

@torch.no_grad()
def benchmark_reasoning(model, tokenizer):
    print("\n--- 3. Reasoning & Logic (Math) ---")
    prompt = (
        "प्रश्न: राम के पास 5 सेब हैं। उसने 2 सेब खा लिए। उसके पास कितने सेब बचे?\nउत्तर: 3\n"
        "प्रश्न: सीता के पास 10 पेन हैं। उसने 4 पेन अपने दोस्त को दे दिए। उसके पास कितने पेन बचे?\nउत्तर: 6\n"
        "प्रश्न: एक टोकरी में 8 केले हैं। 3 केले और डाल दिए गए। कुल कितने केले हुए?\nउत्तर: "
    )
    prompt = arabic_to_devanagari(prompt)
    
    enc = tokenizer.encode(prompt)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    generated_ids = []
    for _ in range(10):
        logits, _ = model(x)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_str = tokenizer.decode([next_token.item()])
        if '\n' in token_str or '</s>' in token_str:
            break
        generated_ids.append(next_token.item())
        x = torch.cat((x, next_token), dim=1)
        
    answer = tokenizer.decode(generated_ids).strip()
    answer = devanagari_to_arabic(answer)
    print(f"Few-Shot Math Prompt:\n{prompt}")
    print(f"Model generated answer: '{answer}'")
    
    # Strict regex match for '11' or exact word match for Hindi 11
    is_correct = bool(re.search(r'\b11\b', answer)) or answer.strip() == "ग्यारह"
    print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
    return is_correct

def benchmark_tokenizer_efficiency(custom_tokenizer):
    print("\n--- 4. Tokenizer Efficiency vs Llama-3 ---")
    try:
        llama_tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
    except Exception as e:
        print(f"\n[WARNING] Could not load Llama 3 tokenizer. You may need to log in to HuggingFace via `huggingface-cli login` or pass a token. Skipping this test.\nError details: {e}")
        return None

    test_sentences = [
        "भारत एक महान देश है।",
        "अन्तर्राष्ट्रीय अंतरिक्ष स्टेशन में अनुसंधान चल रहा है।",
        "यह एक परीक्षण वाक्य है जिसमें संयुक्ताक्षर और मात्राएं हैं।",
        "विज्ञान और प्रौद्योगिकी के क्षेत्र में निरंतर प्रगति हो रही है।"
    ]
    
    total_chars = 0
    total_custom_tokens = 0
    total_llama_tokens = 0
    
    for text in test_sentences:
        total_chars += len(text)
        enc_custom = custom_tokenizer.encode(text)
        ids_custom = enc_custom.ids if hasattr(enc_custom, 'ids') else enc_custom
        total_custom_tokens += len(ids_custom)
        
        ids_llama = llama_tok.encode(text)
        total_llama_tokens += len(ids_llama)
        
    custom_ratio = total_chars / total_custom_tokens
    llama_ratio = total_chars / total_llama_tokens
    
    print(f"Custom Tokenizer: {total_custom_tokens} tokens ({custom_ratio:.2f} chars/token)")
    print(f"Llama-3 Tokenizer: {total_llama_tokens} tokens ({llama_ratio:.2f} chars/token)")
    
    if total_llama_tokens > total_custom_tokens:
        efficiency = total_llama_tokens / total_custom_tokens
        print(f"Efficiency Gain: The Custom tokenizer uses {efficiency:.2f}x fewer tokens than Llama-3 for Hindi!")
    else:
        efficiency = total_custom_tokens / total_llama_tokens
        print(f"Efficiency Loss: The Custom tokenizer uses {efficiency:.2f}x more tokens than Llama-3 for Hindi.")
    
    return custom_ratio, llama_ratio

@torch.no_grad()
def benchmark_latency(model, tokenizer):
    print("\n--- 5. Operational Latency (Tokens/sec) ---")
    
    # Warmup pass
    print("Running GPU warmup...")
    warmup_prompt = "नमस्ते"
    enc = tokenizer.encode(warmup_prompt)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    x_warmup = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(10):
        logits, _ = model(x_warmup[:, -context_length:])
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x_warmup = torch.cat((x_warmup, next_token), dim=1)
    if "cuda" in DEVICE:
        torch.cuda.synchronize()

    prompt = "भारत के इतिहास में"
    enc = tokenizer.encode(prompt)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    target_tokens = 200
    generated = 0
    
    print(f"Generating {target_tokens} tokens...")
    start_time = time.time()
    
    for _ in range(target_tokens):
        x_cond = x[:, -context_length:]
        logits, _ = model(x_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat((x, next_token), dim=1)
        generated += 1
        
    if "cuda" in DEVICE:
        torch.cuda.synchronize()
        
    end_time = time.time()
    duration = end_time - start_time
    tps = generated / duration
    
    print(f"Generated {generated} tokens in {duration:.2f} seconds.")
    print(f"Throughput: {tps:.2f} tokens/second")
    return tps

@torch.no_grad()
def benchmark_mmlu(model, tokenizer):
    print("\n--- 6. MMLU-Style Multiple Choice (Log-Likelihood) ---")
    
    questions = [
        {
            "prompt": "प्रश्न: भारत का प्रथम प्रधानमंत्री कौन था?\nक) महात्मा गांधी\nख) जवाहरलाल नेहरू\nग) सुभाष चंद्र बोस\nघ) लाल बहादुर शास्त्री\nउत्तर: ",
            "answer": "ख"
        },
        {
            "prompt": "प्रश्न: पृथ्वी के वायुमंडल में सबसे अधिक कौन सी गैस है?\nक) ऑक्सीजन\nख) कार्बन डाइऑक्साइड\nग) नाइट्रोजन\nघ) आर्गन\nउत्तर: ",
            "answer": "ग"
        },
        {
            "prompt": "प्रश्न: गुरुत्वाकर्षण का सिद्धांत किसने दिया?\nक) अल्बर्ट आइंस्टीन\nख) गैलीलियो\nग) आइजैक न्यूटन\nघ) निकोला टेस्ला\nउत्तर: ",
            "answer": "ग"
        },
        {
            "prompt": "प्रश्न: कंप्यूटर का मस्तिष्क किसे कहा जाता है?\nक) मॉनिटर\nख) कीबोर्ड\nग) प्रोसेसर\nघ) माउस\nउत्तर: ",
            "answer": "ग"
        },
        {
            "prompt": "प्रश्न: सौरमंडल का सबसे बड़ा ग्रह कौन सा है?\nक) मंगल\nख) पृथ्वी\nग) बृहस्पति\nघ) शनि\nउत्तर: ",
            "answer": "ग"
        }
    ]
    
    correct = 0
    total = len(questions)
    options = ["क", "ख", "ग", "घ"]
    
    for q in questions:
        enc_prompt = tokenizer.encode(q["prompt"])
        prompt_ids = enc_prompt.ids if hasattr(enc_prompt, 'ids') else enc_prompt
        
        best_loss = float('inf')
        best_option = None
        
        for opt in options:
            enc_opt = tokenizer.encode(opt)
            opt_ids = enc_opt.ids if hasattr(enc_opt, 'ids') else enc_opt
            
            # Combine prompt and option
            full_ids = prompt_ids + opt_ids
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=DEVICE)
            y = torch.tensor([full_ids[1:]], dtype=torch.long, device=DEVICE)
            
            # Calculate logits
            logits, _ = model(x)
            
            # We only calculate the loss for the option tokens
            start_idx = len(prompt_ids) - 1
            end_idx = start_idx + len(opt_ids)
            
            loss = F.cross_entropy(
                logits[0, start_idx:end_idx],
                y[0, start_idx:end_idx],
                reduction='mean'
            ).item()
            
            if loss < best_loss:
                best_loss = loss
                best_option = opt
                
        print(f"{q['prompt']}{best_option} (Predicted)")
        
        if best_option == q["answer"]:
            correct += 1
            print("-> Correct\n")
        else:
            print(f"-> Incorrect (Expected {q['answer']})\n")
            
    accuracy = (correct / total) * 100
    print(f"Evaluated {total} MMLU-style Hindi questions using Log-Likelihood.")
    print(f"Accuracy: {accuracy:.1f}%")
    return accuracy

def main():
    print("Initializing Benchmark Suite...")
    model, tokenizer = load_model_and_tokenizer(DEVICE)
    
    ppl = None
    em, f1 = None, None
    #ppl = benchmark_perplexity(model, tokenizer)
    #em, f1 = benchmark_xquad(model, tokenizer)
    math_correct = benchmark_reasoning(model, tokenizer)
    ratios = benchmark_tokenizer_efficiency(tokenizer)
    tps = benchmark_latency(model, tokenizer)
    mmlu_acc = benchmark_mmlu(model, tokenizer)
    
    print("\n========================================================")
    print("               BENCHMARK RESULTS SUMMARY                ")
    print("========================================================")
    print(f"Model Parameters   : 252M")
    print(f"Context Length     : 512")
    print(f"Perplexity (Hi)    : {f'{ppl:.2f}' if ppl is not None else 'N/A'} (Wikipedia-Hi)")
    print(f"XQuAD-Hi (EM/F1)   : {f'{em:.1f}% / {f1:.1f}%' if em is not None else 'N/A'}")
    print(f"Few-Shot Math      : {'Pass' if math_correct else 'Fail'}")
    print(f"Hindi MMLU         : {mmlu_acc:.1f}% Accuracy")
    if ratios:
        print(f"Tokenizer Ratio    : {ratios[0]:.2f} chars/token (Custom)")
        print(f"Llama-3 Tokenizer  : {ratios[1]:.2f} chars/token (Meta Llama 3)")
    print(f"Inference Latency  : {tps:.2f} tokens/second")
    print("========================================================\n")

if __name__ == "__main__":
    main()
