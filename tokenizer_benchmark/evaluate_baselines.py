import sys
import os
import time

try:
    import tiktoken
except ImportError:
    print("Please install tiktoken to run baselines: pip install tiktoken")
    sys.exit(1)

# Add the parent directory to the path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.custom_tokenizer import CustomTokenizer
from tokenizer_benchmark.metrics import calculate_compression_ratio, calculate_subword_fertility

def evaluate_tiktoken(texts: list):
    print("Loading tiktoken (cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    total_chars = 0
    total_words = 0
    
    start_time = time.time()
    for text in texts:
        ids = enc.encode(text)
        total_tokens += len(ids)
        total_chars += len(text)
        total_words += len(text.split())
    end_time = time.time()
    
    comp_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    fertility = total_tokens / total_words if total_words > 0 else 0
    latency = end_time - start_time
    
    print("\n--- Tiktoken (GPT-4) Results ---")
    print(f"Total Text Size: {total_chars} characters ({total_words} words)")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Compression Ratio (Chars/Token): {comp_ratio:.2f}")
    print(f"Subword Fertility (Tokens/Word): {fertility:.2f}")
    print(f"Encoding Latency: {latency:.4f} seconds")

def evaluate_custom_tokenizer(texts: list):
    print("\nLoading CustomTokenizer...")
    try:
        tokenizer = CustomTokenizer()
        tokenizer.load()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    total_tokens = 0
    total_chars = 0
    total_words = 0
    
    start_time = time.time()
    for text in texts:
        output = tokenizer.encode(text)
        ids = output.ids
        total_tokens += len(ids)
        total_chars += len(text)
        total_words += len(text.split())
    end_time = time.time()
    
    comp_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    fertility = total_tokens / total_words if total_words > 0 else 0
    latency = end_time - start_time
    
    print("\n--- Custom Tokenizer Results ---")
    print(f"Total Text Size: {total_chars} characters ({total_words} words)")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Compression Ratio (Chars/Token): {comp_ratio:.2f}")
    print(f"Subword Fertility (Tokens/Word): {fertility:.2f}")
    print(f"Encoding Latency: {latency:.4f} seconds")


if __name__ == "__main__":
    sample_texts = [
        "भारत एक विशाल और विविध देश है। यहाँ कई भाषाएँ बोली जाती हैं।",
        "Artificial Intelligence is transforming the way we interact with technology.",
        "मैं कल Delhi जाऊंगा, and we will have a great time.",
        "ये NLP का project बहुत interesting है, I am enjoying it a lot.",
        "भारतीय प्रौद्योगिकी संस्थान रुड़की (IIT Roorkee) एक प्रमुख संस्थान है।"
    ] * 500  # Duplicate to simulate a larger corpus
    
    print(f"Evaluating Baselines on {len(sample_texts)} sample sentences...\n")
    evaluate_tiktoken(sample_texts)
    evaluate_custom_tokenizer(sample_texts)
