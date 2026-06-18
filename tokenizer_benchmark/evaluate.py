import sys
import os
import time

# Add the parent directory to the path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.custom_tokenizer import CustomTokenizer
from tokenizer_benchmark.metrics import calculate_compression_ratio, calculate_subword_fertility, calculate_oov_rate

def evaluate_custom_tokenizer(texts: list):
    print("Loading CustomTokenizer...")
    try:
        tokenizer = CustomTokenizer()
        tokenizer.load()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure you have trained the tokenizer first using the main LLM scripts.")
        return

    total_tokens = 0
    total_chars = 0
    total_words = 0
    total_unks = 0
    
    start_time = time.time()
    for text in texts:
        output = tokenizer.encode(text)
        ids = output.ids
        tokens = output.tokens
        
        total_tokens += len(ids)
        total_chars += len(text)
        total_words += len(text.split())
        
        # Calculate unks
        unk_token = "<unk>"
        total_unks += sum(1 for t in tokens if t == unk_token)
        
    end_time = time.time()
    
    comp_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    fertility = total_tokens / total_words if total_words > 0 else 0
    oov_rate = (total_unks / total_tokens) * 100 if total_tokens > 0 else 0
    latency = end_time - start_time
    
    print("\n--- Custom Tokenizer Results ---")
    print(f"Total Text Size: {total_chars} characters ({total_words} words)")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Compression Ratio (Chars/Token): {comp_ratio:.2f}")
    print(f"Subword Fertility (Tokens/Word): {fertility:.2f}")
    print(f"OOV Rate: {oov_rate:.4f}%")
    print(f"Encoding Latency: {latency:.4f} seconds")


if __name__ == "__main__":
    # Sample Hindi and English texts for testing
    sample_texts = [
        "भारत एक विशाल और विविध देश है। यहाँ कई भाषाएँ बोली जाती हैं।",
        "Artificial Intelligence is transforming the way we interact with technology.",
        "मैं कल Delhi जाऊंगा, and we will have a great time.",
        "ये NLP का project बहुत interesting है, I am enjoying it a lot.",
        "भारतीय प्रौद्योगिकी संस्थान रुड़की (IIT Roorkee) एक प्रमुख संस्थान है।"
    ] * 100  # Duplicate to simulate a larger corpus
    
    print(f"Evaluating Tokenizer on {len(sample_texts)} sample sentences...")
    evaluate_custom_tokenizer(sample_texts)
