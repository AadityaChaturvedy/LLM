import time
import unicodedata
from collections import Counter

def calculate_compression_ratio(text: str, num_tokens: int) -> float:
    """
    Calculates the compression ratio (characters per token).
    Higher is better, meaning fewer tokens are needed to represent the text.
    """
    if num_tokens == 0:
        return 0
    return len(text) / num_tokens

def calculate_subword_fertility(text: str, tokens: list) -> float:
    """
    Calculates the average number of subwords a word is split into.
    Assumes words are separated by whitespace in the original text.
    """
    words = text.split()
    if len(words) == 0:
        return 0
    
    # Simple heuristic: total tokens / total words
    # A true fertility calculation aligns tokens to words exactly, 
    # but this macro-average is a standard proxy.
    return len(tokens) / len(words)

def calculate_oov_rate(tokens: list, unk_token: str = "<unk>") -> float:
    """
    Calculates the percentage of tokens that are Out-Of-Vocabulary (mapped to UNK).
    """
    if len(tokens) == 0:
        return 0
    unk_count = sum(1 for t in tokens if t == unk_token)
    return (unk_count / len(tokens)) * 100


def word_split(text: str) -> list:
    """Whitespace word split after Unicode NFC normalization."""
    text = unicodedata.normalize("NFC", text)
    return [word for word in text.split() if word]


def character_coverage(original_text: str, decoded_text: str) -> float:
    """Order-insensitive character coverage after ignoring whitespace."""
    original_chars = [char for char in original_text if not char.isspace()]
    decoded_chars = [char for char in decoded_text if not char.isspace()]
    if not original_chars:
        return 1.0

    decoded_counts = Counter(decoded_chars)
    matched = 0
    for char in original_chars:
        if decoded_counts[char] > 0:
            matched += 1
            decoded_counts[char] -= 1
    return matched / len(original_chars)


def safe_decode(wrapper, ids: list) -> str:
    try:
        return wrapper.decode(ids)
    except Exception:
        return ""


def evaluate_tokenizer_on_lines(wrapper, lines: list, repeat_speed: int = 3) -> dict:
    """
    Evaluate a tokenizer wrapper on a fixed list of lines.

    The wrapper must expose:
      - name
      - vocab_size
      - unk_id
      - encode(text) -> list[int]
      - decode(ids) -> str
    """
    normalized_lines = [unicodedata.normalize("NFC", line).strip() for line in lines]
    normalized_lines = [line for line in normalized_lines if line]

    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    unk_count = 0
    empty_count = 0
    continued_words = 0
    coverage_sum = 0.0

    for line in normalized_lines:
        ids = wrapper.encode(line)
        decoded = safe_decode(wrapper, ids)
        words = word_split(line)

        total_tokens += len(ids)
        total_words += len(words)
        total_chars += len(line)
        total_bytes += len(line.encode("utf-8"))
        unk_count += ids.count(wrapper.unk_id) if wrapper.unk_id is not None else 0
        empty_count += 1 if len(ids) == 0 else 0
        coverage_sum += character_coverage(line, decoded)

        for word in words:
            word_ids = wrapper.encode(word)
            if len(word_ids) > 1:
                continued_words += 1

    # Speed is measured separately so metric collection does not pollute timing.
    start = time.perf_counter()
    for _ in range(max(1, repeat_speed)):
        for line in normalized_lines:
            wrapper.encode(line)
    elapsed = time.perf_counter() - start
    speed_denominator = elapsed / max(1, repeat_speed)

    return {
        "tokenizer": wrapper.name,
        "vocab_size": wrapper.vocab_size,
        "num_lines": len(normalized_lines),
        "total_words": total_words,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "fertility": total_tokens / total_words if total_words else 0.0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "unk_rate": (unk_count / total_tokens) * 100 if total_tokens else 0.0,
        "empty_rate": (empty_count / len(normalized_lines)) * 100 if normalized_lines else 0.0,
        "continuation_rate": (continued_words / total_words) * 100 if total_words else 0.0,
        "char_coverage": (coverage_sum / len(normalized_lines)) * 100 if normalized_lines else 0.0,
        "speed_lines_per_sec": len(normalized_lines) / speed_denominator if speed_denominator > 0 else 0.0,
        "speed_chars_per_sec": total_chars / speed_denominator if speed_denominator > 0 else 0.0,
        "speed_mb_per_sec": (total_bytes / (1024 * 1024)) / speed_denominator if speed_denominator > 0 else 0.0,
    }
