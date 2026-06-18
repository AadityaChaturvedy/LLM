import os
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


DEFAULT_HF_TOKENIZERS = {
    "muril": "google/muril-base-cased",
    "sarvam": "sarvamai/sarvam-1",
    "krutrim": "krutrim-ai-labs/Krutrim-2-Instruct-0131",
    "llama": "unsloth/llama-3-8b",
    "qwen": "Qwen/Qwen2.5-7B",
    "mistral": "unsloth/mistral-7b-v0.3",
    "gemma": "unsloth/gemma-2-9b",
}


class TokenizerLoadError(RuntimeError):
    pass


class TokenizerWrapper:
    def __init__(self, name, tokenizer, kind, vocab_size=None, unk_id=None):
        self.name = name
        self.tokenizer = tokenizer
        self.kind = kind
        self.vocab_size = vocab_size
        self.unk_id = unk_id

    def encode(self, text: str) -> list:
        if self.kind == "custom":
            output = self.tokenizer.encode(text)
            return output.ids if hasattr(output, "ids") else list(output)
        if self.kind == "hf":
            return self.tokenizer.encode(text, add_special_tokens=False)
        if self.kind == "tiktoken":
            return self.tokenizer.encode(text)
        if self.kind == "sentencepiece":
            return self.tokenizer.encode(text, out_type=int)
        raise ValueError(f"Unsupported tokenizer kind: {self.kind}")

    def decode(self, ids: list) -> str:
        if self.kind == "custom":
            return self.tokenizer.decode(ids)
        if self.kind == "hf":
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        if self.kind == "tiktoken":
            return self.tokenizer.decode(ids)
        if self.kind == "sentencepiece":
            return self.tokenizer.decode(ids)
        raise ValueError(f"Unsupported tokenizer kind: {self.kind}")


def load_custom_tokenizer(name="custom"):
    from src.config import LANGUAGE, TOKENIZER_MERGES_PATH, TOKENIZER_VOCAB_PATH
    from src.custom_bpe import CustomIndicBPE

    tokenizer = CustomIndicBPE(devanagari_only=(LANGUAGE == "hindi"))
    tokenizer.load(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)
    vocab_size = len(tokenizer.vocab)
    unk_id = tokenizer.token_to_id.get("<unk>")
    return TokenizerWrapper(name, tokenizer, "custom", vocab_size=vocab_size, unk_id=unk_id)


def load_hf_tokenizer(name, model_id):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerLoadError("Install transformers to load HuggingFace tokenizers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else None
    vocab_size = getattr(tokenizer, "vocab_size", None)
    return TokenizerWrapper(name, tokenizer, "hf", vocab_size=vocab_size, unk_id=unk_id)


def load_tiktoken(name="tiktoken", encoding_name="cl100k_base"):
    try:
        import tiktoken
    except ImportError as exc:
        raise TokenizerLoadError("Install tiktoken to load OpenAI tokenizers.") from exc

    tokenizer = tiktoken.get_encoding(encoding_name)
    return TokenizerWrapper(name, tokenizer, "tiktoken", vocab_size=tokenizer.n_vocab, unk_id=None)


def load_sentencepiece(name, model_path):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise TokenizerLoadError("Install sentencepiece to load SentencePiece models.") from exc

    tokenizer = spm.SentencePieceProcessor(model_file=model_path)
    unk_id = tokenizer.unk_id() if tokenizer.unk_id() >= 0 else None
    return TokenizerWrapper(name, tokenizer, "sentencepiece", vocab_size=tokenizer.vocab_size(), unk_id=unk_id)


def build_tokenizers(names, extra_hf=None, sentencepiece=None, strict=False):
    """
    Build tokenizer wrappers.

    names can include:
      custom, tiktoken, muril, sarvam, krutrim, llama, qwen, mistral, gemma
    """
    extra_hf = extra_hf or {}
    sentencepiece = sentencepiece or {}
    hf_tokenizers = dict(DEFAULT_HF_TOKENIZERS)
    hf_tokenizers.update(extra_hf)

    wrappers = []
    failures = []
    for raw_name in names:
        name = raw_name.strip()
        if not name:
            continue
        try:
            if name == "custom":
                wrappers.append(load_custom_tokenizer())
            elif name == "tiktoken":
                wrappers.append(load_tiktoken())
            elif name in hf_tokenizers:
                wrappers.append(load_hf_tokenizer(name, hf_tokenizers[name]))
            elif name in sentencepiece:
                wrappers.append(load_sentencepiece(name, sentencepiece[name]))
            else:
                raise TokenizerLoadError(f"Unknown tokenizer: {name}")
        except Exception as exc:
            message = f"{name}: {exc}"
            failures.append(message)
            if strict:
                raise
            print(f"[WARN] Skipping tokenizer {message}")

    if not wrappers:
        raise TokenizerLoadError("No tokenizers loaded. Check tokenizer names and dependencies.")
    return wrappers, failures
