import json
import os
import random
import unicodedata


SAMPLE_TEXTS = [
    "भारत एक विशाल और विविध देश है। यहाँ कई भाषाएँ बोली जाती हैं।",
    "भारतीय प्रौद्योगिकी संस्थान रुड़की एक प्रमुख संस्थान है।",
    "मैं कल Delhi जाऊंगा और NLP project पर काम करूंगा।",
    "ISRO ने चंद्रयान मिशन से भारत की अंतरिक्ष क्षमता दिखाई।",
    "Artificial Intelligence और भाषा तकनीक तेजी से बदल रही है।",
]


def normalize_lines(lines):
    normalized = []
    for line in lines:
        if line is None:
            continue
        line = unicodedata.normalize("NFC", str(line)).strip()
        if line:
            normalized.append(line)
    return normalized


def limit_and_shuffle(lines, limit=None, seed=42):
    lines = list(lines)
    random.Random(seed).shuffle(lines)
    if limit is not None:
        lines = lines[:limit]
    return lines


def load_local_text(path, limit=None, seed=42):
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return limit_and_shuffle(normalize_lines(lines), limit=limit, seed=seed)


def load_local_jsonl(path, text_fields=None, limit=None, seed=42):
    text_fields = text_fields or ["text", "sentence", "context", "question"]
    lines = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            parts = []
            for field in text_fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            if parts:
                lines.append(" ".join(parts))
    return limit_and_shuffle(normalize_lines(lines), limit=limit, seed=seed)


def _load_dataset_or_raise(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load public benchmark datasets.") from exc
    return load_dataset(*args, **kwargs)


def load_flores(limit=None, seed=42):
    errors = []
    candidates = [
        ("openlanguagedata/flores_plus", "hin_Deva", "dev", "sentence"),
        ("facebook/flores", "hin_Deva", "dev", "sentence"),
        ("tomasmajercik/flores-parquet", "hin_Deva", "validation", "sentence"),
    ]
    for dataset_name, config, split, field in candidates:
        try:
            ds = _load_dataset_or_raise(dataset_name, config, split=split)
            return limit_and_shuffle(normalize_lines(row[field] for row in ds), limit=limit, seed=seed)
        except Exception as exc:
            errors.append(f"{dataset_name}: {exc}")
    raise RuntimeError("Could not load FLORES Hindi. " + " | ".join(errors))


def load_xlsum(limit=None, seed=42):
    import json
    import tarfile
    from huggingface_hub import hf_hub_download

    archive_path = hf_hub_download(
        repo_id="csebuetnlp/xlsum",
        repo_type="dataset",
        filename="data/hindi_XLSum_v2.0.tar.bz2",
    )
    lines = []
    with tarfile.open(archive_path, "r:bz2") as tar:
        member = next(m for m in tar.getmembers() if "test" in m.name and m.name.endswith(".jsonl"))
        for raw_line in tar.extractfile(member):
            row = json.loads(raw_line)
            if row.get("text"):
                lines.append(row["text"])
    return limit_and_shuffle(normalize_lines(lines), limit=limit, seed=seed)

def load_xquad(limit=None, seed=42):
    ds = _load_dataset_or_raise("google/xtreme", "XQuAD.hi", split="validation")
    lines = []
    for row in ds:
        context = row.get("context", "")
        question = row.get("question", "")
        lines.append(f"{context} {question}")
    return limit_and_shuffle(normalize_lines(lines), limit=limit, seed=seed)


def load_indicglue_csqa(limit=None, seed=42):
    ds = _load_dataset_or_raise("ai4bharat/indic_glue", "csqa.hi", split="test")
    lines = []
    for row in ds:
        question = row.get("question", "")
        options = row.get("options", [])
        if isinstance(options, list):
            lines.append(question + " " + " ".join(str(option) for option in options))
        else:
            lines.append(question)
    return limit_and_shuffle(normalize_lines(lines), limit=limit, seed=seed)


def load_hinglish_sample(limit=None, seed=42):
    lines = [
        "आज meeting में NLP tokenizer benchmark discuss करेंगे.",
        "Mujhe Hindi aur English mixed data par model test karna hai.",
        "ISRO launch successful tha and everyone was proud.",
        "ये project research paper ke liye useful हो सकता है.",
        "Delhi campus में AI workshop बहुत interesting थी.",
    ]
    return limit_and_shuffle(normalize_lines(lines), limit=limit, seed=seed)


def load_sample(limit=None, seed=42):
    return limit_and_shuffle(normalize_lines(SAMPLE_TEXTS), limit=limit, seed=seed)


def load_named_dataset(name, limit=None, seed=42, local_path=None):
    name = name.strip().lower()
    if name == "sample":
        return load_sample(limit=limit, seed=seed)
    if name == "hinglish":
        return load_hinglish_sample(limit=limit, seed=seed)
    if name == "flores":
        return load_flores(limit=limit, seed=seed)
    if name == "xlsum":
        return load_xlsum(limit=limit, seed=seed)
    if name == "xquad":
        return load_xquad(limit=limit, seed=seed)
    if name in {"indicglue", "csqa"}:
        return load_indicglue_csqa(limit=limit, seed=seed)
    if name == "local":
        if not local_path:
            raise ValueError("--local-path is required for dataset 'local'.")
        if local_path.endswith(".jsonl"):
            return load_local_jsonl(local_path, limit=limit, seed=seed)
        return load_local_text(local_path, limit=limit, seed=seed)
    if os.path.exists(name):
        if name.endswith(".jsonl"):
            return load_local_jsonl(name, limit=limit, seed=seed)
        return load_local_text(name, limit=limit, seed=seed)
    raise ValueError(f"Unknown dataset: {name}")


def save_dataset_snapshot(dataset_name, lines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset_name}_lines.txt")
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.replace("\n", " ") + "\n")
    return path
