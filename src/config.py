import os

#Generate
max_new_tokens = 128
temperature = 0.7
top_k = 50

# Training Loop
train_loop = 122100

# Evaluation & Checkpointing
eval_every = 2000
eval_steps = 50
save_every = 10000
max_grad_norm = 1.0

# Tokenizer & Dataset Rows
TOTAL_ROWS = 10_000_000
batch_size_tokenizer = 1000

# Language Mode: 'english' or 'hindi' or 'hinglish'
LANGUAGE = "hindi"

hindi_vocab_size = 0
english_vocab_size = 0

# Model Dimensions (based on LANGUAGE mode)
if LANGUAGE == "hindi":
    hindi_vocab_size = 64_000
    vocab_size = hindi_vocab_size
    embedding_dim = 1792
    num_layers = 24
    num_heads = 16
    d_model = 1792
    hidden_dim_ffn = 4864
    batch_size_encoder = 2
    accumulation_steps = 32
    context_length = 512
elif LANGUAGE == "hinglish":
    hindi_vocab_size = 100_000
    english_vocab_size = 28_000
    vocab_size = hindi_vocab_size + english_vocab_size 
    embedding_dim = 1024
    num_layers = 24
    num_heads = 16
    d_model = 1024
    hidden_dim_ffn = 4096
    batch_size_encoder = 32
    accumulation_steps = 4
    context_length = 512
else:
    vocab_size = 52_000
    embedding_dim = 768
    num_layers = 12
    num_heads = 12
    d_model = 768
    hidden_dim_ffn = 3072     
    batch_size_encoder = 16       
    accumulation_steps = 8       
    context_length = 256      

# File Paths
TOKENIZER_DIR = os.path.join("data", LANGUAGE)
TOKENIZER_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "model-vocab.json")
TOKENIZER_MERGES_PATH = os.path.join(TOKENIZER_DIR, "model-merges.txt")
TOKENIZER_JSON_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json")
TOKENIZED_DATA_PATH = os.path.join(TOKENIZER_DIR, "tokens.npy")
