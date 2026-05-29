import os
import re
import json
import heapq
import unicodedata
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Devanagari pre-tokenizer
# ---------------------------------------------------------------------------
_DEVA_BLOCK  = r'\u0900-\u097F'
_DEVA_DIGITS = r'\u0966-\u096F'

_DEVA_CLUSTER = (
    r'(?:'
        r'[{d}]'
        r'[\u094D{d}]*'
        r'[\u0900-\u0903\u093A-\u094C\u094E-\u094F\u0951-\u0957\u0962\u0963]*'
    r')+'.format(d=_DEVA_BLOCK)
)
_PRETOK_RE = re.compile(
    rf'{_DEVA_CLUSTER}|[{_DEVA_DIGITS}]+|[A-Za-z0-9]+|[^\w\s]',
    re.UNICODE,
)

def pre_tokenize(text: str) -> list:
    text = unicodedata.normalize('NFC', text)
    tokens = []
    for chunk in re.split(r'(\s+)', text):
        if not chunk or chunk.isspace():
            continue
        first = True
        for m in _PRETOK_RE.finditer(chunk):
            sub = m.group(0)
            tokens.append(('Ġ' + sub) if first else sub)
            first = False
    return tokens


# ---------------------------------------------------------------------------
# Fast BPE trainer using an inverted index
#
# Key idea: instead of scanning ALL vocab_dict words every merge, we keep
#   pair_to_words[pair] = set of word-ids that contain that pair
# so each merge only touches O(affected_words) instead of O(all_words).
# ---------------------------------------------------------------------------

class CustomIndicBPE:
    SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    def __init__(self):
        self.vocab        : list             = []
        self.token_to_id  : dict             = {}
        self.id_to_token  : dict             = {}
        self.merges       : list             = []   # [(p1,p2), merged]
        self._cache       : dict             = {}

    # ------------------------------------------------------------------ train
    def train(self, text_iterator, vocab_size, max_docs, recalc_every=500):
        # 1. word frequencies
        print(f"[train] reading up to {max_docs} docs …")
        word_counts: defaultdict = defaultdict(int)
        doc_count = 0
        for doc in tqdm(text_iterator, desc="reading"):
            if not doc or "text" not in doc:
                continue
            for tok in pre_tokenize(doc["text"]):
                word_counts[tok] += 1
            doc_count += 1
            if doc_count >= max_docs:
                break
        print(f"[train] unique pre-tokens={len(word_counts):,}  total={sum(word_counts.values()):,}")
        self._build_merge_ranks()

        # 2. base vocab
        unique_chars: set = set()
        for w in word_counts:
            unique_chars.update(w)
        self.vocab       = list(self.SPECIAL_TOKENS) + sorted(unique_chars)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}

        # 3. word table  (list of lists for O(1) mutation)
        #    words[wid]  = list of token-strings
        #    freqs[wid]  = frequency
        words_list = list(word_counts.keys())
        words = [list(w) for w in words_list]          # mutable sequences
        freqs = [word_counts[w] for w in words_list]
        W = len(words)

        num_merges = vocab_size - len(self.vocab)
        if num_merges <= 0:
            print("[train] base vocab already >= target."); return

        # 4. pair counts + inverted index
        #    pair_counts[pair]   = total frequency
        #    pair_to_wids[pair]  = set of word-ids containing that pair
        print("[train] building inverted index …")
        pair_counts  : defaultdict = defaultdict(int)
        pair_to_wids : defaultdict = defaultdict(set)

        for wid, (word, freq) in enumerate(zip(words, freqs)):
            for i in range(len(word) - 1):
                p = (word[i], word[i+1])
                pair_counts[p]  += freq
                pair_to_wids[p].add(wid)

        heap = [(-cnt, pair) for pair, cnt in pair_counts.items()]
        heapq.heapify(heap)

        self.merges = []
        pbar = tqdm(total=num_merges, desc="BPE merges")

        for merge_idx in range(num_merges):
            # find best pair (lazy-deletion heap)
            while True:
                if not heap:
                    pbar.close()
                    break
                neg_cnt, best_pair = heapq.heappop(heap)
                true_cnt = pair_counts.get(best_pair, 0)
                if -neg_cnt == true_cnt and true_cnt > 0:
                    break
                if true_cnt > 0:
                    heapq.heappush(heap, (-true_cnt, best_pair))
            else:
                break   # heap exhausted

            if pair_counts.get(best_pair, 0) == 0:
                break

            p1, p2 = best_pair
            merged = p1 + p2

            # record
            self.merges.append((best_pair, merged))
            new_id = len(self.vocab)
            self.vocab.append(merged)
            self.token_to_id[merged] = new_id
            self.id_to_token[new_id] = merged

            # apply merge only to affected words
            affected = list(pair_to_wids.get(best_pair, []))
            pair_to_wids.pop(best_pair, None)
            pair_counts[best_pair] = 0   # mark dead

            for wid in affected:
                word = words[wid]
                freq = freqs[wid]
                if freq == 0:
                    continue

                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and word[i] == p1 and word[i+1] == p2:
                        # --- remove old neighbour pairs ---
                        if new_word:
                            lp = (new_word[-1], p1)
                            pair_counts[lp] -= freq
                            pair_to_wids[lp].discard(wid)
                        if i+2 < len(word):
                            rp = (p2, word[i+2])
                            pair_counts[rp] -= freq
                            pair_to_wids[rp].discard(wid)

                        new_word.append(merged)

                        # --- add new left-neighbour pair ---
                        if len(new_word) >= 2:
                            lp2 = (new_word[-2], merged)
                            pair_counts[lp2]  += freq
                            pair_to_wids[lp2].add(wid)
                            heapq.heappush(heap, (-pair_counts[lp2], lp2))

                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                # right-of-merged pair (merged is last token appended, check its right)
                # find last occurrence of merged in new_word and check right neighbour
                for k in range(len(new_word)-1):
                    if new_word[k] == merged:
                        rp2 = (merged, new_word[k+1])
                        pair_counts[rp2]  += freq
                        pair_to_wids[rp2].add(wid)
                        heapq.heappush(heap, (-pair_counts[rp2], rp2))

                words[wid] = new_word

            # periodic full recalc to fix any accumulated drift
            if (merge_idx + 1) % recalc_every == 0:
                pair_counts  = defaultdict(int)
                pair_to_wids = defaultdict(set)
                for wid, (word, freq) in enumerate(zip(words, freqs)):
                    for i in range(len(word)-1):
                        p = (word[i], word[i+1])
                        pair_counts[p]  += freq
                        pair_to_wids[p].add(wid)
                heap = [(-cnt, p) for p, cnt in pair_counts.items() if cnt > 0]
                heapq.heapify(heap)

            pbar.update(1)

        pbar.close()
        print(f"[train] done. vocab={len(self.vocab):,}  merges={len(self.merges):,}")
        self._cache = {}

    def _build_merge_ranks(self):
        """Call once after load/train."""
        self._merge_ranks = {pair: i for i, ((pair), _) in enumerate(self.merges)}

    def _encode_word(self, word_str: str) -> list:
        if word_str in self._cache:
            return self._cache[word_str]
        
        word = list(word_str)
        if len(word) < 2:
            self._cache[word_str] = word
            return word

        pairs = {}
        for i in range(len(word) - 1):
            p = (word[i], word[i+1])
            if p in self._merge_ranks:
                pairs[i] = self._merge_ranks[p]

        while pairs:
            best_i = min(pairs, key=lambda i: pairs[i])
            
            # Guard: stale index after previous merges
            if best_i >= len(word) - 1:
                pairs.pop(best_i)
                continue
            
            p1, p2 = word[best_i], word[best_i + 1]
            if (p1, p2) not in self._merge_ranks:
                pairs.pop(best_i)
                continue
                
            merged = p1 + p2
            word[best_i] = merged
            del word[best_i + 1]

            pairs.pop(best_i, None)
            pairs.pop(best_i - 1, None)

            if best_i - 1 >= 0:
                p = (word[best_i - 1], merged)
                if p in self._merge_ranks:
                    pairs[best_i - 1] = self._merge_ranks[p]

            if best_i < len(word) - 1:
                p = (merged, word[best_i + 1])
                if p in self._merge_ranks:
                    pairs[best_i] = self._merge_ranks[p]
                else:
                    pairs.pop(best_i, None)

        self._cache[word_str] = word
        return word

    def encode(self, text: str):
        tokens = []
        for pt in pre_tokenize(text):
            tokens.extend(self._encode_word(pt))
        unk = self.token_to_id.get("<unk>", 0)
        ids = [self.token_to_id.get(t, unk) for t in tokens]

        class _Out:
            def __init__(self, ids, tokens):
                self.ids = ids; self.tokens = tokens
        return _Out(ids, tokens)

    def decode(self, ids: list) -> str:
        special = set(self.SPECIAL_TOKENS)
        out = ""
        for idx in ids:
            tok = self.id_to_token.get(idx, "<unk>")
            if tok in special:
                continue
            out += (' ' + tok[1:]) if tok.startswith('Ġ') else tok
        return out.strip()

    # ------------------------------------------------------------------ I/O
    def save(self, vocab_path: str, merges_path: str):
        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(merges_path, "w", encoding="utf-8") as f:
            for (p1, p2), _ in self.merges:
                f.write(f"{p1} {p2}\n")
        print(f"[save] {vocab_path}  {merges_path}")

    def load(self, vocab_path: str, merges_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.merges = []
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line: continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        self.merges.append(((parts[0], parts[1]), parts[0]+parts[1]))
        self._cache = {}
        print(f"[load] vocab={len(self.vocab):,}  merges={len(self.merges):,}")
        self._build_merge_ranks()