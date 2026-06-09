import json
import re
from datasets import load_dataset
import os
import multiprocessing

def is_hindi(sample):
    text = sample['context']
    if not re.search(r'[\u0900-\u097F]', text):
        return False
        
    hindi_stopwords = {'है', 'में', 'का', 'की', 'और', 'से', 'को', 'एक', 'यह', 'कि', 'हैं', 'पर', 'ने', 'नहीं'}
    marathi_stopwords = {'आहे', 'आणि', 'च्या', 'चा', 'ची', 'चे', 'मध्ये', 'ला', 'साठी', 'हे', 'तर'}
    
    words = set(text.split())
    h_score = len(words.intersection(hindi_stopwords))
    m_score = len(words.intersection(marathi_stopwords))
    
    return h_score > 2 and h_score > (m_score * 2)

def main():
    print("Loading l3cube-pune/indic-squad train split...")
    try:
        ds = load_dataset("l3cube-pune/indic-squad", split="train")
        print(f"Dataset loaded from cache! Total rows: {len(ds)}")
    except Exception as e:
        print(f"Not cached. {e}")
        return

    print("Filtering for Hindi...")
    # Map/filter with multiprocessing
    hindi_ds = ds.filter(is_hindi, num_proc=multiprocessing.cpu_count())
    
    print(f"Total pure Hindi rows found: {len(hindi_ds)}")
    
    os.makedirs("data", exist_ok=True)
    output_file = "data/hindi_squad_large.jsonl"
    
    # Save all rows
    target_samples = len(hindi_ds)
    hindi_ds = hindi_ds.select(range(target_samples))
    
    hindi_ds.to_json(output_file, force_ascii=False)
    print(f"Saved {target_samples} rows to {output_file}")

if __name__ == "__main__":
    main()
