from datasets import load_dataset
import random

class FineWebDataset:
    def __init__(self, name="sample-10BT", split="train", streaming=True):
        print("Connecting to FineWeb 10B sample...")
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name=name, split=split, streaming=streaming)
        print("Dataset stream ready!")

    def check_data(self, num_samples=3):
        print("Checking the data from datastream...")
        for i, row in enumerate(self.dataset.take(num_samples)):
            print(f"--- Document {i+1} ---")
            print(f"ID: {row['id']}")
            print(f"URL: {row['url']}")
            print(f"Token Count: {row['token_count']}")
            print(f"Text Snippet: {row['text'][:250]}...\n")

class BilingualHindiDataset:
    def __init__(self, hindi_ratio=0.7, split="train", streaming=True):
        print("Connecting to Sangraha (Hindi Verified) and FineWeb (English)...")
        # Load verified Hindi Devanagari subset from Sangraha
        self.hindi_dataset = load_dataset(
            "ai4bharat/sangraha",
            data_dir="verified/hin",
            split=split,
            streaming=streaming
        )
        self.english_dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split=split,
            streaming=streaming
        )
        self.hindi_ratio = hindi_ratio
        self.dataset = self
        print("Bilingual dataset stream ready!")

    def __iter__(self):
        hindi_iter = iter(self.hindi_dataset)
        english_iter = iter(self.english_dataset)
        
        while True:
            if random.random() < self.hindi_ratio:
                try:
                    yield next(hindi_iter)
                except StopIteration:
                    hindi_iter = iter(self.hindi_dataset)
                    yield next(hindi_iter)
            else:
                try:
                    yield next(english_iter)
                except StopIteration:
                    english_iter = iter(self.english_dataset)
                    yield next(english_iter)