from datasets import load_dataset

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