import os
import yaml
import numpy as np
from cs336_basics.Tokenizers.BPE_tokenizer import Tokenizer
import json

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config_path = "./cs336_basics/configures/m4.yaml"
    config = load_config(config_path)
    dataset_cfg = config["dataset"]

    # Paths
    input_path = dataset_cfg["input_path"]
    vocab_path = dataset_cfg["vocab_out"]
    merges_path = dataset_cfg["merges_out"]
    train_out_path = dataset_cfg["train_path"]
    val_out_path = dataset_cfg["val_path"]
    special_tokens = dataset_cfg.get("special_tokens", [])

    # Load vocab and merges


    # Load vocab from JSON
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
        vocab = {v.encode("latin1"): int(k) for k, v in vocab_json.items()}

    # Load merges from text
    with open(merges_path, "r", encoding="utf-8") as f:
        merges = [tuple(line.strip().split()) for line in f]
        merges = [(a.encode("latin1"), b.encode("latin1")) for a, b in merges]

    # Rebuild tokenizer
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # Read and encode text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    ids = np.array(ids, dtype=np.uint16)

    # Split into train/val
    split = int(0.9 * len(ids))
    os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
    ids[:split].tofile(train_out_path)
    ids[split:].tofile(val_out_path)

    print(f"Encoded {len(ids)} tokens.")
    print(f"Saved {split} tokens to {train_out_path}")
    print(f"Saved {len(ids) - split} tokens to {val_out_path}")

if __name__ == "__main__":
    main()