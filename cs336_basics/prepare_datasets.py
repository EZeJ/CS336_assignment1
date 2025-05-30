import os
import json
import numpy as np
from cs336_basics.Tokenizers.BPE_tokenizer import BPETokenizer
from cs336_basics.Tokenizers.BPE_tokenizer import Tokenizer
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_tokenizer(vocab, merges, vocab_out, merges_out):
    # Save vocab as JSON (human-readable)
    vocab_json = {v.decode("latin1"): k for k, v in vocab.items()}
    with open(vocab_out, "w") as f:
        json.dump(vocab_json, f)

    # Save merges as text file
    with open(merges_out, "w") as f:
        for a, b in merges:
            f.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")

def main():
    config = load_config("./cs336_basics/configures/m4.yaml")
    dataset_cfg = config["dataset"]

    input_path = dataset_cfg["input_path"]
    train_path = dataset_cfg["train_path"]
    val_path = dataset_cfg["val_path"]
    vocab_out = dataset_cfg["vocab_out"]
    merges_out = dataset_cfg["merges_out"]
    special_tokens = dataset_cfg["special_tokens"]
    vocab_size = config["model"]["vocab_size"]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    # 1. Train BPE tokenizer
    print("Training BPE tokenizer...")
    trainer = BPETokenizer()
    vocab, merges = trainer.train_BPE(input_path, vocab_size, special_tokens)

    save_tokenizer(vocab, merges, vocab_out, merges_out)

    # 2. Build encoder
    print("Encoding dataset...")
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text)
    ids = np.array(ids, dtype=np.uint16)

    # 3. Split train/val
    split = int(0.9 * len(ids))
    ids[:split].tofile(train_path)
    ids[split:].tofile(val_path)

    print(f"Tokenized {len(ids)} tokens.")
    print(f"Saved {split} tokens to {train_path}")
    print(f"Saved {len(ids) - split} tokens to {val_path}")

if __name__ == "__main__":
    main()
    
    
# import os
# import yaml
# import numpy as np
# from cs336_basics.Tokenizers.BPE_tokenizer import BPETokenizer

# def load_config(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def main():
#     config = load_config("./cs336_basics/configures/m4.yaml")
#     dataset_config = config["dataset"]

#     input_path = dataset_config["input_path"]
#     train_out_path = dataset_config["train_path"]
#     val_out_path = dataset_config["val_path"]
#     vocab_out_path = dataset_config["vocab_out"]
#     merges_out_path = dataset_config["merges_out"]
#     vocab_size = config["model"]["vocab_size"]
#     special_tokens = dataset_config.get("special_tokens", [])

#     os.makedirs("./cs336_basics/dataset", exist_ok=True)

#     # Step 1: Train tokenizer
#     tokenizer = BPETokenizer()
#     vocab, merges = tokenizer.train_BPE(
#         input_path=input_path,
#         vocab_size=vocab_size,
#         special_tokens=special_tokens
#     )

#     # Save vocab + merges
#     np.save(vocab_out_path, vocab, allow_pickle=True)
#     np.save(merges_out_path, merges, allow_pickle=True)

#     # Step 2: Rebuild tokenizer
#     tokenizer = BPETokenizer()
#     tokenizer.vocab = vocab
#     tokenizer.merges = merges

#     # Step 3: Tokenize text
#     with open(input_path, "r", encoding="utf-8") as f:
#         text = f.read()
#     token_ids = tokenizer.encode(text)
#     token_ids = np.array(token_ids, dtype=np.uint16)

#     # Step 4: Split and save
#     split = int(0.9 * len(token_ids))
#     token_ids[:split].tofile(train_out_path)
#     token_ids[split:].tofile(val_out_path)

#     print(f"Token count: total={len(token_ids)}, train={split}, val={len(token_ids)-split}")
#     print(f"Saved to: {train_out_path}, {val_out_path}")

# if __name__ == "__main__":
#     main()