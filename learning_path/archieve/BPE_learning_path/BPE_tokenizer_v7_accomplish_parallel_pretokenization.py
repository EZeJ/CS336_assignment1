from typing import Optional, overload, BinaryIO
from dataclasses import dataclass
from collections import defaultdict, Counter
import regex
import re
import os
import multiprocessing
from tqdm import tqdm
from functools import partial


# === Chunking Utilities ===
def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def get_chunk(input_path: str, desired_num_chunks: int) -> list[str]:
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    return chunks

# === Tokenizer ===
GPT2_TOKENIZER_REGEX = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

@dataclass
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

class BPETokenizer:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, params: BPETokenizerParams) -> None: ...

    def __init__(self, params: Optional[BPETokenizerParams] = None) -> None:
        if params:
            self.vocab = params.vocab
            self.merges = params.merges
        else:
            self.vocab = {}
            self.merges = []

    def process_text_with_pre_tokenize(self, text: str) -> Counter[tuple[bytes, ...]]:
        '''
        Pre-tokenizes text using GPT-2 regex, encodes tokens in UTF-8, and returns a Counter
        of token byte tuples (e.g., (b't', b'h', b'e')) with their frequencies.
        '''
        PAT = GPT2_TOKENIZER_REGEX
        tokens_counter = Counter()

        for match in regex.finditer(PAT, text):
            token = match.group()
            token_bytes = token.encode("utf-8")
            byte_tuple = tuple(bytes([b]) for b in token_bytes)  # tuple of bytes
            tokens_counter[byte_tuple] += 1

        return tokens_counter

    @staticmethod
    def _static_process_chunk(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
        split_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = split_pattern.split(text)  # keeps special tokens
        tokenizer = BPETokenizer()
        tokens_counter = Counter()
        for chunk in chunks:
            chunk_counter = tokenizer.process_text_with_pre_tokenize(chunk)
            tokens_counter.update(chunk_counter)
        return tokens_counter
    
    def count_pair_frequencies(self, tokens_counter: Counter[tuple[bytes]]) -> dict[tuple[bytes, bytes], int]:
        counts = defaultdict(int)
        for word, freq in tokens_counter.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += freq
        return counts

    def find_max_pair(self, counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        return max(counts, key=lambda x: (counts[x], x))

    def merge_tokens(self, tokens_counter: Counter[tuple[bytes]], match1: bytes, match2: bytes) -> Counter[tuple[bytes]]:
        new_counter = Counter()
        merged_token = match1 + match2
        for word, freq in tokens_counter.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == match1 and word[i + 1] == match2:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_counter[tuple(new_word)] += freq
        return new_counter

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.vocab = {}
        self.merges = []

        # Initialize vocab with special tokens first
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        offset = len(special_tokens)
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

        next_index = offset + 256

        # Step 1: Get file chunks
        chunks = get_chunk(input_path, multiprocessing.cpu_count())

        # Step 2: Parallel processing of tokenization
        print("\n################################################")
        print(f"Tokenizing {len(chunks)} chunks using multiprocessing...")
        print("################################################")

        partial_func = partial(BPETokenizer._static_process_chunk, special_tokens=special_tokens)

        with multiprocessing.Pool() as pool:
            results = pool.map(partial_func, chunks)

        tokens_counter = Counter()
        for counter in results:
            tokens_counter.update(counter)

        with tqdm(total=vocab_size - len(self.vocab), desc="Training BPE") as pbar:
            while len(self.vocab) < vocab_size:
                pair_counts = self.count_pair_frequencies(tokens_counter)
                if not pair_counts:
                    break
                match1, match2 = self.find_max_pair(pair_counts)
                tokens_counter = self.merge_tokens(tokens_counter, match1, match2)
                self.vocab[next_index] = match1 + match2
                self.merges.append((match1, match2))
                next_index += 1
                pbar.update(1)
        return self.vocab, self.merges

    def get_params(self) -> BPETokenizerParams:
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)
