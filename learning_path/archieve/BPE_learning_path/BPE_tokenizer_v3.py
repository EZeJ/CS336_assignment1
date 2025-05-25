from typing import Optional, overload
from dataclasses import dataclass
from collections import defaultdict, Counter
import regex
import re

GPT2_TOKENIZER_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

@dataclass
class BPETokenizerParams:
    vocab: dict[int, bytes]  # index -> bytes
    merges: list[tuple[int, int]]  # merges represented as index pairs

class BPETokenizer:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, params: BPETokenizerParams) -> None: ...

    def __init__(self, params: Optional[BPETokenizerParams] = None) -> None:
        if params is not None:
            self.vocab = params.vocab
            self.merges = params.merges
        else:
            self.vocab: dict[int, bytes] = {}
            self.merges: list[tuple[int, int]] = []

    def process_text_with_pre_tokenize(self, text: str) -> Counter[tuple[int, ...]]:
        PAT = GPT2_TOKENIZER_REGEX
        tokens_counter = Counter()

        for match in regex.finditer(PAT, text):
            token_bytes = match.group().encode("utf-8")
            token_tuple = tuple(token_bytes)  # tuple of ints
            tokens_counter[token_tuple] += 1

        return tokens_counter

    def count_pair_frequencies(self, tokens_counter: Counter[tuple[int, ...]]) -> dict[tuple[int, int], int]:
        counts = defaultdict(int)
        for word, freq in tokens_counter.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += freq
        return counts

    def find_max_pair(self, counts: dict[tuple[int, int], int]) -> tuple[int, int]:
        return max(counts, key=lambda x: (counts[x], x))

    def merge_tokens(self, tokens_counter: Counter[tuple[int, ...]], match1: int, match2: int, new_index: int) -> Counter[tuple[int, ...]]:
        new_counter = Counter()
        for word, freq in tokens_counter.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == match1 and word[i + 1] == match2:
                    new_word.append(new_index)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_counter[tuple(new_word)] += freq
        return new_counter

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
        self.vocab = {}
        self.merges = []

        # Step 1: Initialize vocab with special tokens
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        offset = len(special_tokens)
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

        next_index = offset + 256

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        split_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = split_pattern.split(text)

        tokens_counter = Counter()
        for chunk in chunks:
            chunk_counter = self.process_text_with_pre_tokenize(chunk)
            tokens_counter.update(chunk_counter)

        while len(self.vocab) < vocab_size:
            pair_counts = self.count_pair_frequencies(tokens_counter)
            if not pair_counts:
                break

            match1, match2 = self.find_max_pair(pair_counts)
            self.vocab[next_index] = self.vocab[match1] + self.vocab[match2]
            self.merges.append((match1, match2))
            tokens_counter = self.merge_tokens(tokens_counter, match1, match2, next_index)
            next_index += 1

        return self.vocab, self.merges

    def get_params(self) -> BPETokenizerParams:
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)