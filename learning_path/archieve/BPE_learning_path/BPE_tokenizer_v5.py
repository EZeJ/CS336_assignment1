from typing import Optional, overload
from dataclasses import dataclass
from collections import defaultdict, Counter
import regex
import re
import multiprocessing
from collections import Counter

GPT2_TOKENIZER_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

@dataclass
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

class BPETokenizer:
    # Overload 1: no arguments
    @overload
    def __init__(self) -> None: ...
    
    # Overload 2: params only
    @overload
    def __init__(self, params: BPETokenizerParams) -> None: ...

    def __init__(self, params: Optional[BPETokenizerParams] = None) -> None:
        if params is not None:
            self.vocab = params.vocab
            self.merges = params.merges
        else:
            self.vocab: dict[int, bytes] = {}
            self.merges: list[tuple[bytes, bytes]] = []

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

    def count_pair_frequencies(self, tokens_counter: Counter[tuple[bytes]]) -> dict[tuple[bytes, bytes], int]:
        '''
        Count frequencies of adjacent byte pairs across all tokens in the corpus.
        '''
        counts = defaultdict(int)
        for word, freq in tokens_counter.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] += freq
        return counts

    def find_max_pair(self, counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
        '''
        Find the most frequent byte pair, breaking ties lexicographically.
        '''
        return max(counts, key=lambda x: (counts[x], x))

    def merge_tokens(self, tokens_counter: Counter[tuple[bytes]], match1: bytes, match2: bytes) -> Counter[tuple[bytes]]:
        '''
        Merge the most frequent pair in all tokens.
        '''
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

    def _process_chunk_wrapper(self, args):
        instance, chunk = args
        return instance.process_text_with_pre_tokenize(chunk)

    def process_chunks_with_multiprocessing(self, chunks: list[str]) -> Counter:
        '''
        Processes chunks in parallel using all available CPUs.
        Returns a combined Counter of token byte tuples.
        '''
        num_cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(self._process_chunk_wrapper, [(self, chunk) for chunk in chunks])

        tokens_counter = Counter()
        for counter in results:
            tokens_counter.update(counter)

        return tokens_counter

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.vocab = {}
        self.merges = []

        # Step 1: Initialize vocab
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        offset = len(special_tokens)
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

        next_index = offset + 256

        # Step 2: Load text and split on special tokens
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split on special tokens (escaped, joined by |)
        split_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = split_pattern.split(text)  # list of strings between special tokens

        # Step 3: Process each chunk separately
        
        tokens_counter = self.process_chunks_with_multiprocessing(chunks)

        # Step 4: BPE merge loop
        while len(self.vocab) < vocab_size:
            pair_counts = self.count_pair_frequencies(tokens_counter)
            if not pair_counts:
                break

            match1, match2 = self.find_max_pair(pair_counts)
            merged_token = match1 + match2

            tokens_counter = self.merge_tokens(tokens_counter, match1, match2)

            self.vocab[next_index] = merged_token
            self.merges.append((match1, match2))
            next_index += 1

        return self.vocab, self.merges


    def get_params(self) -> BPETokenizerParams:
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)