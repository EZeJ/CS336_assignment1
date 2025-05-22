from typing import Optional, overload
from dataclasses import dataclass
from collections import defaultdict, Counter
import regex

GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    def process_text_with_pre_tokenize(self, text: str) -> dict[tuple[int, int], int]:
        '''
        Pre-tokenizes text, encodes tokens in UTF-8, counts each byte-sequence (as tuple[int]),
        and returns bigram frequency counts (pair of byte values).
        '''
        PAT = GPT2_TOKENIZER_REGEX

        tokens_counter = Counter()

        # Step 1: Find tokens and encode to bytes
        for match in regex.finditer(PAT, text):
            token = match.group()
            token_bytes = token.encode("utf-8")
            byte_tuple = tuple(token_bytes)
            tokens_counter[byte_tuple] += 1
        return tokens_counter

    def count_pair_frequencies(self, tokens_counter: Counter) -> dict:
        counts = defaultdict(int)
        for word, freq in tokens_counter.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                counts[pair] += freq
        return counts

    def find_max_pair(self, counts: dict) -> tuple[str, str]:
        return max(counts, key=lambda x: (counts[x], x))

    def merge_tokens(self, tokens_counter: Counter, match1: int, match2: int) -> Counter:
        new_counter = Counter()
        for word, freq in tokens_counter.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == match1 and word[i+1] == match2:
                    new_word.append(match1 + match2)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_counter[tuple(new_word)] += freq
        return new_counter

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> None:
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []


        offset = len(special_tokens)

        # Add special tokens
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        # Add bytes
        for x in range(256):
            self.vocab[offset + x] = bytes([x])

        with open(input_path, "r", encoding="utf-8") as f:
            texts = f.read()

        tokens_counter = self.process_text_with_pre_tokenize(texts)

        index_tracking = 0
        while len(self.vocab) < vocab_size:
            counts = self.count_pair_frequencies(tokens_counter)
            if not counts:
                break
            match1, match2 = self.find_max_pair(counts)
            tokens_counter = self.merge_tokens(tokens_counter, match1, match2)
            index_tracking += 1
            # update the vocab and merges
            match1_bytes = self.vocab[match1]
            match2_bytes = self.vocab[match2]
            self.vocab[256 + offset + index_tracking] = match1_bytes + match2_bytes
            self.merges.append((match1_bytes, match2_bytes))
        
        return self.vocab, self.merges

    def get_params(self) -> BPETokenizerParams:
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)
    

# Example usage
if __name__ == "__main__":
    import json
    import time

    my_BPE_tokenizer = BPETokenizer()
    input_path = '/Users/ethanj/Documents/CODE/Stanford_CS336/assignment1-basics-main/tests/fixtures/corpus.en'
    start_time = time.time()
    vocab, merges = my_BPE_tokenizer.train_BPE(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5