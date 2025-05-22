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

    def get_raw_tokens(self, text: str) -> list[bytes]:
        """
        Applies GPT-2 regex and encodes each token as bytes.
        """
        return [token.encode("utf-8") for token in regex.findall(GPT2_TOKENIZER_REGEX, text)]

    def get_list_of_byte_values(self, byte_tokens: list[bytes]) -> list[list[int]]:
        """
        Converts list of byte tokens into list of byte integer sequences.
        """
        return [list(token) for token in byte_tokens]

    def count_pair_frequencies(self, tokens_counter: Counter[tuple[int]]) -> dict[tuple[int, int], int]:
        counts = defaultdict(int)
        for word, freq in tokens_counter.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                counts[pair] += freq
        return counts

    def find_max_pair(self, counts: dict[tuple[int, int], int]) -> tuple[int, int]:
        return max(counts, key=lambda x: (counts[x], x))

    def merge_tokens(self, tokens_counter: Counter[tuple[int]], match1: int, match2: int) -> Counter[tuple[int]]:
        new_counter = Counter()
        for word, freq in tokens_counter.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == match1 and word[i+1] == match2:
                    new_word.append(256 + len(self.merges))  # merged token index
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_counter[tuple(new_word)] += freq
        return new_counter

    def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
        self.vocab = {}
        self.merges = []

        offset = len(special_tokens)

        # Add special tokens
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        # Add all 256 byte values
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        raw_bytes = self.get_raw_tokens(text)
        byte_sequences = self.get_list_of_byte_values(raw_bytes)

        tokens_counter = Counter(tuple(seq) for seq in byte_sequences)

        while len(self.vocab) < vocab_size:
            counts = self.count_pair_frequencies(tokens_counter)
            if not counts:
                break
            match1, match2 = self.find_max_pair(counts)

            new_index = len(self.vocab)
            new_bytes = self.vocab[offset + match1] + self.vocab[offset + match2] \
                if match1 < 256 and match2 < 256 \
                else b""

            self.vocab[new_index] = new_bytes
            self.merges.append((match1, match2))
            tokens_counter = self.merge_tokens(tokens_counter, match1, match2)

        return self.vocab, [(self.vocab[a], self.vocab[b]) for a, b in self.merges]

    def get_params(self) -> BPETokenizerParams:
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)
    
# class BPETokenizer:
#     # Overload 1: no arguments
#     @overload
#     def __init__(self) -> None: ...
    
#     # Overload 2: params only
#     @overload
#     def __init__(self, params: BPETokenizerParams) -> None: ...

#     def __init__(self, params: Optional[BPETokenizerParams] = None) -> None:
#         if params is not None:
#             self.vocab = params.vocab
#             self.merges = params.merges
#         else:
#             self.vocab: dict[int, bytes] = {}
#             self.merges: list[tuple[bytes, bytes]] = []

#     def get_raw_tokens(self, text: str) -> str:
#         return " ".join(regex.findall(GPT2_TOKENIZER_REGEX, text))

#     def get_list_of_characters(self, raw_tokens: str):
#         return [list(word) for word in raw_tokens.split()]

#     def count_pair_frequencies(self, tokens_counter: Counter) -> dict:
#         counts = defaultdict(int)
#         for word, freq in tokens_counter.items():
#             for i in range(len(word) - 1):
#                 pair = (word[i], word[i+1])
#                 counts[pair] += freq
#         return counts

#     def find_max_pair(self, counts: dict) -> tuple[str, str]:
#         return max(counts, key=lambda x: (counts[x], x))

#     def merge_tokens(self, tokens_counter: Counter, match1: str, match2: str) -> Counter:
#         new_counter = Counter()
#         for word, freq in tokens_counter.items():
#             new_word = []
#             i = 0
#             while i < len(word):
#                 if i < len(word) - 1 and word[i] == match1 and word[i+1] == match2:
#                     new_word.append(match1 + match2)
#                     i += 2
#                 else:
#                     new_word.append(word[i])
#                     i += 1
#             new_counter[tuple(new_word)] += freq
#         return new_counter

#     def train_BPE(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> None:
#         self.vocab = {}
#         self.merges = []
#         offset = len(special_tokens)

#         # Add special tokens
#         for i, token in enumerate(special_tokens):
#             self.vocab[i] = token.encode("utf-8")

#         # Add bytes
#         for x in range(256):
#             self.vocab[offset + x] = bytes([x])

#         with open(input_path, "r", encoding="utf-8") as f:
#             texts = f.read()

#         raw_tokens = self.get_raw_tokens(texts)
#         char_tokens = self.get_list_of_characters(raw_tokens)
#         tokens_counter = Counter(tuple(word) for word in char_tokens)

#         index_tracking = 0
#         while len(self.vocab) < vocab_size:
#             counts = self.count_pair_frequencies(tokens_counter)
#             if not counts:
#                 break
#             match1, match2 = self.find_max_pair(counts)
#             new_token = (match1 + match2).encode("utf-8")
#             self.vocab[256 + offset + index_tracking] = new_token
#             self.merges.append((match1.encode("utf-8"), match2.encode("utf-8")))
#             tokens_counter = self.merge_tokens(tokens_counter, match1, match2)
#             index_tracking += 1
        
#         return self.vocab, self.merges

#     def get_params(self) -> BPETokenizerParams:
#         return BPETokenizerParams(vocab=self.vocab, merges=self.merges)
    


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


