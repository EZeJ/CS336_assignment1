import regex
from collections import defaultdict, Counter
import cProfile
import pstats



def get_raw_tokens(text, PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
    # Use Unicode letters, then split on whitespace
    tokens = regex.findall(PAT, text)
    return " ".join(tokens)

def get_list_of_characters(pure_word_result):
    # Preserve word boundaries
    return [list(word) for word in pure_word_result.split()]

def count_pair_frequencies(tokens_counter):
    counts = defaultdict(int)
    for word, freq in tokens_counter.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            counts[pair] += freq
    return counts

def find_max_pair(counts):
    # Deterministic: break ties using lexicographic order
    return max(counts, key=lambda x: (counts[x], x))

def merge_tokens(tokens_counter, match1, match2):
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

def print_tokens(tokens_counter):
    print("Current tokens:")
    tokens_list = [list(word) for word in tokens_counter.keys()]
    print(" ".join(["' '".join(word) for word in tokens_list]))
    print()

def BPE_training_naive_version(text, num_merges=6, verbose=False):
    # Step 1: Pre-tokenization and splitting into characters
    new_vocab = []
    pure_word_result = get_raw_tokens(text)
    tokens_list = get_list_of_characters(pure_word_result)
    tokens_counter = Counter(tuple(word) for word in tokens_list)

    # Step 2: Perform merges
    for i in range(num_merges):
        counts = count_pair_frequencies(tokens_counter)
        if not counts:
            break
        match1, match2 = find_max_pair(counts)
        new_vocab.append(match1+match2)
        tokens_counter = merge_tokens(tokens_counter, match1, match2)
        if verbose:
            print(f"Counts: {counts}")
            print(f"Merge {i+1}: ({match1}, {match2})")
            print_tokens(tokens_counter)

    return tokens_counter, new_vocab

# Time function time consumption for optimization

def run_bpe(text, num_merges):
    return BPE_training_naive_version(text, num_merges=num_merges, verbose=False)



def time_consumption(func):
    file_path = "/Users/ethanj/Documents/CODE/Stanford_CS336/assignment1-basics-main/cs336_basics/text_examples/text_ex1.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    profiler = cProfile.Profile()
    profiler.enable()
    final_result, new_vocab = run_bpe(text, 10000)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumtime").print_stats(20)  # Top 20 by cumulative time
    return final_result, new_vocab




example_text_ag1 = """
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
"""


from dataclasses import dataclass

GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

from dataclasses import dataclass

GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_BPE_v1(texts: str, num_merges: int) -> BPETokenizerParams:
    # Initialize parameters
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    # Explanation of the difference between the two dictionary comprehensions:
    # ✅ This one is correct: vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    # It creates a dictionary mapping each integer from 0 to 255 (inclusive)
    # to its corresponding single-byte representation using bytes([x]).
    # bytes([x]) takes a list of integers and returns a bytes object of length 1.
    
    # ❌ This one is incorrect and will raise a TypeError:
    # vocab: dict[int, bytes] = {x: bytes(x) for x in range(256)}  # Incorrect
    # # bytes(x) tries to create a bytes object of length x filled with zero bytes,
    # which is not what we want here. For example, bytes(3) returns b'\x00\x00\x00',
    # not b'\x03'.
    # Also, bytes(x) will fail for x >= 256 due to range constraints.

    # Step 1: Pre-tokenization and splitting into characters
    raw_tokens = get_raw_tokens(texts, GPT2_TOKENIZER_REGEX)
    bytes_indices = get_list_of_characters(raw_tokens)
    tokens_counter = Counter(tuple(word) for word in bytes_indices)


    # Step 2: Perform merges
    for i in range(num_merges):
        counts = count_pair_frequencies(tokens_counter)
        if not counts:
            break
        match1, match2  = find_max_pair(counts)
        new_ID_for_new_token = 256 + i
        bytes_presentation_of_max_pair = (match1+match2).encode('utf8')
        merges.append((match1.encode('utf8'), match2.encode('utf8')))
        # print("merges", merges)
        vocab[new_ID_for_new_token] = bytes_presentation_of_max_pair
        tokens_counter = merge_tokens(tokens_counter, match1, match2)
        print("token_counter", tokens_counter)
        print("Merge", merges)
    
    return vocab, merges
