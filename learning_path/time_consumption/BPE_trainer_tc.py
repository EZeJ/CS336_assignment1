import cProfile
import pstats
import io
import os
import sys

# Setup: Add the tests directory to sys.path
sys.path.insert(0, os.path.abspath('/Users/ethanj/Documents/CODE/Stanford_CS336/assignment1-basics-main/tests/'))

from adapters import run_train_bpe
from common import FIXTURES_PATH, gpt2_bytes_to_unicode

def test_train_bpe_special_tokens(snapshot=None):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    if snapshot:
        snapshot.assert_match(
            {
                "vocab_keys": set(vocab.keys()),
                "vocab_values": set(vocab.values()),
                "merges": merges,
            },
        )

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    # Run the test without snapshot validation
    test_train_bpe_special_tokens(snapshot=None)

    pr.disable()
    s = io.StringIO()
    sort_by = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
    ps.print_stats(30)  # Print top 30 lines
    print(s.getvalue())