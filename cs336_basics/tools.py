import regex as re
from collections import defaultdict

def process_text_with_pre_tokenize(text, PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
    '''
    Pre-tokenize the text using regex to match tokens.
    This function uses a regex pattern to find tokens in the text.
    It returns a dictionary with tuple of characters as keys and their counts as values.
    '''
    token_counts = defaultdict(int)
    for match in re.finditer(PAT, text):
        token = match.group()
        char_tuple = tuple(token)  # Convert string token to tuple of characters
        token_counts[char_tuple] += 1
    return dict(token_counts)


def convert_dict_to_list(tokens_counts):
    '''
    Convert the dictionary of token counts to a list of tuples.
    Each tuple contains a token and its count.
    '''
    return list(tokens_counts.keys())


def break_ties_during_merge_by_lexicographically(list_of_tuples):
    '''
    Break ties during merge by lexicographically sorting the tuples.
    This function return the maximum tuple based on the first element.
    '''
    return max(list_of_tuples)


def convert_utf8_to_int(tokens_representation):
    '''
    Convert the tokens representation from UTF-8 to integers.
    This function checks the type of the first element in the list and
    converts the entire list accordingly.
    '''
    if isinstance(tokens_representation[0], str):
        print("Converting str to int")
        return [list(map(ord, string)) for string in tokens_representation]
    if isinstance(tokens_representation[0], bytes):
        print("Converting bytes to int")
        return [list(bytes_item) for bytes_item in tokens_representation]