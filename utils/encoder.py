import tiktoken

cl100k_base = tiktoken.get_encoding("cl100k_base")
    # In production, load the arguments directly instead of accessing private attributes
    # See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)

def pad_or_trim_encoded_vectors(vector, dim):
    """
    Pads or trims a given vector to match the specified dimension.

    If the length of the vector is greater than or equal to the specified dimension,
    it trims the vector to the specified dimension. If the length is less than the
    specified dimension, it pads the vector with a predefined value (100265) until
    it reaches the specified dimension.

    Args:
        vector (list): The input vector to be padded or trimmed.
        dim (int): The desired dimension of the resulting vector.

    Returns:
        list: The resulting vector after padding or trimming.

    Example:
        >>> pad_or_trim_encoded_vectors([1, 2, 3], 5)
        [1, 2, 3, 100265, 100265]
        >>> pad_or_trim_encoded_vectors([1, 2, 3, 4, 5], 3)
        [1, 2, 3]
    """
    if len(vector) >= dim:
        result = vector[:dim]
    else:
        result = vector
        result.extend([100265] * (dim - len(vector)))
    return result


def trim_special_tokens(text, special_tokens=["<|im_start|>", "<|im_end|>"]):
    """
    Remove special tokens from the given text.

    Args:
        text (str): The input text.
        special_tokens (list): List of special tokens to remove.

    Returns:
        str: The text with special tokens removed.
    """
    for token in special_tokens:
        text = text.replace(token, '')
    return text