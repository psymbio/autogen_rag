def get_chunks(s, maxlength):
    """
    Splits a string into chunks of maximum length while preserving whole words.

    Args:
        s (str): The input string to be split into chunks.
        maxlength (int): The maximum length of each chunk.

    Yields:
        str: Each chunk of the input string, ensuring that no chunk exceeds the maximum length
             and that the splitting is done at word boundaries.

    Notes:
        This function iteratively splits the input string into chunks of maximum length while
        ensuring that each chunk ends at a word boundary to avoid splitting words midway.
        If a chunk would exceed the maximum length before finding a word boundary, it will
        split the string at that point.

    References:
        https://stackoverflow.com/questions/57023348/python-splitting-a-long-text-into-chunks-of-strings-given-character-limit
        https://stackoverflow.com/questions/76633836/what-does-langchain-charactertextsplitters-chunk-size-param-even-do
    """
    start = 0
    end   = 0
    while start + maxlength  < len(s) and end != -1:
        end = s.rfind(" ", start, start + maxlength + 1)
        if end == -1: break
        yield s[start:end]
        start = end +1
    yield s[start:]