from llama_index.core import SimpleDirectoryReader

def load_directory(path):
    """
    Loads documents from a directory specified by the given path.

    Args:
        path (str): The path to the directory containing documents.

    Returns:
        List[Document]: A list of Document objects representing the loaded documents.
    """
    documents = SimpleDirectoryReader(path).load_data()
    return documents