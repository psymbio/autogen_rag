from llama_index.core import SimpleDirectoryReader

def load_directory(path):
    documents = SimpleDirectoryReader(path).load_data()
    return documents