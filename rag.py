from utils import document_loader, encoder, get_chunks
import numpy
from sentence_transformers import SentenceTransformer, util
import torch

# hyperparams
CHUNK_SIZE = 1000
TOP_K = 2

def query_folder(folder, query, logs=False):
    """
    Queries a folder of documents for the most relevant sentences to a given query.

    Args:
        folder (str): The path to the folder containing the documents.
        query (str): The query string for which relevant sentences are to be retrieved.
        logs (bool, optional): Flag indicating whether to print logging information. Default is False.

    Returns:
        List[str]: A list of the most relevant sentences from the documents in the folder.

    Notes:
        This function loads documents from the specified folder, splits them into chunks, and
        computes embeddings for each chunk and the query using the SentenceTransformer model.
        It then calculates cosine similarity scores between the query embedding and the
        embeddings of all chunks, retrieves the top-k most similar chunks, and returns the
        corresponding sentences.
    """
    bpe_enc = encoder.enc
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    documents = document_loader.load_directory(folder)

    encoded_partitions = []
    corpus = []
    for document in documents:
        text = document.text
        chunks = get_chunks.get_chunks(text, CHUNK_SIZE)
        for chunk in chunks:
            corpus.append(chunk)
            encoded_partition = bpe_enc.encode(chunk.lower())
            # should be lower than 512 - if not then change CHUNK_SIZE
            if logs:
                # https://github.com/run-llama/llama_index/issues/613 
                print(f"document: {document.metadata['file_name']}, encoded partition length: {len(encoded_partition)}")
            encoded_partitions.append(encoded_partition)
    
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    if logs:
        print(f"corpus_shape: {corpus_embeddings.shape}, query_shape: {query_embedding.shape}")

    top_k = min(TOP_K, len(corpus))
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    results = []
    if logs:
        print("\n\n======================\n\n")
        print(f"query: {query}")
        print(f"\nTop {TOP_K} most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: {:.4f})".format(score))
    
    for score, idx in zip(top_results[0], top_results[1]):
            results.append(corpus[idx])
    
    return results


if __name__ == "__main__":
    query_folder("test_documents", "are toyota sales doing well?", logs=True)