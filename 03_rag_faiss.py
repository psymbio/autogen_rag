from utils import document_loader, encoder
import numpy as np
import faiss

DIM = 50

def get_chunks(s, maxlength):
    start = 0
    end   = 0
    while start + maxlength  < len(s) and end != -1:
        end = s.rfind(" ", start, start + maxlength + 1)
        if end == -1: break
        yield s[start:end]
        start = end +1
    yield s[start:]
# s = "Well, Prince, so Genoa and Lucca are now just family estates of the Buonapartes. But I warn you, if you don’t tell me that this means war, if you still try to defend the infamies and horrors perpetrated by that Antichrist—I really believe he is Antichrist—I will have nothing more to do with you and you are no longer my friend, no longer my ‘faithful slave,’ as you call yourself! But how do you do? I see I have frightened you—sit down and tell me all the news."
# chunks = get_chunks(s, 25)
# chunks_list = [(n, len(n)) for n in chunks]
# print(chunks_list)

enc = encoder.enc
encoded_data_to_insert = []
documents = document_loader.load_directory("test_documents")
for document in documents:
    text = document.text
    chunks = get_chunks(text, 100)
    # chunks_list = [(n, len(n)) for n in chunks]
    # print(chunks_list)
    # print("\n\n\n")
    for chunk in chunks:
        encoded_partition = enc.encode(chunk.lower())
        encoded_partition_padded = encoder.pad_or_trim_encoded_vectors(encoded_partition, DIM)
        encoded_data_to_insert.append(encoded_partition_padded)
        print(len(encoded_partition), len(encoded_partition_padded))

    # https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772    
    vectors = np.array(encoded_data_to_insert).astype("float32")
    print(vectors.shape, vectors.dtype)

    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    # vectors /= 100265
    # faiss.normalize_L2(vectors)
    index.add(vectors)

    search_text = 'are toyota sales declining?'
    search_encoded = enc.encode(search_text.lower())
    search_padded = encoder.pad_or_trim_encoded_vectors(search_encoded, DIM)
    _vector = np.array([search_padded]).astype("float32")
    # _vector /= 100265
    # faiss.normalize_L2(_vector)

    k = index.ntotal
    distances, indices = index.search(_vector, k=k)

    # for dist, idx in zip(distances[0], indices[0]):
    #     print("Distance:", dist)
    #     print("Corresponding document:", encoder.trim_special_tokens(enc.decode(vectors[idx].astype("int"))))

    sorted_results = sorted(zip(distances[0], indices[0]))

    # Print the documents corresponding to the vectors with the least distances
    i = 0
    max_i = 2
    for dist, idx in sorted_results:
        print("Distance:", dist)
        print("Corresponding document:", encoder.trim_special_tokens(enc.decode(vectors[idx].astype("int"))))
        i += 1
        if i > max_i:
            break