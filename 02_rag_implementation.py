from utils import vector_db, document_loader, encoder
import sys

COLLECTION_NAME = "demo"
ID_FIELD_NAME = "id"
VECTOR_FIELD_NAME = "vector"

# here, DIM is a hyperparameter
DIM = 200
DIM2 = 400

if __name__ == "__main__":
    vector_db.create_connection()

    if vector_db.has_collection(COLLECTION_NAME):
        vector_db.drop_collection(COLLECTION_NAME)
    
    collection = vector_db.create_collection(COLLECTION_NAME, ID_FIELD_NAME, VECTOR_FIELD_NAME, DIM)
    
    documents = document_loader.load_directory("test_documents")
    
    enc = encoder.enc
    encoded_data_to_insert = []

    for document in documents:
        text = document.text
        for i in range(0, len(text), DIM2):
            partition = text[i:i+DIM2]
            encoded_partition = enc.encode(partition)
            encoded_partition_padded = encoder.pad_or_trim_encoded_vectors(encoded_partition, DIM)
            encoded_data_to_insert.append(encoded_partition_padded)
    
    print("length of data:", len(encoded_data_to_insert))
    for encoded_data in encoded_data_to_insert:
        if len(encoded_data) != DIM:
            print("CHECK DATA")
    
    data_to_insert = [
        [i for i in range(len(encoded_data_to_insert))],
        [x for x in encoded_data_to_insert]
    ]

    insert_status = vector_db.insert(collection, data_to_insert)
    print("Insert status: ", insert_status)

    vector_db.create_index(collection, VECTOR_FIELD_NAME)
    vector_db.load_collection(collection)

    search_queries = ["Did the captain's ship sail?", "What is the economy in 2024"]
    
    for search_query in search_queries:
        encoded_search_query = enc.encode(search_query)
        encoded_search_query_padded = encoder.pad_or_trim_encoded_vectors(encoded_search_query, DIM)
        results = vector_db.search(collection, VECTOR_FIELD_NAME, [encoded_search_query_padded])
        print(results)

        