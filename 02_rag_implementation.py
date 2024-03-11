from utils import vector_db, document_loader, encoder
import tiktoken

COLLECTION_NAME = "demo"
ID_FIELD_NAME = "id_field"
VECTOR_FIELD_NAME = "float_vector_field"

# here, DIM is a hyperparameter
DIM = 200

if __name__ == "__main__":
    vector_db.create_connection()

    if vector_db.has_collection(COLLECTION_NAME):
        vector_db.drop_collection(COLLECTION_NAME)
    
    collection = vector_db.create_collection(COLLECTION_NAME, ID_FIELD_NAME, VECTOR_FIELD_NAME, DIM)
    
    documents = document_loader.load_directory("test_documents")
    
    enc = encoder.enc

    for document in documents:
        text = document.text
        for i in range(0, len(text), DIM):
            partition = text[i:i+DIM]
            encoded_partition = enc.encode(partition)

            if len(encoded_partition) >= DIM:
                encoded_partition_padded = encoded_partition[:DIM]
            else:
                encoded_partition_padded = encoded_partition
                encoded_partition_padded.extend([100265] * (DIM - len(encoded_partition)))
            
            print(enc.decode(encoded_partition))
            print(enc.decode(encoded_partition_padded))

        