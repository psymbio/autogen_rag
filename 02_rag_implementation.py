from utils import vector_db

COLLECTION_NAME = "demo"
ID_FIELD_NAME = "id_field"
VECTOR_FIELD_NAME = "float_vector_field"
DIM = 200

if __name__ == "__main__":
    vector_db.create_connection()

    if vector_db.has_collection(COLLECTION_NAME):
        vector_db.drop_collection(COLLECTION_NAME)
    
    collection = vector_db.create_collection(COLLECTION_NAME, ID_FIELD_NAME, VECTOR_FIELD_NAME, DIM)