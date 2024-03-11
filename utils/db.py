from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

HOST = '127.0.0.1'
PORT = '19530'

# https://github.com/milvus-io/pymilvus/issues/911
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=HOST, port=PORT)
    print(f"\nList connections:")
    print(connections.list_connections())

def create_collection(name, id_field, vector_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name=name, data=None, schema=schema)
    print("\ncollection created:", name)
    return collection

