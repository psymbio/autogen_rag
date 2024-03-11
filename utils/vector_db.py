import tiktoken
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

HOST = '127.0.0.1'
PORT = '19530'

METRIC_TYPE = 'L2'
INDEX_TYPE = 'IVF_FLAT'
NLIST = 1024
NPROBE = 16
TOPK = 3

# https://github.com/milvus-io/pymilvus/issues/911
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=HOST, port=PORT)
    print(f"\nList connections:")
    print(connections.list_connections())

def create_collection(name, id_field, vector_field, max_dim):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=max_dim,
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name=name, data=None, schema=schema)
    print("\ncollection created:", name)
    return collection

def has_collection(name):
    return utility.has_collection(name)

def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))

def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())

def insert(collection, data):
    collection.insert(data)
    return True

def create_index(collection, filed_name):
    index_param = {
        "index_type": INDEX_TYPE,
        "params": {"nlist": NLIST},
        "metric_type": METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    print("\nCreated index:\n{}".format(collection.index().params))

def search(collection, vector_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": METRIC_TYPE, "params": {"nprobe": NPROBE}},
        "limit": TOPK,
        "expr": "id > 0"}
    results = collection.search(**search_param)
    # for i, result in enumerate(results):
    #     print("\nSearch result for {}th vector: ".format(i))
    #     for j, res in enumerate(result):
    #         print("Top {}: {}".format(j, res))
    return results

def get_entity_num(collection):
    print("\nThe number of entity:")
    print(collection.num_entities)

def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")

def load_collection(collection):
    collection.load()

def release_collection(collection):
    collection.release()