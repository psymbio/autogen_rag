from pymilvus import (
    connections
)

HOST = '127.0.0.1'
PORT = '19530'

# https://github.com/milvus-io/pymilvus/issues/911
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=HOST, port=PORT)
    print(f"\nList connections:")
    print(connections.list_connections())

if __name__ == "__main__":
    create_connection()