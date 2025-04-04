## Refer to the sample notebook for the implementation details

from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

client = MilvusClient(
    uri=os.environ.get("MILVUS_SERVER"),
    token=os.environ.get("MILVUS_TOKEN")
)

def init_milvus_collection(collection_name: str):
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            fields=[
                {"name": "id", "dtype": DataType.VARCHAR, "is_primary": True, "max_length": 36},
                {"name": "repo_name", "dtype": DataType.VARCHAR, "max_length": 200},
                {"name": "language", "dtype": DataType.VARCHAR, "max_length": 20},
                {"name": "chunk_file", "dtype": DataType.VARCHAR, "max_length": 200},
                {"name": "chunk_lines", "dtype": DataType.VARCHAR, "max_length": 50},
                {"name": "summary", "dtype": DataType.VARCHAR, "max_length": 2000},
                {"name": "embedding", "dtype": DataType.FLOAT_VECTOR, "dim": 1024}
            ]
        )
    return collection_name



def get_client():
    return client
