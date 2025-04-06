## Refer to the sample notebook for the implementation details

from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

client = MilvusClient(
    # uri=os.environ.get("MILVUS_SERVER"),
    # token=os.environ.get("MILVUS_TOKEN")
    # uri="http://0.0.0.0:2379"
    # "./milvus_demo.db"
)

from pymilvus import CollectionSchema, FieldSchema, DataType

def init_milvus_collection(collection_name: str):
    if not client.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=36),
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="chunk_file", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="chunk_lines", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_code", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]

        schema = CollectionSchema(fields=fields, description="Code chunk embeddings with metadata")

        client.create_collection(collection_name=collection_name, schema=schema)

        # Step 1: Prepare the index params
        index_params = client.prepare_index_params(
            field_name="embedding",  # replace with your actual vector field
            index_type="IVF_FLAT",   # or HNSW, IVF_PQ, etc.
            metric_type="COSINE",    # or "L2", "IP"
            params={"nlist": 128}
        )

        # Step 2: Create the index
        client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )


    return collection_name

def query_milvus_by_embedding(embedding, collection_name, top_k=5, output_fields=["summary", "chunk_code", "chunk_file"], repo_name=None):
    """
    Query a Milvus collection with an embedding and return top_k matching results.

    Args:
        embedding: List[float], the embedding vector to search with
        collection_name: str, name of the Milvus collection
        top_k: int, number of top results to return
        output_fields: List[str], fields to include in the results (e.g., ["chunk", "repo"])
        repo_name: Optional[str], if provided, filters results by exact repo_name match

    Returns:
        List[dict]: Top matching results
    """
    embedding = [float(x) for x in embedding]
    client.load_collection(collection_name)

    # Optional filter
    filter_expr = f'repo_name == "{repo_name}"' if repo_name else ""

    search_results = client.search(
        collection_name=collection_name,
        data=[embedding],
        vector_field="embedding",
        limit=top_k,
        output_fields=output_fields,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        filter=filter_expr
    )

    results = [hit for hit in search_results[0]]
    return results



def get_client():
    return client
