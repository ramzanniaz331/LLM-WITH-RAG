import os
from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

# Fetch and validate Qdrant config from environment
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_collection = os.getenv("QDRANT_COLLECTION")

logger.debug(f"QDRANT_HOST: {qdrant_host}")
logger.debug(f"QDRANT_PORT: {qdrant_port}")
logger.debug(f"QDRANT_COLLECTION: {qdrant_collection}")

try:
    qdrant_port = int(qdrant_port)
except (TypeError, ValueError):
    raise ValueError("QDRANT_PORT must be a valid integer.")

if not qdrant_host or not qdrant_collection:
    raise ValueError("QDRANT_HOST and QDRANT_COLLECTION must be set in the .env file.")

# Initialize Qdrant client and vector store
client = QdrantClient(host=qdrant_host, port=qdrant_port)
vector_store = QdrantVectorStore(client=client, collection_name=qdrant_collection)


def create_or_update_index(documents):
    """
    Create or update a vector index from given documents using Qdrant.
    """
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index
