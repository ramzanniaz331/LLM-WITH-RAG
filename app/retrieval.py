import os
from dotenv import load_dotenv
from loguru import logger
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.utils import create_or_update_index

# Load environment variables
load_dotenv()

# Initialize embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logger.info("Embedding Model Initialized")

# Configure LLM with Gemini
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
Settings.llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# Load documents from the uploads directory
uploads_path = os.path.join(os.path.dirname(__file__), "uploads")
documents = SimpleDirectoryReader(uploads_path).load_data()

# Create or update the index
index = create_or_update_index(documents)


# Handle user queries
async def query_index(query: dict):
    try:
        question = query.get("question")
        logger.info(f"Received query: {question}")

        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(question)

        relevant_documents = [
            node.node.metadata["file_name"] for node in response.source_nodes
        ]

        return {
            "response_from_llm": response.response,
            "relevant_documents": relevant_documents,
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}
