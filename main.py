from fastapi import FastAPI
from loguru import logger
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from app.ingestion import upload_pdfs
from app.retrieval import query_index

load_dotenv()
logger.add(sys.stdout, level="INFO")
app = FastAPI()

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

app.post("/upload_pdfs")(upload_pdfs)
app.post("/query")(query_index)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
