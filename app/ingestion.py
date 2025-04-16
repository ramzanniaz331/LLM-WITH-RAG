import os
from pathlib import Path
from typing import List

from fastapi import File, UploadFile
from loguru import logger
from llama_index.core import SimpleDirectoryReader

from app.utils import create_or_update_index

# Directory to save uploaded files
uploads_dir = Path(__file__).parent / "uploads"

logger.info("ingestion.py module loaded")


async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload PDF files and update the LlamaIndex with new documents.
    """
    try:
        logger.info(f"Received {len(files)} file(s) for upload.")

        for file in files:
            file_path = uploads_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            logger.info(f"File uploaded: {file.filename}")

        # Re-index uploaded documents
        new_documents = SimpleDirectoryReader(str(uploads_dir)).load_data()
        create_or_update_index(new_documents)

        logger.info("PDFs uploaded and index updated.")
        return {"message": "PDFs uploaded and index updated successfully."}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"error": str(e)}
