import os
import tempfile

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def process_pdf(file: UploadFile):
    """
    Extract text from uploaded PDF and split into chunks.
    """

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        return chunks

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)