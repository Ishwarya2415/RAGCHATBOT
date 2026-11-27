from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_processor import process_pdf
from rag import create_vector_store, ask_question

app = FastAPI(title="AI Document Question Answering API")

# Allow Streamlit to access FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store vector database
vector_store = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Document Question Answering API is Running"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    try:
        chunks = await process_pdf(file)

        if len(chunks) == 0:
            raise HTTPException(
                status_code=400,
                detail="No text found in PDF."
            )

        vector_store = create_vector_store(chunks)

        return {
            "message": "PDF processed successfully.",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/ask")
async def ask(req: QuestionRequest):

    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    answer, docs = ask_question(
        vector_store,
        req.question
    )

    sources = []

    for doc in docs:

        sources.append({
            "page": doc.metadata.get("page", "Unknown"),
            "source": doc.metadata.get("source", ""),
            "text": doc.page_content
        })

    return {
        "answer": answer,
        "sources": sources
    }