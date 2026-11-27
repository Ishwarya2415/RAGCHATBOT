import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain.chains.retrieval import (
    create_retrieval_chain,
)

load_dotenv()

DB_PATH = "chroma_db"


def create_vector_store(chunks):
    """
    Create embeddings and store them in ChromaDB
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    return vector_store


def ask_question(vector_store, question):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant.

Answer ONLY using the context provided.

If the answer is not available in the context,
reply:

"I could not find this information in the uploaded document."

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    result = retrieval_chain.invoke(
        {
            "input": question
        }
    )

    answer = result["answer"]

    source_docs = result["context"]

    return answer, source_docs