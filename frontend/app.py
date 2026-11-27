import streamlit as st
import requests

# FastAPI URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Document Question Answering",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Document Question Answering using RAG")

st.write("Upload a PDF and ask questions about its contents.")

# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Upload PDF"):

        with st.spinner("Uploading and processing PDF..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            response = requests.post(
                f"{API_URL}/upload",
                files=files
            )

            if response.status_code == 200:

                data = response.json()

                st.success(data["message"])

                st.info(
                    f"Total Chunks Created : {data['chunks']}"
                )

            else:

                st.error(response.json()["detail"])

# -------------------------------
# Ask Question
# -------------------------------

st.divider()

question = st.text_input(
    "Ask a question about the uploaded PDF"
)

if st.button("Ask"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching document..."):

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question
                }
            )

            if response.status_code == 200:

                data = response.json()

                st.subheader("Answer")

                st.success(data["answer"])

                st.subheader("Sources")

                for index, source in enumerate(
                    data["sources"],
                    start=1
                ):

                    with st.expander(
                        f"Source {index} | Page {source['page']}"
                    ):

                        st.write(source["text"])

            else:

                st.error(response.json()["detail"])