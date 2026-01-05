import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="PDF RAG Chatbot")

st.title("Chat with your PDF")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = st.text_input("Ask a question from the PDF")

if query:
    docs = db.similarity_search(query, k=3)
    for doc in docs:
        st.write(doc.page_content)
