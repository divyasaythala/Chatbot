import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

#load your pdf files
loader = PyPDFDirectoryLoader("documents")
docs = loader.load()

 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)

 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

 
vectorstore = FAISS.from_documents(chunks, embeddings)

 
vectorstore.save_local("faiss_index")

print("FAISS index created and saved successfully")

