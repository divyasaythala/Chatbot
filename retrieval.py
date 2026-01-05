from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "Explain the key concepts in the document"

results = vectorstore.similarity_search(query, k=5)

print("\nRESULTS:\n")
for res in results:
    print(res.page_content)
    print("=" * 80)
