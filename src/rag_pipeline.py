from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vector_db = FAISS.load_local(
    "vector_store/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# LLM
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1"
)

def ask_refmind(question):

    docs = vector_db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a football referee assistant.

Context:
{context}

Question:
{question}

Answer using official FIFA rules.
"""

    result = generator(prompt, max_new_tokens=200)

    return result[0]["generated_text"]