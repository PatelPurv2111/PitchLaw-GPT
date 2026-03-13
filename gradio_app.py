import gradio as gr

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vector_db = FAISS.load_local(
    "vector_store/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Small LLM
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)


def ask_pitchlaw(question):

    docs = vector_db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer using FIFA rules.

Context:
{context}

Question:
{question}
"""

    result = generator(prompt, max_new_tokens=200)

    return result[0]["generated_text"]


# UI
interface = gr.Interface(
    fn=ask_pitchlaw,
    inputs="text",
    outputs="text",
    title="⚽ PitchLaw AI",
    description="AI assistant for FIFA rules"
)

interface.launch()