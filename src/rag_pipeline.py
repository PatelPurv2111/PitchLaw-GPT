def ask_refmind(question):

    docs = vector_db.similarity_search(question, k=3)

    # Combine retrieved text
    context = "\n".join([doc.page_content for doc in docs])

    # Clean text
    context = context.replace("\n", " ")
    context = context[:1500]

    prompt = f"""
You are a professional football referee assistant.

Use FIFA rules to answer clearly and shortly.
Rules:
1. Do not copy the context text directly.
2. Explain in simple sentences.
3. Provide a short answer.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt, max_new_tokens=150)

    return result[0]["generated_text"]