import streamlit as st
from src.rag_pipeline import ask_refmind
st.title("PITCHLAW 👑 ⚽️")
question=st.text_input(
    "Ask a FIFA Rule Question"
)
if question:
    answer=ask_refmind(question)
    st.write(answer)