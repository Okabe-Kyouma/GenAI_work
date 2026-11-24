import os

import streamlit as st

from rag_utility import process_document_to_chromadb, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("Document Question Answering with RAG")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:

    save_path = os.path.join(working_dir, uploaded_file.name)
    with open (save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.spinner("Processing document Please wait...")

    process_document = process_document_to_chromadb(uploaded_file.name)

    st.info("Document processed and stored in ChromaDB.")


user_question = st.text_input("Ask a question about the document:")

if st.button("Get Answer"): 
    if not uploaded_file:
        st.warning("Please upload a PDF document first.")
    elif user_question:
        answer = answer_question(user_question)
        st.markdown("GROQ LLM Answer:")
        st.markdown(f"Answer: {answer}")
    else:
        st.warning("Please enter a question.")