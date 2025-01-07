import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceApi
import faiss

def load_pdf(file):
    reader = PdfReader(file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

def create_faiss_index(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    sentences = text.split(". ")  # Split text into sentences
    embeddings = model.encode(sentences)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, sentences, model

def retrieve_answer(query, index, sentences, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=1)
    return sentences[indices[0][0]]

def generate_response(context, query, mistral_model="mistralai/Mistral-7B-v0.1"):
    inference = InferenceApi(repo_id=mistral_model, token="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")
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = inference(prompt)
    return result

# Streamlit UI setup
st.title("PDF Q&A with FAISS and Huggingface LLM")

# File uploader
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file is not None:
    with st.spinner("Processing PDF..."):
        # Load the PDF
        pdf_text = load_pdf(pdf_file)

        # Create FAISS index
        index, sentences, model = create_faiss_index(pdf_text)
    st.success("PDF processed successfully! You can now ask questions.")

    # User question
    query = st.text_input("Ask a question about the uploaded PDF")

    if query:
        with st.spinner("Generating answer..."):
            # Retrieve relevant context and generate response
            context = retrieve_answer(query, index, sentences, model)
            answer = generate_response(context, query)
        st.write("**Answer:**", answer)

        # Optionally show the relevant context
        with st.expander("Show relevant context"):
            st.write(context)