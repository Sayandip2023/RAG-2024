import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the tokenizer and language model
model_name = "facebook/opt-1.3b"  # or any other language model
tokenizer = AutoTokenizer.from_pretrained(model_name)
lm_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load SciBERT model
scibert_model = SentenceTransformer('allenai/scibert_scivocab_uncased')

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to retrieve relevant chunks using FAISS
def retrieve(query, top_k=5):
    query_embedding = scibert_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [text_chunks[idx] for idx in indices[0]]
    return results

# Function to generate responses using the language model
def generate_response(query, top_k=5, max_new_tokens=50, max_length=512):
    retrieved_chunks = retrieve(query, top_k)
    context = " ".join(retrieved_chunks)
    
    # Truncate context if necessary
    input_text = query + " " + context
    input_ids = tokenizer.encode(input_text, truncation=True, max_length=max_length)
    
    # Generate response
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    outputs = lm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit interface
st.title("Scientific Paper Query System")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_text = extract_text_from_pdf("uploaded_file.pdf")
    
    # Split the extracted text into chunks
    text_chunks = split_text(pdf_text)
    
    # Generate embeddings
    embeddings = scibert_model.encode(text_chunks, show_progress_bar=True)
    
    # Initialize FAISS
    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    st.success("PDF processed and embeddings generated successfully!")

    query = st.text_input("Enter your query")
    
    if query:
        try:
            response = generate_response(query)
            st.subheader("Response")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
