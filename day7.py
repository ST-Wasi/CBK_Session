import streamlit as st
import requests, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ddgs import DDGS

CHUNK_SIZE= 500
CHUNK_OVERLAP = 80
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:1b" # Qwen/Qwen2.5-1.5B-Instruct

# Models load ONCE, stay cached for the whole session
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

embedder, reranker = load_models()

def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file_path.lower().endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type — use .pdf, .docx, or .txt")
