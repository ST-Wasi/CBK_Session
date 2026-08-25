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

def load_document(file_path):
    global current_index, current_chunks, current_file_name

    print("1. Extracting text...")
    text = extract_text(file_path)

    print("2. Extracted text length:", len(text))

    if not text.strip():
        raise ValueError(
            "PDF se koi text extract nahi hua. "
            "PDF scanned/image-based ho sakti hai."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    new_chunks = splitter.split_text(text)

    print("3. Number of chunks:", len(new_chunks))

    if not new_chunks:
        raise ValueError("Document se chunks generate nahi hue.")

    print("4. Generating embeddings...")

    new_embeddings = embedder.encode(
        new_chunks,
        show_progress_bar=True
    )

    print("5. Embedding type:", type(new_embeddings))
    print("6. Embedding shape:", getattr(new_embeddings, "shape", None))

    if len(new_embeddings) == 0:
        raise ValueError("Embeddings generate nahi hue.")

    if len(new_embeddings.shape) != 2:
        raise ValueError(
            f"Unexpected embedding shape: {new_embeddings.shape}"
        )

    new_index = faiss.IndexFlatL2(new_embeddings.shape[1])

    new_index.add(np.asarray(new_embeddings, dtype="float32"))

    current_index = new_index
    current_chunks = new_chunks
    current_file_name = file_path

    print(
        f"\n'{file_path}' loaded — "
        f"{len(new_chunks)} chunks indexed and ready."
    )

def retrieve(query, top_k=3, fetch_k=10):
    if current_index is None:
        raise RuntimeError("No document loaded yet — run Step 3 first.")
    q_emb = embedder.encode([query])
    _, idx = current_index.search(np.array(q_emb), min(fetch_k, len(current_chunks)))
    candidates = [current_chunks[i] for i in idx[0]]
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]

def safe_invoke(prompt_text, max_new_tokens=300):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def rag_answer(question, top_k=3, verbose=True):
    if current_index is None:
        print("⚠️ No document loaded yet — run Step 3 to upload a file first.")
        return None
    retrieved_chunks = retrieve(question, top_k=top_k)
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = (
        "You are a training content assistant. Answer using ONLY the context below.\n"
        "If the answer is not in the context, say so explicitly. "
        "ALWAYS return the output in JSON format with keys 'answer' and 'found_in_context' (true/false).\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    answer = safe_invoke(prompt)

    found = True
    try:
      parsed = json.loads(answer[answer.find("{"):answer.rfind("}")+1])
      found = parsed.get("found_in_context", True)
    except Exception:
      pass


    if not found:
      if verbose:
        print(" Not Found in Document - Searching back to web search...")
      web_results = web_search(question)
      web_prompt = (
          f"Answer this question using web search result below, "
          f"Mention that this result came from the web search, not from the uploaded document"
          f"Web Result: \n{web_results}\n\nQuestion: {question}"
      )
      answer = safe_invoke(web_prompt)
      source = "Web"
    else:
      source = "document"


    if verbose:
        print(f"[Document: {current_file_name}]")
        print(f"Q: {question}")

        print(f"A: {answer}\n")
        print(f"Source used: {source}")
    return answer

print("Functions ready.")
