import streamlit as st
import requests, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ddgs import DDGS
 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:1b"
 
# ---------- Models load ONCE, stay cached for the whole session ----------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker
 
embedder, reranker = load_models()
 
# ---------- Same functions as your notebook ----------
def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.lower().endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Unsupported file type")
 
def load_document(file_path):
    text = extract_text(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_text(text)
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype="float32"))
    return index, chunks
 
def retrieve(query, index, chunks, top_k=3, fetch_k=10):
    q_emb = embedder.encode([query])
    _, idx = index.search(np.array(q_emb), min(fetch_k, len(chunks)))
    candidates = [chunks[i] for i in idx[0]]
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
 
def safe_invoke(prompt_text, max_new_tokens=300):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        }, timeout=60)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"[Ollama error: {e}]"
 
def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web results found."
        return "\n\n".join(f"[{r['title']}]({r['href']})\n{r.get('body','')}" for r in results)
    except Exception as e:
        return f"Web search failed: {e}"
 
def rag_answer(question, index, chunks):
    retrieved = retrieve(question, index, chunks)
    context = "\n\n---\n\n".join(retrieved)
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
        web_results = web_search(question)
        web_prompt = (
            f"Answer using the web result below. Mention it came from the web, not the document.\n\n"
            f"Web Result:\n{web_results}\n\nQuestion: {question}"
        )
        answer = safe_invoke(web_prompt)
 
    return answer
 
# ---------- UI — only what we used today, nothing else ----------
st.title("RAG Assistant")
 
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.index, st.session_state.chunks = load_document(uploaded_file.name)
    st.write(f"'{uploaded_file.name}' loaded.")
 
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
 
if question := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
 
    if st.session_state.index is None:
        answer = "Please upload a document first."
    else:
        answer = rag_answer(question, st.session_state.index, st.session_state.chunks)
 
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)