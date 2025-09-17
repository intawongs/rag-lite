import os, io, pickle, requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ===== SETTINGS =====
INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")
EMB_MODEL = "all-MiniLM-L6-v2"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "nvidia/nemotron-nano-9b-v2:free"

# ===== UTILS =====
def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)

def read_pdf(file_like: io.BytesIO) -> str:
    reader = PdfReader(file_like)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def read_docx(file_like: io.BytesIO) -> str:
    doc = DocxDocument(file_like)
    return "\n".join([p.text for p in doc.paragraphs])

def read_text(file_like: io.BytesIO) -> str:
    return file_like.read().decode("utf-8", errors="ignore")

def chunk_text(text, chunk_size=800, overlap=100):
    text = text.replace("\r", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n: break
        start = end - overlap
        if start < 0: start = 0
    return [c.strip() for c in chunks if c.strip()]

# ===== VECTOR INDEX =====
class RAGIndex:
    def __init__(self, emb_model=EMB_MODEL):
        self.embedder = SentenceTransformer(emb_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.texts = []
        self.sources = []

    def _l2_normalize(self, vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def _build_index(self):
        self.index = faiss.IndexFlatIP(self.dim)

    def add(self, texts, sources):
        if self.index is None:
            self._build_index()
        emb = self.embedder.encode(texts, show_progress_bar=True, batch_size=64)
        emb = np.array(emb).astype("float32")
        emb = self._l2_normalize(emb)
        self.index.add(emb)
        self.texts.extend(texts)
        self.sources.extend(sources)

    def save(self):
        faiss.write_index(self.index, FAISS_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump({"texts": self.texts, "sources": self.sources, "dim": self.dim}, f)

    def load(self):
        if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(FAISS_PATH)
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            self.texts = meta["texts"]
            self.sources = meta["sources"]
            self.dim = meta["dim"]
            return True
        return False

    def search(self, query, k=5):
        if self.index is None or self.index.ntotal == 0: return []
        q = self.embedder.encode([query]).astype("float32")
        q = self._l2_normalize(q)
        scores, idxs = self.index.search(q, k)
        return [{"score": float(s), "source": self.sources[i], "text": self.texts[i]}
                for s, i in zip(scores[0], idxs[0]) if i != -1]

# ===== LLM (OpenRouter) =====
def generate_answer(prompt):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "กรุณาตั้งค่า OPENROUTER_API_KEY ใน .env"
    client = OpenAI(base_url=OPENROUTER_URL, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "ใช้เฉพาะข้อมูลใน Context ตอบเป็นภาษาไทยอย่างกระชับ"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenRouter error: {e}"

def build_prompt(query, contexts):
    ctx = ""
    for i, c in enumerate(contexts, 1):
        safe_text = c['text'].replace('\n', ' ')
        ctx += f"[{i}] ({c['source']}) {safe_text}\n"
    return f"คำถาม: {query}\n\nContext:\n{ctx}\n\nหากไม่มีข้อมูลตรง ให้ตอบว่า 'ไม่พบข้อมูลในคลังเอกสาร'"

# ===== STREAMLIT APP =====
def main():
    ensure_dirs()
    st.set_page_config(page_title="NeuraRAG CoPilot", layout="wide")
    st.title("🧠 NeuraRAG CoPilot")

    chunk_size = st.sidebar.number_input("Chunk size", 200, 2000, 800, step=50)
    overlap = st.sidebar.number_input("Overlap", 0, 400, 100, step=10)
    top_k = st.sidebar.slider("Top-K", 1, 10, 5)
    score_thresh = st.sidebar.slider("Score Threshold", 0.0, 1.0, 0.2, 0.05)

    # ==== Upload & Preview ====
    uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์ (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif uploaded_file.name.lower().endswith(".docx"):
            text = read_docx(uploaded_file)
        else:
            text = read_text(uploaded_file)

        if st.button("🔍 Preview Chunk"):
            chunks = chunk_text(text, chunk_size, overlap)
            st.session_state["chunks_preview"] = chunks
            st.session_state["chunk_index"] = 0

    if "chunks_preview" in st.session_state:
        chunks = st.session_state["chunks_preview"]
        idx = st.session_state.get("chunk_index", 0)
        st.markdown(f"**Chunk {idx+1}/{len(chunks)}**")
        st.text(chunks[idx])

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⬅️ ก่อนหน้า") and idx > 0:
                st.session_state["chunk_index"] -= 1
        with c2:
            if st.button("ถัดไป ➡️") and idx < len(chunks)-1:
                st.session_state["chunk_index"] += 1
        with c3:
            if st.button("✅ ยืนยันและสร้าง Index"):
                rag = RAGIndex()
                loaded = rag.load()
                if not loaded: rag._build_index()
                rag.add(chunks, [uploaded_file.name]*len(chunks))
                rag.save()
                st.success(f"สร้าง Index สำเร็จ: {len(chunks)} chunks")

    if st.button("🧹 ล้าง Index"):
        if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
        if os.path.exists(META_PATH): os.remove(META_PATH)
        st.success("ล้าง Index เรียบร้อย")

    # ==== Ask Question ====
    query = st.text_input("พิมพ์คำถามเพื่อถาม LLM")
    if st.button("🔎 ถาม LLM"):
        rag = RAGIndex()
        if not rag.load():
            st.error("ยังไม่มี Index")
            st.stop()
        hits = rag.search(query, k=top_k)
        hits = [h for h in hits if h['score'] >= score_thresh]
        if not hits:
            st.warning("ไม่พบข้อมูลที่เกี่ยวข้อง")
            st.stop()
        with st.expander("Context"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[{i}] {h['source']}** · score={h['score']:.3f}")
                st.write(h['text'][:800] + ("..." if len(h['text'])>800 else ""))

        prompt = build_prompt(query, hits)
        with st.spinner("กำลังให้ LLM สรุปคำตอบ..."):
            answer = generate_answer(prompt)
        st.markdown("### ✅ คำตอบ")
        st.write(answer)

if __name__ == "__main__":
    main()
