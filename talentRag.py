import os, io, pickle
import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document
import faiss
from sentence_transformers import SentenceTransformer

# ===== Settings =====
INDEX_DIR = "index_resume"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")
EMB_NAME = "all-MiniLM-L6-v2"

def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)

def read_pdf(file_like):
    reader = PdfReader(file_like)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def read_docx(file_like):
    doc = Document(file_like)
    return "\n".join([p.text for p in doc.paragraphs])

def read_text(file_like):
    return file_like.read().decode("utf-8", errors="ignore")

def chunk_text(text, size=800, overlap=150):
    text = text.replace("\r","")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start+size, len(text))
        chunks.append(text[start:end])
        if end == len(text): break
        start = end - overlap
        if start < 0: start = 0
    return [c.strip() for c in chunks if c.strip()]

class ResumeIndex:
    def __init__(self, emb_model=EMB_NAME):
        self.embedder = SentenceTransformer(emb_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.texts, self.sources = [], []

    def _l2_normalize(self, v):
        norms = np.linalg.norm(v, axis=1, keepdims=True)+1e-12
        return v / norms

    def _build_index(self):
        self.index = faiss.IndexFlatIP(self.dim)

    def add_resumes(self, texts, sources):
        if self.index is None:
            self._build_index()
        emb = self.embedder.encode(texts, show_progress_bar=True, batch_size=64)
        emb = np.array(emb).astype("float32")
        emb = self._l2_normalize(emb)
        self.index.add(emb)
        self.texts.extend(texts)
        self.sources.extend(sources)

    def search(self, query, k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self.embedder.encode([query]).astype("float32")
        q = self._l2_normalize(q)
        scores, idxs = self.index.search(q, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1: continue
            out.append({"score": float(score), "source": self.sources[idx], "text": self.texts[idx]})
        return out

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

def main():
    ensure_dirs()
    st.title("🚀 TalentRAG – Resume Finder")
    st.write("อัปโหลด Resume หลายไฟล์แล้วค้นหาผู้สมัครตามทักษะ")

    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    top_k = st.sidebar.slider("Top-K", 1, 10, 5)
    score_threshold = st.sidebar.slider("Score Threshold", 0.0, 1.0, 0.35, 0.05)
    chunk_size = st.sidebar.number_input("Chunk size", 200, 2000, 800, step=50)
    overlap = st.sidebar.number_input("Overlap", 0, 400, 150, step=10)

    # Upload resumes
    st.subheader("📤 อัปโหลด Resume")
    uploaded = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("➕ เพิ่มเข้า Index"):
            if not uploaded:
                st.warning("กรุณาเลือกไฟล์")
            else:
                rag = ResumeIndex()
                loaded = rag.load()
                if loaded:
                    st.info(f"โหลด Index เดิมแล้ว: {rag.index.ntotal} ชิ้น")
                new_texts, new_sources = [], []
                for file in uploaded:
                    name = file.name
                    data = file.read(); file.seek(0)
                    try:
                        if name.lower().endswith(".pdf"):
                            text = read_pdf(io.BytesIO(data))
                        elif name.lower().endswith(".docx"):
                            text = read_docx(io.BytesIO(data))
                        else:
                            text = read_text(io.BytesIO(data))
                    except Exception as e:
                        st.error(f"อ่านไฟล์ {name} ไม่ได้: {e}")
                        continue
                    chunks = chunk_text(text, size=chunk_size, overlap=overlap)
                    new_texts.extend(chunks)
                    new_sources.extend([name]*len(chunks))
                if not new_texts:
                    st.warning("ไม่มีข้อมูลเพิ่ม")
                else:
                    with st.spinner("กำลังฝังเวกเตอร์..."):
                        if not loaded: rag._build_index()
                        rag.add_resumes(new_texts, new_sources)
                        rag.save()
                    st.success(f"เพิ่มแล้ว {len(new_texts)} ชิ้น | ทั้งหมด {rag.index.ntotal}")

    with col2:
        if st.button("🧹 ล้าง Index"):
            if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
            if os.path.exists(META_PATH): os.remove(META_PATH)
            st.success("ล้าง Index เรียบร้อย")

    st.markdown("---")
    st.subheader("🔎 ค้นหาผู้สมัครตามทักษะ")
    query = st.text_input("ระบุทักษะ เช่น 'Figma', 'Python และ Machine Learning'")
    if st.button("ค้นหา"):
        rag = ResumeIndex()
        if not rag.load():
            st.error("ยังไม่มี Index ข้อมูล")
            st.stop()
        with st.spinner("ค้นหา..."):
            hits = rag.search(query, k=top_k)
            filtered_hits = [h for h in hits if h["score"] >= score_threshold]

        if not filtered_hits:
            st.warning("ไม่พบผู้สมัครที่เกี่ยวข้อง")
        else:
            st.success(f"พบ {len(filtered_hits)} ผู้สมัครที่เกี่ยวข้อง:")
            for i, h in enumerate(filtered_hits,1):
                st.markdown(f"**[{i}] {h['source']}** · score={h['score']:.3f}")
                st.write(h["text"][:500] + ("..." if len(h["text"])>500 else ""))

if __name__ == "__main__":
    main()
