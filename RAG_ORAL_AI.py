import io
import re
import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="College Knowledge Assistant", page_icon="🎓", layout="wide")
st.title("🎓 College Knowledge Assistant")
st.write("Upload at least five official documents, then ask questions grounded in their content.")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 25)
    top_k = st.slider("Passages to retrieve", 1, 10, 5)
    distance_limit = st.slider(
        "Maximum retrieval distance", 0.1, 2.0, 1.0, 0.1,
        help="Tune this using answerable and unanswerable test questions. Lower is stricter."
    )

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def read_uploads(files):
    docs, errors = [], []
    for file in files:
        try:
            if file.name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(file.getvalue()))
                for page_no, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": file.name, "page": page_no}
                        ))
            elif file.name.lower().endswith(".txt"):
                text = file.getvalue().decode("utf-8-sig")
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file.name, "page": "N/A"}
                    ))
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")
    return docs, errors


def normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_conflicts(docs):
    """Flag chunks that contain opposing numeric percentages or dates for the same topic.
    This is a conservative heuristic, not complete contradiction detection."""
    conflicts = []
    patterns = [
        (r"(.{0,70}(?:attendance|minimum attendance).{0,70}?)(\d{1,3}\s*%)", "percentage"),
        (r"(.{0,70}(?:deadline|last date|due date).{0,70}?)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "date"),
    ]
    for pattern, kind in patterns:
        found = []
        for doc in docs:
            for match in re.finditer(pattern, doc.page_content, flags=re.I):
                topic = normalise(match.group(1))
                value = normalise(match.group(2))
                found.append((topic, value, doc.metadata.get("source", "Unknown"),
                              doc.metadata.get("page", "N/A"), match.group(0)))
        for i, first in enumerate(found):
            for second in found[i + 1:]:
                if first[2] != second[2] and first[1] != second[1]:
                    # Only flag if topic words overlap; avoid comparing unrelated facts.
                    words1 = set(re.findall(r"[a-z]{4,}", first[0]))
                    words2 = set(re.findall(r"[a-z]{4,}", second[0]))
                    if len(words1 & words2) >= 2:
                        pair = (first[2], first[3], second[2], second[3], first[1], second[1])
                        if pair not in conflicts:
                            conflicts.append(pair)
    return conflicts


st.subheader("1. Upload official documents")
uploads = st.file_uploader(
    "Select at least five PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)
