import os
import uuid
import logging as log
from PyPDF2 import PdfReader
import docx

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ---------------- LOGGING ----------------
log.basicConfig(
    level=log.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = log.getLogger(__name__)

# ---------------- CONFIG ----------------
BATCH_SIZE = 32
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------- FILE READERS ----------------
def read_file(path: str) -> str:
    ext = path.lower()
    try:
        if ext.endswith(".txt") or ext.endswith(".md"):
            return open(path, encoding="utf-8", errors="ignore").read()

        if ext.endswith(".pdf"):
            reader = PdfReader(path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)

        if ext.endswith(".docx"):
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")

    return ""

def chunk_text(text: str, size=500):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])

# ---------------- INDEXING ----------------
def start_local_index(data_dir, collection, embed_dim, recreate=False):
    qdrant = QdrantClient(host="localhost", port=6333)

    if recreate:
        if qdrant.collection_exists(collection):
            qdrant.delete_collection(collection)

        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=embed_dim,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Collection '{collection}' created")

    model = SentenceTransformer(MODEL_NAME)

    texts = []
    payloads = []
    total_chunks = 0

    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            text = read_file(path)
            if not text.strip():
                continue

            for chunk in chunk_text(text):
                texts.append(chunk)
                payloads.append({"path": path, "text": chunk})
                total_chunks += 1

                # batch embed + upsert
                if len(texts) >= BATCH_SIZE:
                    _embed_and_upsert(qdrant, model, collection, texts, payloads)
                    texts, payloads = [], []

    # flush remainder
    if texts:
        _embed_and_upsert(qdrant, model, collection, texts, payloads)

    logger.info(f"Indexed {total_chunks} chunks")

def _embed_and_upsert(qdrant, model, collection, texts, payloads):
    vectors = model.encode(texts, show_progress_bar=False)
    points = [
        {
            "id": str(uuid.uuid4()),
            "vector": vec.tolist(),
            "payload": payloads[i]
        }
        for i, vec in enumerate(vectors)
    ]
    qdrant.upsert(collection_name=collection, points=points)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    start_local_index(
        data_dir="./data",
        collection="local_docs",
        embed_dim=384,
        recreate=True   # set False for incremental indexing
    )
