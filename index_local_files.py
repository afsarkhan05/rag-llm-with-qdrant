import os
import uuid
import logging as log
from PyPDF2 import PdfReader
import docx
import whisper  # NEW
from PIL import Image # NEW

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# from qdrant_client.models import VectorParams, Distance, PointStruct

# ---------------- CONFIG ----------------
log.basicConfig(level=log.INFO, format="{asctime} - {levelname} - {message}", style="{")
logger = log.getLogger(__name__)

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"  # Dim: 384
CLIP_MODEL_NAME = "clip-ViT-B-32"      # Dim: 512

# Load models once
text_model = SentenceTransformer(TEXT_MODEL_NAME)
clip_model = SentenceTransformer(CLIP_MODEL_NAME)
whisper_model = whisper.load_model("base") # "base" is a good balance of speed/accuracy

# ---------------- MULTIMODAL READERS ----------------

def read_multimodal(path: str):
    """Processes any file type and returns (text_content, image_obj)"""
    ext = path.lower()
    try:
        # 1. Standard Text/Docs
        if ext.endswith((".txt", ".md")):
            return open(path, encoding="utf-8", errors="ignore").read(), None
        if ext.endswith(".pdf"):
            reader = PdfReader(path)
            return "\n".join(p.extract_text() or "" for p in reader.pages), None
        if ext.endswith(".docx"):
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs), None

        # 2. Audio & Video (Transcription)
        if ext.endswith((".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi")):
            logger.info(f"Transcribing media: {os.path.basename(path)}")
            result = whisper_model.transcribe(path)
            return result['text'], None

        # 3. Images (Visual Processing)
        if ext.endswith((".jpg", ".jpeg", ".png")):
            return None, Image.open(path)

    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
    return None, None

def chunk_text(text: str, size=500):
    if not text: return
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])

# ---------------- INDEXING ----------------

def create_multimodal_collection(collection_name, qdrant):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            # Named vector for standard text search (MiniLM)
            "text-vec": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
            # Named vector for image/multimodal search (CLIP)
            "clip-vec": models.VectorParams(
                size=512,
                distance=models.Distance.COSINE
            ),
        }
    )
    print(f"Collection '{collection_name}' created with Hybrid Multimodal config.")


def start_multimodal_index(data_dir, collection, recreate=False):
    qdrant = QdrantClient(host="localhost", port=6333)

    if recreate:
        if qdrant.collection_exists(collection):
            qdrant.delete_collection(collection)

        create_multimodal_collection(collection, qdrant)


    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            text_data, img_data = read_multimodal(path)
            points = []

            # Case A: We found text (from docs or transcription)
            if text_data:
                for chunk in chunk_text(text_data):
                    vector = text_model.encode(chunk).tolist()
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"text-vec": vector},
                        payload={"path": path, "text": chunk, "type": "text"}
                    ))

            # Case B: We found an image
            if img_data:
                vector = clip_model.encode(img_data).tolist()
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"clip-vec": vector},
                    payload={"path": path, "text": f"Image: {f}", "type": "image"}
                ))

            if points:
                qdrant.upsert(collection_name=collection, points=points)
                logger.info(f"Indexed {f}")

if __name__ == "__main__":
    start_multimodal_index("./data", "multimodal_rag", recreate=True)