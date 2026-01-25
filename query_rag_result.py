import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Initialize models
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = SentenceTransformer("clip-ViT-B-32")
qdrant = QdrantClient(host="localhost", port=6333)

def retrieve_hybrid(query, collection_name="multimodal_rag", top_k=5):
    # Encode query for both lanes
    text_emb = text_model.encode(query).tolist()
    clip_emb = clip_model.encode(query).tolist()

    # The Hybrid Query
    result = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            # Lane 1: Search the text index
            models.Prefetch(query=text_emb, using="text-vec", limit=top_k),
            # Lane 2: Search the CLIP index (can find images with text query!)
            models.Prefetch(query=clip_emb, using="clip-vec", limit=top_k),
        ],
        # Fuse results: points that rank high in BOTH lanes get a boost
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True
    )
    return result.points

def chat_with_phi(query, hits):
    # Combine text and image metadata for the LLM
    context = ""
    for i, h in enumerate(hits):
        p = h.payload
        # If it's an image, we use the description; if text, the chunk
        content = p.get("text") or p.get("description") or "Image Data"
        context += f"Source {i+1} ({p.get('type')}): {content}\n\n"

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    print(f'Prompt: {prompt}')

    response = ollama.chat(model="phi3:mini", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Example Usage
if __name__ == "__main__":
    user_q = "How many colors of cat do we have?"
    relevant_hits = retrieve_hybrid(user_q)
    answer = chat_with_phi(user_q, relevant_hits)
    print(answer)