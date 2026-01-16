from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

COLLECTION = "local_docs"

qdrant = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=5):
    q_vec = embed_model.encode(query).tolist()
    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=top_k
    )
    return result.points

def build_prompt(query, hits):
    context = "\n\n".join(
        f"Source {i+1}:\n{h.payload['text']}"
        for i, h in enumerate(hits)
    )

    return f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

if __name__ == "__main__":
    while True:
        query = input("\nAsk (or exit): ").strip()
        if query.lower() == "exit":
            break

        hits = retrieve(query)
        print(f'hits: {hits}')
        if not hits:
            print("No relevant context found.")
            continue

        prompt = build_prompt(query, hits)

        #
        # ✅ response is ALWAYS defined here
        response = ollama.chat(
            #model="llama3.1:8b",
            model="phi3:mini",
            messages=[{"role": "user", "content": prompt}]
        )

        print("\nAnswer:\n", response["message"]["content"])
        print("\nSources:")
        for h in hits:
            print("-", h.payload["path"])
