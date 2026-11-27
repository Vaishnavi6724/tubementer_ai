import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "data/vector_db/faiss.index"
METADATA_FILE = "data/vector_db/metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading FAISS database...")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("FAISS vectors:", index.ntotal)

model = SentenceTransformer(MODEL_NAME)

query = input("\nEnter your question: ")

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

k = 5

scores, indices = index.search(query_embedding, k)

print("\n========== SEARCH RESULTS ==========\n")

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):

    result = metadata[idx]

    print(f"Result {rank}")
    print(f"Score: {score:.4f}")
    print(f"Title: {result['title']}")
    print(f"Video ID: {result['video_id']}")
    print(f"Text: {result['text'][:500]}")
    print("-" * 70)