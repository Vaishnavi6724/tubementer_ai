import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/chunks.json"
VECTOR_DB_DIR = "data/vector_db"

INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss.index")
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading chunks...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))

texts = [chunk["text"] for chunk in chunks]

print("\nLoading embedding model:")
print(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

print("\nCreating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype("float32")

print("Embedding shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS vectors:", index.ntotal)

# Create directory
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Save FAISS index
faiss.write_index(index, INDEX_FILE)

# Save metadata
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n================================")
print("FAISS DATABASE CREATED SUCCESSFULLY")
print("================================")
print("Index:", INDEX_FILE)
print("Metadata:", METADATA_FILE)
print("Vectors:", index.ntotal)