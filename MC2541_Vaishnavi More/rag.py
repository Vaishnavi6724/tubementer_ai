import json
import faiss
from sentence_transformers import SentenceTransformer

from gemini_ai import ask_gemini


# ==========================================
# CONFIGURATION
# ==========================================

INDEX_FILE = "data/vector_db/faiss.index"
METADATA_FILE = "data/vector_db/metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


# ==========================================
# LOAD FAISS DATABASE
# ==========================================

print("Loading TubeMentor AI dataset...")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Loaded {index.ntotal} vectors.")


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("RAG system ready!")


# ==========================================
# SEARCH DATASET
# ==========================================

def search_dataset(question, top_k=5):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        result = metadata[idx]

        results.append({
            "score": float(score),
            "title": result["title"],
            "video_id": result["video_id"],
            "url": result["url"],
            "text": result["text"]
        })

    return results


# ==========================================
# RAG QUESTION ANSWERING
# ==========================================

def ask_rag(question, name="tube_mentor"):

    results = search_dataset(
        question,
        top_k=5
    )

    if not results:
        return ask_gemini(question)

    # Build context from retrieved videos
    context_parts = []

    for result in results:

        context_parts.append(
            f"""
Video Title: {result['title']}

Transcript:
{result['text']}
"""
        )

    context = "\n\n".join(context_parts)

    # Ask Gemini using retrieved context
    answer = ask_gemini(
        question,
        context=context
    )

    return answer


# ==========================================
# OLD FUNCTION
# ==========================================

def build_knowledge_base(text, name="tube_mentor"):

    """
    Kept for compatibility with app.py.

    The main dataset is already stored
    in the FAISS database.
    """

    return len(text.split())