import os
import pickle
import faiss
import numpy as np

from embeddings import create_embeddings, create_embedding

VECTOR_DB_DIR = "vector_db"

os.makedirs(VECTOR_DB_DIR, exist_ok=True)


def create_vector_store(chunks, name="tube_mentor"):

    embeddings = create_embeddings(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.asarray(
            embeddings,
            dtype="float32"
        )
    )

    index_path = os.path.join(
        VECTOR_DB_DIR,
        f"{name}.index"
    )

    chunks_path = os.path.join(
        VECTOR_DB_DIR,
        f"{name}.pkl"
    )

    faiss.write_index(index, index_path)

    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)

    return index


def load_vector_store(name="tube_mentor"):

    index_path = os.path.join(
        VECTOR_DB_DIR,
        f"{name}.index"
    )

    chunks_path = os.path.join(
        VECTOR_DB_DIR,
        f"{name}.pkl"
    )

    if not os.path.exists(index_path):
        return None, None

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def search_vector_store(query, name="tube_mentor", top_k=5):

    index, chunks = load_vector_store(name)

    if index is None:
        return []

    query_vector = create_embedding(query)

    query_vector = np.asarray(
        [query_vector],
        dtype="float32"
    )

    distances, indices = index.search(
        query_vector,
        min(top_k, len(chunks))
    )

    results = []

    for index_number in indices[0]:

        if index_number != -1:
            results.append(
                chunks[index_number]
            )

    return results