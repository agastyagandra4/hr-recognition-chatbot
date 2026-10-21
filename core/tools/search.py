import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "recognitions"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """Load the embedding model once, then reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    """Turn a list of texts into lists of numbers (embeddings)."""
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def get_collection():
    """Open (or create) the ChromaDB collection that stores the messages."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def search_messages(query, department=None, category=None, limit=5):
    """Find the messages closest in meaning to the query, with optional filters."""
    collection = get_collection()
    if collection.count() == 0:
        return {"error": "The search index is empty. Run scripts/build_index.py first."}

    # Build the optional filters
    filters = []
    if department:
        filters.append({"receiver_department": department})
    if category:
        filters.append({"category_name": category})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    results = collection.query(
        query_embeddings=embed([query]),
        n_results=limit,
        where=where,
    )

    matches = []
    for record_id, message, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "record_id": record_id,
            "similarity": round(1 - distance, 3),
            "receiver_department": meta["receiver_department"],
            "giver_department": meta["giver_department"],
            "category": meta["category_name"],
            "subcategory": meta["subcategory_name"],
            "award_title": meta["award_title"],
            "message": message if len(message) <= 400 else message[:400] + "...",
        })

    if not matches:
        return {"query": query, "error": "No messages matched these filters."}

    return {
        "query": query,
        "filters": {"department": department, "category": category},
        "results": matches,
    }