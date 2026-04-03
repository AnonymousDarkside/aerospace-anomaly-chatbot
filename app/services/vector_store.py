"""
Qdrant Cloud vector store operations.

Handles collection creation, upserting page embeddings with metadata,
and similarity search for both text and image queries.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from app.config.settings import get_settings

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _client


def ensure_collection(recreate: bool = False) -> None:
    """Create the collection if it does not already exist.
    If recreate=True, drop and recreate (useful when embedding dim changes).
    """
    settings = get_settings()
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection_name in collections:
        if recreate:
            client.delete_collection(settings.qdrant_collection_name)
            print(f"[VectorStore] Dropped existing collection '{settings.qdrant_collection_name}'")
        else:
            print(f"[VectorStore] Collection '{settings.qdrant_collection_name}' already exists")
            return

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"[VectorStore] Created collection '{settings.qdrant_collection_name}' (dim={settings.embedding_dim})")


def upsert_page_vectors(
    vectors: list[list[float]],
    payloads: list[dict],
    ids: list[str],
) -> None:
    """Upsert a batch of page vectors with metadata payloads."""
    settings = get_settings()
    client = get_qdrant_client()

    points = [
        PointStruct(
            id=_deterministic_id(uid),
            vector=vector,
            payload=payload,
        )
        for uid, vector, payload in zip(ids, vectors, payloads)
    ]

    client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=points,
    )


def search(query_vector: list[float], top_k: int | None = None) -> list[dict]:
    """Search for the top_k most similar vectors. Returns payload dicts."""
    settings = get_settings()
    client = get_qdrant_client()
    k = top_k or settings.top_k

    results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        limit=k,
        with_payload=True,
    )
    return [
        {
            "score": hit.score,
            **hit.payload,
        }
        for hit in results.points
    ]


def _deterministic_id(string_id: str) -> int:
    """Convert a string ID to a deterministic positive integer for Qdrant."""
    import hashlib
    return int(hashlib.sha256(string_id.encode()).hexdigest()[:16], 16)
