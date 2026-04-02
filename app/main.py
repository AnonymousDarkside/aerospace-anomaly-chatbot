"""
FastAPI Backend — Aerospace Anomaly & Telemetry Assistant

Endpoints:
    POST /chat   — RAG query: embed → search Qdrant → generate via external LLM
    GET  /health — Readiness check for VLM and Qdrant
    GET  /image/{source_id}/{filename} — Serve extracted page images
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RetrievedContext,
)
from app.services.embedding import get_embedding_service
from app.services.vector_store import ensure_collection, search
from app.services import llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: warm the VLM and verify Qdrant ──
    get_embedding_service()
    ensure_collection()
    print("[API] Ready.")
    yield


app = FastAPI(
    title="Aerospace Anomaly & Telemetry Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG pipeline:
    1. Embed the user query with the local VLM.
    2. Retrieve top-K matching page vectors from Qdrant.
    3. Build a context string from retrieved metadata.
    4. Generate an answer via the user-selected external LLM.
    """
    embedder = get_embedding_service()

    # 1. Embed query
    query_vector = embedder.embed_text(request.query)

    # 2. Retrieve from Qdrant
    settings = get_settings()
    results = search(query_vector, top_k=settings.top_k)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No matching documents found. Run the ingestion pipeline first.",
        )

    # 3. Build context for the LLM
    contexts = []
    context_parts = []
    for i, hit in enumerate(results, 1):
        ctx = RetrievedContext(
            source_name=hit.get("source_name", "Unknown"),
            page_number=hit.get("page_number", 0),
            total_pages=hit.get("total_pages", 0),
            image_path=hit.get("image_path", ""),
            text_snippet=hit.get("text_snippet", ""),
            score=hit.get("score", 0.0),
        )
        contexts.append(ctx)
        context_parts.append(
            f"[Source {i}] {ctx.source_name} — Page {ctx.page_number}/{ctx.total_pages}\n"
            f"{ctx.text_snippet}\n"
        )

    context_text = "\n---\n".join(context_parts)

    # 4. Generate answer via external LLM
    try:
        answer, model_used = llm.generate(
            query=request.query,
            context_text=context_text,
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error ({request.provider}): {str(e)}",
        )

    return ChatResponse(
        answer=answer,
        provider=request.provider,
        model_used=model_used,
        contexts=contexts,
    )


@app.get("/image/{source_id}/{filename}")
async def serve_image(source_id: str, filename: str):
    """Serve an extracted page image for display in the frontend."""
    settings = get_settings()
    image_path = Path(settings.data_dir) / source_id / filename

    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Prevent path traversal
    resolved = image_path.resolve()
    data_root = Path(settings.data_dir).resolve()
    if not str(resolved).startswith(str(data_root)):
        raise HTTPException(status_code=403, detail="Forbidden")

    return FileResponse(str(resolved), media_type="image/png")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if the VLM is loaded and Qdrant is reachable."""
    from app.services.vector_store import get_qdrant_client

    vlm_ok = False
    qdrant_ok = False

    try:
        get_embedding_service()
        vlm_ok = True
    except Exception:
        pass

    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (vlm_ok and qdrant_ok) else "degraded",
        vlm_loaded=vlm_ok,
        qdrant_connected=qdrant_ok,
    )
