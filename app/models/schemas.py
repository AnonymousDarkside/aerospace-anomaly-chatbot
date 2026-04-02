"""Request and response schemas for the FastAPI backend."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's text query")
    provider: str = Field(..., pattern="^(openrouter|google)$", description="LLM provider")
    api_key: str = Field(..., min_length=1, description="API key for the selected provider")
    model: str | None = Field(
        default=None,
        description="Model override. Defaults to provider-specific default.",
    )


class RetrievedContext(BaseModel):
    source_name: str
    page_number: int
    total_pages: int
    image_path: str
    text_snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    provider: str
    model_used: str
    contexts: list[RetrievedContext]


class HealthResponse(BaseModel):
    status: str = "ok"
    vlm_loaded: bool
    qdrant_connected: bool
