from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Qdrant Cloud ──
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "aerospace_docs"

    # ── VLM Configuration (jina-clip-v2: unified text + image embeddings) ──
    vlm_model_name: str = "jinaai/jina-clip-v2"
    vlm_device: str = "cpu"
    embedding_dim: int = 1024

    # ── Paths ──
    data_dir: str = "app/data"
    sources_path: str = "app/config/sources.json"

    # ── Ingestion ──
    pdf_dpi: int = 200
    batch_size: int = 2
    vlm_dtype: str = "float32"

    # ── API ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    top_k: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
