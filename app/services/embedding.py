"""
Unified Vision-Language Embedding Service.

Wraps jina-clip-v2 (or compatible VLM) to produce embeddings for both
text queries and document page images in a shared vector space.
Runs on CPU by default (configurable via VLM_DEVICE env var).
Designed as a singleton for reuse across ingestion and serving.
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from app.config.settings import get_settings

_instance: "EmbeddingService | None" = None

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.device = settings.vlm_device
        self.model_name = settings.vlm_model_name
        self.dtype = _DTYPE_MAP.get(settings.vlm_dtype, torch.float16)

        print(f"[EmbeddingService] Loading {self.model_name} on {self.device} ({settings.vlm_dtype})...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        print(f"[EmbeddingService] Model loaded. Embedding dim: {settings.embedding_dim}")

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings. Returns list of float vectors."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        return text_embeddings.cpu().tolist()

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed a batch of PIL images. Returns list of float vectors."""
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        image_embeddings = self.model.get_image_features(**inputs)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
        return image_embeddings.cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_image(self, image: Image.Image) -> list[float]:
        """Embed a single PIL image."""
        return self.embed_images([image])[0]


def get_embedding_service() -> EmbeddingService:
    """Return the singleton EmbeddingService, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = EmbeddingService()
    return _instance
