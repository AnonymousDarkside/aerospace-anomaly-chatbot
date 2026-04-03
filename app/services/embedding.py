"""
Unified Vision-Language Embedding Service.

Uses OpenAI CLIP (clip-vit-base-patch32) via HuggingFace transformers.
CPU-only, amd64-compatible. No trust_remote_code, no custom model code.
"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from app.config.settings import get_settings

_instance: "EmbeddingService | None" = None


def _to_tensor(output) -> torch.Tensor:
    """Handle both old (raw tensor) and new (BaseModelOutput) return types."""
    if isinstance(output, torch.Tensor):
        return output
    # transformers >= 5.x returns BaseModelOutputWithPooling
    return output.pooler_output


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.device = settings.vlm_device
        self.model_name = settings.vlm_model_name

        print(f"[EmbeddingService] Loading {self.model_name} on {self.device}...")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        proj_dim = self.model.config.projection_dim
        assert proj_dim == settings.embedding_dim, (
            f"Model projection_dim={proj_dim} != settings.embedding_dim={settings.embedding_dim}"
        )
        print(f"[EmbeddingService] Ready. dim={proj_dim}")

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        raw = self.model.get_text_features(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
        )
        embeddings = _to_tensor(raw)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu().tolist()

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt")
        raw = self.model.get_image_features(
            pixel_values=inputs["pixel_values"].to(self.device),
        )
        embeddings = _to_tensor(raw)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_image(self, image: Image.Image) -> list[float]:
        return self.embed_images([image])[0]


def get_embedding_service() -> EmbeddingService:
    global _instance
    if _instance is None:
        _instance = EmbeddingService()
    return _instance
