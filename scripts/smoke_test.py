#!/usr/bin/env python3
"""
Smoke test — validates the CLIP model loads, embeds text and images,
and produces correct-dimension vectors. No Qdrant or API keys needed.

Exit code 0 = all checks pass.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _to_tensor(output):
    """Handle both old (raw tensor) and new (BaseModelOutput) return types."""
    import torch
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def main():
    errors = []

    # ── 1. Settings ──
    print("[1/5] Loading settings...")
    try:
        from app.config.settings import Settings
        settings = Settings(qdrant_url="http://fake:6333", qdrant_api_key="fake")
        print(f"  model={settings.vlm_model_name} device={settings.vlm_device} dim={settings.embedding_dim}")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # ── 2. Model load ──
    print("[2/5] Loading CLIP model...")
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(settings.vlm_model_name)
        model.eval()
        processor = CLIPProcessor.from_pretrained(settings.vlm_model_name)
        proj_dim = model.config.projection_dim
        print(f"  projection_dim={proj_dim}")
        assert proj_dim == settings.embedding_dim
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # ── 3. Text embedding ──
    print("[3/5] Embedding text...")
    try:
        texts = ["spacecraft anomaly detection", "telemetry data analysis"]
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            raw = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        emb = _to_tensor(raw)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        assert emb.shape == (2, settings.embedding_dim), f"bad shape: {emb.shape}"
        print(f"  shape={list(emb.shape)} norm={emb[0].norm().item():.4f}")
    except Exception as e:
        errors.append(f"Text embed: {e}")
        print(f"  FAIL: {e}")

    # ── 4. Image embedding ──
    print("[4/5] Embedding image...")
    try:
        from PIL import Image
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = processor(images=[dummy_img], return_tensors="pt")
        with torch.no_grad():
            raw = model.get_image_features(pixel_values=inputs["pixel_values"])
        emb = _to_tensor(raw)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        assert emb.shape == (1, settings.embedding_dim), f"bad shape: {emb.shape}"
        print(f"  shape={list(emb.shape)}")
    except Exception as e:
        errors.append(f"Image embed: {e}")
        print(f"  FAIL: {e}")

    # ── 5. Cross-modal similarity ──
    print("[5/5] Cross-modal similarity...")
    try:
        from PIL import Image
        t_in = processor(text=["a photo of a rocket"], return_tensors="pt", padding=True, truncation=True)
        i_in = processor(images=[Image.new("RGB", (224, 224), color=(200, 50, 50))], return_tensors="pt")
        with torch.no_grad():
            t_emb = _to_tensor(model.get_text_features(input_ids=t_in["input_ids"], attention_mask=t_in["attention_mask"]))
            i_emb = _to_tensor(model.get_image_features(pixel_values=i_in["pixel_values"]))
            t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
            i_emb = torch.nn.functional.normalize(i_emb, dim=-1)
            sim = (t_emb @ i_emb.T).item()
        print(f"  cosine_similarity={sim:.4f}")
        assert sim != 0.0
    except Exception as e:
        errors.append(f"Cross-modal: {e}")
        print(f"  FAIL: {e}")

    print()
    if errors:
        print(f"FAILED ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
