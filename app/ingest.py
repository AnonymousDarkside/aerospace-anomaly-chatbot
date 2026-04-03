"""
Ingestion Pipeline — Phase 1

Downloads aerospace PDFs from sources.json, extracts page images,
generates embeddings via the local VLM (CPU), and upserts vectors
to Qdrant Cloud.

Usage:
    python -m app.ingest                     # Ingest all sources
    python -m app.ingest --source-id <id>    # Ingest a single source
"""

import argparse
import json
import os
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from PIL import Image
from tqdm import tqdm

from app.config.settings import get_settings
from app.services.embedding import get_embedding_service
from app.services.vector_store import ensure_collection, upsert_page_vectors


def load_sources(source_id: str | None = None) -> list[dict]:
    """Load source definitions from sources.json."""
    settings = get_settings()
    with open(settings.sources_path, "r") as f:
        sources = json.load(f)["sources"]
    if source_id:
        sources = [s for s in sources if s["id"] == source_id]
        if not sources:
            raise ValueError(f"Source '{source_id}' not found in {settings.sources_path}")
    return sources


def download_pdf(url: str, dest: Path) -> Path:
    """Download a PDF if it doesn't already exist locally."""
    if dest.exists():
        print(f"  [Download] Already exists: {dest.name}")
        return dest

    print(f"  [Download] {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
    print(f"  [Download] Saved: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


def extract_pages(pdf_path: Path, images_dir: Path, dpi: int) -> list[dict]:
    """
    Extract each page of a PDF as a PNG image + raw text.
    Returns a list of dicts with page metadata.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        image_filename = f"{pdf_path.stem}_page_{page_num + 1:04d}.png"
        image_path = images_dir / image_filename
        pix.save(str(image_path))

        text = page.get_text("text").strip()

        pages.append({
            "page_number": page_num + 1,
            "total_pages": len(doc),
            "image_path": str(image_path),
            "image_filename": image_filename,
            "text_content": text[:2000],  # truncate for payload size limits
            "has_text": len(text) > 50,
        })

    doc.close()
    return pages


def ingest_source(source: dict) -> int:
    """
    Full ingestion pipeline for a single source document.
    Returns the number of pages ingested.
    """
    settings = get_settings()
    embedder = get_embedding_service()

    source_id = source["id"]
    source_name = source["name"]
    url = source["url"]

    print(f"\n{'='*60}")
    print(f"Ingesting: {source_name}")
    print(f"{'='*60}")

    # ── 1. Download PDF ──
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = data_dir / f"{source_id}.pdf"
    download_pdf(url, pdf_path)

    # ── 2. Extract page images ──
    images_dir = data_dir / source_id
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Extract] Extracting pages at {settings.pdf_dpi} DPI...")
    pages = extract_pages(pdf_path, images_dir, settings.pdf_dpi)
    print(f"  [Extract] {len(pages)} pages extracted")

    # ── 3. Generate embeddings in batches ──
    print(f"  [Embed] Generating image embeddings (batch_size={settings.batch_size})...")
    all_vectors = []
    all_payloads = []
    all_ids = []

    for i in tqdm(range(0, len(pages), settings.batch_size), desc="  Embedding"):
        batch = pages[i : i + settings.batch_size]
        images = [Image.open(p["image_path"]).convert("RGB") for p in batch]

        vectors = embedder.embed_images(images)

        for page_meta, vector in zip(batch, vectors):
            point_id = f"{source_id}:page:{page_meta['page_number']}"
            payload = {
                "source_id": source_id,
                "source_name": source_name,
                "page_number": page_meta["page_number"],
                "total_pages": page_meta["total_pages"],
                "image_path": page_meta["image_path"],
                "image_filename": page_meta["image_filename"],
                "text_snippet": page_meta["text_content"][:500],
                "has_text": page_meta["has_text"],
                "modality": "image",
            }
            all_vectors.append(vector)
            all_payloads.append(payload)
            all_ids.append(point_id)

        # Close images to free memory
        for img in images:
            img.close()

    # ── 4. Upsert to Qdrant ──
    print(f"  [Qdrant] Upserting {len(all_vectors)} vectors...")
    batch_size = 100
    for i in range(0, len(all_vectors), batch_size):
        upsert_page_vectors(
            vectors=all_vectors[i : i + batch_size],
            payloads=all_payloads[i : i + batch_size],
            ids=all_ids[i : i + batch_size],
        )
    print(f"  [Qdrant] Done. {len(all_vectors)} vectors upserted for '{source_name}'")

    return len(all_vectors)


def main():
    parser = argparse.ArgumentParser(description="Aerospace document ingestion pipeline")
    parser.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="Ingest only the source with this ID (default: all sources)",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the Qdrant collection (use when embedding dim changes)",
    )
    args = parser.parse_args()

    # Validate environment
    settings = get_settings()
    if not settings.qdrant_url or not settings.qdrant_api_key:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env")
        print("  Copy .env.example to .env and fill in your Qdrant Cloud credentials.")
        return

    # Ensure collection exists
    ensure_collection(recreate=args.recreate_collection)

    # Load and process sources
    sources = load_sources(args.source_id)
    total_pages = 0

    for source in sources:
        total_pages += ingest_source(source)

    print(f"\n{'='*60}")
    print(f"Ingestion complete. Total vectors upserted: {total_pages}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
