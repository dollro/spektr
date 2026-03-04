from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass

import pymupdf

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".xml", ".html", ".yaml", ".yml"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


@dataclass
class Page:
    image_bytes: bytes  # PNG bytes for image/PDF pages, empty for text
    text: str  # text content for text pages, empty for image
    page_number: int
    content_type: str  # "pdf" | "image" | "text"


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_number: int


def file_to_pages(filename: str, content: bytes) -> list[Page]:
    """MIME-classify file and convert to a list of Pages.

    PDF -> multiple Pages with PNG bytes (300 DPI).
    Image -> single Page with original bytes.
    Text -> single Page with text content.
    Unknown -> empty list + log warning.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    ext = _get_extension(filename)

    if mime_type == "application/pdf" or ext == ".pdf":
        return _pdf_to_pages(content)
    if ext in _IMAGE_EXTENSIONS or (mime_type and mime_type.startswith("image/")):
        return [Page(image_bytes=content, text="", page_number=1, content_type="image")]
    if ext in _TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
        return [
            Page(
                image_bytes=b"",
                text=content.decode("utf-8"),
                page_number=1,
                content_type="text",
            )
        ]

    logger.warning("Unknown file type for %s (mime=%s), skipping.", filename, mime_type)
    return []


def semantic_chunk(
    text: str,
    max_chunk_size: int = 512,
    page_number: int = 1,
) -> list[TextChunk]:
    """Split text into chunks preserving paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[TextChunk] = []
    current = ""
    idx = 0

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= max_chunk_size:
            current = current + "\n\n" + para
        else:
            chunks.append(TextChunk(text=current, chunk_index=idx, page_number=page_number))
            idx += 1
            current = para

    # Handle remaining text and paragraphs exceeding max_chunk_size
    if current:
        if len(current) <= max_chunk_size:
            chunks.append(TextChunk(text=current, chunk_index=idx, page_number=page_number))
        else:
            for sub in _split_long_text(current, max_chunk_size):
                chunks.append(TextChunk(text=sub, chunk_index=idx, page_number=page_number))
                idx += 1

    return chunks


def _pdf_to_pages(content: bytes) -> list[Page]:
    """Convert PDF to Pages with text extraction (PyMuPDF) + image rendering (150 DPI).

    Each page gets:
    - text: extracted from PDF text layer (empty if scanned)
    - image_bytes: PNG rendered at 150 DPI for visual embeddings
    """
    doc = pymupdf.open(stream=content, filetype="pdf")
    pages: list[Page] = []

    for i, fitz_page in enumerate(doc):
        text = fitz_page.get_text("text").strip()

        mat = pymupdf.Matrix(150 / 72, 150 / 72)
        pix = fitz_page.get_pixmap(matrix=mat)
        image_bytes = pix.tobytes("png")

        pages.append(
            Page(
                image_bytes=image_bytes,
                text=text,
                page_number=i + 1,
                content_type="pdf",
            )
        )

    doc.close()
    return pages


def _split_long_text(text: str, max_size: int) -> list[str]:
    """Split text that exceeds max_size on word boundaries."""
    words = text.split()
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def _get_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        return ""
    return filename[dot_idx:].lower()
