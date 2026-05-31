"""
src/parser/pdf_extractor.py — PDF Text Extraction
Uses PyMuPDF (fitz) for robust layout-aware text extraction from PDF resumes.
Handles multi-column layouts by sorting text blocks by vertical position.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Union


def extract_text_from_pdf(source: Union[str, Path, bytes, io.BytesIO]) -> str:
    """
    Extract plain text from a PDF file.

    Args:
        source: File path (str/Path), raw bytes, or BytesIO object.

    Returns:
        A clean, concatenated string of all text from the PDF.

    Raises:
        ImportError: If PyMuPDF is not installed.
        ValueError: If the source type is unsupported.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: pip install PyMuPDF"
        ) from e

    # Open document from various source types
    if isinstance(source, (str, Path)):
        doc = fitz.open(str(source))
    elif isinstance(source, bytes):
        doc = fitz.open(stream=source, filetype="pdf")
    elif isinstance(source, io.BytesIO):
        doc = fitz.open(stream=source.read(), filetype="pdf")
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Use "blocks" layout to handle multi-column resumes correctly
        # Each block is a tuple: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")

        # Sort blocks top-to-bottom (by y0), then left-to-right (by x0)
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))

        page_text_parts = []
        for block in blocks_sorted:
            if block[6] == 0:  # block_type 0 = text (not image)
                text = block[4].strip()
                if text:
                    page_text_parts.append(text)

        pages_text.append("\n".join(page_text_parts))

    doc.close()
    full_text = "\n\n".join(pages_text)
    return _post_process(full_text)


def _post_process(text: str) -> str:
    """Clean up common PDF extraction artifacts."""
    import re

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize line endings
    text = re.sub(r"\r\n|\r", "\n", text)
    # Remove more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
