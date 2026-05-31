# src/parser/__init__.py
from .pdf_extractor import extract_text_from_pdf
from .docx_extractor import extract_text_from_docx

__all__ = ["extract_text_from_pdf", "extract_text_from_docx"]
