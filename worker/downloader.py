import requests
from src.config import config


def download_file(document_key: str) -> tuple[bytes, str]:
    """Download a file from the CDN and return (bytes, extension)."""
    base = config.S3_ENDPOINT.rstrip("/")
    url = f"{base}{document_key}"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    ext = document_key.rsplit(".", 1)[-1].lower()
    if ext not in ("pdf", "docx"):
        raise ValueError(f"Unsupported file type: .{ext}")

    return resp.content, ext
