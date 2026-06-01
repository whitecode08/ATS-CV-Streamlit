import json
import logging

from src.config import config
from src.scoring.hybrid_engine import run_scoring
from src.parser.pdf_extractor import extract_text_from_pdf
from src.parser.docx_extractor import extract_text_from_docx
from worker.downloader import download_file

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("document_key", "job_description")


def validate(body: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if not body.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def process(body: dict) -> dict:
    document_key = body["document_key"]
    job_id = body.get("job_id", document_key)
    jd_text = body["job_description"]

    logger.info("[%s] Downloading %s", job_id, document_key)
    file_bytes, ext = download_file(document_key)

    if ext == "pdf":
        resume_text = extract_text_from_pdf(file_bytes)
    else:
        resume_text = extract_text_from_docx(file_bytes)

    logger.info("[%s] Scoring...", job_id)
    result = run_scoring(
        resume_text=resume_text,
        jd_text=jd_text,
        alpha=config.ALPHA_WEIGHT,
        model_name=config.SENTENCE_MODEL,
        extract_ner=config.EXTRACT_NER,
    )

    return result.to_dict()


def print_result(job_id: str, result: dict) -> None:
    label = result.get("score_label", "")
    pct = result.get("hybrid_percentage", 0.0)
    sparse = result.get("sparse_score", 0.0)
    dense = result.get("dense_score", 0.0)
    hybrid = result.get("hybrid_score", 0.0)
    alpha = result.get("alpha", config.ALPHA_WEIGHT)
    matched = result.get("matched_keywords", [])
    missing = result.get("missing_keywords", [])

    matched_preview = ", ".join(matched[:8]) + (" ..." if len(matched) > 8 else "")
    missing_preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")

    print("━" * 52)
    print(f"[JOB {job_id}]  {label} — {pct:.1f}%")
    print(f"  Sparse  (BM25):     {sparse:.3f}")
    print(f"  Dense   (semantic): {dense:.3f}")
    print(f"  Hybrid:             {hybrid:.3f}  (alpha={alpha})")
    print()
    print(f"  Matched keywords ({len(matched)}): {matched_preview}")
    print(f"  Missing keywords ({len(missing)}): {missing_preview}")
    print("━" * 52)
