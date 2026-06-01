"""
src/config.py — Hyperparameters & Environment Configuration
Loads .env variables and provides a central config object for the ATS engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")


class Config:
    """Central configuration for the ATS AI Engine."""

    # ── Scoring Weights ───────────────────────────────────────────────────────
    # alpha: weight for BM25 sparse score (0.0 = pure Dense, 1.0 = pure Sparse)
    ALPHA_WEIGHT: float = float(os.getenv("ALPHA_WEIGHT", "0.5"))

    # ── Model Names ───────────────────────────────────────────────────────────
    SENTENCE_MODEL: str = os.getenv("SENTENCE_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")

    # ── Paths ─────────────────────────────────────────────────────────────────
    ROOT_DIR: Path = _root
    DATA_DIR: Path = _root / "data"
    RESUMES_DIR: Path = _root / "data" / "resumes"
    JD_DIR: Path = _root / "data" / "job_descriptions"
    TAXONOMY_DIR: Path = _root / "data" / "taxonomy"

    # ── AMQP (RabbitMQ) ──────────────────────────────────────────────────────
    AMQP_URL: str      = os.getenv("AMQP_URL", "amqp://localhost:5672")
    AMQP_USERNAME: str = os.getenv("AMQP_USERNAME", "guest")
    AMQP_PASSWORD: str = os.getenv("AMQP_PASSWORD", "guest")
    AMQP_QUEUE: str    = os.getenv("AMQP_QUEUE", "ats_check_queue")

    # ── CDN / Object Storage ─────────────────────────────────────────────────
    S3_ENDPOINT: str   = os.getenv("S3_ENDPOINT", "")

    # ── Worker Scoring Defaults ───────────────────────────────────────────────
    EXTRACT_NER: bool  = os.getenv("EXTRACT_NER", "true").lower() == "true"

    # ── Optional API Keys ─────────────────────────────────────────────────────
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # ── Score Thresholds (for UI color coding) ────────────────────────────────
    SCORE_HIGH: float = 0.70    # ≥ 70% → Green (Strong Match)
    SCORE_MEDIUM: float = 0.45  # ≥ 45% → Yellow (Moderate Match)
    # < 45% → Red (Weak Match)


# Singleton instance — import this in other modules
config = Config()
