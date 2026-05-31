"""
src/scoring/dense_vector.py — Semantic Dense Scoring
Uses SentenceTransformers to encode the JD and Resume into embeddings,
then computes cosine similarity. Returns a score in [0, 1].
"""

from __future__ import annotations
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load and cache the SentenceTransformer model (only once per session)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for dense scoring. "
            "Install it with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name)


def dense_score(
    resume_text: str,
    jd_text: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """
    Calculate semantic similarity between a resume and job description
    using sentence embeddings and cosine similarity.

    Args:
        resume_text: Raw or lightly cleaned resume text.
        jd_text: Raw or lightly cleaned job description text.
        model_name: SentenceTransformer model name.

    Returns:
        Cosine similarity score in range [0.0, 1.0].
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    model = _load_model(model_name)

    # Encode both texts as embeddings
    embeddings = model.encode(
        [resume_text[:8000], jd_text[:8000]],  # Truncate for speed
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for faster cosine sim
        show_progress_bar=False,
    )

    resume_emb = embeddings[0]
    jd_emb = embeddings[1]

    # Cosine similarity: since embeddings are L2-normalized, it's just dot product
    similarity = float(np.dot(resume_emb, jd_emb))

    # Clamp to [0, 1] (cosine similarity can be slightly negative for very dissimilar texts)
    return round(max(0.0, min(1.0, similarity)), 4)


def get_section_scores(
    resume_sections: dict[str, str],
    jd_text: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, float]:
    """
    Calculate dense scores for individual resume sections against the JD.

    Args:
        resume_sections: Dict mapping section name → section text.
        jd_text: Job description text.
        model_name: SentenceTransformer model name.

    Returns:
        Dict mapping section name → cosine similarity score.
    """
    if not resume_sections or not jd_text.strip():
        return {}

    model = _load_model(model_name)
    jd_emb = model.encode(
        [jd_text[:8000]],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    results = {}
    for section_name, section_text in resume_sections.items():
        if not section_text.strip():
            results[section_name] = 0.0
            continue
        sec_emb = model.encode(
            [section_text[:2000]],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sim = float(np.dot(sec_emb, jd_emb))
        results[section_name] = round(max(0.0, min(1.0, sim)), 4)

    return results
