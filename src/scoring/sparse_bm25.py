"""
src/scoring/sparse_bm25.py — BM25 Keyword Scoring (Aligned with Advanced ATS PDF)

Custom BM25 implementation for single-document ATS scoring.

Architecture (per PDF §1):
  - BM25 returns RAW unbounded scores
  - Normalization to [0,1] happens LATER in the hybrid engine via sigmoid
  - This avoids the scale mismatch: cosine [0,1] vs BM25 [0, ∞)

Why custom BM25 instead of rank_bm25:
  - Standard BM25Okapi with 1-2 documents produces degenerate IDF values
  - IDF = log((N-df+0.5)/(df+0.5)): with N=1 or N=2, matched terms get IDF ≤ 0
  - Our fix: use IDF=1 for all matched terms (uniform importance), letting the
    TF-saturation component do the heavy lifting for relevance ranking
  - This is equivalent to BM15 (BM25 without IDF weighting), which is appropriate
    for single-document-vs-query matching
"""

from __future__ import annotations
import math
from collections import Counter


def bm25_raw_score(
    resume_tokens: list[str],
    jd_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Calculate the RAW (unbounded) BM25 score of the resume against the JD.

    Uses BM25 TF-saturation formula with IDF=1 for all matched terms.
    This produces scores proportional to both keyword coverage and frequency.

    The raw score scales with the number of JD terms found in the resume,
    making it unbounded — exactly as described in the PDF (e.g. 2.4, 15.6, 54.2).
    Normalization happens later in the hybrid engine.

    Args:
        resume_tokens: Tokenized resume text.
        jd_tokens: Tokenized job description text (query).
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        Raw BM25 score (unbounded, >= 0). NOT normalized to [0,1].
    """
    if not resume_tokens or not jd_tokens:
        return 0.0

    resume_tf = Counter(resume_tokens)
    jd_unique = set(jd_tokens)

    doc_len = len(resume_tokens)
    avg_dl = doc_len  # single document, avgdl = doc_len

    # BM25 length normalization denominator component
    len_norm = 1 - b + b * (doc_len / max(avg_dl, 1))

    score = 0.0
    for term in jd_unique:
        tf = resume_tf.get(term, 0)
        if tf == 0:
            continue

        # BM25 TF saturation with IDF=1
        # Full BM25: IDF * (tf * (k1+1)) / (tf + k1 * len_norm)
        # We set IDF = 1 since we have a single document
        tf_component = (tf * (k1 + 1)) / (tf + k1 * len_norm)
        score += tf_component

    return round(score, 4)


def bm25_normalized_score(
    resume_tokens: list[str],
    jd_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Self-normalized BM25 score in [0, 1].

    Divides raw score by the maximum achievable score (if every JD term
    appeared in the resume with the same TF it currently has, plus any missing
    terms at TF=1). This is used as a bounded fallback.

    Args:
        resume_tokens: Tokenized resume text.
        jd_tokens: Tokenized job description text (query).

    Returns:
        Normalized BM25 score in range [0.0, 1.0].
    """
    if not resume_tokens or not jd_tokens:
        return 0.0

    resume_tf = Counter(resume_tokens)
    jd_unique = set(jd_tokens)

    doc_len = len(resume_tokens)
    avg_dl = doc_len
    len_norm = 1 - b + b * (doc_len / max(avg_dl, 1))

    score = 0.0
    max_possible = 0.0

    for term in jd_unique:
        tf = resume_tf.get(term, 0)

        # Max contribution per term: using max(tf, 1) to estimate achievable
        max_tf = max(tf, 1)
        max_component = (max_tf * (k1 + 1)) / (max_tf + k1 * len_norm)
        max_possible += max_component

        if tf == 0:
            continue

        tf_component = (tf * (k1 + 1)) / (tf + k1 * len_norm)
        score += tf_component

    if max_possible <= 0:
        return 0.0

    normalized = score / max_possible
    return round(min(max(normalized, 0.0), 1.0), 4)


def get_top_matching_terms(
    resume_tokens: list[str],
    jd_tokens: list[str],
    top_n: int = 15,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """
    Return the top-N JD terms by their individual BM25 TF contribution.

    Useful for explaining which keywords drove the BM25 score.

    Args:
        resume_tokens: Tokenized resume.
        jd_tokens: Tokenized JD.
        top_n: Number of top terms to return.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        List of (term, score) tuples sorted by contribution descending.
    """
    if not resume_tokens or not jd_tokens:
        return []

    resume_tf = Counter(resume_tokens)
    doc_len = len(resume_tokens)
    avg_dl = doc_len
    len_norm = 1 - b + b * (doc_len / max(avg_dl, 1))

    term_scores: list[tuple[str, float]] = []
    seen: set[str] = set()

    for token in jd_tokens:
        if token in seen:
            continue
        seen.add(token)

        tf = resume_tf.get(token, 0)
        if tf == 0:
            continue

        tf_component = (tf * (k1 + 1)) / (tf + k1 * len_norm)
        term_scores.append((token, round(tf_component, 4)))

    return sorted(term_scores, key=lambda x: x[1], reverse=True)[:top_n]
