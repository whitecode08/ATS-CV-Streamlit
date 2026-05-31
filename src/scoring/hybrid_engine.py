"""
src/scoring/hybrid_engine.py — Hybrid ATS Scoring Engine
(Aligned with Advanced ATS Scoring PDF)

Scoring Pipeline:
  1. Parse & Tokenize resume + JD
  2. BM25 Sparse Score (raw unbounded → Min-Max/sigmoid normalized)
  3. Dense Semantic Score (cosine similarity, already [0,1])
  4. Hybrid Combination: α × norm_BM25 + (1-α) × Dense
  5. RRF Score: Reciprocal Rank Fusion alternative (per PDF §3)
  6. Keyword overlap analysis
  7. NER entity extraction

Per PDF recommendations:
  - BM25 raw scores are Min-Max normalized before alpha weighting (§1)
  - Alpha default favors exact keywords: 0.6-0.7 (§2)
  - RRF provides a parameter-free alternative to alpha tuning (§3)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import sys
import os

# Add project root to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.nlp.cleaner import tokenize, extract_keyword_overlap
from src.nlp.entity_tagger import extract_entities, ExtractedEntities
from src.scoring.sparse_bm25 import bm25_raw_score, bm25_normalized_score, get_top_matching_terms
from src.scoring.dense_vector import dense_score


@dataclass
class ScoringResult:
    """Complete result from the hybrid ATS scoring engine."""
    # Core scores (all in [0, 1] except raw_bm25)
    sparse_score: float = 0.0       # Normalized BM25 keyword score [0,1]
    dense_score: float = 0.0        # Semantic embedding score [0,1]
    hybrid_score: float = 0.0       # α × sparse + (1-α) × dense
    rrf_score: float = 0.0          # Reciprocal Rank Fusion score [0,1]

    # Raw BM25 (unbounded, for transparency)
    raw_bm25_score: float = 0.0

    # Hybrid weight used
    alpha: float = 0.6

    # Percentage representation
    hybrid_percentage: float = 0.0  # hybrid_score * 100

    # Keyword analysis
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    keyword_match_rate: float = 0.0  # matched / total_jd_keywords

    # Top BM25 terms
    top_bm25_terms: list[tuple[str, float]] = field(default_factory=list)

    # Entity extraction from resume
    resume_entities: ExtractedEntities | None = None

    # Score label for UI
    score_label: str = ""       # "Strong Match", "Moderate Match", "Weak Match"
    score_color: str = ""       # "green", "orange", "red"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (for API / display)."""
        d = {
            "raw_bm25_score": self.raw_bm25_score,
            "sparse_score": self.sparse_score,
            "dense_score": self.dense_score,
            "hybrid_score": self.hybrid_score,
            "rrf_score": self.rrf_score,
            "alpha": self.alpha,
            "hybrid_percentage": self.hybrid_percentage,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
            "keyword_match_rate": self.keyword_match_rate,
            "top_bm25_terms": self.top_bm25_terms,
            "score_label": self.score_label,
            "score_color": self.score_color,
        }
        if self.resume_entities:
            d["resume_entities"] = {
                "skills": self.resume_entities.skills,
                "organizations": self.resume_entities.organizations,
                "locations": self.resume_entities.locations,
                "years_of_experience": self.resume_entities.years_of_experience,
            }
        return d


def _sigmoid_normalize(raw_score: float, k: float) -> float:
    """
    Sigmoid normalization: maps unbounded [0, ∞) to bounded [0, 1).

    Formula: score / (score + k)

    This is the single-resume analog of Min-Max normalization (PDF §1).
    Min-Max requires a pool of candidates; sigmoid works for a single score.

    Args:
        raw_score: Raw unbounded BM25 score.
        k: Smoothing constant (controls the midpoint of the sigmoid).

    Returns:
        Normalized score in [0, 1).
    """
    if raw_score <= 0:
        return 0.0
    return raw_score / (raw_score + k)


def _compute_rrf(
    sparse_score: float,
    dense_score_val: float,
    k: int = 60,
) -> float:
    """
    Reciprocal Rank Fusion (RRF) — per PDF §3.

    For a single resume, we convert scores to pseudo-ranks:
      rank = 1 / (score + epsilon)  (higher score → lower rank → better)

    Then: RRF = 1/(k + rank_sparse) + 1/(k + rank_dense)

    The result is normalized to [0, 1] by dividing by the maximum possible
    RRF value (which occurs when both ranks = 1).

    Args:
        sparse_score: Normalized BM25 score in [0, 1].
        dense_score_val: Dense cosine similarity score in [0, 1].
        k: RRF smoothing constant (default 60 per industry standard).

    Returns:
        RRF score normalized to [0, 1].
    """
    eps = 1e-6

    # Convert scores to pseudo-ranks (higher score = rank closer to 1)
    # If score = 1.0 → rank = 1 (best)
    # If score = 0.5 → rank = 2
    # If score = 0.0 → rank = very large (worst)
    rank_sparse = 1.0 / (sparse_score + eps)
    rank_dense = 1.0 / (dense_score_val + eps)

    # RRF formula: sum of 1/(k + rank_i)
    rrf_raw = 1.0 / (k + rank_sparse) + 1.0 / (k + rank_dense)

    # Max possible RRF (both scores = 1.0 → both ranks = 1)
    max_rrf = 2.0 / (k + 1.0)

    # Normalize to [0, 1]
    rrf_normalized = rrf_raw / max_rrf if max_rrf > 0 else 0.0

    return round(min(max(rrf_normalized, 0.0), 1.0), 4)


def run_scoring(
    resume_text: str,
    jd_text: str,
    alpha: float = 0.6,
    model_name: str = "all-MiniLM-L6-v2",
    extract_ner: bool = True,
) -> ScoringResult:
    """
    Run the full hybrid ATS scoring pipeline (aligned with Advanced ATS PDF).

    Pipeline Steps:
      1. Tokenize both texts (NLP cleaner)
      2. BM25 raw score → sigmoid normalize to [0,1]
      3. Dense semantic score (cosine similarity, already [0,1])
      4. Hybrid = α × normalized_BM25 + (1-α) × Dense
      5. RRF = Reciprocal Rank Fusion (parameter-free alternative)
      6. Keyword overlap analysis
      7. Top BM25 term contributions
      8. NER entity extraction on resume
      9. Score classification (Strong/Moderate/Weak)

    Args:
        resume_text: Parsed resume text.
        jd_text: Job description text.
        alpha: Weight for BM25 score (0.0-1.0). PDF recommends 0.6-0.7.
        model_name: SentenceTransformer model name.
        extract_ner: Whether to run NER entity extraction.

    Returns:
        ScoringResult with all scores and analysis.
    """
    result = ScoringResult(alpha=alpha)

    # ── Step 1: Tokenize ──────────────────────────────────────────────────
    resume_tokens = tokenize(resume_text)
    jd_tokens = tokenize(jd_text)

    # ── Step 2: BM25 Sparse Score ─────────────────────────────────────────
    # Get raw unbounded BM25 score
    result.raw_bm25_score = bm25_raw_score(resume_tokens, jd_tokens)

    # Normalize to [0, 1] via sigmoid (PDF §1: "Min-Max normalize before alpha")
    # k is calibrated based on JD query length
    num_unique_jd = len(set(jd_tokens))
    k_normalizer = max(num_unique_jd * 0.4, 1.0)
    result.sparse_score = round(
        _sigmoid_normalize(result.raw_bm25_score, k_normalizer), 4
    )

    # ── Step 3: Dense (Semantic) Score ────────────────────────────────────
    result.dense_score = dense_score(resume_text, jd_text, model_name)

    # ── Step 4: Hybrid Combination (PDF §1 & §2) ─────────────────────────
    # hybrid = α × normalized_BM25 + (1-α) × Dense
    result.hybrid_score = alpha * result.sparse_score + (1 - alpha) * result.dense_score
    result.hybrid_score = round(min(max(result.hybrid_score, 0.0), 1.0), 4)
    result.hybrid_percentage = round(result.hybrid_score * 100, 1)

    # ── Step 5: RRF Score (PDF §3) ────────────────────────────────────────
    result.rrf_score = _compute_rrf(result.sparse_score, result.dense_score)

    # ── Step 6: Keyword Overlap Analysis ─────────────────────────────────
    overlap = extract_keyword_overlap(resume_tokens, jd_tokens)
    result.matched_keywords = overlap["matched"]
    result.missing_keywords = overlap["missing"]
    total_jd_kw = len(set(jd_tokens))
    result.keyword_match_rate = (
        round(len(result.matched_keywords) / total_jd_kw, 4)
        if total_jd_kw > 0 else 0.0
    )

    # ── Step 7: Top BM25 Terms ────────────────────────────────────────────
    result.top_bm25_terms = get_top_matching_terms(resume_tokens, jd_tokens)

    # ── Step 8: NER Extraction ────────────────────────────────────────────
    if extract_ner:
        result.resume_entities = extract_entities(resume_text)

    # ── Step 9: Score Label & Color ───────────────────────────────────────
    result.score_label, result.score_color = _classify_score(result.hybrid_score)

    return result


def _classify_score(score: float) -> tuple[str, str]:
    """Map a hybrid score to a human-readable label and UI color."""
    if score >= 0.70:
        return "Strong Match", "green"
    elif score >= 0.45:
        return "Moderate Match", "orange"
    else:
        return "Weak Match", "red"
