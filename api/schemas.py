"""
api/schemas.py — Pydantic Models for FastAPI (Phase 4)
Defines request/response schemas for the ATS scoring API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ScoreRequest(BaseModel):
    """Request body for the /score endpoint."""
    resume_text: str = Field(..., description="Extracted resume text", min_length=10)
    jd_text: str = Field(..., description="Job description text", min_length=10)
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="BM25 weight (0.0=pure dense, 1.0=pure sparse)"
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name"
    )
    extract_ner: bool = Field(
        default=True,
        description="Whether to run NER entity extraction on the resume"
    )


class KeywordOverlap(BaseModel):
    matched: list[str]
    missing: list[str]
    match_rate: float


class ResumeEntities(BaseModel):
    skills: list[str]
    organizations: list[str]
    locations: list[str]
    years_of_experience: Optional[float] = None


class ScoreResponse(BaseModel):
    """Response body for the /score endpoint."""
    sparse_score: float = Field(..., description="BM25 keyword score [0, 1]")
    dense_score: float = Field(..., description="Semantic similarity score [0, 1]")
    hybrid_score: float = Field(..., description="Weighted hybrid score [0, 1]")
    hybrid_percentage: float = Field(..., description="hybrid_score * 100")
    alpha: float
    score_label: str = Field(..., description="Strong/Moderate/Weak Match")
    score_color: str = Field(..., description="green/orange/red")
    keywords: KeywordOverlap
    entities: Optional[ResumeEntities] = None
    top_bm25_terms: list[tuple[str, float]] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
