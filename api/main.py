"""
api/main.py — FastAPI Application (Phase 4 Stub)
This is the microservice layer that wraps the ATS scoring engine.
Currently scaffolded; activate in Phase 4 once Streamlit prototype is proven.

Run with: uvicorn api.main:app --reload --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io

from api.schemas import ScoreRequest, ScoreResponse, HealthResponse, KeywordOverlap, ResumeEntities
from src.scoring.hybrid_engine import run_scoring
from src.parser.pdf_extractor import extract_text_from_pdf
from src.parser.docx_extractor import extract_text_from_docx

app = FastAPI(
    title="ATS AI Engine API",
    description=(
        "Hybrid ATS scoring engine: BM25 sparse + SentenceTransformer dense scoring. "
        "Upload resumes against job descriptions to get AI-powered match scores."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow React/Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check that the API is up and running."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/score/text", response_model=ScoreResponse, tags=["Scoring"])
async def score_from_text(request: ScoreRequest):
    """
    Score a resume against a job description (both provided as text).

    - **resume_text**: Extracted plain text from the candidate's resume
    - **jd_text**: Job description text
    - **alpha**: Weight for BM25 (0.0–1.0), default 0.5
    """
    try:
        result = run_scoring(
            resume_text=request.resume_text,
            jd_text=request.jd_text,
            alpha=request.alpha,
            model_name=request.model_name,
            extract_ner=request.extract_ner,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    entities = None
    if result.resume_entities:
        entities = ResumeEntities(
            skills=result.resume_entities.skills,
            organizations=result.resume_entities.organizations,
            locations=result.resume_entities.locations,
            years_of_experience=result.resume_entities.years_of_experience,
        )

    return ScoreResponse(
        sparse_score=result.sparse_score,
        dense_score=result.dense_score,
        hybrid_score=result.hybrid_score,
        hybrid_percentage=result.hybrid_percentage,
        alpha=result.alpha,
        score_label=result.score_label,
        score_color=result.score_color,
        keywords=KeywordOverlap(
            matched=result.matched_keywords,
            missing=result.missing_keywords,
            match_rate=result.keyword_match_rate,
        ),
        entities=entities,
        top_bm25_terms=result.top_bm25_terms,
    )


@app.post("/score/upload", response_model=ScoreResponse, tags=["Scoring"])
async def score_from_upload(
    resume_file: UploadFile = File(..., description="PDF or DOCX resume file"),
    jd_text: str = "",
    alpha: float = 0.5,
):
    """
    Score an uploaded resume file against a job description.

    - **resume_file**: Upload a PDF or DOCX file
    - **jd_text**: Job description as plain text
    - **alpha**: Hybrid weight for BM25
    """
    filename = resume_file.filename or ""
    content = await resume_file.read()

    try:
        if filename.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(io.BytesIO(content))
        elif filename.lower().endswith((".docx", ".doc")):
            resume_text = extract_text_from_docx(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF or DOCX."
            )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    return await score_from_text(ScoreRequest(
        resume_text=resume_text,
        jd_text=jd_text,
        alpha=alpha,
    ))
