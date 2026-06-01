# 🎯 Advanced ATS AI Engine

> **High-Performance Hybrid Resume Scoring** — Exact BM25 Keyword Search meets Dense Vector Semantic AI, coupled with Multi-Domain Entity Extraction and Reciprocal Rank Fusion sorting.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NLP](https://img.shields.io/badge/spaCy-v3.7-09A3D5?style=flat&logo=spacy&logoColor=white)](https://spacy.io)
[![Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-blueviolet?style=flat)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

A beautiful, professional, production-ready **Applicant Tracking System (ATS) AI Engine** that evaluates how well a candidate's resume matches a job description using an advanced hybrid scoring pipeline.

---

## 🧮 Mathematical Scoring Pipeline

To resolve standard ATS issues (such as raw BM25 inflation and keyword stuffing), this engine integrates two complementary search architectures with advanced normalization and ranking fusion:

```
                  ┌───────────────────────┐
                  │ Uploaded Resume & JD  │
                  └───────────┬───────────┘
                              ▼
                ┌─────────────┴─────────────┐
                │    NLP Parser & Cleaner   │
                └──────┬─────────────┬──────┘
                       │             │
        ┌──────────────▼──────┐      └──────────────▼──────┐
        │  BM25 Sparse Score  │      │ Dense Semantic Score│
        │   (Exact Keywords)  │      │  (Transformers AI)  │
        └──────────────┬──────┘      └──────────────┬──────┘
                       │                            │
                       ▼                            ▼
             Sigmoid Normalization          Cosine Similarity
             score / (score + k)                 [0, 1]
                       │                            │
                       └─────────────┬──────────────┘
                                     ▼
                      ┌──────────────┴──────────────┐
                      │    Hybrid Blend (α slider)  │
                      │  α × Sparse + (1-α) × Dense │
                      └──────────────┬──────────────┘
                                     ▼
                        ┌────────────┴────────────┐
                        │   Reciprocal Rank       │
                        │    Fusion (RRF)         │
                        │ (Behind-the-scenes sort)│
                        └─────────────────────────┘
```

### 1. Sparse Matcher: BM25 with Sigmoid Normalization
*   **Method**: Evaluates exact term frequencies and document lengths.
*   **Normalization**: Raw BM25 scores are unbounded $[0, \infty)$ and inflate with longer resumes. To map them to a calibrated $[0, 1)$ percentage range, the engine applies **Sigmoid Normalization**:
    $$\text{Sparse Score} = \frac{\text{Raw Score}}{\text{Raw Score} + k}$$
    *Where $k$ is dynamically calibrated based on the number of unique keywords in the Job Description.*

### 2. Dense Matcher: Semantic Vector Search
*   **Method**: Sentence-Transformers (`all-MiniLM-L6-v2`) encode the semantic context of both documents.
*   **Calculation**: Calculates **Cosine Similarity** between the embedding vectors to capture intent, conceptual match, and synonym overlaps (already naturally bounded within $[0, 1]$).

### 3. Hybrid Combination & Tuning
*   **Method**: Blends exact match precision with deep conceptual understanding using a weighted average:
    $$\text{Hybrid Score} = \alpha \times \text{Sparse Score} + (1 - \alpha) \times \text{Dense Score}$$
*   **Calibration**: The default hybrid weight is calibrated at **$\alpha = 0.6$**, striking a professional balance that slightly favors exact keyword validation (crucial for technical compliance) while valuing semantic context.

### 4. Reciprocal Rank Fusion (RRF)
*   **Method**: Multi-engine candidate ranking utilizing reciprocal rank decay:
    $$\text{RRF Raw} = \frac{1}{k_{rrf} + r_{\text{sparse}}} + \frac{1}{k_{rrf} + r_{\text{dense}}}$$
*   **Design Choice**: RRF represents candidate ranks *relative* to a pool, rather than an absolute candidate-to-job match percentage. Therefore, **RRF is used behind the scenes** to mathematically order multiple candidates, ensuring single-candidate dashboard percentages remain accurate, logical, and easy for recruiters to interpret.

---

## ✨ Features

*   **📄 Multi-Format Resume Parser** — Robust extraction of text from PDF and DOCX files (handles multi-column CVs and complex layouts).
*   **🧠 Multi-Domain Skill Extraction** — Transitioned from a tech-only parser to a universal parser using highly comprehensive **`DOMAIN_SKILLS_KEYWORDS`** mappings, encompassing:
    *   *Finance, Healthcare, HR & Recruitment, Sales & Marketing, Operations & Project Management, Legal & Compliance, Data Science & AI, Software Engineering, Design, Education, and more.*
*   **🛑 High-Fidelity Stop-Word Filtering** — Utilizes a highly calibrated `_JD_FLUFF` filter in the NLP cleaner to exclude regulatory boilerplate and general office terminology from matching, eliminating false positives.
*   **📊 Clean Streamlit Dashboard** — A beautiful, professional interface featuring:
    *   A high-contrast visual palette (Green/Orange/Red) indicating strong, moderate, or weak matches.
    *   Plotly gauge and score comparison bar charts.
    *   A glassmorphic sidebar layout for interactive parameter tuning (like the $\alpha$ blend slider).
*   **🔍 Actionable Keyword Gap Analysis** — Grouped displays of matched and missing keywords with concrete suggestions for CV optimization.
*   **📋 JSON Report Export** — Click-to-download structured JSON scoring reports containing precise metadata, extracted entities, and sub-score metrics.
*   **🔌 FastAPI Microservice Stub** — Ready-to-go API layer structure with schemas and endpoints, suitable for production deployment.

---

## 📁 Project Structure

```
ats-cv-streamlit/
├── app.py                       # 🎨 Streamlit Application (Main UI Dashboard)
├── requirements.txt             # Project Python Dependencies
├── .env.example                 # Environment Variable Template
├── .gitignore                   # Excludes caches, secrets, and private CVs
├── LICENSE                      # MIT License
│
├── src/                         # 🧠 Core Scoring & NLP Pipeline
│   ├── __init__.py
│   ├── config.py                # Hyperparameters, Paths, and Thresholds
│   ├── parser/
│   │   ├── pdf_extractor.py     # Robust multi-column PyMuPDF extraction
│   │   └── docx_extractor.py    # python-docx Word document parser
│   ├── nlp/
│   │   ├── cleaner.py           # Tokenizer & Stopword / JD Fluff Filtering
│   │   └── entity_tagger.py     # Multi-Domain Skill Extraction & spaCy NER
│   └── scoring/
│       ├── sparse_bm25.py       # BM25 Keyword Scoring Logic
│       ├── dense_vector.py      # SentenceTransformer Vector Similarity
│       └── hybrid_engine.py     # Hybrid combination and RRF Engine
│
├── api/                         # 🔌 FastAPI Microservice (Production Endpoints)
│   ├── main.py                  # API endpoints
│   └── schemas.py               # Pydantic schemas
│
├── data/
│   ├── resumes/                 # Private folder for uploaded resumes (Git-ignored)
│   ├── job_descriptions/        # Sample JD files
│   └── taxonomy/                # Local skills catalog
│
├── docs/                        # 📚 Methodology & Blueprint PDFs
│   ├── ATS AI Engine - Project Blueprint.pdf
│   └── Advanced ATS Scoring_ Alpha Tuning & RRF.pdf
│
└── scratch/                     # 🧪 Unit Tests & Integrity Verifications
    ├── test_bm25.py             # Validates Sigmoid BM25 normalization
    └── test_hybrid.py           # Verifies mathematical properties of hybrid scoring
```

---

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/<your-username>/ats-cv-streamlit.git
cd ats-cv-streamlit
```

### 2. Create & Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> ⏱️ **Note**: On its first execution, `sentence-transformers` automatically downloads the small and efficient `all-MiniLM-L6-v2` model (~80MB) to your local cache directory.

### 4. Download spaCy NLP Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Setup Configuration
Copy the configuration template to establish environment variables:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
*(The defaults are ready for immediate use. You can edit the file to customize weights or change model names if desired)*

### 6. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser! 🎉

---

## 🧪 Testing & Verification

The core scoring engine features built-in verification scripts in the `scratch/` directory to guarantee mathematical stability and pipeline integrity:

```bash
# Run BM25 Normalization tests
python scratch/test_bm25.py

# Run Hybrid Scoring & RRF alignment tests
python scratch/test_hybrid.py
```

These scripts verify that:
1.  **Normalization bounds** are strictly adhered to ($[0,1)$ for sparse, $[0,1]$ for dense and hybrid).
2.  **No mathematical division-by-zero errors** occur even on empty, low-word, or completely non-matching documents.
3.  **Keyword scoring matches intuition** (more target skills in a resume strictly yields a higher sparse score).

---

## 🔌 FastAPI Integration (Phase 4)

To serve the scoring pipeline programmatically to a frontend or outer microservice, start the high-performance Uvicorn server:

```bash
uvicorn api.main:app --reload --port 8000
```
*   **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Endpoints**:
    *   `GET /health` — Simple health check and uptime monitor.
    *   `POST /score/text` — Returns a hybrid score based on raw string input (JSON body).
    *   `POST /score/upload` — Returns a hybrid score directly from uploaded PDF/DOCX binary documents.

---

## 🤝 Contribution & Standards

We value clean code, high performance, and mathematical soundness. When contributing:
1.  Verify that your changes pass all tests in `scratch/`.
2.  Preserve high-quality text extraction that is resistant to multi-column layout distortions.
3.  Do not commit credentials, `.env` configurations, or actual test resumes (covered under `.gitignore`).

---
