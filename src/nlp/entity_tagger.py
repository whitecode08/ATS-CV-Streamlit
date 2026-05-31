"""
src/nlp/entity_tagger.py — Named Entity Recognition for Resumes
Uses spaCy to extract skills, organizations, dates, and infer years of experience.
Falls back gracefully if spaCy model is not installed.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

# Domain skill keywords for rule-based tagging (supplements spaCy NER)
DOMAIN_SKILLS_KEYWORDS = {
    # Software Engineering & Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "go",
    "rust", "swift", "kotlin", "r", "scala", "php", "ruby", "bash", "dart", "objective-c", "matlab", "perl",
    "html", "css", "assembly", "vba", "groovy", "lua", "shell scripting",
    
    # ML / AI / Data Science
    "machine learning", "deep learning", "nlp", "computer vision", "artificial intelligence", "ai",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm", "catboost",
    "transformers", "bert", "gpt", "llm", "rag", "vertex ai", "mlops", "huggingface", "opencv", 
    "predictive modeling", "feature engineering", "data science", "neural networks", "generative ai",
    "reinforcement learning", "supervised learning", "unsupervised learning", "time series analysis",
    "spacy", "nltk",
    
    # Data Engineering & Databases
    "pandas", "numpy", "sql", "nosql", "spark", "hadoop", "dbt", "pyspark",
    "airflow", "kafka", "elasticsearch", "snowflake", "bigquery", "redshift", "databricks", "etl", "elt", 
    "data warehouse", "data pipeline", "data lake", "data mining",
    "postgresql", "mysql", "mongodb", "redis", "cassandra", "oracle", "sql server", "dynamodb",
    "mariadb", "neo4j", "couchbase", "firebase", "sqlite",
    
    # Cloud & DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd",
    "git", "github", "gitlab", "jenkins", "ansible", "linux", "unix", "bash scripting",
    "puppet", "chef", "circleci", "travis ci", "bitbucket", "prometheus", "grafana", "splunk",
    "datadog", "new relic", "agile", "scrum", "kanban", "sre", "site reliability engineering",
    
    # Web & Mobile Development
    "react", "next.js", "vue", "angular", "fastapi", "django", "flask", "express",
    "node.js", "restful api", "graphql", "spring boot", "react native", "flutter",
    "svelte", "jquery", "bootstrap", "tailwind css", "material ui", "redux", "webpack", "babel",
    "android", "ios", "xamarin", "ionic",
    
    # Analytics & Business Intelligence
    "tableau", "power bi", "looker", "excel", "google analytics", "metabase", "data visualization", "a/b testing",
    "qlikview", "sisense", "domo", "mixpanel", "amplitude", "segment", "adobe analytics", "google tag manager",
    "business intelligence", "bi", "dashboarding", "reporting",
    
    # Finance, Accounting & Banking
    "credit risk", "risk management", "financial services", "lending", "banking", "regulatory", "compliance", 
    "credit scoring", "fintech", "wealth management", "capital markets", "anti-money laundering", "aml", "kyc",
    "financial modeling", "forecasting", "budgeting", "valuation", "gaap", "ifrs", "taxation", "auditing",
    "accounts payable", "accounts receivable", "reconciliation", "general ledger", "cpa", "cfa", "asset management",
    "investment banking", "private equity", "venture capital", "mergers and acquisitions", "m&a",
    
    # Marketing, Sales & PR
    "seo", "sem", "crm", "salesforce", "hubspot", "content marketing", "digital marketing", "b2b", "b2c", "lead generation",
    "email marketing", "social media marketing", "ppc", "google ads", "facebook ads", "copywriting", "public relations",
    "pr", "brand management", "market research", "marketing strategy", "sales operations", "account management",
    "cold calling", "b2b sales", "b2c sales", "customer success", "customer retention", "churn reduction",
    
    # Design, UX & Creative
    "figma", "sketch", "adobe xd", "ui/ux", "user research", "wireframing", "prototyping",
    "adobe creative cloud", "photoshop", "illustrator", "indesign", "premiere pro", "after effects",
    "graphic design", "visual design", "interaction design", "user experience", "user interface", "usability testing",
    "information architecture", "typography", "color theory", "animation", "video editing",
    
    # HR, Recruiting & People Ops
    "talent acquisition", "recruiting", "sourcing", "onboarding", "employee relations", "performance management",
    "payroll", "benefits administration", "hris", "workday", "bamboo hr", "adp", "greenhouse", "lever",
    "employee engagement", "diversity and inclusion", "dei", "compensation", "talent management", "succession planning",
    
    # Operations, Supply Chain & Manufacturing
    "supply chain management", "logistics", "procurement", "inventory management", "six sigma", "lean manufacturing",
    "erp", "sap", "oracle e-business suite", "vendor management", "quality assurance", "qa", "quality control", "qc",
    "continuous improvement", "kaizen", "supply chain optimization", "freight forwarding", "warehousing",
    
    # Product & Project Management
    "product management", "product strategy", "product roadmap", "agile methodologies", "scrum master",
    "project management", "pmp", "prince2", "jira", "trello", "asana", "confluence", "stakeholder management",
    "cross-functional leadership", "go-to-market strategy", "user stories", "backlog grooming",
    
    # Legal, Compliance & Governance
    "corporate law", "intellectual property", "contract negotiation", "compliance", "regulatory affairs",
    "gdpr", "ccpa", "data privacy", "corporate governance", "litigation", "employment law", "mergers & acquisitions",
    
    # Healthcare, Medical & Science
    "patient care", "clinical research", "hipaa", "emr", "epic", "cerner", "medical billing", "medical coding",
    "pharmacovigilance", "fda regulations", "clinical trials", "biostatistics", "epidemiology", "public health",
    "nursing", "triage", "healthcare administration", "life sciences",
    
    # Soft Skills & Miscellaneous
    "communication", "leadership", "problem solving", "teamwork", "critical thinking", "time management",
    "adaptability", "conflict resolution", "emotional intelligence", "decision making", "creativity",
    "strategic planning", "cross-functional", "mentorship", "public speaking", "negotiation", "presentation skills",
    "customer service", "troubleshooting",
    
    # Specialized Data & Visualization Tools
    "streamlit", "jupyter", "matplotlib", "seaborn", "plotly", "dash", "bokeh", "altair", "d3.js", "shiny"
}


@dataclass
class ExtractedEntities:
    """Container for all entities extracted from a resume."""
    skills: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    years_of_experience: float | None = None
    raw_text_length: int = 0


def extract_entities(text: str, use_spacy: bool = True) -> ExtractedEntities:
    """
    Extract named entities and skills from resume text.

    Args:
        text: Raw or cleaned resume text.
        use_spacy: If True, attempt to use spaCy NER (falls back to rule-based).

    Returns:
        ExtractedEntities dataclass with all found entities.
    """
    result = ExtractedEntities(raw_text_length=len(text))

    # ── Rule-based skill extraction (always runs) ──────────────────────────
    result.skills = _extract_skills_rule_based(text)

    # ── Years of experience (regex pattern) ────────────────────────────────
    result.years_of_experience = _extract_years_of_experience(text)

    # ── spaCy NER (optional, for ORG, GPE, DATE entities) ─────────────────
    if use_spacy:
        try:
            spacy_entities = _extract_with_spacy(text)
            result.organizations = spacy_entities.get("ORG", [])
            result.locations = spacy_entities.get("GPE", [])
            result.dates = spacy_entities.get("DATE", [])
        except Exception:
            pass  # spaCy not installed or model missing — silently skip

    return result


def _extract_skills_rule_based(text: str) -> list[str]:
    """Match domain skill keywords (case-insensitive) against the text."""
    text_lower = text.lower()
    found = set()
    for skill in DOMAIN_SKILLS_KEYWORDS:
        # Use word boundary for single-word skills, substring for multi-word
        if " " in skill:
            if skill in text_lower:
                found.add(skill.title())
        else:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                found.add(skill.upper() if len(skill) <= 4 else skill.title())
    return sorted(found)


def _extract_years_of_experience(text: str) -> float | None:
    """
    Parse patterns like '5+ years', '3-5 years', '2 years of experience'.
    Returns the maximum found value (most senior claim).
    """
    patterns = [
        r"(\d+)\s*\+?\s*years?\s*(?:of\s+)?(?:experience|exp)",
        r"(\d+)\s*[-–]\s*\d+\s*years?",
        r"over\s+(\d+)\s*years?",
        r"(\d+)\s*years?\s*(?:of\s+)?(?:professional|working|work)",
    ]
    found_years: list[float] = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for m in matches:
            try:
                found_years.append(float(m))
            except ValueError:
                pass
    return max(found_years) if found_years else None


def _extract_with_spacy(text: str) -> dict[str, list[str]]:
    """Run spaCy NER and return entity lists grouped by label."""
    import spacy
    from src.config import config

    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{config.SPACY_MODEL}' not found. "
            f"Run: python -m spacy download {config.SPACY_MODEL}"
        )

    # spaCy has a token limit; truncate for safety
    doc = nlp(text[:50000])
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        label_text = ent.text.strip()
        if label_text and label_text not in entities[ent.label_]:
            entities[ent.label_].append(label_text)
    return entities
