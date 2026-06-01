"""
src/nlp/entity_tagger.py — Named Entity Recognition for Resumes
Uses spaCy to extract skills, organizations, dates, and infer years of experience.
Falls back gracefully if spaCy model is not installed.

Revisions per ATS NLP Pipeline Implementation Plan:
  - Task 3: EntityRuler for SKILL overrides + ORG post-processing filter
  - Task 4: Deterministic experience calculator from date ranges
"""

from __future__ import annotations
import re
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

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


# ── ORG Exclusion Regex (Task 3b) ─────────────────────────────────────────
# Matches academic metrics, bullet fragments, and known false-positive ORGs
_ORG_EXCLUSION_REGEX = re.compile(
    r'(?i)('
    r'bachelor|master|degree|gpa|score|idr'
    r'|\btoefl\b|\bielts\b'
    r'|\baverage\b|\bavergae\b'     # Typo in resume ("Avergae Score")
    r'|\bscopus\b'
    r'|\bundergraduate\b'
    r'|\bcertification\b'
    r'|\bskill\b|\bskills\b'        # Resume section headers
    r'|\bprofile\b'                  # "Business Profile", "Health Profile"
    r'|\bdata\b\s+\w*\s*skill'       # "Data Analytics Skills"
    r'|\bthinking\b'                 # "Logical Thinking"
    r'|\bachievement\b'              # "Student Activities Unit's Achievement"
    r')',
)

# Patterns that indicate an entity is a PDF/resume parsing artifact
_ORG_ARTIFACT_REGEX = re.compile(
    r'\n'                 # Contains newlines (multi-line junk from PDF)
    r'|^\s*[\u2022\u2023\u25E6\u2043\*\-]'  # Starts with bullet
    r'|^[\W\s]*$'         # Only symbols/whitespace
    r'|\|'                # Contains pipe (e.g., "AI|")
)

# Known technical terms that spaCy often misclassifies as ORG
# Includes DOMAIN_SKILLS_KEYWORDS + additional statistical/ML abbreviations
_SKILL_AS_ORG_SET = {
    s.lower() for s in DOMAIN_SKILLS_KEYWORDS
    if len(s) > 2  # Skip very short ones like "r", "go"
}
# Additional abbreviations and tools not in the main skills set but still not ORGs
_SKILL_AS_ORG_SET.update({
    "eda", "eviews", "spss", "svr", "lgd", "arima", "arima-ann",
    "multivariate", "convolutional neural network", "k-means",
    "probability of default", "customer behavior", "inflation rate",
    "k-means cluster algorithms", "fourier series estimator",
    "statistical modeling & inferences", "risk modeling",
    "team management", "employee data", "deeplearning",
    "freelance data scientist",
})


def _filter_organizations(org_list: list[str]) -> list[str]:
    """
    Post-processing filter for ORG entities (Task 3b+3c).
    Drops academic metrics, bullet fragments, known skills, and short noise.
    """
    filtered = []
    for org in org_list:
        text = org.strip()
        # 3c: Skip very short or empty entities
        if len(text) < 3:
            continue
        # 3c: Skip entities with newlines (PDF multi-line artifacts)
        if _ORG_ARTIFACT_REGEX.search(text):
            continue
        # 3b: Skip academic metrics and false positives
        if _ORG_EXCLUSION_REGEX.search(text):
            continue
        # 3a: Skip known technical terms misclassified as ORG
        if text.lower() in _SKILL_AS_ORG_SET:
            continue
        filtered.append(text)
    return filtered


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

    # ── Rule-based skill extraction (always runs) ──────────────────────
    result.skills = _extract_skills_rule_based(text)

    # ── Years of experience (chronological date-range + regex) ───────────
    result.years_of_experience = _extract_years_of_experience(text)

    # ── spaCy NER (optional, for ORG, GPE, DATE entities) ─────────────
    if use_spacy:
        try:
            spacy_entities = _extract_with_spacy(text)
            # Task 3b: Post-filter ORG entities to remove noise
            raw_orgs = spacy_entities.get("ORG", [])
            result.organizations = _filter_organizations(raw_orgs)
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


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Deterministic Experience Calculator
# ─────────────────────────────────────────────────────────────────────────────

# Date range patterns for employment history
# Matches: "January 2021 - Present", "Mar 2019 – Dec 2020", "Jan. 2018 - Jun 2021"
_DATE_RANGE_MONTH_YEAR = re.compile(
    r'([A-Za-z]+\.?\s+\d{4})\s*[-\u2013\u2014]\s*'
    r'([A-Za-z]+\.?\s+\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow|[Ss]ekarang)',
    re.IGNORECASE,
)

# Year-only ranges: "2018 - 2021", "2019–2023", "2020 - Present"
_DATE_RANGE_YEAR_ONLY = re.compile(
    r'(?<!\d)(\d{4})\s*[-\u2013\u2014]\s*(\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow|[Ss]ekarang)(?!\d)',
    re.IGNORECASE,
)

# Months mapping for manual parsing fallback
_MONTH_MAP = {
    'jan': 1, 'january': 1, 'januari': 1,
    'feb': 2, 'february': 2, 'februari': 2,
    'mar': 3, 'march': 3, 'maret': 3,
    'apr': 4, 'april': 4,
    'may': 5, 'mei': 5,
    'jun': 6, 'june': 6, 'juni': 6,
    'jul': 7, 'july': 7, 'juli': 7,
    'aug': 8, 'august': 8, 'agustus': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'oktober': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12, 'desember': 12,
}


def _parse_date_string(date_str: str) -> datetime | None:
    """
    Parse a date string like 'January 2021', 'Mar 2019', 'Present' into datetime.
    Returns None if parsing fails.
    """
    date_str = date_str.strip().lower().rstrip('.')

    # Handle 'Present', 'Current', 'Now', 'Sekarang' (Indonesian)
    if date_str in ('present', 'current', 'now', 'sekarang'):
        return datetime.now()

    # Try dateutil first (best fuzzy parser)
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(date_str, default=datetime(2000, 1, 1))
    except Exception:
        pass

    # Manual fallback: "Month Year" pattern
    parts = date_str.split()
    if len(parts) == 2:
        month_str, year_str = parts
        month = _MONTH_MAP.get(month_str.rstrip('.'), None)
        try:
            year = int(year_str)
            if month and 1900 <= year <= 2100:
                return datetime(year, month, 1)
        except ValueError:
            pass

    # Year-only: "2021"
    try:
        year = int(date_str)
        if 1900 <= year <= 2100:
            return datetime(year, 1, 1)
    except ValueError:
        pass

    return None


def _merge_overlapping_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """
    Merge overlapping date intervals to prevent double-counting concurrent roles.

    Args:
        intervals: List of (start, end) datetime tuples.

    Returns:
        List of merged non-overlapping intervals.
    """
    if not intervals:
        return []

    # Sort by start date
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping — extend the previous interval
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _calculate_years_from_date_ranges(text: str) -> float | None:
    """
    Extract date ranges from resume text and calculate total years of experience.
    Handles overlapping intervals by merging them before summing.

    Returns:
        Total years of experience (float), or None if no date ranges found.
    """
    intervals: list[tuple[datetime, datetime]] = []

    # Try month+year ranges first (higher precision)
    for match in _DATE_RANGE_MONTH_YEAR.finditer(text):
        start_str, end_str = match.group(1), match.group(2)
        start_dt = _parse_date_string(start_str)
        end_dt = _parse_date_string(end_str)
        if start_dt and end_dt and start_dt < end_dt:
            intervals.append((start_dt, end_dt))

    # Also try year-only ranges
    for match in _DATE_RANGE_YEAR_ONLY.finditer(text):
        start_str, end_str = match.group(1), match.group(2)
        start_dt = _parse_date_string(start_str)
        end_dt = _parse_date_string(end_str)
        if start_dt and end_dt and start_dt < end_dt:
            # Only add if not already covered by a month+year range
            already_covered = any(
                s <= start_dt and end_dt <= e for s, e in intervals
            )
            if not already_covered:
                intervals.append((start_dt, end_dt))

    if not intervals:
        return None

    # Merge overlapping intervals (prevents double-counting concurrent roles)
    merged = _merge_overlapping_intervals(intervals)

    # Sum total days across all non-overlapping intervals
    total_days = sum((end - start).days for start, end in merged)
    total_years = round(total_days / 365.25, 1)

    return total_years if total_years > 0 else None


def _extract_years_explicit_mention(text: str) -> float | None:
    """
    Parse explicit mentions like '5+ years', '3-5 years', '2 years of experience'.
    Returns the maximum found value (most senior claim).
    """
    patterns = [
        r"(\d+)\s*\+?\s*(?:years?|tahun)\s*(?:of\s+)?(?:experience|exp|pengalaman)",
        r"(\d+)\s*[-\u2013]\s*\d+\s*(?:years?|tahun)",
        r"(?:over|lebih\s+dari)\s+(\d+)\s*(?:years?|tahun)",
        r"(\d+)\s*(?:years?|tahun)\s*(?:of\s+)?(?:professional|working|work|kerja|pengalaman)",
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


def _extract_years_of_experience(text: str) -> float | None:
    """
    Calculate years of experience using a two-pass strategy:

    1. Primary: Extract date ranges from employment history ("Jan 2019 - Present")
       and calculate total duration with overlap merging.
    2. Fallback: Match explicit mentions ("5+ years of experience").

    Returns the higher of the two values (if both found), or whichever is available.
    """
    # Primary: Chronological date-range calculation (Task 4)
    date_based = _calculate_years_from_date_ranges(text)

    # Fallback: Explicit mention regex (original approach)
    explicit = _extract_years_explicit_mention(text)

    if date_based is not None and explicit is not None:
        return max(date_based, explicit)
    return date_based or explicit


# ─────────────────────────────────────────────────────────────────────────────
# Task 3a: spaCy NER with EntityRuler
# ─────────────────────────────────────────────────────────────────────────────

_NER_NLP_CACHE = None


def _build_entity_ruler_patterns() -> list[dict]:
    """
    Build EntityRuler patterns from DOMAIN_SKILLS_KEYWORDS.
    Forces known technical terms to be labeled as SKILL instead of ORG.
    """
    patterns = []
    for skill in DOMAIN_SKILLS_KEYWORDS:
        if " " in skill:
            # Multi-word: split into token-level pattern
            pattern = [{"LOWER": w} for w in skill.split()]
        else:
            pattern = [{"LOWER": skill}]
        patterns.append({"label": "SKILL", "pattern": pattern})
    return patterns


def _extract_with_spacy(text: str) -> dict[str, list[str]]:
    """
    Run spaCy NER with EntityRuler pre-processing and return entity lists
    grouped by label. The EntityRuler forces known tech terms to be tagged
    as SKILL, preventing misclassification as ORG.
    """
    global _NER_NLP_CACHE
    import spacy
    from src.config import config

    if _NER_NLP_CACHE is None:
        try:
            nlp = spacy.load(config.SPACY_MODEL)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{config.SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {config.SPACY_MODEL}"
            )

        # Task 3a: Add EntityRuler before NER to override known skills
        try:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(_build_entity_ruler_patterns())
            logger.info("EntityRuler added with %d skill patterns.", len(DOMAIN_SKILLS_KEYWORDS))
        except Exception as e:
            logger.warning("Could not add EntityRuler: %s", e)

        _NER_NLP_CACHE = nlp

    nlp = _NER_NLP_CACHE

    # Pre-clean: strip unicode bullets to prevent multi-line entity spans
    cleaned_text = re.sub(r'[\u2022\u2023\u25E6\u2043\x95]', ' ', text)

    # spaCy has a token limit; truncate for safety
    doc = nlp(cleaned_text[:50000])
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        label_text = ent.text.strip()
        if label_text and label_text not in entities[ent.label_]:
            entities[ent.label_].append(label_text)
    return entities
