"""
src/nlp/cleaner.py — Text Normalization & Tokenization
Cleans raw resume/JD text: lowercasing, punctuation removal, stopword filtering.
"""

from __future__ import annotations
import re
import string

# --- NLTK Setup -----------------------------------------------------------
# We lazy-load NLTK stopwords to avoid import errors on first run
_STOPWORDS: set[str] | None = None


# Common JD and resume fluff that shouldn't be scored as keywords
_JD_FLUFF = {
    "ability", "acceptance", "according", "adoption", "analyzing", "artificial",
    "assessing", "assist", "bonus", "collaborate", "collaboration", "conduct",
    "contact", "contribute", "decisions", "deployed", "develop", "directly",
    "document", "domain", "ensure", "environment", "experience", "exposure",
    "fluent", "goals", "governance", "hiring", "human", "humans", "ideally",
    "improvements", "improving", "increasing", "independently", "information",
    "across", "applications", "business", "communication", "customer",
    "deployment", "development", "engineering", "english", "feature", "final",
    "improve", "internal", "made", "management", "model", "models", "projects",
    "risk", "services", "skills", "strategic", "successfully", "support",
    "team", "tools", "using", "years", "proven", "track", "record", "work",
    "working", "strong", "understanding", "prior", "knowledge", "transfer",
    "parts", "process", "reviewing", "analyzing", "assessing", "responses",
    "recruitment", "replace", "judgment", "ultimately", "processed", "please",
    "us", "may", "like", "more", "about", "how", "your", "data", "is", "we",
    "role", "join", "opportunity", "company", "candidate", "candidates",
    "requirement", "requirements", "responsibility", "responsibilities",
    "job", "description", "apply", "now", "resume", "cv", "application",
    "related", "field", "preferred", "required", "minimum", "maximum",
    "plus", "must", "good", "excellent", "written", "verbal", "highly",
    "motivated", "fast-paced", "dynamic", "inclusive", "diversity", "equal",
    "employer", "status", "disability", "veteran", "sexual", "orientation",
    "gender", "identity", "national", "origin", "color", "religion", "race",
    "age", "marital", "accommodation", "without", "regard", "basis",
    "protected", "law", "including", "pay", "salary", "benefits", "health",
    "dental", "vision", "life", "insurance", "401k", "pto", "paid", "time",
    "off", "vacation", "sick", "leave", "holiday", "holidays", "flexible",
    "remote", "hybrid", "office", "onsite", "location", "based", "located",
    "hq", "headquarters", "global", "international", "local", "regional",
    "national", "state", "city", "country", "world", "worldwide", "travel",
    "build", "building", "create", "creating", "design", "designing", "manage",
    "managing", "lead", "leading", "help", "helping", "drive", "driving",
    "deliver", "delivering", "execute", "executing", "implement", "implementing",
    "maintain", "maintaining", "operate", "operating", "optimize", "optimizing",
    "plan", "planning", "test", "testing", "troubleshoot", "troubleshooting",
    "understand", "understanding", "write", "writing", "year", "month", "day",
    "week", "hour", "minute", "second", "people", "person", "individual",
    "group", "organization", "agency", "firm", "client", "customer", "user",
    "partner", "vendor", "supplier", "contractor", "employee", "staff",
    "member", "manager", "director", "executive", "vp", "president", "ceo",
    "cto", "cfo", "coo", "cmo", "cro", "board", "committee", "council",
    "expert", "specialist", "professional", "junior", "senior", "mid-level",
    "entry-level", "intern", "internship", "apprentice", "apprenticeship",
    "part-time", "full-time", "contract", "freelance", "temporary", "permanent",
    "exempt", "non-exempt", "salary", "hourly", "wage", "commission", "bonus",
    "equity", "stock", "options", "shares", "grant", "vesting", "cliff",
    "base", "variable", "target", "uncapped", "quota", "atb", "mbo", "okr",
    "kpi", "roi", "pnl", "ebitda", "margin", "revenue", "profit", "loss",
    "growth", "scale", "scaling", "expand", "expanding", "market", "industry",
    "sector", "vertical", "horizontal", "b2b", "b2c", "d2c", "smb", "mid-market",
    "enterprise", "startup", "scale-up", "unicorn", "fortune", "global", "amar", "bank",
}

def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                _STOPWORDS = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                _STOPWORDS = set(stopwords.words("english"))
        except ImportError:
            # Fallback minimal stopword list if NLTK not available
            _STOPWORDS = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                "you", "your", "yours", "he", "him", "his", "she", "her",
                "it", "its", "they", "them", "their", "what", "which", "who",
                "whom", "this", "that", "these", "those", "am", "is", "are",
                "was", "were", "be", "been", "being", "have", "has", "had",
                "do", "does", "did", "will", "would", "shall", "should",
                "may", "might", "must", "can", "could", "a", "an", "the",
                "and", "but", "or", "nor", "for", "yet", "so", "in", "on",
                "at", "to", "by", "with", "of", "from", "as",
            }
        
        # Add JD fluff words
        _STOPWORDS.update(_JD_FLUFF)
    return _STOPWORDS


def clean_text(text: str) -> str:
    """
    Normalize raw text for NLP processing:
      - Lowercase
      - Remove special characters (keep alphanumeric + spaces)
      - Collapse whitespace

    Args:
        text: Raw extracted text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Remove punctuation except hyphens (useful for compound terms like "full-stack")
    text = re.sub(r"[^\w\s\-+#]", " ", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Tokenize cleaned text into a list of meaningful words.

    Args:
        text: Input text (ideally already cleaned via clean_text).
        remove_stopwords: If True, filter out common English stopwords.

    Returns:
        List of token strings.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()

    if remove_stopwords:
        stopwords = _get_stopwords()
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

    return tokens


def extract_keyword_overlap(
    resume_tokens: list[str],
    jd_tokens: list[str],
) -> dict[str, list[str]]:
    """
    Find keyword overlap between resume and job description tokens.

    Args:
        resume_tokens: Tokenized resume text.
        jd_tokens: Tokenized job description text.

    Returns:
        Dict with keys:
          - "matched": keywords in JD that appear in resume
          - "missing": keywords in JD that are absent from resume
    """
    resume_set = set(resume_tokens)
    jd_set = set(jd_tokens)

    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)

    return {"matched": matched, "missing": missing}
