"""
src/nlp/cleaner.py — Text Normalization & Tokenization
Cleans raw resume/JD text: lowercasing, punctuation removal, stopword filtering.
Optional POS-aware tokenization via spaCy for higher-quality keyword extraction.
"""

from __future__ import annotations
import re
import string
import logging

logger = logging.getLogger(__name__)

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
    # ── Noise words identified from ATS score report analysis ──────────────
    # These leak through NLTK stopwords and corrupt missing_keywords results
    "would", "take", "new", "similar", "processes", "use", "point",
    "supports", "resumes", "asian", "multicultural",
}

def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                _STOPWORDS = set(stopwords.words("english")).union(set(stopwords.words("indonesian")))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                _STOPWORDS = set(stopwords.words("english")).union(set(stopwords.words("indonesian")))
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
                # Indonesian fallback
                "dan", "atau", "tetapi", "namun", "untuk", "dari", "ke", "di",
                "yang", "ini", "itu", "saya", "kamu", "dia", "mereka", "kita",
                "kami", "akan", "telah", "sedang", "adalah", "sebagai", "dengan",
                "pada", "dalam", "bahwa", "juga", "tidak", "bukan", "belum",
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

    # Strip unicode bullet points and resume formatting artifacts
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\x95\t\*]', ' ', text)
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


# ── spaCy model cache (loaded once per session) ───────────────────────────────
_SPACY_NLP = None
_SPACY_LOAD_FAILED = False


def _get_spacy_nlp():
    """Load and cache spaCy model for POS tagging. Returns None if unavailable."""
    global _SPACY_NLP, _SPACY_LOAD_FAILED
    if _SPACY_LOAD_FAILED:
        return None
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy
        try:
            from src.config import config
            model_name = config.SPACY_MODEL
        except Exception:
            model_name = "en_core_web_sm"
        try:
            _SPACY_NLP = spacy.load(model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. POS tokenization disabled. "
                "Run: python -m spacy download %s", model_name, model_name
            )
            _SPACY_LOAD_FAILED = True
            return None
    except ImportError:
        logger.warning("spaCy not installed. POS tokenization disabled.")
        _SPACY_LOAD_FAILED = True
        return None
    return _SPACY_NLP


# ── Lemma Overrides ─────────────────────────────────────────────────────────
# Override spaCy's lemmatization for certain tech domain terms where the lemma
# is confusing or incorrect (e.g., "data" -> "datum").
_LEMMA_OVERRIDES = {
    "datum": "data",
}


def tokenize_with_pos(
    text: str,
    remove_stopwords: bool = True,
) -> list[str]:
    """
    POS-filtered tokenization using spaCy.

    Keeps only NOUN and PROPN tokens (lemmatized), plus noun_chunks for
    multi-word domain phrases (e.g., "financial markets", "credit risk").
    Falls back to the basic `tokenize()` if spaCy is unavailable.

    Args:
        text: Input text (raw or cleaned).
        remove_stopwords: If True, filter out stopwords and fluff.

    Returns:
        List of lemmatized noun tokens + lowercased noun chunk phrases.
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        # Graceful fallback to basic tokenizer
        return tokenize(text, remove_stopwords=remove_stopwords)

    # Pre-clean the text (unicode bullets, whitespace normalization)
    cleaned = clean_text(text)
    doc = nlp(cleaned[:50000])  # spaCy token limit safety

    stopwords = _get_stopwords() if remove_stopwords else set()
    tokens: list[str] = []
    seen: set[str] = set()

    # ── Helper to get overridden lemma ────────────────────────────────────────
    def _get_lemma(token) -> str:
        lemma = token.lemma_.lower()
        return _LEMMA_OVERRIDES.get(lemma, lemma)

    # ── Extract noun chunks (multi-word domain phrases, max 3 words) ────────
    for chunk in doc.noun_chunks:
        # Lemmatize each word in the chunk, filter stopwords
        chunk_words = [
            _get_lemma(token)
            for token in chunk
            if token.pos_ in ("NOUN", "PROPN", "ADJ")
            and _get_lemma(token) not in stopwords
            and len(_get_lemma(token)) > 1
            and not token.is_stop
        ]
        # Only keep 2-3 word phrases (longer chunks are too specific to match)
        if 2 <= len(chunk_words) <= 3:
            phrase = " ".join(chunk_words)
            if phrase not in seen:
                tokens.append(phrase)
                seen.add(phrase)

    # ── Extract individual NOUN/PROPN tokens (lemmatized) ─────────────────
    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue
        lemma = _get_lemma(token)
        if len(lemma) <= 1:
            continue
        if remove_stopwords and (lemma in stopwords or token.is_stop):
            continue
        if lemma not in seen:
            tokens.append(lemma)
            seen.add(lemma)

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
