"""
app.py — ATS AI Engine | Beautiful Streamlit Application
Hybrid ATS scoring: BM25 sparse + SentenceTransformer dense scoring.
"""

import sys
import os
import io
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="ATS AI Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-username/ats-cv-streamlit",
        "Report a bug": "https://github.com/your-username/ats-cv-streamlit/issues",
        "About": "ATS AI Engine — Hybrid resume scoring powered by BM25 + SentenceTransformers",
    },
)

# ── Project root on path ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset & Theme ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1530 50%, #0a0f1e 100%);
    min-height: 100vh;
}

/* ── Main content padding ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1530 0%, #0a0f1e 100%) !important;
    border-right: 1px solid rgba(0, 212, 180, 0.15) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00d4b4 !important;
}

/* Glass containers for Sidebar parameters */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stSidebar"] div[style*="border"] {
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2) !important;
}

/* Glass containers for Main layout widgets */
.main [data-testid="stVerticalBlockBorderWrapper"],
.main div[style*="border"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15) !important;
    transition: all 0.3s ease !important;
}
.main [data-testid="stVerticalBlockBorderWrapper"]:hover,
.main div[style*="border"]:hover {
    border-color: rgba(0, 212, 180, 0.25) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 212, 180, 0.08) !important;
}

/* ── Headers ── */
h1 { 
    color: #ffffff !important; 
    font-weight: 800 !important; 
    letter-spacing: -0.5px;
}
h2, h3 { 
    color: #e2e8f0 !important; 
    font-weight: 600 !important; 
}

/* ── Sleek Flex Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(0, 212, 180, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%);
    border: 1px solid rgba(0, 212, 180, 0.15);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-content {
    flex: 1;
    min-width: 300px;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4b4 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -0.8px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.25rem;
    font-weight: 400;
}
.hero-badges-container {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: #94a3b8;
    padding: 4px 10px;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 500;
    transition: all 0.2s ease;
}
.hero-badge:hover {
    border-color: rgba(0, 212, 180, 0.3);
    color: #00d4b4;
    background: rgba(0, 212, 180, 0.05);
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(0,212,180,0.3);
    transform: translateY(-1px);
}

/* ── Score Gauge Container ── */
.score-container {
    background: linear-gradient(135deg, rgba(0,212,180,0.08) 0%, rgba(124,58,237,0.08) 100%);
    border: 1px solid rgba(0,212,180,0.2);
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.score-value {
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #00d4b4 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.score-label-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 0.75rem;
}
.score-strong { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.score-moderate { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.score-weak { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover { border-color: rgba(0,212,180,0.4); }
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4b4;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}
.metric-sublabel {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.2rem;
}

/* ── Keyword Chips ── */
.chip-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.75rem; }
.chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    transition: transform 0.15s ease;
}
.chip:hover { transform: scale(1.05); }
.chip-matched { 
    background: rgba(16,185,129,0.12); 
    color: #34d399; 
    border: 1px solid rgba(16,185,129,0.25); 
}
.chip-missing { 
    background: rgba(239,68,68,0.1); 
    color: #f87171; 
    border: 1px solid rgba(239,68,68,0.2); 
}
.chip-skill { 
    background: rgba(124,58,237,0.12); 
    color: #a78bfa; 
    border: 1px solid rgba(124,58,237,0.25); 
}

/* ── Streamlit Widgets Overrides ── */
.stButton button {
    background: linear-gradient(135deg, #00d4b4, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0,212,180,0.25) !important;
}
.stButton button:hover {
    box-shadow: 0 6px 30px rgba(0,212,180,0.4) !important;
    transform: translateY(-1px) !important;
}

.stSlider [data-testid="stSlider"] { padding: 0.5rem 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,180,0.2), rgba(124,58,237,0.2)) !important;
    color: #00d4b4 !important;
    font-weight: 600 !important;
}

/* Text Area */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0,212,180,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,212,180,0.1) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(0,212,180,0.3) !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,180,0.6) !important;
    background: rgba(0,212,180,0.05) !important;
}

/* Expanders */
details {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Section labels */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #00d4b4;
    margin-bottom: 0.5rem;
}

/* Info boxes */
.info-box {
    background: rgba(0,212,180,0.06);
    border-left: 3px solid #00d4b4;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #94a3b8;
}

/* Warning box */
.warn-box {
    background: rgba(245,158,11,0.06);
    border-left: 3px solid #f59e0b;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #94a3b8;
}

/* Spinner override */
.stSpinner > div {
    border-top-color: #00d4b4 !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4b4, #7c3aed) !important;
    border-radius: 4px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(0,212,180,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,180,0.5); }

/* Select box */
.stSelectbox select, [data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
}

/* Markdown text */
.stMarkdown p { color: #94a3b8; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_sentence_model(model_name: str):
    """Cache the SentenceTransformer model across reruns."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def parse_uploaded_file(uploaded_file) -> str:
    """Extract text from uploaded PDF or DOCX file."""
    from src.parser.pdf_extractor import extract_text_from_pdf
    from src.parser.docx_extractor import extract_text_from_docx

    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(io.BytesIO(file_bytes))
    elif name.endswith((".docx", ".doc")):
        return extract_text_from_docx(io.BytesIO(file_bytes))
    else:
        st.error("Unsupported file type. Please upload a PDF or DOCX file.")
        return ""


def load_sample_jds() -> dict[str, str]:
    """Load all .txt JD files from data/job_descriptions/."""
    jd_dir = ROOT / "data" / "job_descriptions"
    jds = {}
    if jd_dir.exists():
        for f in sorted(jd_dir.glob("*.txt")):
            jds[f.stem.replace("_", " ").title()] = f.read_text(encoding="utf-8")
    return jds


def make_gauge_chart(score: float, label: str, color: str) -> go.Figure:
    """Create a sleek Plotly gauge chart for the score."""
    color_map = {
        "green": "#10b981",
        "orange": "#f59e0b",
        "red": "#ef4444",
    }
    bar_color = color_map.get(color, "#00d4b4")

    # Gradient steps
    steps = [
        {"range": [0, 45], "color": "rgba(239,68,68,0.15)"},
        {"range": [45, 70], "color": "rgba(245,158,11,0.15)"},
        {"range": [70, 100], "color": "rgba(16,185,129,0.15)"},
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={
            "suffix": "%",
            "font": {"size": 52, "color": bar_color, "family": "Inter"},
        },
        title={
            "text": f"<b>{label}</b>",
            "font": {"size": 15, "color": "#94a3b8", "family": "Inter"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.15)",
                "tickfont": {"color": "#475569", "size": 11},
            },
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.8,
                "value": round(score * 100, 1),
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=30, r=30),
        height=280,
        font={"family": "Inter"},
    )
    return fig


def make_score_comparison_chart(sparse: float, dense: float, hybrid: float) -> go.Figure:
    """Horizontal bar chart comparing all three scores."""
    categories = ["🔑 BM25 Sparse", "🤖 Dense Semantic", "⚖️ Hybrid Score"]
    values = [round(sparse * 100, 1), round(dense * 100, 1), round(hybrid * 100, 1)]
    colors = ["#7c3aed", "#00d4b4", "#f59e0b"]

    fig = go.Figure()
    for cat, val, col in zip(categories, values, colors):
        fig.add_trace(go.Bar(
            x=[val],
            y=[cat],
            orientation="h",
            marker=dict(
                color=col,
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[f"{val}%"],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=13, family="Inter"),
            hovertemplate=f"<b>{cat}</b><br>Score: {val}%<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            ticksuffix="%",
            tickfont=dict(color="#475569", size=11),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=12, family="Inter"),
            showgrid=False,
        ),
        margin=dict(t=10, b=10, l=10, r=60),
        height=220,
        barmode="group",
        bargap=0.25,
    )
    return fig


def render_keyword_chips(keywords: list[str], chip_class: str, max_show: int = 40) -> str:
    """Render keywords as HTML chip badges."""
    if not keywords:
        return "<p style='color:#475569;font-size:0.85rem'>None found</p>"
    display = keywords[:max_show]
    chips_html = "".join(
        f'<span class="chip {chip_class}">{kw}</span>'
        for kw in display
    )
    extra = ""
    if len(keywords) > max_show:
        extra = f'<span class="chip" style="background:rgba(255,255,255,0.06);color:#64748b;border:1px solid rgba(255,255,255,0.1)">+{len(keywords)-max_show} more</span>'
    return f'<div class="chip-container">{chips_html}{extra}</div>'


def score_label_badge(label: str, color: str) -> str:
    css_class = {
        "green": "score-strong",
        "orange": "score-moderate",
        "red": "score-weak",
    }.get(color, "score-moderate")
    icons = {"green": "✅", "orange": "⚠️", "red": "❌"}
    icon = icons.get(color, "⚠️")
    return f'<span class="score-label-badge {css_class}">{icon} {label}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem 0; border-bottom:1px solid rgba(0,212,180,0.15); margin-bottom:1.5rem">
        <div style="font-size:1.6rem; font-weight:800; background:linear-gradient(135deg,#00d4b4,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            🎯 ATS AI Engine
        </div>
        <div style="color:#475569;font-size:0.8rem;margin-top:4px">Hybrid Resume Scorer v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="section-label">⚖️ Scoring Weights</p>', unsafe_allow_html=True)
        alpha = st.slider(
            "Alpha — BM25 Keyword Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="0.0 = Pure Semantic, 1.0 = Pure Keyword. Default 0.6 favors exact keywords per HR Tech best practice.",
            key="alpha_slider",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="text-align:center;padding:8px;background:rgba(124,58,237,0.1);border-radius:10px;border:1px solid rgba(124,58,237,0.2)">
                <div style="font-size:1.3rem;font-weight:700;color:#a78bfa">{int(alpha*100)}%</div>
                <div style="font-size:0.7rem;color:#64748b;margin-top:2px">BM25</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="text-align:center;padding:8px;background:rgba(0,212,180,0.1);border-radius:10px;border:1px solid rgba(0,212,180,0.2)">
                <div style="font-size:1.3rem;font-weight:700;color:#00d4b4">{int((1-alpha)*100)}%</div>
                <div style="font-size:0.7rem;color:#64748b;margin-top:2px">Dense</div>
            </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="section-label">🤖 Model Settings</p>', unsafe_allow_html=True)
        model_name = st.selectbox(
            "SentenceTransformer Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2",
            ],
            index=0,
            help="all-MiniLM-L6-v2 is fast (~80MB). all-mpnet-base-v2 is more accurate but larger.",
        )

        use_ner = st.toggle(
            "Enable NER Extraction",
            value=True,
            help="Extract skills and experience using spaCy. Disable if spaCy model not installed.",
        )

    with st.container(border=True):
        st.markdown('<p class="section-label">📂 Input Source</p>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Resume Input Mode",
            options=["📤 Upload File", "✍️ Paste Text"],
            label_visibility="collapsed",
        )

    st.markdown("""
    <div style="margin-top:2rem;padding-top:1.5rem;border-top:1px solid rgba(255,255,255,0.06)">
        <div style="font-size:0.75rem;color:#334155;line-height:1.8">
            <div>🔑 BM25 — Keyword matching</div>
            <div>🤖 Dense — Semantic similarity</div>
            <div>⚖️ Hybrid — Weighted combination</div>
            <br>
            <div style="color:#1e293b">Score Legend</div>
            <div><span style="color:#10b981">●</span> ≥ 70% Strong Match</div>
            <div><span style="color:#f59e0b">●</span> 45–69% Moderate</div>
            <div><span style="color:#ef4444">●</span> &lt; 45% Weak Match</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <div class="hero-content">
        <div class="hero-title">🎯 ATS AI Engine</div>
        <div class="hero-subtitle">Hybrid Resume Scoring • BM25 Keyword + Semantic AI Analysis</div>
    </div>
    <div class="hero-badges-container">
        <span class="hero-badge">⚡ BM25 Sparse</span>
        <span class="hero-badge">🧠 SentenceTransformers</span>
        <span class="hero-badge">🔀 Hybrid Scoring</span>
        <span class="hero-badge">📊 NER Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_upload, tab_results, tab_ner, tab_about = st.tabs([
    "📄 Input & Score",
    "📊 Score Dashboard",
    "🧠 NER Analysis",
    "ℹ️ About",
])


# ── TAB 1: Input & Score ──────────────────────────────────────────────────────
with tab_upload:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Resume Input ─────────────────────────────────────────────────
    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-label" style="font-size:0.9rem;margin-bottom:1rem">📄 Resume Input</p>', unsafe_allow_html=True)

            resume_text = ""

            if input_mode == "📤 Upload File":
                uploaded = st.file_uploader(
                    "Drop your resume here",
                    type=["pdf", "docx", "doc"],
                    label_visibility="collapsed",
                    key="resume_uploader",
                )
                if uploaded:
                    with st.spinner("📖 Parsing resume..."):
                        resume_text = parse_uploaded_file(uploaded)
                    if resume_text:
                        st.success(f"✅ Parsed **{uploaded.name}**")
                        with st.expander("👁️ Preview extracted text"):
                            st.markdown(f"""
                            <div style="background:rgba(0,0,0,0.2);border-radius:10px;padding:1rem;
                                        max-height:200px;overflow-y:auto;font-size:0.82rem;
                                        color:#94a3b8;font-family:'JetBrains Mono',monospace;
                                        white-space:pre-wrap;line-height:1.6;">
                            {resume_text[:1500]}{'...' if len(resume_text) > 1500 else ''}
                            </div>""", unsafe_allow_html=True)
            else:
                resume_text = st.text_area(
                    "Paste resume text",
                    height=300,
                    placeholder="Paste your resume content here...\n\nExample:\nJohn Doe | Senior Data Scientist\n5+ years of experience in Python, Machine Learning, NLP...",
                    label_visibility="collapsed",
                    key="resume_textarea",
                )
                if resume_text:
                    st.caption(f"📝 {len(resume_text):,} characters")

    # ── RIGHT: Job Description ─────────────────────────────────────────────
    with col_right:
        with st.container(border=True):
            st.markdown('<p class="section-label" style="font-size:0.9rem;margin-bottom:1rem">💼 Job Description</p>', unsafe_allow_html=True)

            # Sample JD loader
            sample_jds = load_sample_jds()
            jd_text = ""

            if sample_jds:
                jd_source = st.radio(
                    "JD Source",
                    ["📂 Load Sample JD", "✍️ Paste Custom JD"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="jd_source_radio",
                )
                if jd_source == "📂 Load Sample JD":
                    selected_jd = st.selectbox(
                        "Select a sample job description",
                        options=list(sample_jds.keys()),
                        label_visibility="collapsed",
                        key="jd_selector",
                    )
                    jd_text = sample_jds[selected_jd]
                    st.caption(f"📋 {len(jd_text):,} characters")
                    with st.expander("👁️ Preview Job Description"):
                        st.markdown(f"""
                        <div style="background:rgba(0,0,0,0.2);border-radius:10px;padding:1rem;
                                    max-height:200px;overflow-y:auto;font-size:0.82rem;
                                    color:#94a3b8;line-height:1.6;">
                        {jd_text[:800]}{'...' if len(jd_text) > 800 else ''}
                        </div>""", unsafe_allow_html=True)
                else:
                    jd_text = st.text_area(
                        "Paste JD",
                        height=300,
                        placeholder="Paste the job description here...",
                        label_visibility="collapsed",
                        key="jd_textarea",
                    )
            else:
                jd_text = st.text_area(
                    "Paste the job description",
                    height=300,
                    placeholder="Paste the job description here...\n\nTip: Add sample JDs to data/job_descriptions/ for quick loading.",
                    label_visibility="collapsed",
                    key="jd_textarea_only",
                )

    # ── Score Button ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    score_col, _ = st.columns([2, 3])
    with score_col:
        run_score = st.button(
            "🚀 Analyze & Score Resume",
            width="stretch",
            key="run_score_btn",
        )

    # Validation
    if run_score:
        if not resume_text.strip():
            st.markdown('<div class="warn-box">⚠️ Please upload a resume or paste resume text to continue.</div>', unsafe_allow_html=True)
        elif not jd_text.strip():
            st.markdown('<div class="warn-box">⚠️ Please provide a job description to score against.</div>', unsafe_allow_html=True)
        elif len(resume_text.strip()) < 50:
            st.markdown('<div class="warn-box">⚠️ Resume text seems too short. Please provide more content.</div>', unsafe_allow_html=True)
        else:
            # ── Run the Scoring Pipeline ─────────────────────────────────
            progress_bar = st.progress(0, text="🔄 Initializing pipeline...")
            status_text = st.empty()

            try:
                from src.scoring.hybrid_engine import run_scoring

                progress_bar.progress(15, text="🔑 Running BM25 keyword analysis...")
                time.sleep(0.3)

                progress_bar.progress(35, text="🤖 Loading semantic model (first run may take a moment)...")

                result = run_scoring(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    alpha=alpha,
                    model_name=model_name,
                    extract_ner=use_ner,
                )

                progress_bar.progress(85, text="📊 Generating analysis...")
                time.sleep(0.2)
                progress_bar.progress(100, text="✅ Complete!")
                time.sleep(0.4)
                progress_bar.empty()

                # Store result in session state
                st.session_state["scoring_result"] = result
                st.session_state["resume_text"] = resume_text
                st.session_state["jd_text"] = jd_text

                # Quick success preview on Tab 1
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    ✅ Scoring complete! Switch to the <strong>📊 Score Dashboard</strong> tab to see your full results.
                </div>
                """, unsafe_allow_html=True)

                # Quick score preview
                preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)
                with preview_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result.hybrid_percentage:.1f}%</div>
                        <div class="metric-label">Hybrid Score</div>
                        <div class="metric-sublabel">{result.score_label}</div>
                    </div>""", unsafe_allow_html=True)
                with preview_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#7c3aed">{result.sparse_score*100:.1f}%</div>
                        <div class="metric-label">BM25 Score</div>
                        <div class="metric-sublabel">Keyword Match</div>
                    </div>""", unsafe_allow_html=True)
                with preview_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#00d4b4">{result.dense_score*100:.1f}%</div>
                        <div class="metric-label">Dense Score</div>
                        <div class="metric-sublabel">Semantic Match</div>
                    </div>""", unsafe_allow_html=True)
                with preview_col4:
                    matched_count = len(result.matched_keywords)
                    total_kw = matched_count + len(result.missing_keywords)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#10b981">{matched_count}/{total_kw}</div>
                        <div class="metric-label">Keywords</div>
                        <div class="metric-sublabel">Matched / Total</div>
                    </div>""", unsafe_allow_html=True)

            except ImportError as e:
                progress_bar.empty()
                st.error(f"❌ Missing dependency: {e}\n\nRun: `pip install -r requirements.txt`")
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Scoring failed: {e}")
                st.exception(e)


# ── TAB 2: Score Dashboard ────────────────────────────────────────────────────
with tab_results:
    if "scoring_result" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#334155">
            <div style="font-size:4rem;margin-bottom:1rem">📊</div>
            <div style="font-size:1.3rem;font-weight:600;color:#475569">No Results Yet</div>
            <div style="font-size:0.95rem;color:#334155;margin-top:0.5rem">
                Go to the <strong style="color:#00d4b4">Input &amp; Score</strong> tab, 
                upload your resume and job description, then click <strong style="color:#00d4b4">Analyze &amp; Score Resume</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state["scoring_result"]

        # ── Score Gauge + Metrics ──────────────────────────────────────────
        gauge_col, metrics_col = st.columns([1, 1], gap="large")

        with gauge_col:
            st.markdown('<p class="section-label">🎯 Overall Match Score</p>', unsafe_allow_html=True)
            gauge_fig = make_gauge_chart(result.hybrid_score, result.score_label, result.score_color)
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
            <div style="text-align:center;margin-top:-1rem">
                {score_label_badge(result.score_label, result.score_color)}
            </div>
            """, unsafe_allow_html=True)

        with metrics_col:
            st.markdown('<p class="section-label">📈 Score Breakdown</p>', unsafe_allow_html=True)
            comparison_fig = make_score_comparison_chart(
                result.sparse_score, result.dense_score, result.hybrid_score
            )
            st.plotly_chart(comparison_fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#7c3aed;font-size:1.6rem">{result.sparse_score*100:.1f}%</div>
                    <div class="metric-label">BM25 Sparse</div>
                    <div class="metric-sublabel">α = {int(alpha*100)}%</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#00d4b4;font-size:1.6rem">{result.dense_score*100:.1f}%</div>
                    <div class="metric-label">Dense Semantic</div>
                    <div class="metric-sublabel">(1-α) = {int((1-alpha)*100)}%</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#f59e0b;font-size:1.6rem">{result.keyword_match_rate*100:.1f}%</div>
                    <div class="metric-label">KW Match Rate</div>
                    <div class="metric-sublabel">{len(result.matched_keywords)} matched</div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Keyword Analysis ───────────────────────────────────────────────
        kw_col1, kw_col2 = st.columns(2, gap="large")

        with kw_col1:
            matched_count = len(result.matched_keywords)
            st.markdown(f'<p class="section-label">✅ Matched Keywords ({matched_count})</p>', unsafe_allow_html=True)
            if result.matched_keywords:
                st.markdown(
                    render_keyword_chips(result.matched_keywords, "chip-matched"),
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="warn-box">No keyword matches found. Consider tailoring your resume to the JD.</div>', unsafe_allow_html=True)

        with kw_col2:
            missing_count = len(result.missing_keywords)
            st.markdown(f'<p class="section-label">❌ Missing Keywords ({missing_count})</p>', unsafe_allow_html=True)
            if result.missing_keywords:
                st.markdown(
                    render_keyword_chips(result.missing_keywords, "chip-missing"),
                    unsafe_allow_html=True
                )
                st.markdown("""
                <div class="warn-box" style="margin-top:1rem;font-size:0.82rem">
                    💡 <strong>Tip:</strong> Consider adding these keywords to your resume where applicable 
                    to improve your ATS match rate.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">🎉 Excellent! No significant missing keywords found.</div>', unsafe_allow_html=True)

        st.divider()

        # ── Top BM25 Terms ─────────────────────────────────────────────────
        if result.top_bm25_terms:
            st.markdown('<p class="section-label">🔑 Top Keyword Contributions (BM25)</p>', unsafe_allow_html=True)
            terms, scores_vals = zip(*result.top_bm25_terms[:12])
            max_score = max(scores_vals) if scores_vals else 1

            bm25_fig = go.Figure(go.Bar(
                x=list(terms),
                y=[round(s / max_score * 100, 1) for s in scores_vals],
                marker=dict(
                    color=[f"rgba(124,58,237,{0.4 + 0.6*(s/max_score):.2f})" for s in scores_vals],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>Relative Score: %{y:.1f}%<extra></extra>",
            ))
            bm25_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#94a3b8", size=11), showgrid=False),
                yaxis=dict(
                    tickfont=dict(color="#475569", size=10),
                    gridcolor="rgba(255,255,255,0.05)",
                    ticksuffix="%",
                ),
                margin=dict(t=10, b=10, l=10, r=10),
                height=200,
            )
            st.plotly_chart(bm25_fig, use_container_width=True, config={"displayModeBar": False})

        # ── Raw JSON Export ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Export Full Scoring Data (JSON)"):
            import json
            result_dict = result.to_dict()
            st.code(json.dumps(result_dict, indent=2), language="json")
            st.download_button(
                label="⬇️ Download JSON Report",
                data=json.dumps(result_dict, indent=2),
                file_name="ats_score_report.json",
                mime="application/json",
                key="download_json",
            )


# ── TAB 3: NER Analysis ───────────────────────────────────────────────────────
with tab_ner:
    if "scoring_result" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#334155">
            <div style="font-size:4rem;margin-bottom:1rem">🧠</div>
            <div style="font-size:1.3rem;font-weight:600;color:#475569">No Analysis Yet</div>
            <div style="font-size:0.95rem;color:#334155;margin-top:0.5rem">
                Run a score first from the <strong style="color:#00d4b4">Input &amp; Score</strong> tab.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state["scoring_result"]
        entities = result.resume_entities

        if not entities or not use_ner:
            st.markdown("""
            <div class="warn-box">
                NER extraction was disabled. Enable <strong>NER Extraction</strong> in the sidebar and re-run scoring.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<p class="section-label">🧠 Extracted Resume Entities</p>', unsafe_allow_html=True)

            # Summary metrics
            ner_m1, ner_m2, ner_m3, ner_m4 = st.columns(4)
            with ner_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#a78bfa;font-size:2.2rem">{len(entities.skills)}</div>
                    <div class="metric-label">Skills Found</div>
                </div>""", unsafe_allow_html=True)
            with ner_m2:
                yoe = entities.years_of_experience
                yoe_str = f"{yoe:.0f}+" if yoe else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#00d4b4;font-size:2.2rem">{yoe_str}</div>
                    <div class="metric-label">Years Exp.</div>
                </div>""", unsafe_allow_html=True)
            with ner_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#10b981;font-size:2.2rem">{len(entities.organizations)}</div>
                    <div class="metric-label">Organizations</div>
                </div>""", unsafe_allow_html=True)
            with ner_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#f59e0b;font-size:2.2rem">{len(entities.locations)}</div>
                    <div class="metric-label">Locations</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Skills
            st.markdown(f'<p class="section-label">🛠️ Technical Skills Detected ({len(entities.skills)})</p>', unsafe_allow_html=True)
            if entities.skills:
                st.markdown(render_keyword_chips(entities.skills, "chip-skill"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">No skills detected automatically. Try enabling spaCy NER (install en_core_web_sm).</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Org + Locations side by side
            col_org, col_loc = st.columns(2, gap="large")
            with col_org:
                st.markdown('<p class="section-label">🏢 Organizations</p>', unsafe_allow_html=True)
                if entities.organizations:
                    for org in entities.organizations[:15]:
                        st.markdown(f'<span class="chip chip-skill" style="background:rgba(16,185,129,0.1);color:#34d399;border-color:rgba(16,185,129,0.2)">{org}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:#334155;font-size:0.85rem">None detected (spaCy en_core_web_sm needed)</p>', unsafe_allow_html=True)

            with col_loc:
                st.markdown('<p class="section-label">📍 Locations</p>', unsafe_allow_html=True)
                if entities.locations:
                    for loc in entities.locations[:15]:
                        st.markdown(f'<span class="chip chip-skill" style="background:rgba(245,158,11,0.1);color:#fbbf24;border-color:rgba(245,158,11,0.2)">{loc}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color:#334155;font-size:0.85rem">None detected (spaCy en_core_web_sm needed)</p>', unsafe_allow_html=True)

            # Skills vs JD overlap visualization
            if entities.skills and "scoring_result" in st.session_state:
                st.divider()
                st.markdown('<p class="section-label">📊 Skills vs Job Description Coverage</p>', unsafe_allow_html=True)

                jd_text_stored = st.session_state.get("jd_text", "")
                jd_lower = jd_text_stored.lower()

                skill_in_jd = []
                skill_not_in_jd = []
                for skill in entities.skills:
                    if skill.lower() in jd_lower:
                        skill_in_jd.append(skill)
                    else:
                        skill_not_in_jd.append(skill)

                sv_col1, sv_col2 = st.columns(2)
                with sv_col1:
                    st.markdown(f'<p style="color:#10b981;font-size:0.8rem;font-weight:600">✅ Skills Mentioned in JD ({len(skill_in_jd)})</p>', unsafe_allow_html=True)
                    if skill_in_jd:
                        st.markdown(render_keyword_chips(skill_in_jd, "chip-matched"), unsafe_allow_html=True)
                with sv_col2:
                    st.markdown(f'<p style="color:#64748b;font-size:0.8rem;font-weight:600">➖ Skills Not in JD ({len(skill_not_in_jd)})</p>', unsafe_allow_html=True)
                    if skill_not_in_jd:
                        st.markdown(render_keyword_chips(skill_not_in_jd, "chip-skill"), unsafe_allow_html=True)


# ── TAB 4: About ──────────────────────────────────────────────────────────────
with tab_about:
    about_col, _ = st.columns([2, 1])
    with about_col:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:1.5rem">
            <div style="font-size:1.3rem;font-weight:700;color:#00d4b4;margin-bottom:0.75rem">🎯 What is ATS AI Engine?</div>
            <p style="color:#94a3b8;line-height:1.8">
            ATS AI Engine is a hybrid resume scoring system that combines two powerful AI techniques 
            to evaluate how well a candidate's resume matches a given job description:
            </p>
            <ul style="color:#94a3b8;line-height:2">
                <li><strong style="color:#a78bfa">BM25 Sparse Scoring</strong> — Classic information retrieval algorithm that measures keyword frequency and relevance (exact match). Raw BM25 scores are <strong>unbounded</strong> and must be normalized before combining.</li>
                <li><strong style="color:#00d4b4">Dense Semantic Scoring</strong> — Modern sentence embeddings (SentenceTransformers) that capture the <em>meaning</em> behind words, not just exact matches. Outputs bounded cosine similarity [0, 1].</li>
                <li><strong style="color:#f59e0b">Hybrid Combination</strong> — BM25 scores are <strong>normalized via sigmoid</strong> (analog of Min-Max for single-resume), then weighted: <code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px">score = α × norm_BM25 + (1-α) × Dense</code></li>
                <li><strong style="color:#38bdf8">Reciprocal Rank Fusion (RRF)</strong> — Industry-standard parameter-free alternative (used by Elasticsearch, Pinecone) that combines <em>rankings</em> instead of raw scores, bypassing the scale mismatch problem entirely.</li>
            </ul>
        </div>

        <div class="glass-card" style="margin-bottom:1.5rem">
            <div style="font-size:1.3rem;font-weight:700;color:#00d4b4;margin-bottom:0.75rem">🏗️ Architecture</div>
            <div style="background:rgba(0,0,0,0.3);border-radius:10px;padding:1rem;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#475569;line-height:1.8">
            Resume/JD Input<br>
            &nbsp;&nbsp;&nbsp;↓<br>
            📄 Parser (PyMuPDF / python-docx)<br>
            &nbsp;&nbsp;&nbsp;↓<br>
            🧹 NLP Cleaner (NLTK stopwords, tokenization)<br>
            &nbsp;&nbsp;&nbsp;↓<br>
            ┌──────────────────────────────┐<br>
            │ BM25 Sparse  │  Dense Vector │<br>
            │ (rank-bm25)  │ (STransformers)│<br>
            │ RAW score    │ cosine [0,1]  │<br>
            └──────────────────────────────┘<br>
            &nbsp;&nbsp;&nbsp;↓<br>
            📐 Sigmoid Normalization (BM25 → [0,1])<br>
            &nbsp;&nbsp;&nbsp;↓<br>
            ┌──────────────────────────────┐<br>
            │ ⚖️ Hybrid  │  🔀 RRF       │<br>
            │ α×BM25+    │ Rank Fusion   │<br>
            │ (1-α)×Dense│ (k=60)        │<br>
            └──────────────────────────────┘<br>
            &nbsp;&nbsp;&nbsp;↓<br>
            📊 Streamlit Dashboard
            </div>
        </div>

        <div class="glass-card" style="margin-bottom:1.5rem">
            <div style="font-size:1.3rem;font-weight:700;color:#00d4b4;margin-bottom:0.75rem">📐 Scoring Methodology</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#a78bfa;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">🔑 BM25 (Sparse)</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">• BM25Okapi with IDF fix<br>• Raw unbounded score<br>• Sigmoid normalized</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#00d4b4;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">🤖 Dense (Semantic)</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">• SentenceTransformers<br>• Cosine similarity [0,1]<br>• 384-dim embeddings</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#f59e0b;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">⚖️ Hybrid (α-blend)</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">• α default = 0.6<br>• Favors exact keywords<br>• α×BM25 + (1-α)×Dense</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#38bdf8;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">🔀 RRF (Rank Fusion)</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">• k=60 smoothing<br>• No tuning needed<br>• Industry standard</div>
                </div>
            </div>
        </div>

        <div class="glass-card">
            <div style="font-size:1.3rem;font-weight:700;color:#00d4b4;margin-bottom:0.75rem">⚙️ Tech Stack</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#a78bfa;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">🎨 UI</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">Streamlit<br>Plotly<br>Custom CSS</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#00d4b4;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">🧠 AI/ML</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">SentenceTransformers<br>rank-bm25<br>spaCy + NLTK</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#f59e0b;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">📄 Parsing</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">PyMuPDF (fitz)<br>python-docx<br>Multi-column support</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.75rem;border:1px solid rgba(255,255,255,0.06)">
                    <div style="color:#10b981;font-weight:600;font-size:0.85rem;margin-bottom:0.5rem">🔌 API (Phase 4)</div>
                    <div style="color:#64748b;font-size:0.82rem;line-height:1.8">FastAPI<br>Pydantic v2<br>Uvicorn</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1.5rem;color:#1e293b;font-size:0.8rem">
        Built with ❤️ using Streamlit + SentenceTransformers + BM25 &nbsp;•&nbsp; 
        <a href="https://github.com/your-username/ats-cv-streamlit" style="color:#00d4b4;text-decoration:none">GitHub</a>
    </div>
    """, unsafe_allow_html=True)
