"""
CognifyX – AI Resume Intelligence System
Premium Streamlit UI with keyword-based ATS scoring, multi-model LLM pipeline, and visual rankings.
"""
import streamlit as st
import pandas as pd
import os
import sys
import time
import zipfile
import shutil
import base64
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# -- FIX MODULE PATHS --
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.resume_parser import ResumeParser
from pipeline.ats_filter import ATSFilter
from pipeline.llm_extractor import LLMExtractor
from pipeline.candidate_ranker import CandidateRanker
from pipeline.final_analysis import FinalAnalysis
from agents.llm_agent import LLMAgent
from database.resume_db import ResumeDB

# -- PAGE CONFIG --
st.set_page_config(
    page_title="CognifyX – AI Resume Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- CUSTOM CSS --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    h1, h2, h3, h4, h5, h6, .stMarkdown p {
        color: #0f172a !important;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] .stTextArea textarea {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #1e293b;
        color: #e2e8f0;
    }

    /* Log Panel */
    .log-container {
        background: #020617;
        color: #94a3b8;
        padding: 16px;
        border-radius: 12px;
        height: 360px;
        overflow-y: auto;
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 0.8rem;
        border: 1px solid #1e293b;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    .log-entry { margin-bottom: 4px; padding-bottom: 3px; border-bottom: 1px solid #0f172a; }
    .log-timestamp { color: #475569; margin-right: 8px; font-weight: 300; }
    .log-info { color: #38bdf8; }
    .log-success { color: #4ade80; }
    .log-error { color: #f87171; }
    .log-warning { color: #fbbf24; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #06b6d4 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        animation: headerShimmer 8s ease-in-out infinite;
    }
    @keyframes headerShimmer {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(10%, 10%); }
    }
    .app-header h1 { color: white !important; font-weight: 800; letter-spacing: -1px; position: relative; }
    .app-header p { color: #e2e8f0 !important; font-size: 1rem; position: relative; }

    /* Dashboard Cards */
    .dashboard-card {
        background: white;
        padding: 20px 24px;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 16px;
        border: 1px solid #e2e8f0;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .dashboard-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-1px);
    }

    /* Candidate Cards */
    .candidate-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 16px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        transition: all 0.3s ease;
    }
    .candidate-card:hover {
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.12);
        transform: translateY(-2px);
    }
    .candidate-card.gold { border-left-color: #f59e0b; }
    .candidate-card.silver { border-left-color: #94a3b8; }
    .candidate-card.bronze { border-left-color: #cd7c32; }
    .candidate-card.shortlisted { border-left-color: #10b981; }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 16px 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { font-size: 0.8rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Skill Tags */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #ede9fe, #e0e7ff);
        color: #4338ca;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 3px;
    }
    .skill-tag.matched {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
    }
    .skill-tag.missed {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
    }

    /* Score Badge */
    .score-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .score-high { background: #d1fae5; color: #065f46; }
    .score-mid { background: #fef3c7; color: #92400e; }
    .score-low { background: #fee2e2; color: #991b1b; }

    /* Progress bars */
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #4f46e5, #06b6d4); }

    label { color: #334155 !important; font-weight: 600 !important; }

    /* Timing badge */
    .timing-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-left: 8px;
    }

    .stTable { border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; }

    /* Insights panel */
    .insights-panel {
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 24px;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# -- INITIALIZATION --
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'pipeline_logs' not in st.session_state:
    st.session_state.pipeline_logs = []

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'temp_uploads')

# -- HEADER --
st.markdown("""
<div class='app-header'>
    <h1>🧠 CognifyX – AI Resume Intelligence System</h1>
    <p>Upload resumes • Define keywords • AI scores, shortlists & ranks candidates in seconds</p>
</div>
""", unsafe_allow_html=True)

# -- DASHBOARD LAYOUT --
col_logs, col_stages = st.columns([1, 1])

with col_logs:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>📡 Live Activity Log</h3>", unsafe_allow_html=True)
    log_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with col_stages:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>📊 Pipeline Stages</h3>", unsafe_allow_html=True)
    st.write("**Stage 1 – Resume Parsing**")
    prog_parsing = st.progress(0)
    st.write("**Stage 2 – ATS Keyword Filtering**")
    prog_ats = st.progress(0)
    st.write("**Stage 3 – LLM Extraction (Qwen)**")
    prog_extraction = st.progress(0)
    st.write("**Stage 4 – Candidate Scoring (Mistral)**")
    prog_ranking = st.progress(0)
    st.write("**Stage 5 – Final AI Insights (LLaMA)**")
    prog_insights = st.progress(0)
    st.markdown("</div>", unsafe_allow_html=True)

# -- LOGGING --
def push_log(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": timestamp, "msg": message, "level": level})
    try:
        log_placeholder.markdown(get_log_html(), unsafe_allow_html=True)
    except Exception:
        pass

def get_log_html():
    html = "<div class='log-container'>"
    for log in st.session_state.logs:
        level = log['level'].lower()
        color_class = f"log-{level}" if level in ('info', 'success', 'error', 'warning') else 'log-info'
        html += f"<div class='log-entry'><span class='log-timestamp'>[{log['time']}]</span> <span class='{color_class}'>{log['msg']}</span></div>"
    html += "</div>"
    return html


# -- HELPER FUNCTIONS --
def get_score_class(score):
    if score >= 70:
        return "score-high"
    elif score >= 40:
        return "score-mid"
    return "score-low"

def get_medal(rank):
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    return medals.get(rank, f"#{rank}")


# -- CORE PIPELINE FUNCTIONS --
def load_dataset(uploaded_files):
    """Handle ZIP, PDF, DOCX, TXT uploads."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    file_count = 0
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(TEMP_DIR)
                for root, dirs, files in os.walk(TEMP_DIR):
                    file_count += len([f for f in files if f.lower().endswith(('.pdf', '.docx', '.txt'))])
        else:
            with open(os.path.join(TEMP_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_count += 1

    push_log(f"📁 Dataset loaded: {file_count} resumes staged.", "SUCCESS")
    return file_count


def run_full_pipeline(uploaded_files, jd, keywords_input, shortlist_count):
    """Execute the full CognifyX pipeline."""
    st.session_state.is_running = True
    st.session_state.logs = []
    st.session_state.results = None

    # Reset progress bars
    prog_parsing.progress(0)
    prog_ats.progress(0)
    prog_extraction.progress(0)
    prog_ranking.progress(0)
    prog_insights.progress(0)

    try:
        # ── Load Files ──
        count = load_dataset(uploaded_files)
        if count == 0:
            push_log("❌ No valid resumes found in upload.", "ERROR")
            return

        # Initialize components
        db = ResumeDB()
        parser = ResumeParser()
        ats_filter = ATSFilter(top_n=40)
        llm_agent = LLMAgent()
        extractor = LLMExtractor(llm_agent)
        ranker_obj = CandidateRanker(llm_agent)
        analyzer = FinalAnalysis(llm_agent)

        # Check Ollama
        ollama_up, available_models = llm_agent.check_ollama_status()
        if ollama_up:
            push_log(f"✅ Ollama connected. Models: {', '.join(available_models[:5])}", "SUCCESS")
        else:
            push_log("⚠️ Ollama not detected. Using heuristic analysis.", "WARNING")

        # ── Stage 1: Parse ──
        push_log("📄 Stage 1: Parsing resumes...", "INFO")
        prog_parsing.progress(20)
        resumes = parser.parse_directory(TEMP_DIR)
        prog_parsing.progress(100)
        push_log(f"   ✓ Parsed {len(resumes)} resumes successfully.", "SUCCESS")

        if not resumes:
            push_log("❌ No readable resumes found.", "ERROR")
            return

        # Store in DB
        for r in resumes:
            db.store_resume(r['filename'], r['text'])

        # ── Stage 2: ATS Keyword Filtering ──
        push_log("🔍 Stage 2: ATS Keyword + Similarity Scoring...", "INFO")
        prog_ats.progress(30)
        top_resumes = ats_filter.calculate_scores(resumes, jd, keywords_input)
        prog_ats.progress(100)

        keywords_list = [k.strip() for k in keywords_input.replace('\n', ',').split(',') if k.strip()]
        push_log(f"   ✓ {len(keywords_list)} keywords matched against {len(resumes)} resumes.", "SUCCESS")
        push_log(f"   ✓ Top {len(top_resumes)} candidates passed ATS filter.", "SUCCESS")

        # ── Stage 3: LLM Extraction (Qwen) ──
        push_log(f"🧠 Stage 3: Extracting structured data via Qwen...", "INFO")
        structured_profiles = []
        total = len(top_resumes)

        for i, resume in enumerate(top_resumes):
            p_val = int(((i + 1) / total) * 100)
            prog_extraction.progress(p_val)

            cached = db.get_extraction(resume['text'])
            if cached:
                profile = cached
                push_log(f"   [Cache] {resume['filename']}", "INFO")
            else:
                profile = extractor.extract(resume['text'], identifier=resume['filename'])
                db.save_extraction(resume['text'], profile)
                push_log(f"   [Extracted] {resume['filename']}", "INFO")

            profile['filename'] = resume['filename']
            if not profile.get('name') or profile.get('name') == "Sample Candidate":
                profile['name'] = resume['filename'].replace('.pdf', '').replace('.docx', '').replace('_', ' ')

            profile['ats_score'] = resume.get('score', 0)
            profile['keyword_score'] = resume.get('keyword_score', 0)
            profile['keywords_matched'] = resume.get('keywords_matched', [])
            profile['keywords_missed'] = resume.get('keywords_missed', [])
            profile['tfidf_score'] = resume.get('tfidf_score', 0)
            profile['full_text'] = resume['text']
            structured_profiles.append(profile)

        push_log(f"   ✓ Extraction complete for {total} candidates.", "SUCCESS")

        # ── Stage 4: LLM Scoring (Mistral) ──
        push_log("📊 Stage 4: Scoring candidates via Mistral...", "INFO")

        def process_profile(profile):
            profile_brief = {k: v for k, v in profile.items() if k != 'full_text'}
            p_json = json.dumps(profile_brief, default=str)

            cached = db.get_scoring(jd, p_json)
            if cached:
                analysis = cached
            else:
                analysis = ranker_obj.rank(jd, profile, identifier=profile['filename'])
                db.save_scoring(jd, p_json, analysis)

            profile['llm_analysis'] = analysis
            profile['llm_score'] = CandidateRanker.extract_llm_score(analysis)
            profile['composite_score'] = CandidateRanker.compute_composite_score(
                profile['ats_score'], profile['keyword_score'], profile['llm_score']
            )
            return profile

        with ThreadPoolExecutor(max_workers=4) as executor:
            results_list = list(executor.map(process_profile, structured_profiles))

        for i, res in enumerate(results_list):
            prog_ranking.progress(int(((i + 1) / total) * 100))

        push_log(f"   ✓ Scoring complete for {total} candidates.", "SUCCESS")

        # ── Stage 5: Rank + Shortlist ──
        ranked = CandidateRanker.rank_candidates(results_list)
        shortlisted = ranked[:shortlist_count]
        for p in shortlisted:
            p['shortlisted'] = True
        for p in ranked[shortlist_count:]:
            p['shortlisted'] = False

        push_log(f"   ✓ Top {shortlist_count} candidates shortlisted.", "SUCCESS")

        # ── Stage 6: Final AI Insights (LLaMA) ──
        push_log("🤖 Stage 5: Generating executive AI insights (LLaMA)...", "INFO")
        prog_insights.progress(40)
        top_data = [{k: v for k, v in p.items() if k != 'full_text'} for p in shortlisted]
        insights = analyzer.analyze(json.dumps(top_data, default=str)[:4000])
        prog_insights.progress(100)
        push_log("   ✓ Final insights generated.", "SUCCESS")

        # ── Store Results ──
        run_id = db.create_run(jd, keywords_input, len(resumes))
        for p in ranked:
            text_hash = db._get_hash(p['full_text'])
            resume_id = db.get_resume_id(text_hash)
            if resume_id:
                db.store_score(
                    run_id, resume_id,
                    ats_score=p.get('ats_score', 0),
                    keyword_score=p.get('keyword_score', 0),
                    llm_score=p.get('llm_score', 0),
                    composite_score=p.get('composite_score', 0),
                    llm_analysis=str(p.get('llm_analysis', '')),
                    final_rank=p.get('final_rank', 0),
                    shortlisted=1 if p.get('shortlisted') else 0
                )
        db.update_run(run_id, shortlist_count)

        st.session_state.results = {
            'candidates': ranked,
            'shortlisted': shortlisted,
            'insights': insights,
            'total_parsed': len(resumes),
            'total_filtered': len(top_resumes),
            'shortlist_count': shortlist_count,
            'keywords': keywords_list,
        }

        push_log("🎉 Pipeline execution finished successfully!", "SUCCESS")

    except Exception as e:
        push_log(f"❌ Pipeline Failed: {str(e)}", "ERROR")
        import traceback
        push_log(f"   {traceback.format_exc()[:300]}", "ERROR")
    finally:
        st.session_state.is_running = False


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <span style='font-size: 3rem;'>🧠</span>
        <h2 style='margin: 5px 0; color: white !important;'>CognifyX</h2>
        <p style='color: #94a3b8 !important; font-size: 0.85rem;'>AI Resume Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📁 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drop ZIP / PDF / DOCX / TXT files",
        accept_multiple_files=True,
        type=['zip', 'pdf', 'docx', 'txt']
    )
    if uploaded_files:
        st.success(f"✓ {len(uploaded_files)} file(s) uploaded")

    st.markdown("---")

    st.subheader("📝 Job Description")
    jd_content = st.text_area(
        "Paste full JD here",
        height=150,
        placeholder="We are looking for a Senior Software Engineer with 5+ years Python experience..."
    )

    st.subheader("🔑 ATS Keywords")
    keywords_input = st.text_area(
        "Comma-separated keywords",
        height=100,
        placeholder="python, react, sql, docker, machine learning, aws, 5 years experience"
    )
    if keywords_input:
        kw_list = [k.strip() for k in keywords_input.replace('\n', ',').split(',') if k.strip()]
        st.caption(f"📋 {len(kw_list)} keywords defined")

    st.markdown("---")

    st.subheader("⚙️ Settings")
    shortlist_count = st.slider("Shortlist top N candidates", 3, 20, 5)

    st.markdown("---")

    if st.button("🚀 Start AI Screening", disabled=st.session_state.is_running, use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please upload resume files.")
        elif not jd_content:
            st.warning("⚠️ Please provide a job description.")
        elif not keywords_input:
            st.warning("⚠️ Please enter ATS keywords.")
        else:
            run_full_pipeline(uploaded_files, jd_content, keywords_input, shortlist_count)


# ═══════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════
if st.session_state.is_running:
    st.info("⏳ Pipeline in progress... Please wait.")


# ═══════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if st.session_state.results:
    res = st.session_state.results
    st.markdown("---")

    # Metric cards
    st.markdown("## 📈 Pipeline Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{res['total_parsed']}</div>
            <div class='metric-label'>Resumes Parsed</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{res['total_filtered']}</div>
            <div class='metric-label'>ATS Filtered</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(res['keywords'])}</div>
            <div class='metric-label'>Keywords Used</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{res['shortlist_count']}</div>
            <div class='metric-label'>Shortlisted</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        avg_score = 0
        if res['shortlisted']:
            avg_score = round(sum(c.get('composite_score', 0) for c in res['shortlisted']) / len(res['shortlisted']), 1)
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{avg_score}</div>
            <div class='metric-label'>Avg Composite Score</div>
        </div>""", unsafe_allow_html=True)

    # ── Shortlisted Candidates ──
    st.markdown("---")
    st.markdown("## 🏆 Shortlisted Candidates")

    for i, c in enumerate(res['shortlisted']):
        rank = c.get('final_rank', i + 1)
        medal = get_medal(rank)
        card_class = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "shortlisted")
        composite = c.get('composite_score', 0)
        score_class = get_score_class(composite)

        st.markdown(f"<div class='candidate-card {card_class}'>", unsafe_allow_html=True)

        col_info, col_scores, col_btn = st.columns([3, 2, 1])

        with col_info:
            st.markdown(f"### {medal} {c.get('name', c.get('filename', 'Unknown'))}")
            st.caption(f"📄 {c.get('filename', '')}  •  🏷️ {c.get('domain', 'General')}")

            # Skills
            skills = c.get('skills', [])
            if skills:
                skill_html = " ".join([f"<span class='skill-tag'>{s}</span>" for s in skills[:8]])
                st.markdown(f"**Skills:** {skill_html}", unsafe_allow_html=True)

            # Keyword matches
            matched = c.get('keywords_matched', [])
            missed = c.get('keywords_missed', [])
            if matched:
                match_html = " ".join([f"<span class='skill-tag matched'>✓ {k}</span>" for k in matched[:6]])
                st.markdown(f"**Keywords Matched:** {match_html}", unsafe_allow_html=True)
            if missed:
                miss_html = " ".join([f"<span class='skill-tag missed'>✗ {k}</span>" for k in missed[:4]])
                st.markdown(f"**Missing:** {miss_html}", unsafe_allow_html=True)

        with col_scores:
            st.markdown(f"""
            <div style='text-align: center; padding-top: 10px;'>
                <div class='score-badge {score_class}' style='font-size: 1.4rem; padding: 10px 24px;'>
                    {composite}
                </div>
                <div style='color: #64748b; font-size: 0.75rem; margin-top: 4px;'>Composite Score</div>
            </div>
            """, unsafe_allow_html=True)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("ATS", f"{c.get('ats_score', 0)}")
            sc2.metric("Keywords", f"{c.get('keyword_score', 0)}")
            sc3.metric("LLM", f"{c.get('llm_score', 0)}")

        with col_btn:
            file_path = os.path.join(TEMP_DIR, c['filename'])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="📄 Download",
                        data=f,
                        file_name=c['filename'],
                        mime="application/octet-stream",
                        key=f"dl_short_{i}"
                    )

        # Expandable AI Analysis
        with st.expander("🔍 View AI Analysis"):
            st.write(c.get('llm_analysis', 'No analysis available.'))

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Full Rankings Table ──
    st.markdown("---")
    st.markdown("## 📋 Full Candidate Rankings")

    show_all = st.checkbox("Show all candidates (not just shortlisted)", value=False)
    display_list = res['candidates'] if show_all else res['shortlisted']

    table_data = []
    for c in display_list:
        table_data.append({
            "Rank": c.get('final_rank', '-'),
            "Name": c.get('name', c.get('filename', '')),
            "File": c.get('filename', ''),
            "Domain": c.get('domain', 'General'),
            "Experience": f"{c.get('experience_years', '?')} yrs",
            "ATS Score": c.get('ats_score', 0),
            "Keyword Match": f"{c.get('keyword_score', 0)}%",
            "LLM Score": c.get('llm_score', 0),
            "Composite": c.get('composite_score', 0),
            "Shortlisted": "✅" if c.get('shortlisted') else "—",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # CSV Download
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv_data,
        file_name=f'cognifyx_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv',
        key="global_csv"
    )

    # ── Final AI Insights ──
    st.markdown("---")
    st.markdown("## 🤖 Final AI Insights & Recommendation")
    st.markdown(f"<div class='insights-panel'>{res['insights']}</div>", unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<br><hr>
<center style='color: #94a3b8; font-size: 0.8rem;'>
    © 2026 CognifyX AI Resume Intelligence | Powered by Ollama (Qwen • Mistral • LLaMA)
</center>
""", unsafe_allow_html=True)
