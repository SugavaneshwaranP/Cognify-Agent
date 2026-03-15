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
from agents.reflection_agent import ReflectionAgent
from agents.chat_agent import ChatAgent
from agents.skill_agent import SkillAgent
from agents.experience_agent import ExperienceAgent
from agents.culture_agent import CultureAgent
from agents.debate_moderator import DebateModerator
from agents.planner_agent import PlannerAgent
from agents.report_agent import ReportAgent
from pipeline.report_generator import PDFReportGenerator, DOCXReportGenerator
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

    /* ═══ GLOBAL DARK THEME ═══ */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background: #0b0f1a;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: #cbd5e1 !important;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #131b2e 50%, #0f1629 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #e0e7ff !important;
    }
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stTextInput input {
        background: #1a2236;
        color: #e2e8f0;
        border: 1px solid #2d3a56;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .stTextArea textarea:focus,
    [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #1a2236;
        color: #e2e8f0;
        border: 1px solid #2d3a56;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }

    /* ═══ LIVE LOG TERMINAL ═══ */
    .log-container {
        background: #080c15;
        color: #94a3b8;
        padding: 18px;
        border-radius: 14px;
        height: 360px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        font-size: 0.78rem;
        border: 1px solid #1e2a42;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.4);
    }
    .log-entry { margin-bottom: 4px; padding-bottom: 3px; border-bottom: 1px solid rgba(30,42,66,0.5); }
    .log-timestamp { color: #4b5e80; margin-right: 8px; font-weight: 300; }
    .log-info    { color: #60a5fa; }
    .log-success { color: #34d399; }
    .log-error   { color: #fb7185; }
    .log-warning { color: #fbbf24; }

    /* ═══ APP HEADER ═══ */
    .app-header {
        background: linear-gradient(135deg, #1e1145 0%, #312876 30%, #1a1042 60%, #0d2847 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.15), 0 0 80px rgba(99, 102, 241, 0.05);
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.12) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(34,211,238,0.08) 0%, transparent 50%);
        animation: headerGlow 10s ease-in-out infinite;
    }
    @keyframes headerGlow {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.6; }
        50% { transform: translate(5%, 5%) scale(1.05); opacity: 1; }
    }
    .app-header h1 {
        color: white !important;
        font-weight: 800;
        letter-spacing: -1px;
        position: relative;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }
    .app-header p {
        color: #c7d2fe !important;
        font-size: 1rem;
        position: relative;
    }

    /* ═══ DASHBOARD CARDS ═══ */
    .dashboard-card {
        background: #111827;
        padding: 22px 26px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        margin-bottom: 16px;
        border: 1px solid #1f2b45;
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }
    .dashboard-card:hover {
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
        border-color: rgba(99, 102, 241, 0.25);
    }
    .dashboard-card h3 {
        color: #e0e7ff !important;
    }

    /* ═══ CANDIDATE CARDS ═══ */
    .candidate-card {
        background: linear-gradient(135deg, #111827 0%, #151f32 100%);
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        margin-bottom: 18px;
        border: 1px solid #1f2b45;
        border-left: 4px solid #6366f1;
        transition: all 0.3s ease;
    }
    .candidate-card:hover {
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .candidate-card.gold   { border-left-color: #fbbf24; box-shadow: 0 4px 20px rgba(251,191,36,0.1); }
    .candidate-card.silver { border-left-color: #94a3b8; }
    .candidate-card.bronze { border-left-color: #cd7c32; }
    .candidate-card.shortlisted { border-left-color: #34d399; }

    /* ═══ METRIC CARDS ═══ */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #151f32 100%);
        padding: 18px 22px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #1f2b45;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* ═══ SKILL TAGS ═══ */
    .skill-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 600;
        margin: 2px 3px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .skill-tag.matched {
        background: rgba(52, 211, 153, 0.15);
        color: #6ee7b7;
        border-color: rgba(52, 211, 153, 0.25);
    }
    .skill-tag.missed {
        background: rgba(251, 113, 133, 0.15);
        color: #fda4af;
        border-color: rgba(251, 113, 133, 0.25);
    }

    /* ═══ SCORE BADGES ═══ */
    .score-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .score-high { background: rgba(52,211,153,0.15); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }
    .score-mid  { background: rgba(251,191,36,0.15); color: #fde68a; border: 1px solid rgba(251,191,36,0.3); }
    .score-low  { background: rgba(251,113,133,0.15); color: #fda4af; border: 1px solid rgba(251,113,133,0.3); }

    /* ═══ PROGRESS BARS ═══ */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #22d3ee);
    }

    label { color: #94a3b8 !important; font-weight: 600 !important; }

    /* ═══ TIMING BADGE ═══ */
    .timing-badge {
        display: inline-block;
        background: rgba(99,102,241,0.12);
        color: #a5b4fc;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-left: 8px;
    }

    .stTable { border-radius: 12px; overflow: hidden; border: 1px solid #1f2b45; }

    /* ═══ INSIGHTS PANEL ═══ */
    .insights-panel {
        background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(34,211,238,0.06));
        border: 1px solid rgba(52,211,153,0.2);
        border-radius: 18px;
        padding: 26px;
        color: #a7f3d0;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    /* ═══ REFLECTION CARDS ═══ */
    .reflection-card {
        background: #111827;
        padding: 18px 22px;
        border-radius: 14px;
        border: 1px solid #1f2b45;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: all 0.2s ease;
    }
    .reflection-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        transform: translateY(-1px);
    }
    .reflection-card.corrected {
        border-left: 4px solid #fbbf24;
        background: linear-gradient(135deg, rgba(251,191,36,0.06), #111827);
    }
    .reflection-card.validated {
        border-left: 4px solid #34d399;
        background: linear-gradient(135deg, rgba(52,211,153,0.06), #111827);
    }
    .anomaly-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 2px 3px;
    }
    .anomaly-high   { background: rgba(251,113,133,0.15); color: #fda4af; }
    .anomaly-medium { background: rgba(251,191,36,0.15); color: #fde68a; }
    .anomaly-low    { background: rgba(99,102,241,0.15); color: #a5b4fc; }

    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .confidence-HIGH   { background: rgba(52,211,153,0.15); color: #6ee7b7; }
    .confidence-MEDIUM { background: rgba(251,191,36,0.15); color: #fde68a; }
    .confidence-LOW    { background: rgba(251,113,133,0.15); color: #fda4af; }

    .score-correction {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .score-original  { color: #64748b; text-decoration: line-through; }
    .score-arrow     { color: #94a3b8; }
    .score-corrected { color: #34d399; }

    /* ═══ CHAT INTERFACE ═══ */
    .chat-container {
        background: linear-gradient(135deg, #0f1629 0%, #131b2e 100%);
        border-radius: 22px;
        padding: 28px;
        margin-top: 8px;
        border: 1px solid #1f2b45;
        box-shadow: 0 10px 50px rgba(0,0,0,0.3);
    }
    .chat-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid #1f2b45;
    }
    .chat-header-icon {
        font-size: 1.8rem;
        background: linear-gradient(135deg, #818cf8, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-header-title {
        color: #e0e7ff !important;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 !important;
    }
    .chat-header-badge {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .chat-messages {
        max-height: 450px;
        overflow-y: auto;
        padding: 8px 0;
        scrollbar-width: thin;
        scrollbar-color: #2d3a56 transparent;
    }
    .chat-msg {
        margin-bottom: 16px;
        display: flex;
        gap: 10px;
        animation: chatFadeIn 0.3s ease;
    }
    @keyframes chatFadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-msg.user      { justify-content: flex-end; }
    .chat-msg.assistant  { justify-content: flex-start; }
    .chat-avatar {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .chat-avatar.user-av { background: linear-gradient(135deg, #6366f1, #8b5cf6); order: 2; }
    .chat-avatar.ai-av   { background: linear-gradient(135deg, #22d3ee, #34d399); }
    .chat-bubble {
        max-width: 80%;
        padding: 14px 18px;
        border-radius: 16px;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .chat-bubble.user-bubble {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white;
        border-bottom-right-radius: 4px;
    }
    .chat-bubble.ai-bubble {
        background: rgba(17, 24, 39, 0.8);
        color: #e2e8f0;
        border: 1px solid #1f2b45;
        border-bottom-left-radius: 4px;
        backdrop-filter: blur(10px);
    }
    .chat-bubble.ai-bubble strong { color: #818cf8; }
    .chat-bubble.ai-bubble code { color: #34d399; background: rgba(0,0,0,0.4); padding: 1px 5px; border-radius: 4px; }
    .chat-proactive {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(34,211,238,0.06));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #c7d2fe;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .chat-proactive:hover {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(34,211,238,0.12));
        border-color: rgba(99,102,241,0.4);
        transform: translateX(4px);
    }

    /* ═══ DEBATE CARDS ═══ */
    .debate-card {
        background: #111827;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 16px;
        border: 1px solid #1f2b45;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .debate-agent {
        background: rgba(17, 24, 39, 0.6);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 3px solid #475569;
    }
    .debate-agent.skill      { border-left-color: #22d3ee; background: rgba(34,211,238,0.06); }
    .debate-agent.experience { border-left-color: #a78bfa; background: rgba(167,139,250,0.06); }
    .debate-agent.culture    { border-left-color: #fbbf24; background: rgba(251,191,36,0.06); }
    .debate-agent.devil      { border-left-color: #fb7185; background: rgba(251,113,133,0.06); }
    .debate-agent.moderator  { border-left-color: #34d399; background: rgba(52,211,153,0.06); }
    .debate-agent-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .debate-agent-name {
        font-weight: 700;
        font-size: 0.9rem;
        color: #e0e7ff;
    }
    .debate-score-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(99,102,241,0.15);
        color: #a5b4fc;
    }

    /* ═══ PLANNER / REACT CARDS ═══ */
    .plan-card {
        background: linear-gradient(135deg, #0f1629 0%, #1a1042 50%, #0d2040 100%);
        border-radius: 18px;
        padding: 26px;
        border: 1px solid rgba(99,102,241,0.25);
        color: #e0e7ff;
        box-shadow: 0 4px 20px rgba(99,102,241,0.1);
    }
    .react-step {
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #cbd5e1;
    }
    .react-think   { background: rgba(99,102,241,0.1); border-left: 3px solid #818cf8; }
    .react-act     { background: rgba(52,211,153,0.1); border-left: 3px solid #34d399; }
    .react-observe { background: rgba(251,191,36,0.08); border-left: 3px solid #fbbf24; }
    .react-llm     { background: rgba(244,114,182,0.08); border-left: 3px solid #f472b6; }
    .plan-weight-table { width: 100%; border-collapse: collapse; }
    .plan-weight-table td, .plan-weight-table th {
        padding: 6px 12px;
        text-align: left;
        font-size: 0.82rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .plan-weight-table th { color: #818cf8; font-weight: 600; }
    .plan-weight-table td { color: #cbd5e1; }

    /* ═══ BUTTONS ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6366f1, #818cf8);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2);
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
    }

    /* ═══ INPUTS & EXPANDERS ═══ */
    .stTextArea textarea, .stTextInput input {
        background: #111827 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1f2b45 !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
    }
    .stCheckbox label span { color: #cbd5e1 !important; }
    div[data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #1f2b45;
        border-radius: 14px;
    }
    div[data-testid="stExpander"] summary {
        color: #e0e7ff !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #818cf8 !important;
    }

    /* ═══ DATAFRAMES ═══ */
    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ═══ TABS & METRICS ═══ */
    [data-testid="stMetric"] {
        background: #111827;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #1f2b45;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #e0e7ff !important; }

    /* ═══ FILE UPLOADER ═══ */
    [data-testid="stFileUploader"] {
        background: #111827;
        border: 2px dashed #2d3a56;
        border-radius: 14px;
        padding: 16px;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
    }

    /* ═══ LARGE TABS ═══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0d1220;
        padding: 8px 12px;
        border-radius: 16px;
        border: 1px solid #1f2b45;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        background: transparent;
        border-radius: 12px !important;
        border: 1px solid transparent;
        padding: 0 24px !important;
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c7d2fe !important;
        background: rgba(99,102,241,0.08);
        border-color: rgba(99,102,241,0.2);
    }
    .stTabs [aria-selected="true"] {
        color: #e0e7ff !important;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)) !important;
        border-color: rgba(99,102,241,0.4) !important;
        box-shadow: 0 0 20px rgba(99,102,241,0.15);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        height: 3px;
        border-radius: 3px;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 28px;
    }

    /* ═══ DIVIDERS ═══ */
    hr {
        border-color: #1f2b45 !important;
        opacity: 0.6;
    }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2d3a56; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3d4f72; }
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
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_agent' not in st.session_state:
    st.session_state.chat_agent = ChatAgent(LLMAgent())
if 'proactive_questions' not in st.session_state:
    st.session_state.proactive_questions = None
if 'chat_input_key' not in st.session_state:
    st.session_state.chat_input_key = 0

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
    st.write("**Stage 0 – 🧠 Autonomous Planning (ReAct)**")
    prog_planning = st.progress(0)
    st.write("**Stage 1 – Resume Parsing**")
    prog_parsing = st.progress(0)
    st.write("**Stage 2 – ATS Keyword Filtering**")
    prog_ats = st.progress(0)
    st.write("**Stage 3 – LLM Extraction (Qwen)**")
    prog_extraction = st.progress(0)
    st.write("**Stage 4 – Candidate Scoring (Mistral)**")
    prog_ranking = st.progress(0)
    st.write("**Stage 4.5 – Self-Correction & Reflection**")
    prog_reflection = st.progress(0)
    st.write("**Stage 4.75 – Multi-Agent Debate**")
    prog_debate = st.progress(0)
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
    prog_planning.progress(0)
    prog_parsing.progress(0)
    prog_ats.progress(0)
    prog_extraction.progress(0)
    prog_ranking.progress(0)
    prog_reflection.progress(0)
    prog_debate.progress(0)
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
        reflection_agent = ReflectionAgent(llm_agent)
        skill_agent = SkillAgent(llm_agent)
        experience_agent = ExperienceAgent(llm_agent)
        culture_agent = CultureAgent(llm_agent)
        debate_mod = DebateModerator(skill_agent, experience_agent, culture_agent, llm_agent)
        planner = PlannerAgent(llm_agent)

        # Check Ollama
        ollama_up, available_models = llm_agent.check_ollama_status()
        if ollama_up:
            push_log(f"✅ Ollama connected. Models: {', '.join(available_models[:5])}", "SUCCESS")
        else:
            push_log("⚠️ Ollama not detected. Using heuristic analysis.", "WARNING")

        # ── Stage 0: Autonomous Planning (ReAct) ──
        push_log("🧠 Stage 0: Autonomous Planning...", "INFO")
        prog_planning.progress(20)
        plan = planner.plan(jd, keywords_input, resume_count=count)
        prog_planning.progress(80)

        push_log(f"   Strategy: {plan.strategy_label}", "SUCCESS")
        push_log(f"   Complexity: {plan.jd_complexity} | Est: {plan.estimated_time}", "INFO")
        for rec in plan.recommendations:
            push_log(f"   💡 {rec}", "INFO")

        # Apply plan's ATS top_n
        ats_filter = ATSFilter(top_n=plan.ats_top_n)
        prog_planning.progress(100)

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
                profile['ats_score'], profile['keyword_score'], profile['llm_score'],
                w_ats=plan.composite_weights['ats'],
                w_kw=plan.composite_weights['keyword'],
                w_llm=plan.composite_weights['llm']
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

        # ── Stage 4.5: Self-Correction & Reflection ──
        if plan.run_reflection:
            push_log("🔄 Stage 4.5: Running Self-Correction & Reflection...", "INFO")
            prog_reflection.progress(20)
            reflection_results = reflection_agent.reflect_batch(
                ranked[:shortlist_count], jd, use_llm=True
            )
            prog_reflection.progress(60)

            corrected_count = 0
            for candidate, reflection in zip(ranked[:shortlist_count], reflection_results):
                if reflection['was_corrected']:
                    old_score = reflection['original_score']
                    candidate['llm_score'] = reflection['corrected_score']
                    candidate['composite_score'] = CandidateRanker.compute_composite_score(
                        candidate['ats_score'], candidate['keyword_score'], candidate['llm_score']
                    )
                    candidate['reflection'] = reflection
                    corrected_count += 1
                    push_log(
                        f"   ⚠️ {candidate.get('name', candidate['filename'])}: "
                        f"{old_score} → {reflection['corrected_score']} "
                        f"({len(reflection['anomalies'])} anomalies)",
                        "WARNING"
                    )
                else:
                    candidate['reflection'] = reflection
                    push_log(
                        f"   ✅ {candidate.get('name', candidate['filename'])}: Score validated.",
                        "SUCCESS"
                    )

            ranked = CandidateRanker.rank_candidates(ranked)
            shortlisted = ranked[:shortlist_count]

            prog_reflection.progress(100)
            push_log(
                f"   ✓ Reflection complete: {corrected_count}/{len(reflection_results)} scores corrected.",
                "SUCCESS"
            )
        else:
            reflection_results = []
            prog_reflection.progress(100)
            push_log("⏭️ Stage 4.5: Reflection SKIPPED (planner decision).", "INFO")

        # ── Stage 4.75: Multi-Agent Debate ──
        if plan.run_debate:
            push_log("🏛️ Stage 4.75: Running Multi-Agent Debate...", "INFO")
            prog_debate.progress(15)

            debate_mod.WEIGHTS = plan.debate_weights

            debate_results = debate_mod.debate_batch(shortlisted, jd)
            prog_debate.progress(60)

            for candidate, debate in zip(shortlisted, debate_results):
                old_composite = candidate['composite_score']
                debate_score = debate['consensus_score']
                blended = round(old_composite * 0.60 + debate_score * 0.40, 2)
                candidate['composite_score'] = blended
                candidate['debate'] = debate
                push_log(
                    f"   🏛️ {candidate.get('name', candidate['filename'])}: "
                    f"Debate={debate_score} | S:{debate['skill_score']} E:{debate['experience_score']} "
                    f"C:{debate['culture_score']} | Blended={blended}",
                    "INFO"
                )

            ranked = CandidateRanker.rank_candidates(ranked)
            shortlisted = ranked[:shortlist_count]

            prog_debate.progress(100)
            push_log(f"   ✓ Multi-Agent Debate complete.", "SUCCESS")
        else:
            debate_results = []
            prog_debate.progress(100)
            push_log("⏭️ Stage 4.75: Debate SKIPPED (planner decision).", "INFO")

        # ── Stage 5: Final AI Insights (LLaMA) ──
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
            'reflection_results': reflection_results,
            'debate_results': debate_results,
            'plan': plan,
            'total_parsed': len(resumes),
            'total_filtered': len(top_resumes),
            'shortlist_count': shortlist_count,
            'keywords': keywords_list,
        }

        push_log("🎉 Pipeline execution finished successfully!", "SUCCESS")

        # Generate proactive questions from chat agent
        st.session_state.proactive_questions = st.session_state.chat_agent.get_proactive_questions(
            st.session_state.results
        )
        # Add welcome message to chat
        if not st.session_state.chat_messages:
            shortlisted = st.session_state.results.get('shortlisted', [])
            top_name = shortlisted[0].get('name', 'Unknown') if shortlisted else 'Unknown'
            welcome = (
                f"👋 Pipeline complete! I've analyzed **{st.session_state.results.get('total_parsed', 0)}** "
                f"resumes and shortlisted **{len(shortlisted)}** candidates. "
                f"**{top_name}** is the top pick.\n\n"
                f"Ask me anything — compare candidates, explain scores, or refine the rankings!"
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": welcome})

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
        <h2 style='margin: 5px 0; color: #f1f5f9 !important;'>CognifyX</h2>
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
    m1, m2, m3, m4, m5, m6 = st.columns(6)
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
    with m6:
        corrected = sum(1 for r in res.get('reflection_results', []) if r.get('was_corrected'))
        total_reflected = len(res.get('reflection_results', []))
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{corrected}/{total_reflected}</div>
            <div class='metric-label'>Scores Corrected</div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # TABBED RESULTS DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    tab_strategy, tab_shortlisted, tab_rankings, tab_insights, tab_report, tab_chat = st.tabs([
        "🧠 AI Strategy & Autonomous Reasoning",
        "🏆 Shortlisted Candidates",
        "📋 Full Candidate Rankings",
        "🤖 Final AI Insights & Recommendation",
        "📝 Automated Hiring Report",
        "💬 CognifyX AI Assistant",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1: AI Strategy & Autonomous Reasoning
    # ══════════════════════════════════════════════════════════════
    with tab_strategy:
        plan = res.get('plan')
        if plan:
            st.markdown("## 🧠 AI Strategy & Autonomous Reasoning")

            st.markdown(f"""
            <div class='plan-card'>
                <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>
                    <div>
                        <span style='font-size:1.6rem; font-weight:800; color:#e0e7ff;'>{plan.strategy_label}</span>
                        <br><span style='font-size:0.85rem; color:#a5b4fc;'>
                            Complexity: <b>{plan.jd_complexity.title()}</b> &nbsp;|&nbsp;
                            Est. Time: <b>{plan.estimated_time}</b> &nbsp;|&nbsp;
                            ATS Top-N: <b>{plan.ats_top_n}</b>
                        </span>
                    </div>
                    <div style='text-align:right;'>
                        <span style='font-size:0.75rem; background:rgba(99,102,241,0.3); padding:4px 12px;
                            border-radius:8px; color:#c7d2fe;'>🧠 ReAct Reasoning</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            plan_col1, plan_col2 = st.columns([1, 1])

            with plan_col1:
                st.markdown("#### 💡 Strategic Recommendations")
                for rec in plan.recommendations:
                    st.markdown(f"- {rec}")

                st.markdown("#### ⚖️ Dynamic Weight Configuration")
                st.markdown(f"""
                <table class='plan-weight-table'>
                    <tr><th>Component</th><th>Weight</th></tr>
                    <tr><td>ATS Score</td><td><b>{plan.composite_weights['ats']:.0%}</b></td></tr>
                    <tr><td>Keyword Match</td><td><b>{plan.composite_weights['keyword']:.0%}</b></td></tr>
                    <tr><td>LLM Score</td><td><b>{plan.composite_weights['llm']:.0%}</b></td></tr>
                    <tr><td>Debate: Skill</td><td><b>{plan.debate_weights.get('skill', 0.40):.0%}</b></td></tr>
                    <tr><td>Debate: Experience</td><td><b>{plan.debate_weights.get('experience', 0.35):.0%}</b></td></tr>
                    <tr><td>Debate: Culture</td><td><b>{plan.debate_weights.get('culture', 0.25):.0%}</b></td></tr>
                </table>
                """, unsafe_allow_html=True)

            with plan_col2:
                st.markdown("#### 🔄 ReAct Reasoning Trace")
                for step in plan.reasoning_steps:
                    phase = step.get('phase', '')
                    if phase == 'THINK':
                        st.markdown(f"<div class='react-step react-think'><b>💭 THINK:</b> {step.get('observation', '')}<br><span style='color:#a5b4fc;'>→ {step.get('thought', '')}</span></div>", unsafe_allow_html=True)
                    elif phase == 'ACT':
                        st.markdown(f"<div class='react-step react-act'><b>⚡ ACT:</b> <code>{step.get('action', '')}</code><br><span style='color:#6ee7b7;'>→ {step.get('detail', '')}</span></div>", unsafe_allow_html=True)
                    elif phase == 'OBSERVE':
                        st.markdown(f"<div class='react-step react-observe'><b>👁️ OBSERVE:</b> {step.get('result', '')}</div>", unsafe_allow_html=True)
                    elif phase == 'LLM_REASONING':
                        st.markdown(f"<div class='react-step react-llm'><b>🤖 LLM Validation:</b> {step.get('output', '')[:400]}</div>", unsafe_allow_html=True)

                st.markdown("#### 🔧 Stage Configuration")
                stages_status = [
                    ("ATS Filter", plan.run_ats_filter, f"top_n={plan.ats_top_n}"),
                    ("LLM Extraction", plan.run_extraction, "Qwen"),
                    ("LLM Scoring", plan.run_scoring, "Mistral"),
                    ("Self-Correction", plan.run_reflection, "Reflection"),
                    ("Multi-Agent Debate", plan.run_debate, "3 Agents"),
                    ("Final Insights", plan.run_insights, "LLaMA"),
                ]
                for label, enabled, detail in stages_status:
                    icon = "✅" if enabled else "⏭️"
                    status = "Enabled" if enabled else "**Skipped**"
                    st.markdown(f"- {icon} **{label}** — {status} ({detail})")
        else:
            st.info("No planning data available. Run the pipeline to see AI strategy details.")

    # ══════════════════════════════════════════════════════════════
    # TAB 2: Shortlisted Candidates
    # ══════════════════════════════════════════════════════════════
    with tab_shortlisted:
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
                skills = c.get('skills', [])
                if skills:
                    skill_html = " ".join([f"<span class='skill-tag'>{s}</span>" for s in skills[:8]])
                    st.markdown(f"**Skills:** {skill_html}", unsafe_allow_html=True)
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
                    <div class='score-badge {score_class}' style='font-size: 1.4rem; padding: 10px 24px;'>{composite}</div>
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
                        st.download_button(label="📄 Download", data=f, file_name=c['filename'], mime="application/octet-stream", key=f"dl_short_{i}")

            with st.expander("🔍 View AI Analysis"):
                st.write(c.get('llm_analysis', 'No analysis available.'))
            st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3: Full Candidate Rankings
    # ══════════════════════════════════════════════════════════════
    with tab_rankings:
        st.markdown("## 📋 Full Candidate Rankings")
        show_all = st.checkbox("Show all candidates (not just shortlisted)", value=False)
        display_list = res['candidates'] if show_all else res['shortlisted']
        table_data = []
        for c in display_list:
            table_data.append({
                "Rank": c.get('final_rank', '-'), "Name": c.get('name', c.get('filename', '')),
                "File": c.get('filename', ''), "Domain": c.get('domain', 'General'),
                "Experience": f"{c.get('experience_years', '?')} yrs",
                "ATS Score": c.get('ats_score', 0), "Keyword Match": f"{c.get('keyword_score', 0)}%",
                "LLM Score": c.get('llm_score', 0), "Composite": c.get('composite_score', 0),
                "Shortlisted": "✅" if c.get('shortlisted') else "—",
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=600)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Full Report (CSV)", data=csv_data, file_name=f'cognifyx_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', mime='text/csv', key="global_csv")

        # Self-Correction Report
        reflection_data = res.get('reflection_results', [])
        if reflection_data:
            st.markdown("---")
            st.markdown("### 🔄 AI Self-Correction Report")
            st.markdown("*The Reflection Agent audited each candidate's LLM score against hard facts from the JD and profile, correcting inflated or deflated scores.*")
            total_r = len(reflection_data)
            corrected_r = sum(1 for r in reflection_data if r.get('was_corrected'))
            validated_r = total_r - corrected_r
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #f59e0b;'>{corrected_r}</div><div class='metric-label'>Scores Corrected</div></div>", unsafe_allow_html=True)
            with rc2:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #10b981;'>{validated_r}</div><div class='metric-label'>Scores Validated</div></div>", unsafe_allow_html=True)
            with rc3:
                avg_adj = 0
                if corrected_r > 0:
                    avg_adj = round(sum(r.get('total_adjustment', 0) for r in reflection_data if r.get('was_corrected')) / corrected_r, 1)
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #ef4444;'>-{avg_adj}</div><div class='metric-label'>Avg Score Adjustment</div></div>", unsafe_allow_html=True)

            for ref in reflection_data:
                card_class = "corrected" if ref.get('was_corrected') else "validated"
                status_icon = "⚠️" if ref.get('was_corrected') else "✅"
                confidence = ref.get('confidence', 'MEDIUM')
                st.markdown(f"<div class='reflection-card {card_class}'>", unsafe_allow_html=True)
                ref_col1, ref_col2 = st.columns([3, 1])
                with ref_col1:
                    st.markdown(f"#### {status_icon} {ref.get('candidate', 'Unknown')}")
                    if ref.get('was_corrected'):
                        st.markdown(f"<div class='score-correction'><span class='score-original'>{ref.get('original_score', '?')}</span><span class='score-arrow'>→</span><span class='score-corrected'>{ref.get('corrected_score', '?')}</span><span style='font-size:0.8rem; color:#64748b;'>/100 (LLM Score)</span></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"Score **{ref.get('original_score', '?')}/100** — No issues found.")
                with ref_col2:
                    st.markdown(f"<div style='text-align:right; padding-top:10px;'><span class='confidence-badge confidence-{confidence}'>Confidence: {confidence}</span></div>", unsafe_allow_html=True)
                anomalies = ref.get('anomalies', [])
                if anomalies:
                    with st.expander(f"🔍 View {len(anomalies)} Anomalies Detected"):
                        for a in anomalies:
                            severity = a.get('severity', 'LOW')
                            st.markdown(f"<span class='anomaly-tag anomaly-{severity.lower()}'>{severity}</span> **{a.get('type', 'UNKNOWN').replace('_', ' ')}**", unsafe_allow_html=True)
                            st.write(f"  {a.get('description', '')}")
                llm_reflection = ref.get('llm_reflection')
                if llm_reflection:
                    with st.expander("🤖 View AI Reflection Analysis"):
                        st.write(llm_reflection)
                st.markdown("</div>", unsafe_allow_html=True)

        # Multi-Agent Debate Transcripts
        debate_data = res.get('debate_results', [])
        if debate_data:
            st.markdown("---")
            st.markdown("### 🏛️ Multi-Agent Debate Transcripts")
            st.markdown("*Three specialized agents debated each candidate. A Devil's Advocate challenged their conclusions, and the Moderator synthesized the final consensus score.*")
            for di, d in enumerate(debate_data):
                with st.expander(f"🏛️ {d.get('candidate', 'Unknown')} — Consensus: {d.get('consensus_score', '?')}/100 | Skill: {d.get('skill_score', '?')} | Exp: {d.get('experience_score', '?')} | Culture: {d.get('culture_score', '?')}{'  ⚡' + str(d.get('penalty', 0)) + 'pt penalty' if d.get('penalty', 0) > 0 else ''}", expanded=(di == 0)):
                    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                    with dcol1:
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #06b6d4;'>{d.get('skill_score', 0)}</div><div class='metric-label'>🔧 Skill</div></div>", unsafe_allow_html=True)
                    with dcol2:
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #8b5cf6;'>{d.get('experience_score', 0)}</div><div class='metric-label'>💼 Experience</div></div>", unsafe_allow_html=True)
                    with dcol3:
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #f59e0b;'>{d.get('culture_score', 0)}</div><div class='metric-label'>🤝 Culture</div></div>", unsafe_allow_html=True)
                    with dcol4:
                        penalty = d.get('penalty', 0)
                        penalty_text = f" (-{penalty})" if penalty > 0 else ""
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #10b981;'>{d.get('consensus_score', 0)}</div><div class='metric-label'>🏆 Consensus{penalty_text}</div></div>", unsafe_allow_html=True)
                    evaluations = d.get('evaluations', [])
                    for ev in evaluations:
                        agent = ev.get('agent', 'Unknown')
                        if 'Skill' in agent: css_class, icon = 'skill', '🔧'
                        elif 'Experience' in agent: css_class, icon = 'experience', '💼'
                        else: css_class, icon = 'culture', '🤝'
                        st.markdown(f"<div class='debate-agent {css_class}'><div class='debate-agent-header'><span class='debate-agent-name'>{icon} {agent} ({ev.get('model', '')})</span><span class='debate-score-pill'>{ev.get('score', 0)}/100</span></div><div style='font-size:0.83rem; color:#475569; white-space: pre-wrap;'>{ev.get('argument', '')[:500]}</div></div>", unsafe_allow_html=True)
                    devils = d.get('devils_advocacy', {})
                    challenges = devils.get('challenges', [])
                    if challenges:
                        challenge_text = ""
                        for ch in challenges:
                            sev = ch.get('severity', 'LOW')
                            sev_color = {'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#64748b'}.get(sev, '#64748b')
                            challenge_text += f"<span style='color:{sev_color}; font-weight:700;'>[{sev}]</span> {ch.get('challenge', '')}<br>"
                        st.markdown(f"<div class='debate-agent devil'><div class='debate-agent-header'><span class='debate-agent-name'>⚡ Devil's Advocate</span><span class='debate-score-pill' style='background:#fee2e2; color:#991b1b;'>{devils.get('challenge_count', 0)} challenges</span></div><div style='font-size:0.83rem; color:#475569;'>{challenge_text}</div></div>", unsafe_allow_html=True)
                    mod_summary = d.get('moderator_summary', '')
                    if mod_summary:
                        st.markdown(f"<div class='debate-agent moderator'><div class='debate-agent-header'><span class='debate-agent-name'>🏆 Moderator Verdict</span><span class='debate-score-pill' style='background:#d1fae5; color:#065f46;'>Consensus: {d.get('consensus_score', 0)}/100</span></div><div style='font-size:0.83rem; color:#475569; white-space: pre-wrap;'>{mod_summary[:600]}</div></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4: Final AI Insights & Recommendation
    # ══════════════════════════════════════════════════════════════
    with tab_insights:
        st.markdown("## 🤖 Final AI Insights & Recommendation")
        st.markdown(f"<div class='insights-panel'>{res['insights']}</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 5: Automated Hiring Report
    # ══════════════════════════════════════════════════════════════
    with tab_report:
        st.markdown("## 📝 Automated Hiring Report")
        st.markdown("*Generate a polished, professional hiring report with visualizations, comparisons, and actionable recommendations.*")
        if 'report_data' not in st.session_state:
            st.session_state.report_data = None
        gen_col1, gen_col2, gen_col3 = st.columns([2, 1, 1])
        with gen_col1:
            if st.button("🧠 Generate Full Report", key="gen_report", use_container_width=True):
                with st.spinner("Generating report sections..."):
                    report_agent = ReportAgent()
                    st.session_state.report_data = report_agent.generate_full_report(res, job_description=st.session_state.get('last_jd', ''))
                st.success("✅ Report generated!")
                st.rerun()
        report = st.session_state.report_data
        if report:
            with gen_col2:
                try:
                    pdf_gen = PDFReportGenerator()
                    pdf_bytes = pdf_gen.generate(report)
                    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=f"CognifyX_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", key="dl_pdf_report", use_container_width=True)
                except Exception as e:
                    st.warning(f"PDF unavailable: {e}")
            with gen_col3:
                try:
                    docx_gen = DOCXReportGenerator()
                    docx_bytes = docx_gen.generate(report)
                    st.download_button("📝 Download DOCX", data=docx_bytes, file_name=f"CognifyX_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_docx_report", use_container_width=True)
                except Exception as e:
                    st.warning(f"DOCX unavailable: {e}")
            exec_sum = report.get('executive_summary', {})
            if exec_sum:
                with st.expander("📋 Executive Summary", expanded=True):
                    st.markdown(exec_sum.get('content', ''))
                    stats = exec_sum.get('stats', {})
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Screened", stats.get('total_screened', 0))
                    sc2.metric("Filtered", stats.get('ats_filtered', 0))
                    sc3.metric("Shortlisted", stats.get('shortlisted', 0))
                    sc4.metric("Top Score", f"{stats.get('top_score', 0)}/100")
            matrix = report.get('comparison_matrix', {})
            matrix_rows = matrix.get('rows', [])
            if matrix_rows:
                with st.expander("📊 Candidate Comparison Matrix"):
                    import pandas as pd
                    df_matrix = pd.DataFrame(matrix_rows)
                    display_cols = ['rank', 'name', 'domain', 'experience_years', 'ats_score', 'keyword_score', 'llm_score', 'composite_score', 'consensus_score']
                    available = [c for c in display_cols if c in df_matrix.columns]
                    st.dataframe(df_matrix[available], use_container_width=True, hide_index=True)
            skill_gap = report.get('skill_gap_analysis', {})
            if skill_gap:
                with st.expander("🔍 Skill Gap Analysis"):
                    gap_col1, gap_col2 = st.columns([2, 1])
                    with gap_col1:
                        coverage = skill_gap.get('coverage', [])
                        if coverage:
                            for c in coverage:
                                status = c['status']
                                if status == 'Strong': icon = '🟢'
                                elif status == 'Moderate': icon = '🟡'
                                else: icon = '🔴'
                                st.markdown(f"{icon} **{c['keyword']}** — {c['coverage_pct']}% coverage ({c['candidates_with']} candidates)")
                    with gap_col2:
                        st.metric("Total Keywords", skill_gap.get('total_keywords', 0))
                        st.metric("Strong Coverage", skill_gap.get('strong_coverage', 0))
                        st.metric("Gaps Found", skill_gap.get('gaps_found', 0))
            iq = report.get('interview_questions', {})
            if iq:
                with st.expander("💬 Interview Question Bank"):
                    st.markdown(f"*{iq.get('total_questions', 0)} personalized questions generated*")
                    for cand in iq.get('candidates', []):
                        st.markdown(f"#### #{cand['rank']} – {cand['candidate']}")
                        for q in cand['questions']:
                            st.markdown(f"**[{q['type']}]** {q['question']}  \n<small style='color:#64748b;'>📌 {q['reason']}</small>", unsafe_allow_html=True)
                        st.markdown("---")
            diversity = report.get('diversity_metrics', {})
            if diversity:
                with st.expander("🌍 Diversity & Inclusion Metrics"):
                    div_col1, div_col2, div_col3 = st.columns(3)
                    div_col1.metric("Diversity Index", f"{diversity.get('diversity_index', 0)}%")
                    div_col2.metric("Unique Domains", diversity.get('unique_domains', 0))
                    div_col3.metric("Total Pool", diversity.get('total_pool', 0))
                    st.markdown("**Domain Distribution:**")
                    for domain, count in diversity.get('domain_distribution', {}).items():
                        st.markdown(f"- **{domain}**: {count} candidates")
                    st.markdown("**Education Distribution:**")
                    for edu, count in diversity.get('education_distribution', {}).items():
                        if count > 0: st.markdown(f"- **{edu}**: {count}")
                    st.markdown("**Experience Distribution:**")
                    for bucket, count in diversity.get('experience_distribution', {}).items():
                        if count > 0: st.markdown(f"- **{bucket}**: {count}")
            salary = report.get('salary_benchmarks', {})
            benchmarks = salary.get('benchmarks', [])
            if benchmarks:
                with st.expander("💰 Salary Benchmarking"):
                    import pandas as pd
                    df_salary = pd.DataFrame(benchmarks)
                    display_cols = ['rank', 'name', 'level', 'salary_min', 'salary_max', 'midpoint', 'composite_score']
                    available = [c for c in display_cols if c in df_salary.columns]
                    st.dataframe(df_salary[available], use_container_width=True, hide_index=True)
                    st.caption(salary.get('note', ''))

    # ══════════════════════════════════════════════════════════════
    # TAB 6: CognifyX AI Assistant
    # ══════════════════════════════════════════════════════════════
    with tab_chat:
        st.markdown("""
        <div class='chat-container'>
            <div class='chat-header'>
                <span class='chat-header-icon'>💬</span>
                <span class='chat-header-title'>CognifyX AI Assistant</span>
                <span class='chat-header-badge'>Agentic AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        proactive_qs = st.session_state.proactive_questions
        if proactive_qs and not any(m.get('role') == 'user' for m in st.session_state.chat_messages):
            st.markdown("**🤔 I have some questions before you proceed:**")
            for i, q in enumerate(proactive_qs):
                if st.button(q, key=f"proactive_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    response = st.session_state.chat_agent.chat(q, st.session_state.results)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.session_state.proactive_questions = None
                    st.rerun()
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-msg user'><div class='chat-bubble user-bubble'>{msg['content']}</div><div class='chat-avatar user-av'>👤</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-msg assistant'><div class='chat-avatar ai-av'>🤖</div><div class='chat-bubble ai-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        chat_col1, chat_col2 = st.columns([5, 1])
        with chat_col1:
            user_input = st.text_input("Ask CognifyX AI...", placeholder="Compare #1 and #2 | Why is #3 ranked here? | Prioritize experience | Summary", key=f"chat_input_{st.session_state.chat_input_key}", label_visibility="collapsed")
        with chat_col2:
            send_clicked = st.button("🚀 Send", use_container_width=True, key="chat_send_btn")
        if (send_clicked or user_input) and user_input and user_input.strip():
            st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
            response = st.session_state.chat_agent.chat(user_input.strip(), st.session_state.results)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.session_state.proactive_questions = None
            st.session_state.chat_input_key += 1
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
        with qa_col1:
            if st.button("📊 Summary", key="qa_summary", use_container_width=True):
                msg = "Give me a summary"
                st.session_state.chat_messages.append({"role": "user", "content": msg})
                response = st.session_state.chat_agent.chat(msg, st.session_state.results)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with qa_col2:
            if st.button("🔍 Compare Top 2", key="qa_compare", use_container_width=True):
                msg = "Compare #1 and #2"
                st.session_state.chat_messages.append({"role": "user", "content": msg})
                response = st.session_state.chat_agent.chat(msg, st.session_state.results)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with qa_col3:
            if st.button("💡 Explain #1", key="qa_explain", use_container_width=True):
                msg = "Explain why #1 is ranked first"
                st.session_state.chat_messages.append({"role": "user", "content": msg})
                response = st.session_state.chat_agent.chat(msg, st.session_state.results)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with qa_col4:
            if st.button("🗑️ Clear Chat", key="qa_clear", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.chat_agent.conversation_history = []
                st.session_state.chat_agent.preferences = {}
                st.session_state.proactive_questions = st.session_state.chat_agent.get_proactive_questions(st.session_state.results)
                st.rerun()

elif not st.session_state.results:
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 20px;
        padding: 40px; text-align: center; border: 1px solid #334155;'>
        <span style='font-size: 3rem;'>💬</span>
        <h3 style='color: #f1f5f9 !important; margin-top: 12px;'>CognifyX AI Assistant</h3>
        <p style='color: #94a3b8 !important; max-width: 500px; margin: 0 auto;'>
            Run the screening pipeline to activate the AI Assistant.
            Ask questions, compare candidates, refine rankings, and get personalized recommendations.
        </p>
        <div style='margin-top: 16px;'>
            <span class='chat-header-badge'>Agentic AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<br><hr>
<center style='color: #94a3b8; font-size: 0.8rem;'>
    © 2026 CognifyX AI Resume Intelligence | Powered by Ollama (Qwen • Mistral • LLaMA)
</center>
""", unsafe_allow_html=True)
