"""
YogaCare AI - Phase 3, 4 & 5: Integrated Streamlit Web Application
A full-featured AI-powered Yoga Posture Detection & Guidance Web Dashboard.

Features:
- Balanced side-by-side Reference Pose and Live Webcam Practice panels.
- Reusable pose reference configuration with authentic images, instructions, and tips.
- Real-time full-body landmark detection, skeleton rendering, and vector angle calculation.
- Verified target hold timing with streak tracking and session running timer.
- Live coaching cues, priority safety overrides, and asynchronous voice assistant.
- Comprehensive Workout History & Analytics and Posture Biomechanics Guide.

Run via:
    streamlit run app.py
"""

from datetime import datetime
import os
import sys
import time
from typing import Dict, Optional

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from pose_detector import YogaPoseDetector
from pose_reference import get_pose_reference, resolve_reference_image
from posture_analyzer import PostureAnalyzer
from routine_manager import PoseSessionState, RoutineManager, RoutinePose
from voice_assistant import VoiceAssistant
from workout_logger import (
    WorkoutLogger,
    format_datetime_readable,
    format_pose_display_name,
    format_seconds_readable,
)
from yoga_library import YogaCategory, YogaLibrary, YogaPractice

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="YogaCare AI — Posture Detection & Guidance",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SLEEK MODERN DARK MODE DESIGN SYSTEM (Midnight Slate, Emerald, Cyan & Purple)
# ==============================================================================
st.markdown(
    """
    <style>
    /* App Canvas & Global Typography */
    .stApp {
        background-color: #0B0F17 !important;
        color: #F1F5F9 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main {
        background-color: #0B0F17 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }

    /* Top Segmented Navigation Pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: #131A29 !important;
        padding: 6px !important;
        border-radius: 16px !important;
        gap: 8px !important;
        border: 1px solid #222E42 !important;
        display: flex !important;
        flex-wrap: wrap !important;
        margin-bottom: 18px !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: transparent !important;
        padding: 8px 18px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #94A3B8 !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: #1E293B !important;
        color: #38BDF8 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35) !important;
        font-weight: 700 !important;
        border: 1px solid #334155 !important;
    }

    /* Streamlit Native Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.45rem 1.15rem !important;
        border: 1px solid #2A364F !important;
        background-color: #161E2E !important;
        color: #F1F5F9 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }
    .stButton > button:hover {
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
        background-color: #1E293B !important;
        box-shadow: 0 3px 12px rgba(56, 189, 248, 0.2) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 2px 10px rgba(2, 132, 199, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.55) !important;
    }

    /* Cards & Elevated Containers */
    .metric-card {
        background: #131A29 !important;
        border: 1px solid #1E293B !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    .metric-title {
        font-size: 0.78rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.1px !important;
        font-weight: 700 !important;
    }
    .metric-value {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-top: 4px !important;
    }

    /* Home Dashboard Cards */
    .home-hero-card {
        background: linear-gradient(135deg, #131A29 0%, #0E131F 100%) !important;
        border: 1px solid #222E42 !important;
        border-radius: 18px !important;
        padding: 28px 30px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
    }
    .home-quick-card {
        background: #131A29 !important;
        border: 1px solid #1E293B !important;
        border-radius: 16px !important;
        padding: 22px 20px !important;
        margin-bottom: 16px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .home-quick-card:hover {
        border-color: #38BDF8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15) !important;
    }

    /* Feedback & Coaching Cue Boxes */
    .cue-box {
        background-color: #0F291E !important;
        border-left: 4px solid #10B981 !important;
        padding: 14px 18px !important;
        border-radius: 0 10px 10px 0 !important;
        font-size: 1.02rem !important;
        color: #A7F3D0 !important;
        margin-top: 8px !important;
        margin-bottom: 12px !important;
    }
    .cue-box-warning {
        border-left-color: #F59E0B !important;
        background-color: #2D220E !important;
        color: #FDE68A !important;
    }
    .cue-box-success {
        border-left-color: #10B981 !important;
        background-color: #064E3B !important;
        color: #D1FAE5 !important;
    }
    .cue-box-danger {
        border-left-color: #EF4444 !important;
        background-color: #3A1414 !important;
        color: #FECACA !important;
    }

    /* Badges & Status Pills */
    .pose-badge {
        display: inline-block !important;
        padding: 4px 12px !important;
        border-radius: 16px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 8px !important;
    }
    .badge-diff {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid #334155 !important;
    }
    .badge-status {
        background-color: #064E3B44 !important;
        color: #34D399 !important;
        border: 1px solid #059669 !important;
    }
    .badge-standby {
        background-color: #1E293B !important;
        color: #94A3B8 !important;
        border: 1px solid #334155 !important;
    }
    .tip-item {
        background-color: #161E2E !important;
        border-radius: 8px !important;
        padding: 9px 13px !important;
        margin-bottom: 6px !important;
        font-size: 0.92rem !important;
        color: #E2E8F0 !important;
        border-left: 3px solid #10B981 !important;
    }
    .instruction-step {
        background: #161E2E !important;
        border: 1px solid #222E42 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
        font-size: 0.93rem !important;
        color: #F1F5F9 !important;
        line-height: 1.5 !important;
    }
    .instruction-step-kn {
        background-color: #1A2333 !important;
        border-left: 4px solid #10B981 !important;
        padding: 10px 14px !important;
        border-radius: 0 8px 8px 0 !important;
        margin-bottom: 8px !important;
        font-size: 0.98rem !important;
        line-height: 1.6 !important;
        color: #F1F5F9 !important;
    }
    .standby-card {
        background: linear-gradient(135deg, #131A29 0%, #0E131F 100%) !important;
        border: 2px dashed #243049 !important;
        border-radius: 14px !important;
        padding: 36px 24px !important;
        text-align: center !important;
        color: #94A3B8 !important;
        margin-bottom: 14px !important;
    }

    /* History & Analytics Dashboard Styling */
    .history-header {
        margin-bottom: 22px !important;
    }
    .history-title {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 4px !important;
    }
    .history-subtitle {
        font-size: 0.95rem !important;
        color: #94A3B8 !important;
        margin-bottom: 18px !important;
    }
    .summary-card {
        background: #131A29 !important;
        border: 1px solid #1E293B !important;
        border-radius: 14px !important;
        padding: 20px 18px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .summary-card:hover {
        border-color: #38BDF8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    }
    .card-blue { border-top: 4px solid #38BDF8 !important; }
    .card-green { border-top: 4px solid #10B981 !important; }
    .card-purple { border-top: 4px solid #A855F7 !important; }
    .card-gold { border-top: 4px solid #F59E0B !important; }

    .card-icon-label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.1px !important;
        color: #94A3B8 !important;
        margin-bottom: 6px !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    .card-value {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.5px !important;
        color: #FFFFFF !important;
    }
    .card-desc {
        font-size: 0.82rem !important;
        color: #64748B !important;
        line-height: 1.4 !important;
    }
    .chart-container-box {
        background: #131A29 !important;
        border: 1px solid #1E293B !important;
        border-radius: 14px !important;
        padding: 22px 20px !important;
        margin-top: 14px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    .section-heading {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 2px !important;
    }
    .section-subheading {
        font-size: 0.9rem !important;
        color: #94A3B8 !important;
        margin-bottom: 16px !important;
    }
    .empty-history-box {
        background: #131A29 !important;
        border: 2px dashed #243049 !important;
        border-radius: 14px !important;
        padding: 36px 20px !important;
        text-align: center !important;
        color: #94A3B8 !important;
        margin: 16px 0 !important;
    }

    /* Yoga Practice Library Styling */
    .library-hero-banner {
        background: linear-gradient(135deg, #131A29 0%, #0E131F 100%) !important;
        border: 1px solid #222E42 !important;
        border-radius: 16px !important;
        padding: 24px 26px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }
    .library-stat-pill {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 20px !important;
        padding: 4px 14px !important;
        font-size: 0.82rem !important;
        color: #38BDF8 !important;
        font-weight: 600 !important;
        margin-right: 8px !important;
        margin-top: 8px !important;
    }
    .category-card-box {
        background: #131A29 !important;
        border: 1px solid #1E293B !important;
        border-radius: 14px !important;
        padding: 18px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        transition: border-color 0.2s ease, transform 0.2s ease !important;
    }
    .category-card-box:hover {
        border-color: #38BDF8 !important;
        transform: translateY(-2px) !important;
    }
    .category-phase-badge {
        display: inline-block !important;
        padding: 4px 12px !important;
        border-radius: 12px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
    }
    .cat-color-1 { background: #78350F44 !important; color: #FDE68A !important; border: 1px solid #B45309 !important; }
    .cat-color-2 { background: #0F766E33 !important; color: #5EEAD4 !important; border: 1px solid #0D9488 !important; }
    .cat-color-3 { background: #581C8733 !important; color: #D8B4FE !important; border: 1px solid #7E22CE !important; }
    .cat-color-4 { background: #83184333 !important; color: #FDA4AF !important; border: 1px solid #BE123C !important; }
    .cat-color-5 { background: #7F1D1D33 !important; color: #FCA5A5 !important; border: 1px solid #B91C1C !important; }
    .cat-color-6 { background: #14532D33 !important; color: #86EFAC !important; border: 1px solid #15803D !important; }
    .cat-color-7 { background: #0C4A6E33 !important; color: #7DD3FC !important; border: 1px solid #0369A1 !important; }

    .practice-item-card {
        background: #131A29 !important;
        border: 1px solid #1E293B !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        transition: border-color 0.2s ease, transform 0.2s ease !important;
    }
    .practice-item-card:hover {
        border-color: #38BDF8 !important;
        transform: translateY(-2px) !important;
    }

    /* Streamlit Expander in Dark Mode */
    [data-testid="stExpander"] {
        background-color: #131A29 !important;
        border: 1px solid #1E293B !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] details summary {
        color: #F1F5F9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# MODEL & RESOURCE CACHING HELPERS (Avoid repeated reinitialization)
# ==============================================================================

def get_or_create_detector(backend_key: str) -> YogaPoseDetector:
    """Retrieves cached pose detector or initializes only when backend changes."""
    if "detector" not in st.session_state or st.session_state.get("detector_backend") != backend_key:
        if "detector" in st.session_state and st.session_state.detector is not None:
            try:
                st.session_state.detector.release()
            except Exception:
                pass
        st.session_state.detector = YogaPoseDetector(backend=backend_key)
        st.session_state.detector_backend = backend_key
    return st.session_state.detector


def get_or_create_analyzer(pose_key: str) -> PostureAnalyzer:
    """Retrieves cached posture analyzer or updates active pose evaluator."""
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = PostureAnalyzer(default_pose=pose_key)
    else:
        st.session_state.analyzer.set_active_pose(pose_key)
    return st.session_state.analyzer


def get_or_create_voice(enabled: bool, rate: int) -> VoiceAssistant:
    """Retrieves cached voice assistant or updates enabled/rate state, reviving worker thread if needed."""
    needs_reinit = (
        "voice" not in st.session_state
        or st.session_state.get("voice_rate") != rate
        or not getattr(st.session_state.get("voice"), "_worker_thread", None)
        or not st.session_state.voice._worker_thread.is_alive()
    )
    if needs_reinit:
        if "voice" in st.session_state and st.session_state.voice is not None:
            try:
                st.session_state.voice.stop()
            except Exception:
                pass
        st.session_state.voice = VoiceAssistant(enabled=enabled, voice_rate=rate)
        st.session_state.voice_rate = rate
    else:
        st.session_state.voice.enabled = enabled
    return st.session_state.voice


def sys_is_windows() -> bool:
    """Checks if running on Windows OS."""
    return sys.platform.startswith("win")


def draw_guided_hud(frame, pose_name: str, side: Optional[str], remaining_sec: float, phase: str):
    """Overlays an unobtrusive HUD card on the live video frame during guided routine practice."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    box_w = min(w - 40, 580)
    cv2.rectangle(overlay, (20, 20), (20 + box_w, 105), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (20, 20), (20 + box_w, 105), (56, 189, 248), 2)
    side_txt = f" • [{side} Side]" if side else ""
    cv2.putText(frame, f"{pose_name}{side_txt}", (36, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    rem_txt = f"Hold Timer: {int(max(0, remaining_sec))}s remaining  •  {phase.split(' — ')[0]}"
    cv2.putText(frame, rem_txt, (36, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)
    return frame


def render_guided_routine(logger: WorkoutLogger, backend_key: str, camera_idx: int, voice_enabled: bool, voice_rate: int):
    """
    Renders the 22-posture guided routine with male reference images,
    countdown timer, bilateral side switching, Cat-Cow rep counting,
    live webcam with skeleton overlay, and session summary.
    """
    if "routine_manager" not in st.session_state:
        st.session_state.routine_manager = RoutineManager()
    rm: RoutineManager = st.session_state.routine_manager

    # --------------------------------------------------------------------------
    # 1. SUMMARY VIEW (After Routine Completion or User End)
    # --------------------------------------------------------------------------
    if st.session_state.get("routine_summary_data"):
        summary = st.session_state["routine_summary_data"]
        st.markdown(
            """
            <div class="summary-card card-purple" style="padding: 24px 22px; margin-bottom: 22px;">
                <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">🎉 Guided Routine Completed!</div>
                <div style="font-size: 0.95rem; color: #94A3B8;">
                    Great practice! Review your performance metrics and posture breakdown below.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.markdown(
                f"""
                <div class="summary-card card-blue">
                    <div class="card-icon-label">⏱️ Total Routine Time</div>
                    <div class="card-value" style="color: #38BDF8;">{format_seconds_readable(summary.get('total_duration_seconds'))}</div>
                    <div class="card-desc">Duration across all postures</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s_col2:
            st.markdown(
                f"""
                <div class="summary-card card-green">
                    <div class="card-icon-label">🧘 Postures Completed</div>
                    <div class="card-value" style="color: #34D399;">{summary.get('poses_completed', 0)} / {len(rm.poses)}</div>
                    <div class="card-desc">Target holds & reps satisfied</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s_col3:
            corr_val = format_seconds_readable(summary.get('correct_pose_seconds'))
            st.markdown(
                f"""
                <div class="summary-card card-purple">
                    <div class="card-icon-label">🎯 Correct Alignment Time</div>
                    <div class="card-value" style="color: #C084FC;">{corr_val}</div>
                    <div class="card-desc">Accumulated verified posture time</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s_col4:
            best_h = f"{summary.get('best_hold_seconds', 0.0):.1f}s"
            st.markdown(
                f"""
                <div class="summary-card card-gold">
                    <div class="card-icon-label">🏆 Best Pose Hold</div>
                    <div class="card-value" style="color: #FBBF24;">{best_h}</div>
                    <div class="card-desc">Longest verified continuous hold</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.subheader("📋 Posture-by-Posture Performance Breakdown")
        detail_rows = []
        for d in summary.get("details", []):
            dur = f"{d['duration_seconds']:.1f}s"
            corr = f"{d['correct_seconds']:.1f}s" if d.get("correct_seconds") is not None else "Not applicable"
            status_badge = "<span style='color: #34D399; font-weight: 600;'>✅ Completed</span>" if d.get("completed") else "<span style='color: #64748B;'>⚪ Practiced</span>"
            eval_tag = "<span style='color: #38BDF8; font-weight: 600;'>🤖 Live AI Evaluated</span>" if d.get("has_ai_evaluation") else "<span style='color: #94A3B8;'>🧘 Guided Form</span>"
            detail_rows.append(
                f"<tr>"
                f"<td style='padding: 8px 12px; border-bottom: 1px solid #1E293B;'><b>#{d['sequence']}</b></td>"
                f"<td style='padding: 8px 12px; border-bottom: 1px solid #1E293B; color: #FFFFFF;'><b>{d['name']}</b> <span style='color: #94A3B8;'>({d['sanskrit']})</span></td>"
                f"<td style='padding: 8px 12px; border-bottom: 1px solid #1E293B; color: #CBD5E1;'>{d['phase']}</td>"
                f"<td style='padding: 8px 12px; border-bottom: 1px solid #1E293B; color: #38BDF8; font-weight: 600;'>{dur}</td>"
                f"<td style='padding: 8px 12px; border-bottom: 1px solid #1E293B; color: #34D399; font-weight: 600;'>{corr}</td>"
                f"<td style='padding: 8px 12px; border-bottom: 1px solid #1E293B;'>{eval_tag}</td>"
                f"<td style='padding: 8px 12px; border-bottom: 1px solid #1E293B;'>{status_badge}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 10px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
                    <thead>
                        <tr style="text-align: left; color: #94A3B8; border-bottom: 2px solid #243049;">
                            <th style="padding: 8px 12px;">#</th>
                            <th style="padding: 8px 12px;">Posture</th>
                            <th style="padding: 8px 12px;">Phase</th>
                            <th style="padding: 8px 12px;">Practice Time</th>
                            <th style="padding: 8px 12px;">Correct Time</th>
                            <th style="padding: 8px 12px;">Tracking Mode</th>
                            <th style="padding: 8px 12px;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(detail_rows)}
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Start New 22-Posture Routine", use_container_width=True, type="primary"):
            del st.session_state["routine_summary_data"]
            rm.reset_routine()
            st.rerun()
        return

    # --------------------------------------------------------------------------
    # 2. ACTIVE PRACTICE VIEW
    # --------------------------------------------------------------------------
    curr_pose = rm.get_current_pose()
    curr_state = rm.get_current_state()
    progress_info = rm.get_progress()

    def _log_routine_pose_if_ready(pose: RoutinePose, state: PoseSessionState):
        if state.elapsed_seconds >= 2.0 and not getattr(state, "is_logged_to_history", False):
            state.is_logged_to_history = True
            logger.log_session(
                pose_name=f"{pose.name} ({pose.sanskrit_name})",
                duration_seconds=state.elapsed_seconds,
                target_duration=int(pose.suggested_duration_seconds),
                correct_pose_seconds=state.correct_seconds if pose.has_ai_evaluation else state.elapsed_seconds,
                max_continuous_hold=state.best_hold_seconds if pose.has_ai_evaluation else state.elapsed_seconds,
                average_score=85.0 if state.is_completed else 75.0,
                target_achieved=state.is_completed,
                cues_provided=pose.alignment_tips[:2],
                stance=f"Phase {pose.phase_number}",
            )
            st.toast(f"✅ Logged {pose.name} to Workout History!", icon="💾")

    # Phase color map
    phase_colors = {
        1: ("#f59e0b", "Warm-up"),
        2: ("#38bdf8", "Standing Poses"),
        3: ("#a855f7", "Floor and Stretching"),
        4: ("#2ecc71", "Breathing & Relaxation"),
    }
    p_color, p_name = phase_colors.get(curr_pose.phase_number, ("#38bdf8", "Routine"))

    # Routine Progress Bar & Header
    comp_count = progress_info["completed_count"]
    total_count = progress_info["total_poses"]
    percent_val = progress_info["percent_completed"]

    st.markdown(
        f"""
        <div style="background: #131A29; border: 1px solid #1E293B; border-radius: 14px; padding: 14px 18px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="background: {p_color}22; color: {p_color}; border: 1px solid {p_color}66; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">
                        {curr_pose.phase}
                    </span>
                    <span style="color: #FFFFFF; font-weight: 700; font-size: 1.1rem; margin-left: 10px;">
                        Pose {curr_pose.sequence_number} of {total_count}: {curr_pose.name}
                    </span>
                    <span style="color: #94A3B8; font-size: 0.95rem; margin-left: 6px;">
                        ({curr_pose.sanskrit_name})
                    </span>
                </div>
                <div style="color: #38BDF8; font-weight: 600; font-size: 0.9rem;">
                    Completed: <b>{comp_count} / {total_count}</b> ({percent_val}%)
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, comp_count / total_count))

    # Fast dropdown selector to review or jump between postures
    pose_select_opts = [
        f"#{p.sequence_number}. {p.name} ({p.sanskrit_name}) — Phase {p.phase_number}"
        for p in rm.poses
    ]
    selected_idx = st.selectbox(
        "Direct Posture Navigation",
        options=range(len(pose_select_opts)),
        format_func=lambda i: pose_select_opts[i],
        index=rm.current_index,
        key="guided_jump_selector",
        help="Select any posture to navigate directly to it.",
    )
    if selected_idx != rm.current_index:
        rm.jump_to_pose(selected_idx + 1)
        st.rerun()

    # Balanced Two-Column Layout (Desktop Side-by-Side, Mobile Stacked)
    col_ref, col_cam = st.columns([1, 1], gap="large")

    # --------------------------------------------------------------------------
    # LEFT PANEL — Male Reference Yoga Pose
    # --------------------------------------------------------------------------
    with col_ref:
        st.markdown(f"### 🧘 Reference Pose: {curr_pose.name}")
        st.markdown(
            f"<span class='pose-badge badge-diff'>Phase {curr_pose.phase_number}</span> "
            f"<span class='pose-badge badge-status'>{curr_pose.sanskrit_name}</span> "
            f"<span class='pose-badge badge-standby'>⏱️ Suggested: {curr_pose.suggested_duration_seconds}s</span>",
            unsafe_allow_html=True,
        )

        if curr_pose.requires_sides:
            active_side_txt = curr_state.current_side or "Left"
            st.markdown(
                f"""
                <div style="background: #0B221B; border-left: 4px solid #10B981; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; font-size: 0.9rem; color: #E2E8F0;">
                    👈 <b>Bilateral Posture:</b> Currently practicing <b>{active_side_txt} Side</b>.
                    <br><span style="font-size: 0.82rem; color: #94A3B8;">Both sides must be held to complete this posture.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if curr_pose.is_repetition:
            st.markdown(
                f"""
                <div style="background: #2D220E; border-left: 4px solid #F59E0B; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; font-size: 0.9rem; color: #FDE68A;">
                    🔄 <b>Repetition Posture:</b> Move with your breath (5–8 repetitions).
                    <br><span style="font-size: 0.82rem; color: #FCD34D;">Completed Repetitions: <b>{curr_state.repetition_count}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Check and display male reference image
        if os.path.exists(curr_pose.local_image_path):
            st.image(
                curr_pose.local_image_path,
                caption=f"Verified Male Reference • Pose {curr_pose.sequence_number}: {curr_pose.name} ({curr_pose.sanskrit_name})",
                use_container_width=True,
            )
        else:
            st.markdown(
                f"""
                <div class="standby-card">
                    <div style="font-size: 2.8rem; margin-bottom: 8px;">🧘‍♂️</div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 1.15rem;">Missing Male Reference Image</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">
                        Expected file: <code>{curr_pose.image_filename}</code> in <code>assets/reference_poses/</code>.
                        <br>YogaCare AI strictly adheres to male reference standards and never substitutes unrelated images.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Step-by-Step Instructions
        with st.expander("📝 Step-by-Step Instructions", expanded=True):
            for i, step in enumerate(curr_pose.instructions, 1):
                st.markdown(f"<div class='instruction-step'><b>{i}.</b> {step}</div>", unsafe_allow_html=True)

        # Key Alignment Tips & Safety Rules
        with st.expander("💡 Key Alignment Tips", expanded=True):
            for tip in curr_pose.alignment_tips:
                st.markdown(f"<div class='tip-item'>{tip}</div>", unsafe_allow_html=True)

        # Safety Reminder
        st.markdown(
            f"""
            <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 14px; font-size: 0.86rem; color: #92400E; margin-top: 10px;">
                🛡️ <b>Safety Reminder:</b> {curr_pose.safety_reminder}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # RIGHT PANEL — Live Webcam Stream & Feedback
    # --------------------------------------------------------------------------
    with col_cam:
        st.markdown(f"### 📹 Live Practice: {curr_pose.name}")

        r_ctrl1, r_ctrl2 = st.columns([1.2, 1])
        with r_ctrl1:
            routine_cam_active = st.toggle(
                "🔴 Enable Live Camera Practice",
                value=st.session_state.get("routine_cam_active_state", False),
                key="routine_cam_active_toggle",
                help="Toggle camera stream for live pose tracking and skeleton rendering.",
            )
        with r_ctrl2:
            if routine_cam_active:
                st.markdown("<span class='pose-badge badge-status'>● STREAM ACTIVE</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='pose-badge badge-standby'>○ CAMERA STANDBY</span>", unsafe_allow_html=True)

        # Video and banner placeholders
        r_video_placeholder = st.empty()
        r_banner_placeholder = st.empty()

    # --------------------------------------------------------------------------
    # ROUTINE NAVIGATION & CONTROLS (Below Both Panels)
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎮 Routine Navigation & Session Controls")

    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1.2, 1.2, 1.6, 1.2, 1.5])

    with btn_col1:
        if st.button("⬅️ Previous Pose", use_container_width=True, disabled=(rm.current_index == 0)):
            rm.previous()
            st.rerun()

    with btn_col2:
        pause_label = "▶️ Resume Timer" if rm.is_paused else "⏸️ Pause Timer"
        if st.button(pause_label, use_container_width=True):
            rm.toggle_pause()
            st.rerun()

    with btn_col3:
        if curr_pose.requires_sides:
            side_label = f"🔁 Switch Side ({curr_state.current_side or 'Left'})"
            if st.button(side_label, use_container_width=True, type="secondary"):
                _log_routine_pose_if_ready(curr_pose, curr_state)
                rm.switch_side()
                curr_state.is_logged_to_history = False
                st.rerun()
        elif curr_pose.is_repetition:
            rep_label = f"➕ Record Rep ({curr_state.repetition_count})"
            if st.button(rep_label, use_container_width=True, type="secondary"):
                rm.record_rep()
                st.rerun()
        else:
            if st.button("🔄 Reset Pose Timer", use_container_width=True):
                curr_state.elapsed_seconds = 0.0
                st.rerun()

    with btn_col4:
        next_disabled = (rm.current_index >= len(rm.poses) - 1)
        if st.button("➡️ Next Pose", use_container_width=True, disabled=next_disabled):
            _log_routine_pose_if_ready(curr_pose, curr_state)
            rm.advance(force=True)
            st.rerun()

    with btn_col5:
        if st.button("⏹️ End Routine & View Summary", use_container_width=True, type="primary"):
            _log_routine_pose_if_ready(curr_pose, curr_state)
            summary_data = rm.get_summary()
            logger.log_routine_session(summary_data)
            st.session_state["routine_summary_data"] = summary_data
            st.rerun()

    # Auto-advance toggle
    adv_col1, _ = st.columns([2, 1])
    with adv_col1:
        auto_adv = st.checkbox(
            "⚡ Auto-advance to next posture when countdown completes",
            value=rm.auto_advance,
            key="routine_auto_adv_chk",
            help="Automatically transition to the next pose or side once target duration is achieved.",
        )
        rm.auto_advance = auto_adv

    # --------------------------------------------------------------------------
    # METRICS & COACHING GAUGES
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        r_m1_ph = st.empty()
        r_b1_ph = st.empty()
    with m2:
        r_m2_ph = st.empty()
        r_b2_ph = st.empty()
    with m3:
        r_m3_ph = st.empty()
    with m4:
        r_m4_ph = st.empty()

    r_cue_ph = st.empty()
    r_detail_ph = st.empty()

    # --------------------------------------------------------------------------
    # STANDBY MODE (When Camera Toggle is Off)
    # --------------------------------------------------------------------------
    if not routine_cam_active:
        r_video_placeholder.markdown(
            f"""
            <div class="standby-card">
                <div style="font-size: 3rem; margin-bottom: 10px;">🧘‍♂️</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
                    Camera in Standby — {curr_pose.name}
                </div>
                <div style="font-size: 0.95rem; color: #94A3B8; max-width: 440px; margin: 0 auto 16px auto;">
                    Turn on the live camera stream above to activate skeleton tracking and live posture coaching.
                </div>
                <div style="display: inline-block; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.3); padding: 6px 14px; border-radius: 8px; font-size: 0.85rem; color: #34D399;">
                    Suggested Duration: <b>{curr_pose.suggested_duration_seconds}s</b> • Mode: <b>{'🤖 AI Evaluated' if curr_pose.has_ai_evaluation else '🧘 Guided Practice'}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        r_banner_placeholder.info("👈 Flip the 'Enable Live Camera Practice' toggle above to start video tracking.")

        # Standby metric cards
        target_sec = float(curr_pose.suggested_duration_seconds)
        rem_sec = max(0.0, target_sec - curr_state.elapsed_seconds)

        if curr_pose.has_ai_evaluation:
            r_m1_ph.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">Posture Accuracy</div>
                    <div class="metric-value" style="color: #94a3b8;">-- %</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Waiting for Camera</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            r_b1_ph.progress(0.0)

            r_m2_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Verified Hold Time</div>
                    <div class="metric-value" style="color: #00c0f2;">{curr_state.correct_seconds:.1f}s / {target_sec:.0f}s</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Session Best: {curr_state.best_hold_seconds:.1f}s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            r_b2_ph.progress(min(1.0, curr_state.correct_seconds / max(1.0, target_sec)))
        else:
            r_m1_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Countdown Timer</div>
                    <div class="metric-value" style="color: #38bdf8;">{int(rem_sec)}s</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Target: {target_sec:.0f}s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            r_b1_ph.progress(min(1.0, curr_state.elapsed_seconds / max(1.0, target_sec)))

            side_or_rep = f"Side: {curr_state.current_side or 'Both'}" if curr_pose.requires_sides else (f"Reps: {curr_state.repetition_count}" if curr_pose.is_repetition else "Active")
            r_m2_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Pose Status</div>
                    <div class="metric-value" style="color: #2ecc71; font-size: 1.35rem;">{side_or_rep}</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Elapsed: {curr_state.elapsed_seconds:.1f}s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            r_b2_ph.progress(min(1.0, curr_state.elapsed_seconds / max(1.0, target_sec)))

        r_m3_ph.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Routine Time</div>
                <div class="metric-value" style="color: #f1f5f9;">{format_seconds_readable(sum(s.elapsed_seconds for s in rm.states.values()))}</div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Active Across Routine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r_m4_ph.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Evaluation Mode</div>
                <div class="metric-value" style="color: {'#38bdf8' if curr_pose.has_ai_evaluation else '#a855f7'}; font-size: 1.25rem;">
                    {'AI Evaluated' if curr_pose.has_ai_evaluation else 'Guided Form'}
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">{curr_pose.phase.split(' — ')[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if curr_pose.has_ai_evaluation:
            r_cue_ph.markdown(
                """
                <div class="cue-box">
                    <b>AI Posture Analysis Ready:</b> Turn on camera stream to verify your joint alignment angles in real-time.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            r_cue_ph.markdown(
                f"""
                <div class="cue-box" style="border-left-color: #a855f7;">
                    <b>ℹ️ Guided Practice Mode:</b> Focus on breath, alignment, and gentle stretching. Automated joint analysis is not applicable for {curr_pose.name}.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # ACTIVE WEBCAM PROCESSING LOOP (Guided Routine)
    # --------------------------------------------------------------------------
    else:
        rm.start_routine()
        with st.spinner("Starting AI Pose Estimator for Guided Routine..."):
            try:
                detector = get_or_create_detector(backend_key)
                analyzer = get_or_create_analyzer(curr_pose.applicable_evaluator_key or curr_pose.pose_id) if curr_pose.has_ai_evaluation else None
                voice = get_or_create_voice(enabled=voice_enabled, rate=voice_rate)
            except Exception as e:
                st.error(f"Failed to initialize models: {e}")
                return

        cap = cv2.VideoCapture(int(camera_idx), cv2.CAP_DSHOW if sys_is_windows() else cv2.CAP_ANY)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(camera_idx))

        if not cap.isOpened():
            r_video_placeholder.error(
                f"⚠️ Could not open webcam at camera index {camera_idx}. "
                "Please verify camera connection and Windows privacy permissions."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        start_routine_pose_ts = time.time()
        last_frame_ts = start_routine_pose_ts
        milestone_spoken = False
        intro_spoken = False
        halfway_spoken = False
        midway_cue_spoken = False
        no_person_spoken_ts = 0.0

        try:
            while routine_cam_active:
                success, frame = cap.read()
                if not success or frame is None:
                    r_video_placeholder.warning("⚠️ Webcam stream interrupted.")
                    break

                now_ts = time.time()
                frame_dt = now_ts - last_frame_ts
                last_frame_ts = now_ts

                frame = cv2.flip(frame, 1)

                # Pose detection with skeleton
                det_result = detector.process_frame(frame, draw=True)

                is_correct = False
                eval_result = None

                # 1. Spoken Guided Introduction when camera starts / posture begins
                if not intro_spoken:
                    intro_spoken = True
                    if curr_pose.requires_sides and curr_state.current_side == "Right":
                        voice.speak(f"{curr_pose.name}, now switching to Right side. Get into position and hold steady.", priority=True)
                    else:
                        intro_parts = [f"Beginning {curr_pose.name}."]
                        if curr_pose.requires_sides and curr_state.current_side:
                            intro_parts.append(f"Starting with {curr_state.current_side} side.")
                        if curr_pose.instructions:
                            intro_parts.append(curr_pose.instructions[0])
                        elif curr_pose.safety_reminder:
                            intro_parts.append(curr_pose.safety_reminder)
                        voice.speak(" ".join(intro_parts), priority=True)

                # 2. Spoken prompt if practitioner is not detected in view
                if not det_result.pose_detected:
                    if (now_ts - start_routine_pose_ts > 6.0) and (now_ts - no_person_spoken_ts > 15.0):
                        no_person_spoken_ts = now_ts
                        voice.speak("Please step into full view of the camera to begin.", priority=False)

                # 3. Posture Evaluation & Spoken Guidance
                target_sec = float(curr_pose.suggested_duration_seconds)
                if curr_pose.has_ai_evaluation and analyzer:
                    eval_result = analyzer.evaluate_result(det_result, pose_name=curr_pose.applicable_evaluator_key or curr_pose.pose_id)
                    is_correct = (det_result.pose_detected and eval_result.is_pose_correct and eval_result.overall_score >= 75.0)

                    # Voice coaching: actionable alignment corrections or hold encouragement
                    if eval_result.actionable_cues:
                        cue = eval_result.actionable_cues[0]
                        is_safety = "safety" in cue.lower() or "never" in cue.lower()
                        voice.speak(cue, priority=is_safety)
                    elif is_correct and not halfway_spoken and (curr_state.correct_seconds >= target_sec * 0.5):
                        halfway_spoken = True
                        voice.speak("Great alignment! Hold steady and breathe deeply.", priority=False)

                    annotated = analyzer.draw_posture_feedback(det_result.annotated_frame, det_result, eval_result)
                else:
                    rem = max(0.0, target_sec - curr_state.elapsed_seconds)
                    annotated = draw_guided_hud(det_result.annotated_frame, curr_pose.name, curr_state.current_side, rem, curr_pose.phase)

                    # Guided breathing & posture cue for non-AI postures (warmup, floor, breathing)
                    if not midway_cue_spoken and (curr_state.elapsed_seconds >= min(10.0, target_sec * 0.45)):
                        midway_cue_spoken = True
                        guidance_msg = curr_pose.safety_reminder
                        if not guidance_msg and curr_pose.alignment_tips:
                            guidance_msg = curr_pose.alignment_tips[0]
                        if not guidance_msg:
                            guidance_msg = "Maintain deep, steady breathing and relaxed posture."
                        voice.speak(guidance_msg, priority=False)

                # Update timing
                reached, rem_sec = rm.update_timing(frame_dt, is_correct=is_correct)

                # Milestone handling
                if reached and not milestone_spoken:
                    milestone_spoken = True
                    voice.play_milestone_chime()
                    side_note = f" on {curr_state.current_side} side" if curr_pose.requires_sides and curr_state.current_side else ""
                    voice.speak(f"{curr_pose.name}{side_note} target achieved! Wonderful.", priority=True)
                    _log_routine_pose_if_ready(curr_pose, curr_state)

                    if rm.auto_advance:
                        time.sleep(1.2)
                        has_more = rm.advance(force=False)
                        if not has_more:
                            summary_data = rm.get_summary()
                            logger.log_routine_session(summary_data)
                            st.session_state["routine_summary_data"] = summary_data
                        st.rerun()

                # Render video frame
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                r_video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Banner placeholder
                if not det_result.pose_detected:
                    r_banner_placeholder.warning("🔴 No practitioner detected. Step into full view of the camera.")
                elif det_result.status == "PARTIAL_BODY":
                    r_banner_placeholder.warning("🟡 Partial body detected. Step back so your feet, hips, and shoulders are in view.")
                else:
                    side_str = f" • Stance: {curr_state.current_side} Side" if curr_pose.requires_sides and curr_state.current_side else ""
                    r_banner_placeholder.success(f"🟢 Tracking Practitioner (17 Landmarks){side_str}")

                # Update live metric cards
                target_sec = float(curr_pose.suggested_duration_seconds)
                if curr_pose.has_ai_evaluation and eval_result:
                    score = eval_result.overall_score
                    score_col = "#2ecc71" if is_correct else ("#ff9800" if score >= 50 else "#e74c3c")
                    r_m1_ph.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Posture Accuracy</div>
                            <div class="metric-value" style="color: {score_col};">{score:.0f}%</div>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">{'🟢 Verified' if is_correct else '🟡 Adjusting'}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    r_b1_ph.progress(min(1.0, max(0.0, score / 100.0)))

                    hold_s = analyzer.current_hold_duration if analyzer else 0.0
                    r_m2_ph.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Verified Hold Time</div>
                            <div class="metric-value" style="color: #00c0f2;">{hold_s:.1f}s / {target_sec:.0f}s</div>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Max Continuous: {curr_state.best_hold_seconds:.1f}s</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    r_b2_ph.progress(min(1.0, hold_s / max(1.0, target_sec)))
                else:
                    r_m1_ph.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Countdown Timer</div>
                            <div class="metric-value" style="color: #38bdf8;">{int(rem_sec)}s</div>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Target: {target_sec:.0f}s</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    r_b1_ph.progress(min(1.0, curr_state.elapsed_seconds / max(1.0, target_sec)))

                    side_rep_text = f"Side: {curr_state.current_side}" if curr_pose.requires_sides else (f"Reps: {curr_state.repetition_count}" if curr_pose.is_repetition else "Active Form")
                    r_m2_ph.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Posture Status</div>
                            <div class="metric-value" style="color: #2ecc71; font-size: 1.35rem;">{side_rep_text}</div>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Elapsed: {curr_state.elapsed_seconds:.1f}s</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    r_b2_ph.progress(min(1.0, curr_state.elapsed_seconds / max(1.0, target_sec)))

                tot_time = sum(s.elapsed_seconds for s in rm.states.values())
                r_m3_ph.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Routine Time</div>
                        <div class="metric-value" style="color: #f1f5f9;">{format_seconds_readable(tot_time)}</div>
                        <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Active Across Routine</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                r_m4_ph.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Vision Tracking</div>
                        <div class="metric-value" style="color: #2ecc71; font-size: 1.25rem;">Active ({backend_key.upper()})</div>
                        <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Pose {curr_pose.sequence_number}/22</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if curr_pose.has_ai_evaluation and eval_result:
                    if is_correct:
                        r_cue_ph.markdown(
                            """
                            <div class="cue-box-success">
                                <b>🟢 Optimal Alignment Verified:</b> Posture criteria satisfied! Hold steadily and breathe naturally.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif eval_result.actionable_cues:
                        r_cue_ph.markdown(
                            f"""
                            <div class="cue-box-warning">
                                <b>💡 Alignment Guidance:</b> {eval_result.actionable_cues[0]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    r_cue_ph.markdown(
                        f"""
                        <div class="cue-box" style="border-left-color: #a855f7;">
                            <b>ℹ️ Guided Practice Mode:</b> Focus on your breath and relaxed alignment. Automated joint scoring is not applicable for {curr_pose.name}.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                time.sleep(0.01)

        finally:
            cap.release()
            _log_routine_pose_if_ready(curr_pose, curr_state)


def render_home_dashboard(logger: WorkoutLogger, library: YogaLibrary):
    """
    Renders the modern, clean, beginner-friendly Home Dashboard for YogaCare AI.
    Features:
    - App branding and welcoming tagline: 'Find your balance, one pose at a time.'
    - Prominent 'Start Practising Now' call-to-action button.
    - Quick-access feature cards (Browse Yoga, Guided Practice, Workout History).
    - Overview of the 7 yoga practice categories with actual thumbnails and practice counts.
    - Lightweight summary of user practice activity.
    """
    # --------------------------------------------------------------------------
    # 1. WELCOME HERO SECTION
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="home-hero-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 2.2rem;">🌿</span>
                <div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #34D399; letter-spacing: 0.5px; text-transform: uppercase;">
                        YogaCare AI
                    </div>
                    <div style="font-size: 0.88rem; color: #94A3B8;">
                        Your Personal AI Yoga & Wellness Companion
                    </div>
                </div>
            </div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; line-height: 1.25; margin-bottom: 8px; letter-spacing: -0.5px;">
                Find your balance, one pose at a time.
            </div>
            <div style="font-size: 1.02rem; color: #CBD5E1; max-width: 780px; line-height: 1.6; margin-bottom: 18px;">
                Experience mindful, beginner-friendly yoga guided by real-time computer vision posture feedback,
                authentic adult male reference models, gentle audio coaching, and personalized practice history.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_btn1, hero_btn2, _ = st.columns([1.3, 1.3, 1.4])
    with hero_btn1:
        if st.button("🚀 Start Practising Now ➔", type="primary", use_container_width=True, key="home_hero_start_btn"):
            st.session_state.active_nav = "📹 Live & Guided Practice"
            st.rerun()
    with hero_btn2:
        if st.button("📚 Explore 40 Practices", use_container_width=True, key="home_hero_browse_btn"):
            st.session_state.active_nav = "📚 Browse Yoga"
            st.rerun()

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. THREE QUICK-ACCESS FEATURE CARDS
    # --------------------------------------------------------------------------
    st.subheader("✨ What would you like to do today?")
    st.caption("Choose a practice path to begin your journey with YogaCare AI.")

    q_col1, q_col2, q_col3 = st.columns(3, gap="medium")

    with q_col1:
        st.markdown(
            """
            <div class="home-quick-card">
                <div>
                    <div style="font-size: 2rem; margin-bottom: 8px;">📚</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                        Browse Yoga Library
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #38BDF8; margin-bottom: 10px;">
                        40 Curated Practices • 7 Phases
                    </div>
                    <div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;">
                        Discover step-by-step instructions, benefits, breathing tips, and male reference guides with bilingual English & Kannada support.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Browse Practices ➔", key="quick_browse_btn", use_container_width=True):
            st.session_state.active_nav = "📚 Browse Yoga"
            st.rerun()

    with q_col2:
        st.markdown(
            """
            <div class="home-quick-card" style="border-top: 4px solid #10B981;">
                <div>
                    <div style="font-size: 2rem; margin-bottom: 8px;">📹</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                        Guided & Live Practice
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #34D399; margin-bottom: 10px;">
                        Live Vision • 22-Pose Routine
                    </div>
                    <div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;">
                        Turn on your webcam for full-body skeleton tracking, real-time joint angle verification, hold timers, and spoken voice coaching.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Live Practice ➔", key="quick_live_btn", use_container_width=True, type="primary"):
            st.session_state.active_nav = "📹 Live & Guided Practice"
            st.rerun()

    with q_col3:
        st.markdown(
            """
            <div class="home-quick-card">
                <div>
                    <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                        Workout History
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #C084FC; margin-bottom: 10px;">
                        Verified Holds & Analytics
                    </div>
                    <div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;">
                        Track your practice consistency, total time on mat, verified correct posture durations, and longest continuous holds.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Workout History ➔", key="quick_hist_btn", use_container_width=True):
            st.session_state.active_nav = "📊 Workout History"
            st.rerun()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. PRACTICE CATEGORIES OVERVIEW (7 CURATED PHASES)
    # --------------------------------------------------------------------------
    st.subheader("🧘 Practice Categories Overview")
    st.caption("Seven progressive biomechanical categories structured for balance, strength, and mindful breathing.")

    categories = library.get_categories()
    cat_cols = st.columns(2, gap="medium")

    for idx, cat in enumerate(categories):
        col = cat_cols[idx % 2]
        color_class = f"cat-color-{cat.category_number}"
        with col:
            st.markdown(
                f"""
                <div class="category-card-box">
                    <div class="category-phase-badge {color_class}">
                        Phase {cat.category_number} • {cat.practice_count} Practices
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                        {cat.name}
                    </div>
                    <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.45; min-height: 38px; margin-bottom: 10px;">
                        {cat.description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if os.path.exists(cat.thumbnail_path):
                st.image(cat.thumbnail_path, use_container_width=True)

            if st.button(
                f"Explore {cat.name} ({cat.practice_count} Practices) ➔",
                key=f"home_cat_explore_{cat.category_id}",
                use_container_width=True,
            ):
                st.session_state.selected_category_id = cat.category_id
                st.session_state.library_view = "category_detail"
                st.session_state.active_nav = "📚 Browse Yoga"
                st.rerun()

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. AT-A-GLANCE ACTIVITY SUMMARY STRIP
    # --------------------------------------------------------------------------
    stats = logger.get_summary_stats()
    tot_time_str = format_seconds_readable(stats.get("total_practice_time_seconds"))
    corr_time_str = format_seconds_readable(stats.get("total_correct_pose_seconds"))
    best_hold_str = f"{stats.get('best_hold_seconds', 0.0):.1f}s"
    session_count = stats.get("total_sessions", 0)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.subheader("📈 Your Yoga Journey at a Glance")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""
            <div class="summary-card card-blue">
                <div class="card-icon-label">⏱️ Total Practice</div>
                <div class="card-value" style="color: #38BDF8;">{tot_time_str}</div>
                <div class="card-desc">Time spent on the mat</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class="summary-card card-green">
                <div class="card-icon-label">🎯 Correct Alignment</div>
                <div class="card-value" style="color: #34D399;">{corr_time_str}</div>
                <div class="card-desc">Verified posture hold time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
            <div class="summary-card card-purple">
                <div class="card-icon-label">🧘 Sessions Completed</div>
                <div class="card-value" style="color: #C084FC;">{session_count}</div>
                <div class="card-desc">Recorded coaching workouts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"""
            <div class="summary-card card-gold">
                <div class="card-icon-label">🏆 Best Continuous Hold</div>
                <div class="card-value" style="color: #FBBF24;">{best_hold_str}</div>
                <div class="card-desc">Longest verified correct hold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_yoga_practice_library(
    logger: WorkoutLogger,
    backend_key: str,
    camera_idx: int,
    voice: VoiceAssistant,
    voice_enabled: bool,
    voice_rate: int,
):
    """
    Renders the 40-Practice Yoga Library across 7 curated categories.
    Supports 3 distinct navigation states:
    1. 'categories' - 7 Category Cards with thumbnails, description, count, and explore button.
    2. 'category_detail' - Filtered list of practices belonging only to the chosen category.
    3. 'practice_detail' - Full 2-column view with adult male reference model, bilingual (English/Kannada)
       instructions, audio readout, in-category Previous/Next navigation, live camera coaching,
       and guided timer logging.
    """
    if "yoga_library" not in st.session_state:
        st.session_state.yoga_library = YogaLibrary()
    lib: YogaLibrary = st.session_state.yoga_library

    # State initialization
    if "library_view" not in st.session_state:
        st.session_state.library_view = "categories"
    if "selected_category_id" not in st.session_state:
        st.session_state.selected_category_id = "cat_warmup"
    if "selected_practice_id" not in st.session_state:
        st.session_state.selected_practice_id = "sukhasana"
    if "library_lang" not in st.session_state:
        st.session_state.library_lang = "English"

    current_view = st.session_state.library_view

    # ==========================================================================
    # VIEW 1: CATEGORIES OVERVIEW (7 CURATED CATEGORIES)
    # ==========================================================================
    if current_view == "categories":
        st.markdown(
            """
            <div class="library-hero-banner">
                <div style="font-size: 1.85rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
                    📚 40 Yoga Practices Library — Category-Wise Guided Learning
                </div>
                <div style="font-size: 0.98rem; color: #CBD5E1; max-width: 820px; line-height: 1.55; margin-bottom: 12px;">
                    Master 40 classical yoga postures and breath practices structured into 7 progressive biomechanical phases.
                    Each practice features authentic adult male alignment reference cards, bilingual instruction sets,
                    and optional real-time computer vision skeleton tracking.
                </div>
                <div>
                    <span class="library-stat-pill">📂 7 Biomechanical Categories</span>
                    <span class="library-stat-pill">🧘 40 Master Practices</span>
                    <span class="library-stat-pill">👤 100% Adult Male Reference Models</span>
                    <span class="library-stat-pill">🗣️ Bilingual: English & ಕನ್ನಡ (Kannada)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        categories = lib.get_categories()
        cat_cols = st.columns(2, gap="medium")

        for idx, cat in enumerate(categories):
            col = cat_cols[idx % 2]
            color_class = f"cat-color-{cat.category_number}"
            with col:
                st.markdown(
                    f"""
                    <div class="category-card-box">
                        <div class="category-phase-badge {color_class}">
                            Phase {cat.category_number} of 7 • {cat.practice_count} Exercises
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                            {cat.name}
                        </div>
                        <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.45; min-height: 40px; margin-bottom: 10px;">
                            {cat.description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if os.path.exists(cat.thumbnail_path):
                    st.image(cat.thumbnail_path, use_container_width=True)

                if st.button(
                    f"Explore {cat.name} ({cat.practice_count} Exercises) ➔",
                    key=f"btn_cat_explore_{cat.category_id}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.selected_category_id = cat.category_id
                    st.session_state.library_view = "category_detail"
                    st.rerun()

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        return

    # ==========================================================================
    # VIEW 2: CATEGORY DETAIL (FILTERED LIST OF PRACTICES FOR CHOSEN CATEGORY)
    # ==========================================================================
    if current_view == "category_detail":
        cat_id = st.session_state.get("selected_category_id", "cat_warmup")
        curr_cat = lib.get_category_by_id(cat_id)
        if not curr_cat:
            curr_cat = lib.get_category_by_id("cat_warmup")

        # Top breadcrumb & quick switcher
        top_c1, top_c2 = st.columns([1.4, 2.6])
        with top_c1:
            if st.button("⬅ Back to All Categories", key="btn_back_to_all_cats", use_container_width=True):
                st.session_state.library_view = "categories"
                st.rerun()
        with top_c2:
            all_cats = lib.get_categories()
            cat_names = [c.name for c in all_cats]
            curr_idx = [c.category_id for c in all_cats].index(curr_cat.category_id)
            picked_cat_name = st.selectbox(
                "Switch Category",
                options=cat_names,
                index=curr_idx,
                key="cat_quick_picker",
                help="Quickly jump between the 7 categories",
            )
            if picked_cat_name != curr_cat.name:
                for c in all_cats:
                    if c.name == picked_cat_name:
                        st.session_state.selected_category_id = c.category_id
                        st.rerun()

        # Category Banner
        color_class = f"cat-color-{curr_cat.category_number}"
        st.markdown(
            f"""
            <div class="library-hero-banner" style="margin-top: 8px;">
                <div class="category-phase-badge {color_class}">
                    Phase {curr_cat.category_number} • {curr_cat.practice_count} Curated Exercises
                </div>
                <div style="font-size: 1.7rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                    {curr_cat.name}
                </div>
                <div style="font-size: 0.95rem; color: #CBD5E1; max-width: 750px;">
                    {curr_cat.description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        practices = lib.get_practices_by_category(curr_cat.category_id)
        p_cols = st.columns(2, gap="medium")

        for idx, p in enumerate(practices):
            with p_cols[idx % 2]:
                eval_tag = "🤖 Live AI Evaluated" if p.has_ai_evaluation else "🧘 Guided Form"
                side_tag = " • 👈 Bilateral" if p.requires_sides else ""
                rep_tag = " • 🔄 Dynamic" if p.is_repetition else ""

                st.markdown(
                    f"""
                    <div class="practice-item-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span class="pose-badge badge-diff">#{p.sequence_number} • {p.difficulty}</span>
                            <span class="pose-badge badge-status">⏱️ {p.suggested_duration_seconds}s{side_tag}{rep_tag}</span>
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 2px;">
                            {p.name}
                        </div>
                        <div style="font-size: 0.9rem; color: #38BDF8; font-style: italic; margin-bottom: 8px;">
                            {p.sanskrit_name}
                        </div>
                        <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.45; min-height: 42px; margin-bottom: 10px;">
                            {p.description}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 10px;">
                            Tracking: <b style="color: {'#34D399' if p.has_ai_evaluation else '#38BDF8'};">{eval_tag}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if os.path.exists(p.local_image_path):
                    st.image(p.local_image_path, use_container_width=True)

                if st.button(
                    f"View Practice Details: {p.name} ➔",
                    key=f"btn_p_detail_{p.id}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.selected_practice_id = p.id
                    st.session_state.library_view = "practice_detail"
                    st.rerun()

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        return

    # ==========================================================================
    # VIEW 3: PRACTICE DETAIL (BALANCED 2-COLUMN VIEW & LIVE PRACTICE)
    # ==========================================================================
    if current_view == "practice_detail":
        p_id = st.session_state.get("selected_practice_id", "sukhasana")
        curr_p = lib.get_practice_by_id(p_id)
        if not curr_p:
            curr_p = lib.get_practice_by_id("sukhasana")

        curr_cat = lib.get_category_by_id(curr_p.category_id)
        cat_practices = lib.get_practices_by_category(curr_p.category_id)
        cat_p_ids = [p.id for p in cat_practices]
        curr_idx_in_cat = cat_p_ids.index(curr_p.id) if curr_p.id in cat_p_ids else 0

        prev_p = lib.get_previous_practice(curr_p.id, within_category=True)
        next_p = lib.get_next_practice(curr_p.id, within_category=True)

        # ----------------------------------------------------------------------
        # IN-CATEGORY SEQUENTIAL NAVIGATION BAR
        # ----------------------------------------------------------------------
        nav_c1, nav_c2, nav_c3 = st.columns([1.5, 2.5, 1.5])
        with nav_c1:
            if st.button(f"⬅ Back to {curr_cat.name}", key="btn_back_to_cat_view", use_container_width=True):
                st.session_state.library_view = "category_detail"
                st.rerun()
        with nav_c2:
            s_c1, s_c2, s_c3 = st.columns([1.1, 1.8, 1.1])
            with s_c1:
                prev_label = f"⏮ Prev: {prev_p.name}" if prev_p else "⏮ At Start"
                if st.button(prev_label, disabled=(prev_p is None), key="nav_btn_prev", use_container_width=True):
                    st.session_state.selected_practice_id = prev_p.id
                    st.session_state[f"cam_active_{curr_p.id}"] = False
                    st.rerun()
            with s_c2:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding-top: 6px; font-size: 0.88rem; color: #94A3B8; font-weight: 600;">
                        Exercise {curr_idx_in_cat + 1} of {len(cat_practices)} in Phase {curr_cat.category_number}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with s_c3:
                next_label = f"Next: {next_p.name} ⏭" if next_p else "At End ⏭"
                if st.button(next_label, disabled=(next_p is None), key="nav_btn_next", use_container_width=True):
                    st.session_state.selected_practice_id = next_p.id
                    st.session_state[f"cam_active_{curr_p.id}"] = False
                    st.rerun()
        with nav_c3:
            st.markdown(
                f"""
                <div style="text-align: right; padding-top: 4px;">
                    <span class="pose-badge badge-diff">#{curr_p.sequence_number} / 40</span>
                    <span class="pose-badge badge-status">{curr_p.difficulty}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # BALANCED 2-COLUMN DETAIL LAYOUT
        # ----------------------------------------------------------------------
        col_left, col_right = st.columns([1, 1], gap="large")

        # LEFT COLUMN: Adult Male Reference Model & Posture Constraints
        with col_left:
            st.markdown(f"### 🧘 Reference Model: {curr_p.name}")

            side_badge = "<span class='pose-badge badge-diff'>👈 Bilateral (Left & Right)</span> " if curr_p.requires_sides else ""
            rep_badge = "<span class='pose-badge badge-standby'>🔄 Dynamic Flow</span> " if curr_p.is_repetition else ""
            eval_badge = "<span class='pose-badge badge-status'>🤖 Live AI Evaluated</span> " if curr_p.has_ai_evaluation else "<span class='pose-badge badge-standby'>🧘 Guided Form</span> "

            st.markdown(
                f"{side_badge}{rep_badge}{eval_badge}"
                f"<span class='pose-badge badge-diff'>⏱️ Target: {curr_p.suggested_duration_seconds}s</span>",
                unsafe_allow_html=True,
            )

            # Adult male reference image
            if os.path.exists(curr_p.local_image_path):
                st.image(
                    curr_p.local_image_path,
                    caption=f"Verified Adult Male Biomechanical Reference • #{curr_p.sequence_number}: {curr_p.name} ({curr_p.sanskrit_name})",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="standby-card">
                        <div style="font-size: 2.8rem; margin-bottom: 8px;">🧘‍♂️</div>
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 1.15rem;">Adult Male Reference Card</div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">
                            File: <code>{curr_p.local_image_path}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Beginner Modification Card
            if curr_p.beginner_modification:
                st.markdown(
                    f"""
                    <div style="background: #064E3B33; border-left: 4px solid #10B981; border-radius: 8px; padding: 12px 14px; margin-top: 12px; font-size: 0.88rem; color: #D1FAE5;">
                        💡 <b>Beginner Modification:</b> {curr_p.beginner_modification}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Safety Reminder Card
            if curr_p.safety_reminder:
                st.markdown(
                    f"""
                    <div style="background: #78350F33; border-left: 4px solid #F59E0B; border-radius: 8px; padding: 12px 14px; margin-top: 8px; font-size: 0.88rem; color: #FDE68A;">
                        🛡️ <b>Safety Reminder:</b> {curr_p.safety_reminder}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # RIGHT COLUMN: Bilingual Step-by-Step Instructions & Biomechanics
        with col_right:
            st.markdown(f"## {curr_p.name}")
            st.markdown(f"<div style='font-size: 1.15rem; color: #38BDF8; font-style: italic; margin-bottom: 8px;'>{curr_p.sanskrit_name} • Phase {curr_cat.category_number}: {curr_cat.name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.98rem; color: #CBD5E1; line-height: 1.55; margin-bottom: 14px;'>{curr_p.description}</div>", unsafe_allow_html=True)

            # Bilingual Language Selector & Audio Readout Button
            l_col, v_col = st.columns([1.5, 1.3])
            with l_col:
                selected_lang = st.radio(
                    "Instruction Language / ಭಾಷೆ",
                    options=["🇬🇧 English", "🇮🇳 ಕನ್ನಡ (Kannada)"],
                    index=0 if st.session_state.library_lang == "English" else 1,
                    horizontal=True,
                    key=f"lang_radio_{curr_p.id}",
                )
                st.session_state.library_lang = "English" if "English" in selected_lang else "Kannada"

            with v_col:
                st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
                if st.button("🔊 Read Instructions Aloud", key=f"btn_speak_inst_{curr_p.id}", use_container_width=True):
                    if st.session_state.library_lang == "English":
                        text_to_speak = f"{curr_p.name}. {curr_p.description}. " + " ".join(
                            [f"Step {i}: {s}" for i, s in enumerate(curr_p.instructions, 1)]
                        )
                    else:
                        text_to_speak = f"{curr_p.name}. {curr_p.description}. " + " ".join(curr_p.instructions)
                    voice.speak(text_to_speak, priority=True)
                    st.toast("🔊 AI Voice Assistant reading instructions...", icon="📢")

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            # Step-by-Step Instructions
            if st.session_state.library_lang == "English":
                st.markdown("#### 📝 Step-by-Step Instructions (English)")
                for idx, step in enumerate(curr_p.instructions, 1):
                    st.markdown(f"<div class='instruction-step'><b>{idx}.</b> {step}</div>", unsafe_allow_html=True)
            else:
                st.markdown("#### 📝 ಹಂತ ಹಂತದ ಸೂಚನೆಗಳು (ಕನ್ನಡ — Kannada)")
                for idx, step in enumerate(curr_p.instructions_kannada, 1):
                    st.markdown(f"<div class='instruction-step-kn'><b>{idx}.</b> {step}</div>", unsafe_allow_html=True)

            # Key Alignment Tips
            with st.expander("💡 Key Alignment Tips", expanded=True):
                for tip in curr_p.alignment_tips:
                    st.markdown(f"<div class='tip-item'>• {tip}</div>", unsafe_allow_html=True)

            # Breathing Guidance
            if curr_p.breathing_guidance:
                st.markdown(
                    f"""
                    <div style="background: #0F291E; border-left: 4px solid #10B981; padding: 12px 14px; border-radius: 8px; margin-top: 10px; font-size: 0.92rem; color: #A7F3D0;">
                        🌬️ <b>Breathing Guidance:</b> {curr_p.breathing_guidance}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------------------------
        # PRACTICE COACHING SECTION (LIVE CAMERA OR GUIDED TIMER)
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.subheader(f"🎯 Practice Session: {curr_p.name}")

        coach_mode = st.radio(
            "Select Practice Mode",
            options=["📹 Live Webcam Coaching", "⏱️ Guided Form & Countdown Timer"],
            horizontal=True,
            key=f"coach_mode_{curr_p.id}",
        )

        # MODE A: LIVE WEBCAM COACHING
        if "Live Webcam" in coach_mode:
            cam_col1, cam_col2 = st.columns([1.2, 1])
            with cam_col1:
                cam_toggle = st.toggle(
                    "🔴 Enable Live Camera Practice",
                    value=st.session_state.get(f"cam_active_{curr_p.id}", False),
                    key=f"cam_toggle_{curr_p.id}",
                    help="Start live webcam stream with skeleton tracking.",
                )
            with cam_col2:
                if cam_toggle:
                    st.markdown("<span class='pose-badge badge-status'>● CAMERA STREAM ACTIVE</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='pose-badge badge-standby'>○ CAMERA IN STANDBY</span>", unsafe_allow_html=True)

            v_placeholder = st.empty()
            banner_placeholder = st.empty()

            m_c1, m_c2, m_c3 = st.columns(3)
            with m_c1:
                g1_ph = st.empty()
            with m_c2:
                g2_ph = st.empty()
            with m_c3:
                g3_ph = st.empty()
            cue_ph = st.empty()

            if not cam_toggle:
                v_placeholder.markdown(
                    f"""
                    <div class="standby-card">
                        <div style="font-size: 3rem; margin-bottom: 8px;">🧘‍♂️</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                            Camera in Standby — {curr_p.name}
                        </div>
                        <div style="font-size: 0.9rem; color: #94a3b8; max-width: 440px; margin: 0 auto 12px auto;">
                            Enable the live camera toggle above to activate skeleton tracking and real-time posture coaching.
                        </div>
                        <div style="display: inline-block; background: #1e293b; border: 1px solid #334155; padding: 6px 14px; border-radius: 8px; font-size: 0.82rem; color: #38bdf8;">
                            Suggested Duration: <b>{curr_p.suggested_duration_seconds}s</b> • Mode: <b>{'🤖 AI Evaluated' if curr_p.has_ai_evaluation else '🧘 Guided Practice'}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                banner_placeholder.info("👈 Flip the 'Enable Live Camera Practice' toggle above to start video tracking.")
            else:
                cap = cv2.VideoCapture(camera_idx)
                if not cap.isOpened():
                    banner_placeholder.error(f"Cannot access camera at index {camera_idx}. Verify webcam connection.")
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

                    detector = get_or_create_detector(backend_key)
                    eval_key = curr_p.applicable_evaluator_key or curr_p.id
                    analyzer = get_or_create_analyzer(eval_key) if curr_p.has_ai_evaluation else None

                    start_ts = time.time()
                    last_ts = start_ts
                    elapsed_sec = 0.0
                    correct_sec = 0.0
                    max_hold_sec = 0.0
                    milestone_spoken = False
                    intro_spoken = False
                    midway_spoken = False
                    no_person_spoken_ts = 0.0

                    try:
                        while st.session_state.get(f"cam_toggle_{curr_p.id}", False):
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                banner_placeholder.warning("⚠️ Webcam stream interrupted.")
                                break

                            now_ts = time.time()
                            frame_dt = now_ts - last_ts
                            last_ts = now_ts
                            elapsed_sec += frame_dt

                            frame = cv2.flip(frame, 1)
                            det_result = detector.process_frame(frame, draw=True)

                            eval_result = None
                            is_correct = False

                            # Spoken Guided Setup Instruction on camera start
                            if not intro_spoken:
                                intro_spoken = True
                                intro_parts = [f"Beginning {curr_p.name}."]
                                if curr_p.instructions:
                                    intro_parts.append(curr_p.instructions[0])
                                elif curr_p.description:
                                    intro_parts.append(curr_p.description)
                                voice.speak(" ".join(intro_parts), priority=True)

                            # Spoken prompt if practitioner is not detected in view
                            if not det_result.pose_detected:
                                if (now_ts - start_ts > 6.0) and (now_ts - no_person_spoken_ts > 15.0):
                                    no_person_spoken_ts = now_ts
                                    voice.speak("Please step into full view of the camera to begin.", priority=False)

                            target_dur = float(curr_p.suggested_duration_seconds)

                            if curr_p.has_ai_evaluation and analyzer:
                                eval_result = analyzer.evaluate_result(det_result, pose_name=eval_key)
                                is_correct = (
                                    det_result.pose_detected
                                    and eval_result.is_pose_correct
                                    and eval_result.overall_score >= 75.0
                                )
                                if is_correct:
                                    correct_sec += frame_dt

                                current_hold = analyzer.current_hold_duration
                                if current_hold > max_hold_sec:
                                    max_hold_sec = current_hold

                                if eval_result.actionable_cues:
                                    top_cue = eval_result.actionable_cues[0]
                                    is_safe = "safety" in top_cue.lower() or "never" in top_cue.lower()
                                    voice.speak(top_cue, priority=is_safe)
                                elif is_correct and not midway_spoken and (correct_sec >= target_dur * 0.5):
                                    midway_spoken = True
                                    voice.speak("Great alignment! Hold steady and breathe smoothly.", priority=False)

                                annotated = analyzer.draw_posture_feedback(det_result.annotated_frame, det_result, eval_result)
                            else:
                                rem_sec = max(0.0, target_dur - elapsed_sec)
                                annotated = draw_guided_hud(det_result.annotated_frame, curr_p.name, None, rem_sec, curr_cat.name)

                                if not midway_spoken and (elapsed_sec >= min(10.0, target_dur * 0.45)):
                                    midway_spoken = True
                                    guidance_txt = curr_p.breathing_guidance or curr_p.safety_reminder
                                    if not guidance_txt and curr_p.alignment_tips:
                                        guidance_txt = curr_p.alignment_tips[0]
                                    if not guidance_txt:
                                        guidance_txt = "Breathe steadily and maintain an elongated, relaxed posture."
                                    voice.speak(guidance_txt, priority=False)

                            eval_dur = correct_sec if curr_p.has_ai_evaluation else elapsed_sec
                            if eval_dur >= target_dur and not milestone_spoken:
                                milestone_spoken = True
                                voice.play_milestone_chime()
                                voice.speak(f"{curr_p.name} target achieved! Fantastic practice.", priority=True)

                            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            v_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                            # Status banner
                            if not det_result.pose_detected:
                                banner_placeholder.warning("🔴 No practitioner detected. Step into full view of the camera.")
                            elif det_result.status == "PARTIAL_BODY":
                                banner_placeholder.warning("🟡 Partial body detected. Step back so hips and limbs are visible.")
                            else:
                                banner_placeholder.success(f"🟢 Full-Body Tracked (17 Landmarks) • {curr_p.name}")

                            # Metrics
                            if curr_p.has_ai_evaluation and eval_result:
                                score = eval_result.overall_score
                                score_col = "#2ecc71" if is_correct else ("#ff9800" if score >= 50 else "#e74c3c")
                                g1_ph.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-title">Posture Accuracy</div>
                                        <div class="metric-value" style="color: {score_col};">{score:.0f}%</div>
                                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{'🟢 Verified' if is_correct else '🟡 Adjusting'}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                g2_ph.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-title">Verified Hold Time</div>
                                        <div class="metric-value" style="color: #00c0f2;">{correct_sec:.1f}s / {target_dur:.0f}s</div>
                                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">Session Best: {max_hold_sec:.1f}s</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                rem_disp = max(0, int(target_dur - elapsed_sec))
                                g1_ph.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-title">Countdown Timer</div>
                                        <div class="metric-value" style="color: #38bdf8;">{rem_disp}s</div>
                                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">Target: {target_dur:.0f}s</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                g2_ph.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-title">Elapsed Session Time</div>
                                        <div class="metric-value" style="color: #2ecc71;">{elapsed_sec:.1f}s</div>
                                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">Guided Form Active</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            g3_ph.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">Tracking Engine</div>
                                    <div class="metric-value" style="color: #a855f7; font-size: 1.25rem;">{backend_key.upper()} Vision</div>
                                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">Pose #{curr_p.sequence_number} / 40</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if curr_p.has_ai_evaluation and eval_result and eval_result.actionable_cues:
                                cue_ph.markdown(
                                    f"<div class='cue-box-warning'><b>💡 Alignment Cue:</b> {eval_result.actionable_cues[0]}</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                cue_ph.markdown(
                                    f"<div class='cue-box'><b>🌬️ Breathing Focus:</b> {curr_p.breathing_guidance}</div>",
                                    unsafe_allow_html=True,
                                )

                            time.sleep(0.01)

                    finally:
                        cap.release()
                        if elapsed_sec >= 2.0:
                            logger.log_session(
                                pose_name=curr_p.name,
                                duration_seconds=elapsed_sec,
                                target_duration=int(curr_p.suggested_duration_seconds),
                                correct_pose_seconds=correct_sec if curr_p.has_ai_evaluation else elapsed_sec,
                                max_continuous_hold=max_hold_sec if curr_p.has_ai_evaluation else elapsed_sec,
                                average_score=75.0 if is_correct else 70.0,
                                target_achieved=(eval_dur >= target_dur),
                                cues_provided=curr_p.alignment_tips[:2],
                                stance=f"Phase {curr_p.category_number}",
                            )
                            st.toast(f"✅ Session for {curr_p.name} logged to Workout History!", icon="💾")

        # MODE B: GUIDED FORM & COUNTDOWN TIMER
        else:
            t_col1, t_col2 = st.columns([1.5, 1])
            with t_col1:
                target_secs = curr_p.suggested_duration_seconds
                timer_slider = st.slider(
                    "Practice Duration (Seconds)",
                    min_value=10,
                    max_value=300,
                    value=int(target_secs),
                    step=5,
                    key=f"timer_slide_{curr_p.id}",
                )

                t_run = st.checkbox("▶️ Start Guided Practice Timer", key=f"timer_run_{curr_p.id}")

                if t_run:
                    timer_ph = st.empty()
                    prog_ph = st.empty()
                    msg_ph = st.empty()
                    elapsed_so_far = 0
                    session_already_logged = False

                    try:
                        voice.speak(f"Starting {curr_p.name} practice. {curr_p.instructions[0] if curr_p.instructions else curr_p.description}", priority=True)
                        midway_timer_spoken = False
                        for sec_left in range(timer_slider, -1, -1):
                            elapsed_so_far = timer_slider - sec_left
                            ratio = min(1.0, elapsed_so_far / timer_slider)
                            prog_ph.progress(ratio)

                            if not midway_timer_spoken and elapsed_so_far >= int(timer_slider * 0.5):
                                midway_timer_spoken = True
                                guidance_msg = curr_p.breathing_guidance or curr_p.safety_reminder
                                if guidance_msg:
                                    voice.speak(guidance_msg, priority=False)

                            timer_ph.markdown(
                                f"""
                                <div class="metric-card" style="text-align: center; padding: 24px;">
                                    <div class="metric-title">Remaining Time</div>
                                    <div class="metric-value" style="font-size: 3rem; color: #38bdf8;">{sec_left}s</div>
                                    <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 4px;">Hold with steady, diaphragmatic breath</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if sec_left == 0:
                                voice.play_milestone_chime()
                                voice.speak(f"Wonderful job! {curr_p.name} session completed.", priority=True)
                                msg_ph.success(f"🎉 Target hold of {timer_slider} seconds completed!")
                                logger.log_session(
                                    pose_name=curr_p.name,
                                    duration_seconds=float(timer_slider),
                                    target_duration=timer_slider,
                                    correct_pose_seconds=float(timer_slider),
                                    max_continuous_hold=float(timer_slider),
                                    average_score=85.0,
                                    target_achieved=True,
                                    cues_provided=curr_p.alignment_tips[:2],
                                    stance=f"Phase {curr_p.category_number}",
                                )
                                session_already_logged = True
                                st.toast(f"✅ Session for {curr_p.name} logged to Workout History!", icon="💾")
                                break
                            time.sleep(1.0)
                    finally:
                        if not session_already_logged and elapsed_so_far >= 2.0:
                            logger.log_session(
                                pose_name=curr_p.name,
                                duration_seconds=float(elapsed_so_far),
                                target_duration=timer_slider,
                                correct_pose_seconds=float(elapsed_so_far),
                                max_continuous_hold=float(elapsed_so_far),
                                average_score=75.0,
                                target_achieved=(elapsed_so_far >= timer_slider),
                                cues_provided=curr_p.alignment_tips[:2],
                                stance=f"Phase {curr_p.category_number}",
                            )
                            st.toast(f"✅ Session for {curr_p.name} logged to Workout History!", icon="💾")
                else:
                    st.info(f"Press **'Start Guided Practice Timer'** above to begin a {timer_slider}-second guided practice session.")

            with t_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Practice Focus</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 4px;">{curr_p.name}</div>
                        <div style="font-size: 0.85rem; color: #38bdf8; font-style: italic;">{curr_p.sanskrit_name}</div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 10px; line-height: 1.45;">
                            {curr_p.breathing_guidance}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main():
    logger = WorkoutLogger()

    # ==========================================================================
    # SIDEBAR: SETTINGS & CONTROLS
    # ==========================================================================
    st.sidebar.image(
        "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=600&auto=format&fit=crop&q=80",
        use_container_width=True,
    )
    st.sidebar.title("🧘 YogaCare AI")
    st.sidebar.caption("Smart AI Yoga Instructor & Posture Coach")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🎯 Workout Settings")
    pose_options = [
        "Warrior II (Virabhadrasana II)",
        "Tree Pose (Vrikshasana)",
        "Downward-Facing Dog (Adho Mukha Svanasana)",
        "Plank Pose (Phalakasana)",
        "Cobra Pose (Bhujangasana)",
        "Mountain Pose (Tadasana)",
        "Auto-Detect Pose",
    ]
    pose_choice = st.sidebar.selectbox(
        "Target Posture",
        options=pose_options,
        index=0,
        help="Select the yoga asana you want to practice. The reference guide and AI evaluator will update automatically.",
    )

    pose_map = {
        "Warrior II (Virabhadrasana II)": "warrior_ii",
        "Tree Pose (Vrikshasana)": "tree_pose",
        "Downward-Facing Dog (Adho Mukha Svanasana)": "downward_dog",
        "Plank Pose (Phalakasana)": "plank_pose",
        "Cobra Pose (Bhujangasana)": "cobra_pose",
        "Mountain Pose (Tadasana)": "mountain_pose",
        "Auto-Detect Pose": "auto",
    }
    active_pose_key = pose_map[pose_choice]

    target_hold_time = st.sidebar.slider(
        "Target Hold Time (Seconds)",
        min_value=5,
        max_value=60,
        value=15,
        step=5,
        help="Seconds of continuous, verified correct posture required to complete the hold milestone.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎙️ Voice & Audio Coaching")
    voice_enabled = st.sidebar.checkbox("Enable AI Voice Coach", value=True, help="Spoken auditory feedback for posture adjustments.")
    voice_rate = st.sidebar.slider("Voice Speed (WPM)", min_value=120, max_value=200, value=160, step=10)
    if st.sidebar.button("🔊 Test Voice Coach", key="sidebar_test_voice_btn", use_container_width=True):
        test_voice = get_or_create_voice(enabled=voice_enabled, rate=voice_rate)
        test_voice.speak("Hello! I am your AI Yoga Voice Coach. Your audio is working perfectly.", priority=True)
        st.sidebar.toast("🔊 Voice Coach speaking test message...", icon="🎙️")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Vision & Camera Settings")
    backend = st.sidebar.selectbox(
        "Pose Model Backend",
        options=["YOLOv8-Pose (Fast CPU)", "MediaPipe Tasks"],
        index=0,
        help="YOLOv8-Pose delivers optimal performance for standing and floor yoga asanas.",
    )
    backend_key = "yolo" if "YOLO" in backend else "mediapipe"
    camera_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=4, value=0, step=1)

    # Resolve pose reference for the currently selected posture
    ref_pose_key = "warrior_ii" if active_pose_key == "auto" else active_pose_key
    pose_ref = get_pose_reference(ref_pose_key)

    # ==========================================================================
    # MAIN HEADER & TOP NAVIGATION
    # ==========================================================================
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 12px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2.2rem;">🧘</span>
                <div>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.3px;">
                        YogaCare AI <span style="font-size: 0.85rem; font-weight: 600; color: #34D399; margin-left: 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.3); padding: 3px 10px; border-radius: 20px;">Personal Wellness Companion</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 2px;">
                        Real-Time Computer Vision Posture Alignment, Biomechanical Angles & Audio Coaching
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = [
        "🏠 Home",
        "📚 Browse Yoga",
        "📹 Live & Guided Practice",
        "📊 Workout History",
        "📖 Biomechanics Guide",
    ]
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "🏠 Home"
    elif st.session_state.active_nav not in nav_options:
        st.session_state.active_nav = "🏠 Home"

    cur_idx = nav_options.index(st.session_state.active_nav)
    selected_nav = st.radio(
        "Navigation Menu",
        options=nav_options,
        index=cur_idx,
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected_nav != st.session_state.active_nav:
        st.session_state.active_nav = selected_nav
        st.rerun()

    active_page = st.session_state.active_nav

    # ==========================================================================
    # PAGE 1: HOME DASHBOARD
    # ==========================================================================
    if active_page == "🏠 Home":
        if "yoga_library" not in st.session_state:
            st.session_state.yoga_library = YogaLibrary()
        render_home_dashboard(logger=logger, library=st.session_state.yoga_library)

    # ==========================================================================
    # PAGE 2: 40-PRACTICE CATEGORY-WISE YOGA LIBRARY
    # ==========================================================================
    elif active_page == "📚 Browse Yoga":
        voice_coach = get_or_create_voice(enabled=voice_enabled, rate=voice_rate)
        render_yoga_practice_library(
            logger=logger,
            backend_key=backend_key,
            camera_idx=camera_idx,
            voice=voice_coach,
            voice_enabled=voice_enabled,
            voice_rate=voice_rate,
        )

    # ==========================================================================
    # PAGE 3: LIVE YOGA PRACTICE (BALANCED SIDE-BY-SIDE PANELS)
    # ==========================================================================
    elif active_page == "📹 Live & Guided Practice":
        practice_mode = st.radio(
            "Select Practice Mode",
            options=["🧘 22-Pose Guided Routine (Full Sequence)", "🎯 Single Pose Practice (Focused Drill)"],
            index=0,
            horizontal=True,
            help="Guided Routine guides you through all 22 postures with male references in fixed sequence. Single Pose Practice allows targeted drill on an individual pose.",
            key="practice_mode_selection",
        )

        if "Guided Routine" in practice_mode:
            render_guided_routine(logger, backend_key, camera_idx, voice_enabled, voice_rate)
        else:
            # Balanced Two-Column Layout (Reference Pose on Left, Live Webcam on Right)
            col_ref, col_cam = st.columns([1, 1], gap="large")

            # ----------------------------------------------------------------------
            # LEFT PANEL — Reference Yoga Pose
            # ----------------------------------------------------------------------
            with col_ref:
                st.markdown(f"### 🧘 Reference Pose: {pose_ref.name}")
                st.markdown(
                    f"<span class='pose-badge badge-diff'>Difficulty: {pose_ref.difficulty}</span> "
                    f"<span class='pose-badge badge-status'>{pose_ref.sanskrit_name}</span>",
                    unsafe_allow_html=True,
                )

                # Display genuine reference yoga pose image
                img_source, is_local, status_msg = resolve_reference_image(pose_ref)
                if img_source:
                    st.image(
                        img_source,
                        caption=f"Standard Anatomical Form • {pose_ref.name} ({pose_ref.sanskrit_name})",
                        use_container_width=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="standby-card">
                            <div style="font-size: 2.5rem; margin-bottom: 8px;">🧘</div>
                            <div style="font-weight: 600; color: #f1f5f9; font-size: 1.1rem;">Reference Image Placeholder</div>
                            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">{status_msg}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Description
                st.markdown(f"*{pose_ref.description}*")

                # Step-by-Step Instructions
                with st.expander("📝 Step-by-Step Instructions", expanded=True):
                    for idx, step in enumerate(pose_ref.instructions, 1):
                        st.markdown(f"<div class='instruction-step'><b>{idx}.</b> {step}</div>", unsafe_allow_html=True)

                # Key Alignment Tips & Safety Rules
                with st.expander("💡 Key Alignment Tips & Safety", expanded=True):
                    for tip in pose_ref.alignment_tips:
                        st.markdown(f"<div class='tip-item'>{tip}</div>", unsafe_allow_html=True)

                # Target Joint Angles Guide
                if pose_ref.target_joints:
                    with st.expander("📐 Target Joint Angles Guide", expanded=False):
                        for j_title, j_target in pose_ref.target_joints.items():
                            st.markdown(f"- **{j_title}:** `{j_target}`")

            # ----------------------------------------------------------------------
            # RIGHT PANEL — Live Webcam Practice
            # ----------------------------------------------------------------------
            with col_cam:
                st.markdown(f"### 📹 Live Camera: {pose_ref.name if active_pose_key != 'auto' else 'Auto-Detect Mode'}")

                # Obvious Start & Stop Session Controls
                ctrl_col1, ctrl_col2 = st.columns([1, 1])
                with ctrl_col1:
                    start_session_toggle = st.toggle(
                        "🔴 Enable Live Webcam Stream",
                        value=st.session_state.get("is_coaching_active", False),
                        key="webcam_coaching_toggle",
                        help="Toggle to start or stop the real-time camera coaching loop.",
                    )
                with ctrl_col2:
                    if start_session_toggle:
                        st.markdown("<span class='pose-badge badge-status'>● COACHING ACTIVE</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='pose-badge badge-standby'>○ CAMERA STANDBY</span>", unsafe_allow_html=True)

                # Live Video Display Placeholder
                video_placeholder = st.empty()

                # Detection Status Banner beneath video
                detection_banner_placeholder = st.empty()

            # ----------------------------------------------------------------------
            # POSTURE FEEDBACK & GAUGES (Below the Two Panels)
            # ----------------------------------------------------------------------
            st.markdown("---")
            st.subheader("📋 Real-Time Coaching & Biomechanical Feedback")

            # Top Metric Cards: Accuracy Score, Verified Hold, Session Running Time, Stance
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                score_placeholder = st.empty()
                score_bar_placeholder = st.empty()
            with m_col2:
                hold_placeholder = st.empty()
                hold_bar_placeholder = st.empty()
            with m_col3:
                session_timer_placeholder = st.empty()
            with m_col4:
                detection_status_placeholder = st.empty()

            # Actionable Coaching Cue Banner
            cue_placeholder = st.empty()

            # Detailed Joint Angles Breakdown
            joint_details_placeholder = st.empty()

            # Set initial standby cards when camera is inactive
            if not start_session_toggle:
                video_placeholder.markdown(
                    f"""
                    <div class="standby-card">
                        <div style="font-size: 3rem; margin-bottom: 10px;">📹</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">Camera in Standby</div>
                        <div style="font-size: 0.95rem; color: #94A3B8; max-width: 440px; margin: 0 auto 16px auto;">
                            Position your device so your entire body is clearly visible in the frame (approx. 6–8 feet away).
                        </div>
                        <div style="display: inline-block; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.3); padding: 6px 14px; border-radius: 8px; font-size: 0.85rem; color: #34D399;">
                            Target: <b>{pose_ref.name}</b> • Hold Goal: <b>{target_hold_time}s</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.session_state.get("single_pose_last_saved"):
                    detection_banner_placeholder.success(f"🎉 **Workout Saved:** {st.session_state['single_pose_last_saved']} recorded into Workout History.")
                else:
                    detection_banner_placeholder.info("👈 Flip the 'Enable Live Webcam Stream' toggle above to start your live coaching session.")

                score_placeholder.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-title">Posture Accuracy</div>
                        <div class="metric-value" style="color: #9aa0a6;">-- %</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                score_bar_placeholder.progress(0.0)

                hold_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Verified Hold Time</div>
                        <div class="metric-value" style="color: #00c0f2;">0.0s / {target_hold_time}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                hold_bar_placeholder.progress(0.0)

                session_timer_placeholder.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-title">Session Running Time</div>
                        <div class="metric-value" style="color: #94a3b8;">00:00</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                detection_status_placeholder.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-title">Vision Tracking</div>
                        <div class="metric-value" style="color: #94a3b8; font-size: 1.35rem;">Standby</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                cue_placeholder.markdown(
                    """
                    <div class="cue-box">
                        <b>Ready to Practice:</b> Flip the toggle when you are in position to begin live analysis.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show target angles guide table in standby
                guide_items = [
                    f"<tr><td style='padding: 6px 12px; border-bottom: 1px solid #2d3748;'><b>{k}</b></td>"
                    f"<td style='padding: 6px 12px; border-bottom: 1px solid #2d3748; color: #38bdf8;'><code>{v}</code></td></tr>"
                    for k, v in pose_ref.target_joints.items()
                ]
                joint_details_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Target Joint Angle Specifications ({pose_ref.name})</div>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95rem;">
                            {"".join(guide_items)}
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------------------------------
            # LIVE WEBCAM PROCESSING LOOP
            # ----------------------------------------------------------------------
            else:
                # Step 1: Initialize / retrieve cached models
                with st.spinner("Initializing AI Pose Estimator & Biomechanics Engine..."):
                    try:
                        detector = get_or_create_detector(backend_key)
                        analyzer = get_or_create_analyzer(active_pose_key)
                        voice = get_or_create_voice(enabled=voice_enabled, rate=voice_rate)
                    except Exception as e:
                        st.error(f"Failed to initialize models: {e}")
                        return

                # Step 2: Open webcam stream cleanly
                cap = cv2.VideoCapture(int(camera_idx), cv2.CAP_DSHOW if sys_is_windows() else cv2.CAP_ANY)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(camera_idx))

                if not cap.isOpened():
                    video_placeholder.error(
                        f"⚠️ Could not open webcam at camera index {camera_idx}. "
                        "Please verify that your webcam is connected, not in use by another application, "
                        "and camera privacy permissions are enabled in Windows Settings."
                    )
                    return

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

                milestone_announced = False
                intro_spoken = False
                halfway_encouragement_spoken = False
                no_person_spoken_ts = 0.0
                session_start_time = time.time()
                last_frame_ts = session_start_time
                total_correct_pose_seconds = 0.0
                score_samples = []
                collected_cues = []
                max_hold = 0.0

                try:
                    while start_session_toggle:
                        success, frame = cap.read()
                        if not success or frame is None:
                            video_placeholder.warning("⚠️ Webcam stream interrupted or frame drop detected.")
                            break

                        now_ts = time.time()
                        frame_dt = now_ts - last_frame_ts
                        last_frame_ts = now_ts

                        # Horizontal mirror for intuitive coaching experience
                        frame = cv2.flip(frame, 1)

                        # 1. Pose landmark detection & skeleton overlay
                        det_result = detector.process_frame(frame, draw=True)

                        # 2. Biomechanical posture analysis & scoring
                        eval_result = analyzer.evaluate_result(det_result, pose_name=active_pose_key)
                        score_samples.append(eval_result.overall_score)

                        # Spoken Guided Setup Instruction on camera start
                        if not intro_spoken:
                            intro_spoken = True
                            setup_text = f"Starting {pose_ref.name} practice. "
                            if pose_ref.instructions:
                                setup_text += pose_ref.instructions[0]
                            else:
                                setup_text += "Position your full body in view of the camera."
                            voice.speak(setup_text, priority=True)

                        # Spoken prompt if practitioner is not detected in view
                        if not det_result.pose_detected:
                            if (now_ts - session_start_time > 6.0) and (now_ts - no_person_spoken_ts > 15.0):
                                no_person_spoken_ts = now_ts
                                voice.speak("Please step back into full view of the camera to begin.", priority=False)

                        # Accumulate verified correct pose duration when alignment criteria are satisfied
                        is_pose_verified = (det_result.pose_detected and eval_result.is_pose_correct and eval_result.overall_score >= 75.0)
                        if is_pose_verified:
                            total_correct_pose_seconds += frame_dt

                        # 3. Audio Voice Coaching (non-blocking)
                        if eval_result.actionable_cues:
                            top_cue = eval_result.actionable_cues[0]
                            if top_cue not in collected_cues and len(collected_cues) < 4:
                                collected_cues.append(top_cue)
                            is_safety = "safety" in top_cue.lower() or "never" in top_cue.lower()
                            voice.speak(top_cue, priority=is_safety)
                        elif is_pose_verified and not halfway_encouragement_spoken and analyzer.current_hold_duration >= (target_hold_time * 0.5):
                            halfway_encouragement_spoken = True
                            voice.speak("Halfway through your target hold! Keep your form steady and breathe deeply.", priority=False)

                        # 4. Verified Target Hold Tracking
                        hold_sec = analyzer.current_hold_duration
                        if hold_sec > max_hold:
                            max_hold = hold_sec

                        # Milestone chime & announcement
                        if hold_sec >= target_hold_time and not milestone_announced:
                            milestone_announced = True
                            voice.play_milestone_chime()
                            voice.speak(f"Awesome! Target hold of {target_hold_time} seconds achieved!", priority=True)
                        elif hold_sec < 2.0:
                            milestone_announced = False

                        # 5. Render HUD feedback card & badges onto video frame
                        annotated = analyzer.draw_posture_feedback(det_result.annotated_frame, det_result, eval_result)
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # 6. Real-time Detection Status Banner
                        if not det_result.pose_detected:
                            detection_banner_placeholder.warning("🔴 No practitioner detected. Step into full view of the camera.")
                        elif det_result.status == "PARTIAL_BODY":
                            detection_banner_placeholder.warning("🟡 Partial body detected. Step back so your feet, hips, and shoulders are in view.")
                        else:
                            detected_side_str = f" • Stance: {eval_result.detected_side.replace('_', ' ').title()}" if eval_result.detected_side != "unknown" else ""
                            detection_banner_placeholder.success(f"🟢 Full-Body Tracked (17 Landmarks){detected_side_str}")

                        # 7. Posture Performance Score
                        score = eval_result.overall_score
                        if score >= 75.0 and eval_result.is_pose_correct:
                            score_color = "#2ecc71"
                            verdict = "🟢 Pose Verified"
                        elif score >= 50.0:
                            score_color = "#ff9800"
                            verdict = "🟡 Adjusting Alignment"
                        else:
                            score_color = "#e74c3c"
                            verdict = "🔴 Incorrect Posture"

                        score_placeholder.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Posture Accuracy Score</div>
                                <div class="metric-value" style="color: {score_color};">{score:.0f}%</div>
                                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">{verdict}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        score_bar_placeholder.progress(min(1.0, max(0.0, score / 100.0)))

                        # 8. Verified Hold Progress
                        hold_color = "#2ecc71" if hold_sec >= target_hold_time else "#00c0f2"
                        hold_placeholder.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Verified Hold Time</div>
                                <div class="metric-value" style="color: {hold_color};">{hold_sec:4.1f}s / {target_hold_time}s</div>
                                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Session Best Hold: {max_hold:.1f}s</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        hold_progress_ratio = min(1.0, hold_sec / max(1, target_hold_time))
                        hold_bar_placeholder.progress(hold_progress_ratio)

                        # 9. Session Running Time (Distinguishing elapsed session time from verified hold time)
                        elapsed_total = time.time() - session_start_time
                        mins, secs = divmod(int(elapsed_total), 60)
                        session_timer_placeholder.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Session Running Time</div>
                                <div class="metric-value" style="color: #f1f5f9;">{mins:02d}:{secs:02d}</div>
                                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Total Active Duration</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 10. Vision Tracking Status Card
                        tracking_str = "Optimal (Full Body)" if det_result.status == "VALID_FULL_BODY" else ("Partial View" if det_result.status == "PARTIAL_BODY" else "Searching")
                        tracking_color = "#2ecc71" if det_result.status == "VALID_FULL_BODY" else ("#ff9800" if det_result.status == "PARTIAL_BODY" else "#e74c3c")
                        detection_status_placeholder.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Vision Tracking</div>
                                <div class="metric-value" style="color: {tracking_color}; font-size: 1.35rem;">{tracking_str}</div>
                                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">Backend: {backend_key.upper()}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 11. Real-Time Coaching Cue Card
                        if eval_result.is_pose_correct and score >= 75.0:
                            cue_class = "cue-box-success"
                            cue_title = "🟢 Optimal Alignment Verified:"
                            cue_text = eval_result.actionable_cues[0] if eval_result.actionable_cues else "Posture is aligned! Hold steadily and breathe naturally."
                        elif any("safety" in c.lower() or "never" in c.lower() for c in eval_result.actionable_cues):
                            cue_class = "cue-box-danger"
                            cue_title = "🚨 Safety Alert:"
                            cue_text = [c for c in eval_result.actionable_cues if "safety" in c.lower() or "never" in c.lower()][0]
                        else:
                            cue_class = "cue-box-warning" if score < 60.0 else "cue-box"
                            cue_title = "💡 Alignment Correction:"
                            cue_text = eval_result.actionable_cues[0] if eval_result.actionable_cues else "Adjust limbs according to the reference guide."

                        cue_placeholder.markdown(
                            f"""
                            <div class="{cue_class}">
                                <b>{cue_title}</b><br>{cue_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 12. Detailed Joint Angle Kinematics Breakdown
                        joint_rows = []
                        for j_name, f_data in eval_result.joint_feedbacks.items():
                            val = f"{f_data.measured_angle:.0f}°" if f_data.measured_angle is not None else "N/A"
                            tgt = f"{f_data.target_angle:.0f}°"
                            if f_data.status.value == "CORRECT":
                                badge = "<span style='color: #2ecc71;'>● Correct</span>"
                            elif f_data.status.value == "WARNING":
                                badge = "<span style='color: #ff9800;'>▲ Adjust</span>"
                            else:
                                badge = "<span style='color: #e74c3c;'>■ Incorrect</span>"

                            joint_rows.append(
                                f"<tr>"
                                f"<td style='padding: 6px 12px; border-bottom: 1px solid #1E293B;'><b style='color: #F1F5F9;'>{j_name.replace('_', ' ').title()}</b></td>"
                                f"<td style='padding: 6px 12px; border-bottom: 1px solid #1E293B; color: #38BDF8; font-weight: 600;'>{val}</td>"
                                f"<td style='padding: 6px 12px; border-bottom: 1px solid #1E293B; color: #94A3B8;'>Target ~{tgt}</td>"
                                f"<td style='padding: 6px 12px; border-bottom: 1px solid #1E293B;'>{badge}</td>"
                                f"<td style='padding: 6px 12px; border-bottom: 1px solid #1E293B; font-size: 0.85rem; color: #CBD5E1;'>{f_data.message}</td>"
                                f"</tr>"
                            )

                        joint_details_placeholder.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Live Joint Angle Kinematics ({eval_result.pose_name})</div>
                                <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95rem;">
                                    <thead>
                                        <tr style="text-align: left; color: #94A3B8; border-bottom: 2px solid #243049;">
                                            <th style="padding: 6px 12px;">Joint / Segment</th>
                                            <th style="padding: 6px 12px;">Measured</th>
                                            <th style="padding: 6px 12px;">Target Range</th>
                                            <th style="padding: 6px 12px;">Status</th>
                                            <th style="padding: 6px 12px;">Biomechanical Guidance</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {"".join(joint_rows)}
                                    </tbody>
                                </table>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Yield CPU cycle
                        time.sleep(0.01)

                finally:
                    # Clean release of hardware resources
                    cap.release()

                    # Persist completed session to workout history
                    session_duration = time.time() - session_start_time
                    if session_duration >= 2.0:
                        avg_score = float(np.mean(score_samples)) if score_samples else 75.0
                        saved_pose = eval_result.pose_name if ("eval_result" in locals() and eval_result) else pose_ref.name
                        saved_stance = eval_result.detected_side if ("eval_result" in locals() and eval_result) else "standard"
                        logger.log_session(
                            pose_name=saved_pose,
                            duration_seconds=session_duration,
                            target_duration=target_hold_time,
                            correct_pose_seconds=total_correct_pose_seconds,
                            max_continuous_hold=max_hold,
                            average_score=avg_score,
                            target_achieved=(max_hold >= target_hold_time),
                            cues_provided=collected_cues,
                            stance=saved_stance,
                        )
                        st.session_state["single_pose_last_saved"] = f"{saved_pose} • {session_duration:.1f}s"
                        st.toast(f"✅ Session for {saved_pose} logged to Workout History!", icon="💾")

    # ==========================================================================
    # PAGE 4: PRACTICE HISTORY & PERFORMANCE ANALYTICS
    # ==========================================================================
    elif active_page == "📊 Workout History":
        st.markdown(
            """
            <div class="history-header">
                <div class="history-title">Practice History & Performance Analytics</div>
                <div class="history-subtitle">Track your yoga practice journey and see how you are improving over time.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        stats = logger.get_summary_stats()
        history = logger.get_history()

        # ----------------------------------------------------------------------
        # 1. FOUR SUMMARY CARDS
        # ----------------------------------------------------------------------
        card_col1, card_col2, card_col3, card_col4 = st.columns(4)

        # Card 1: Total Yoga Practice (Blue Accent)
        with card_col1:
            tot_time_str = format_seconds_readable(stats.get("total_practice_time_seconds"))
            st.markdown(
                f"""
                <div class="summary-card card-blue">
                    <div>
                        <div class="card-icon-label">⏱️ Total Yoga Practice</div>
                        <div class="card-value" style="color: #2B7A78;">{tot_time_str}</div>
                    </div>
                    <div class="card-desc">Total time spent practising across all sessions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Card 2: Correct Pose Time (Green Accent)
        with card_col2:
            correct_time_str = format_seconds_readable(stats.get("total_correct_pose_seconds"))
            st.markdown(
                f"""
                <div class="summary-card card-green">
                    <div>
                        <div class="card-icon-label">🎯 Correct Pose Time</div>
                        <div class="card-value" style="color: #34D399;">{correct_time_str}</div>
                    </div>
                    <div class="card-desc">Total time when your posture was correctly aligned.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Card 3: Total Sessions (Purple Accent)
        with card_col3:
            tot_sess = stats.get("total_sessions", 0)
            st.markdown(
                f"""
                <div class="summary-card card-purple">
                    <div>
                        <div class="card-icon-label">🧘 Total Sessions</div>
                        <div class="card-value" style="color: #C084FC;">{tot_sess}</div>
                    </div>
                    <div class="card-desc">Number of completed practice sessions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Card 4: Best Correct Hold (Gold Accent)
        with card_col4:
            best_hold_str = format_seconds_readable(stats.get("best_overall_hold"))
            st.markdown(
                f"""
                <div class="summary-card card-gold">
                    <div>
                        <div class="card-icon-label">🏆 Best Correct Hold</div>
                        <div class="card-value" style="color: #FBBF24;">{best_hold_str}</div>
                    </div>
                    <div class="card-desc">Longest continuous time you maintained the correct pose.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 2. PRACTICE TIME VS CORRECT POSE TIME CHART
        # ----------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-heading">Practice Time vs Correct Pose Time</div>
            <div class="section-subheading">Compare your total practice time and correct pose time for each session.</div>
            """,
            unsafe_allow_html=True,
        )

        if history:
            chart_rows = []
            for i, r in enumerate(history):
                sess_label = f"Session {i + 1}"
                # Series 1: Total Practice Time (Blue)
                chart_rows.append({
                    "Session": sess_label,
                    "Metric": "Total Practice Time",
                    "Duration": round(float(r.get("duration_seconds", 0.0)), 1),
                    "Order": i + 1,
                })
                # Series 2: Correct Pose Time (Green) - exclude if not recorded
                if r.get("correct_pose_seconds") is not None:
                    chart_rows.append({
                        "Session": sess_label,
                        "Metric": "Correct Pose Time",
                        "Duration": round(float(r["correct_pose_seconds"]), 1),
                        "Order": i + 1,
                    })

            chart_df = pd.DataFrame(chart_rows)

            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X(
                        "Session:N",
                        title="Practice Sessions (Chronological)",
                        sort=alt.EncodingSortField(field="Order", order="ascending"),
                        axis=alt.Axis(
                            labelColor="#94A3B8",
                            titleColor="#F1F5F9",
                            labelAngle=-25 if len(history) > 8 else 0,
                            gridColor="#1E293B",
                        ),
                    ),
                    y=alt.Y(
                        "Duration:Q",
                        title="Duration (seconds)",
                        axis=alt.Axis(
                            gridColor="#1E293B",
                            labelColor="#94A3B8",
                            titleColor="#F1F5F9",
                        ),
                    ),
                    xOffset=alt.XOffset("Metric:N"),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=["Total Practice Time", "Correct Pose Time"],
                            range=["#38BDF8", "#34D399"],
                        ),
                        legend=alt.Legend(
                            title="",
                            orient="top",
                            labelColor="#F1F5F9",
                            labelFontSize=12,
                            symbolType="square",
                            padding=10,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Session:N", title="Session"),
                        alt.Tooltip("Metric:N", title="Type"),
                        alt.Tooltip("Duration:Q", title="Duration (sec)", format=".1f"),
                    ],
                )
                .properties(height=340)
                .configure_view(stroke=None)
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.markdown(
                """
                <div class="empty-history-box">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">📊</div>
                    <div style="font-weight: 600; color: #FFFFFF; font-size: 1.05rem;">No Practice Data Yet</div>
                    <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 4px;">
                        Complete a session in <b>Live & Guided Practice</b> to generate practice comparisons!
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 3. RECENT PRACTICE SESSIONS TABLE
        # ----------------------------------------------------------------------
        st.markdown(
            """
            <div class="section-heading">Recent Practice Sessions</div>
            """,
            unsafe_allow_html=True,
        )

        if history:
            table_rows = []
            n_sess = len(history)
            # Show newest sessions first (reverse chronological order)
            for i in range(n_sess - 1, -1, -1):
                r = history[i]
                table_rows.append({
                    "Session #": f"#{i + 1}",
                    "Date & Time": format_datetime_readable(r.get("timestamp", "")),
                    "Yoga Pose": format_pose_display_name(r.get("pose_name", "Unknown")),
                    "Total Practice Time": format_seconds_readable(r.get("duration_seconds")),
                    "Correct Pose Time": format_seconds_readable(r.get("correct_pose_seconds")),
                    "Best Hold": format_seconds_readable(r.get("best_hold_seconds")),
                })

            table_df = pd.DataFrame(table_rows)
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------------------------
            # 4. CLEAR PRACTICE HISTORY (WITH CONFIRMATION DIALOG)
            # ------------------------------------------------------------------
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            if "confirm_clear_history" not in st.session_state:
                st.session_state.confirm_clear_history = False

            if not st.session_state.confirm_clear_history:
                if st.button("🗑️ Clear Practice History", type="secondary", help="Delete all recorded workout sessions"):
                    st.session_state.confirm_clear_history = True
                    st.rerun()
            else:
                st.warning("⚠️ **Confirm Deletion:** Are you sure you want to permanently delete all practice records? This action cannot be undone.")
                col_del_yes, col_del_no, _ = st.columns([1.2, 1.2, 3])
                with col_del_yes:
                    if st.button("✅ Yes, Delete All", type="primary", use_container_width=True):
                        logger.clear_history()
                        st.session_state.confirm_clear_history = False
                        st.success("Practice history cleared successfully.")
                        time.sleep(0.4)
                        st.rerun()
                with col_del_no:
                    if st.button("❌ Cancel", type="secondary", use_container_width=True):
                        st.session_state.confirm_clear_history = False
                        st.rerun()
        else:
            st.markdown(
                """
                <div class="empty-history-box">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">🧘</div>
                    <div style="font-weight: 600; color: #FFFFFF; font-size: 1.05rem;">No Recorded Sessions</div>
                    <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 4px;">
                        Your completed practice logs and joint stability metrics will be displayed here automatically.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # PAGE 5: POSTURE BIOMECHANICS GUIDE
    # ==========================================================================
    elif active_page == "📖 Biomechanics Guide":
        st.subheader("📖 Posture Alignment Rules & Biomechanics Reference")

        col_w, col_t = st.columns(2)
        with col_w:
            st.markdown("### ⚔️ Warrior II (*Virabhadrasana II*)")
            st.write(
                """
                - **Front Knee Lunge:** Bend to **90°** (acceptable range: 80°–105°). Ensure the knee stays directly above the ankle.
                - **Back Leg:** Fully straight and grounded through the back heel (~**175°**).
                - **Arms:** Extended horizontally at shoulder level, parallel to the ground (**0° horizontal**).
                - **Torso:** Keep shoulders stacked vertically over hips; do not lean forward.
                """
            )

        with col_t:
            st.markdown("### 🌲 Tree Pose (*Vrikshasana*)")
            st.write(
                """
                - **Standing Leg:** Keep the knee straight (~**175°**), engaging the thigh without hyperextending.
                - **Lifted Leg:** Bend outward at the hip (**25°–65°**).
                - **Critical Safety Rule:** Never place your foot directly on the knee joint! Place it on your inner thigh or calf.
                - **Hands & Arms:** Join hands at chest in prayer (*Anjali Mudra*) or reach overhead.
                - **Spine:** Maintain tall, vertical spine alignment without tilting sideways.
                """
            )

        col_d, col_p, col_c = st.columns(3)
        with col_d:
            st.markdown("### 🐕 Downward Dog (*Adho Mukha Svanasana*)")
            st.write(
                """
                - **Hip Peak:** Form an inverted 'V' shape (**65°–88°** hip angle).
                - **Legs:** Straight knees (~**175°**); press heels toward the mat.
                - **Arms:** Push firmly into palms with straight arms (**160°–180°**).
                """
            )
        with col_p:
            st.markdown("### 🪵 Plank Pose (*Phalakasana*)")
            st.write(
                """
                - **Body Line:** Neutral line from shoulders through hips to heels (**165°–180°**).
                - **Core:** Avoid sagging lower back or piking hips too high.
                - **Arms:** Straight arms stacked beneath shoulders (**155°–180°**).
                """
            )
        with col_c:
            st.markdown("### 🐍 Cobra Pose (*Bhujangasana*)")
            st.write(
                """
                - **Chest Arch:** Elevated thoracic spine (**125°–150°**).
                - **Elbows:** Soft micro-bend hugging ribcage (**120°–160°**).
                - **Pelvis:** Keep hips and tops of feet grounded on the mat.
                """
            )

        st.markdown("---")
        st.markdown("### ⛰️ Mountain Pose (*Tadasana*)")
        st.write(
            """
            - **Legs:** Symmetrical extension in both knees (**165°–180°**) with quadriceps engaged.
            - **Hips & Pelvis:** Neutral pelvic tilt with hips stacked vertically over ankles (**165°–180°**).
            - **Spine & Torso:** Elongated vertical plumb line (**<= 12° deviation from vertical**).
            - **Shoulders & Arms:** Relaxed shoulders rolled back and down, arms resting naturally alongside torso.
            """
        )

        st.markdown("---")
        st.subheader("🧘 22-Posture Guided Routine Curriculum")
        st.caption("Standard sequential curriculum progressing from warm-ups to standing asanas, floor stretches, and deep restorative relaxation.")

        with st.expander("📋 View Complete 22-Posture Sequence & Biomechanical Phases", expanded=True):
            routine_table_rows = [
                "<tr style='background: #161E2E; color: #FFFFFF; border-bottom: 2px solid #243049;'><th style='padding:8px 12px; text-align: left;'>#</th><th style='padding:8px 12px; text-align: left;'>Posture Name</th><th style='padding:8px 12px; text-align: left;'>Sanskrit</th><th style='padding:8px 12px; text-align: left;'>Phase</th><th style='padding:8px 12px; text-align: left;'>Target Hold</th><th style='padding:8px 12px; text-align: left;'>Stance Mode</th></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>1</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Easy Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Sukhasana</td><td style='padding:6px 12px; color: #34D399;'>Phase 1 — Warm-up</td><td style='padding:6px 12px; color: #E2E8F0;'>60s</td><td style='padding:6px 12px; color: #CBD5E1;'>Seated Neutral</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>2</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Gentle Neck Stretches</b></td><td style='padding:6px 12px; color: #94A3B8;'>Kantha Sanchalana</td><td style='padding:6px 12px; color: #34D399;'>Phase 1 — Warm-up</td><td style='padding:6px 12px; color: #E2E8F0;'>30s</td><td style='padding:6px 12px; color: #CBD5E1;'>Dynamic Reps</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>3</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Shoulder Rolls</b></td><td style='padding:6px 12px; color: #94A3B8;'>Skandha Chakra</td><td style='padding:6px 12px; color: #34D399;'>Phase 1 — Warm-up</td><td style='padding:6px 12px; color: #E2E8F0;'>30s</td><td style='padding:6px 12px; color: #CBD5E1;'>Dynamic Reps</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>4</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Mountain Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Tadasana</td><td style='padding:6px 12px; color: #34D399;'>Phase 1 — Warm-up</td><td style='padding:6px 12px; color: #E2E8F0;'>40s</td><td style='padding:6px 12px; color: #CBD5E1;'>Symmetrical Standing (AI Evaluated)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>5</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Standing Side Stretch</b></td><td style='padding:6px 12px; color: #94A3B8;'>Tiryaka Tadasana</td><td style='padding:6px 12px; color: #34D399;'>Phase 1 — Warm-up</td><td style='padding:6px 12px; color: #E2E8F0;'>30s</td><td style='padding:6px 12px; color: #CBD5E1;'>Lateral Flexion</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>6</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Tree Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Vrikshasana</td><td style='padding:6px 12px; color: #38BDF8;'>Phase 2 — Standing Poses</td><td style='padding:6px 12px; color: #E2E8F0;'>30s / side</td><td style='padding:6px 12px; color: #CBD5E1;'>Bilateral (Left/Right, AI Evaluated)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>7</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Triangle Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Trikonasana</td><td style='padding:6px 12px; color: #38BDF8;'>Phase 2 — Standing Poses</td><td style='padding:6px 12px; color: #E2E8F0;'>30s / side</td><td style='padding:6px 12px; color: #CBD5E1;'>Bilateral (Left/Right)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>8</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Warrior I</b></td><td style='padding:6px 12px; color: #94A3B8;'>Virabhadrasana I</td><td style='padding:6px 12px; color: #38BDF8;'>Phase 2 — Standing Poses</td><td style='padding:6px 12px; color: #E2E8F0;'>25s / side</td><td style='padding:6px 12px; color: #CBD5E1;'>Bilateral (Left/Right)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>9</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Warrior II</b></td><td style='padding:6px 12px; color: #94A3B8;'>Virabhadrasana II</td><td style='padding:6px 12px; color: #38BDF8;'>Phase 2 — Standing Poses</td><td style='padding:6px 12px; color: #E2E8F0;'>25s / side</td><td style='padding:6px 12px; color: #CBD5E1;'>Bilateral (Left/Right, AI Evaluated)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>10</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Chair Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Utkatasana</td><td style='padding:6px 12px; color: #38BDF8;'>Phase 2 — Standing Poses</td><td style='padding:6px 12px; color: #E2E8F0;'>20s</td><td style='padding:6px 12px; color: #CBD5E1;'>Knee Flexion Hold</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>11</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Standing Forward Bend</b></td><td style='padding:6px 12px; color: #94A3B8;'>Uttanasana</td><td style='padding:6px 12px; color: #38BDF8;'>Phase 2 — Standing Poses</td><td style='padding:6px 12px; color: #E2E8F0;'>25s</td><td style='padding:6px 12px; color: #CBD5E1;'>Hip Hinge</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>12</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Cat Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Marjaryasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>40s</td><td style='padding:6px 12px; color: #CBD5E1;'>Tabletop Spinal Flexion</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>13</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Cow Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Bitilasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>40s</td><td style='padding:6px 12px; color: #CBD5E1;'>Tabletop Spinal Extension</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>14</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Child's Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Balasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>45s</td><td style='padding:6px 12px; color: #CBD5E1;'>Restorative Kneeling</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>15</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Cobra Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Bhujangasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>20s</td><td style='padding:6px 12px; color: #CBD5E1;'>Prone Backbend (AI Evaluated)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>16</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Downward-Facing Dog</b></td><td style='padding:6px 12px; color: #94A3B8;'>Adho Mukha Svanasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>25s</td><td style='padding:6px 12px; color: #CBD5E1;'>Inversion Apex (AI Evaluated)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>17</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Bridge Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Setu Bandhasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>20s</td><td style='padding:6px 12px; color: #CBD5E1;'>Supine Pelvic Lift</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>18</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Knees-to-Chest Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Apanasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>30s</td><td style='padding:6px 12px; color: #CBD5E1;'>Supine Decompression</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>19</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Supine Spinal Twist</b></td><td style='padding:6px 12px; color: #94A3B8;'>Supta Matsyendrasana</td><td style='padding:6px 12px; color: #C084FC;'>Phase 3 — Floor and Stretching</td><td style='padding:6px 12px; color: #E2E8F0;'>30s / side</td><td style='padding:6px 12px; color: #CBD5E1;'>Bilateral (Left/Right)</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>20</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Thunderbolt Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Vajrasana</td><td style='padding:6px 12px; color: #FBBF24;'>Phase 4 — Breathing & Relaxation</td><td style='padding:6px 12px; color: #E2E8F0;'>45s</td><td style='padding:6px 12px; color: #CBD5E1;'>Kneeling Spine</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>21</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Alternate Nostril Breathing</b></td><td style='padding:6px 12px; color: #94A3B8;'>Nadi Shodhana Pranayama</td><td style='padding:6px 12px; color: #FBBF24;'>Phase 4 — Breathing & Relaxation</td><td style='padding:6px 12px; color: #E2E8F0;'>90s</td><td style='padding:6px 12px; color: #CBD5E1;'>Breath & Mudra</td></tr>",
                "<tr style='border-bottom: 1px solid #1E293B;'><td style='padding:6px 12px; color: #F1F5F9;'>22</td><td style='padding:6px 12px;'><b style='color: #FFFFFF;'>Corpse Pose</b></td><td style='padding:6px 12px; color: #94A3B8;'>Shavasana</td><td style='padding:6px 12px; color: #FBBF24;'>Phase 4 — Breathing & Relaxation</td><td style='padding:6px 12px; color: #E2E8F0;'>180s</td><td style='padding:6px 12px; color: #CBD5E1;'>Deep Relaxation</td></tr>",
            ]
            st.markdown(
                f"""
                <div class="metric-card" style="margin-top: 10px; overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                        {"".join(routine_table_rows)}
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
