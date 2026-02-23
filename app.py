"""
SurgiPath — AI-guided surgical training coach.

Combines YOLOv11 (tool detection) + MediaPipe (hand tracking).

Three phases:
  1. SETUP   — Camera calibration + tool checklist (all required tools visible?)
  2. PRACTICE — Real-time coaching: grip, stability, smoothness, rule violations
  3. REPORT  — Post-session summary with mastery score + error log

Video paths:
  - WebRTC (live webcam, 15-30 FPS via callback thread)
  - OpenCV fallback (live/upload, ~1 FPS via @st.fragment)
  - Demo mode (synthetic detections, no camera needed)

Usage:
  streamlit run app.py
"""

import os
import queue
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from styles import load_css
from src.constants import (
    MODEL_PATH, RECIPE_PATH,
    CONF_MIN_DEFAULT, IMGSZ_DEFAULT, FRAME_SKIP_DEFAULT,
    SMOOTH_WINDOW_SIZE, EVIDENCE_WINDOW, MISSING_GRACE_SEC,
    AUTO_TRANSITION_ENABLED, WEBRTC_QUEUE_MAXSIZE,
    KEY_NAV, KEY_PREOP_SMOOTHER, KEY_PREOP_STABLE_START,
    KEY_ALERTS_LOG, KEY_LAST_DETECTIONS, KEY_LAST_COUNTS,
    KEY_FRAME_INDEX, KEY_CONFIG_CONF_MIN, KEY_CONFIG_IMGSZ,
    KEY_CONFIG_FRAME_SKIP, KEY_CAMERA,
    KEY_VIDEO_SOURCE, KEY_UPLOADED_FILE,
    KEY_STREAK_SECONDS, KEY_STREAK_BEST,
    KEY_SESSION_START, KEY_HAND_STABILITY,
    KEY_DEMO_MODE, KEY_DEMO_TICK,
    KEY_EVIDENCE, KEY_COACH_PROMPTS, KEY_OVERRIDES,
    KEY_PROMPT_COUNTER, KEY_LAST_PROMPT_TS, KEY_AUTO_TRANSITION,
    KEY_CALIBRATION_DONE, KEY_STREAM_START_TS,
    KEY_PREV_HANDS, KEY_HELD_TOOLS, KEY_TECHNIQUE, KEY_BIMANUAL_HISTORY,
    KEY_WEBRTC_QUEUE, KEY_WEBRTC_ACTIVE, KEY_WEBRTC_ENABLED,
    KEY_PERF_SAMPLES, KEY_LAST_DISPLAY_TS,
    KEY_TIP_HISTORY, KEY_WRIST_PATH, KEY_JERK_DATA, KEY_ECONOMY_DATA,
)
from src.state import init_state, get_mode, set_mode
from src.detector import get_model, infer_tools, count_tools
from src.utils import load_recipe, ToolPresenceSmoother
from src.evidence import EvidenceState

from ui.video import (
    configure as configure_video,
    display_frame, generate_demo_detections, render_demo_frame,
    process_frame, make_webrtc_callback, drain_webrtc_queue,
)
from ui.setup import render_setup_tab
from ui.practice import render_practice_tab
from ui.report import render_report_tab

# --- .env loading ---

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
except Exception:
    pass

# --- Brain (Gemini AI) — optional ---

try:
    from brain import generate_final_critique, generate_dynamic_syllabus, SyllabusError
    BRAIN_AVAILABLE = True
    BRAIN_IMPORT_ERROR = ""
except Exception:
    generate_final_critique = None
    generate_dynamic_syllabus = None
    SyllabusError = None
    BRAIN_AVAILABLE = False
    import traceback
    BRAIN_IMPORT_ERROR = "brain.py import failed: " + traceback.format_exc().splitlines()[-1]


def resolve_brain_import() -> bool:
    """Try importing brain.py lazily — useful if deps were installed after app start."""
    global generate_final_critique, generate_dynamic_syllabus, SyllabusError
    global BRAIN_AVAILABLE, BRAIN_IMPORT_ERROR
    if BRAIN_AVAILABLE and generate_dynamic_syllabus is not None:
        return True
    try:
        from brain import (
            generate_final_critique as _gfc,
            generate_dynamic_syllabus as _gds,
            SyllabusError as _se,
        )
        generate_final_critique = _gfc
        generate_dynamic_syllabus = _gds
        SyllabusError = _se
        BRAIN_AVAILABLE = True
        BRAIN_IMPORT_ERROR = ""
        return True
    except Exception:
        import traceback
        BRAIN_AVAILABLE = False
        BRAIN_IMPORT_ERROR = "brain.py import failed: " + traceback.format_exc().splitlines()[-1]
        return False


# --- Page config ---

st.set_page_config(page_title="SurgiPath", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
load_css()

# --- Config helpers ---


def get_config() -> dict:
    if KEY_CONFIG_CONF_MIN not in st.session_state:
        st.session_state[KEY_CONFIG_CONF_MIN] = CONF_MIN_DEFAULT
    if KEY_CONFIG_IMGSZ not in st.session_state:
        st.session_state[KEY_CONFIG_IMGSZ] = IMGSZ_DEFAULT
    if KEY_CONFIG_FRAME_SKIP not in st.session_state:
        st.session_state[KEY_CONFIG_FRAME_SKIP] = FRAME_SKIP_DEFAULT
    return {
        "conf_min": st.session_state[KEY_CONFIG_CONF_MIN],
        "imgsz": st.session_state[KEY_CONFIG_IMGSZ],
        "frame_skip": max(1, st.session_state[KEY_CONFIG_FRAME_SKIP]),
    }


def init_session_state() -> None:
    init_state()
    defaults = {
        KEY_PREOP_SMOOTHER: lambda: ToolPresenceSmoother(window_size=SMOOTH_WINDOW_SIZE),
        KEY_PREOP_STABLE_START: lambda: None,
        KEY_ALERTS_LOG: list, KEY_LAST_DETECTIONS: list, KEY_LAST_COUNTS: dict,
        KEY_FRAME_INDEX: lambda: 0, KEY_NAV: lambda: "Setup",
        KEY_VIDEO_SOURCE: lambda: "Live Webcam", KEY_UPLOADED_FILE: lambda: None,
        KEY_STREAK_SECONDS: lambda: 0.0, KEY_STREAK_BEST: lambda: 0.0,
        KEY_SESSION_START: lambda: None, KEY_HAND_STABILITY: list,
        KEY_DEMO_MODE: lambda: False, KEY_DEMO_TICK: lambda: 0,
        KEY_EVIDENCE: lambda: EvidenceState(window=EVIDENCE_WINDOW, missing_grace_sec=MISSING_GRACE_SEC),
        KEY_COACH_PROMPTS: list, KEY_OVERRIDES: list,
        KEY_PROMPT_COUNTER: lambda: 0, KEY_LAST_PROMPT_TS: dict,
        KEY_AUTO_TRANSITION: lambda: AUTO_TRANSITION_ENABLED,
        KEY_CALIBRATION_DONE: lambda: False, KEY_STREAM_START_TS: lambda: None,
        KEY_PREV_HANDS: list, KEY_HELD_TOOLS: set,
        KEY_TECHNIQUE: dict, KEY_BIMANUAL_HISTORY: list,
        KEY_WEBRTC_QUEUE: lambda: queue.Queue(maxsize=WEBRTC_QUEUE_MAXSIZE),
        KEY_WEBRTC_ACTIVE: lambda: False, KEY_WEBRTC_ENABLED: lambda: True,
        KEY_PERF_SAMPLES: list, KEY_LAST_DISPLAY_TS: lambda: 0.0,
        "_tts_queue": list, "_tts_busy_until": lambda: 0.0,
        KEY_TIP_HISTORY: list, KEY_WRIST_PATH: list,
        KEY_JERK_DATA: dict, KEY_ECONOMY_DATA: dict,
        "_brain_summary": lambda: "", "_gemini_key": lambda: os.getenv("GOOGLE_API_KEY", ""),
        "_ai_reasoning_enabled": lambda: True,
        "_procedure_text": lambda: "", "_procedure_name": lambda: "",
        "_procedure_steps": list,
        "_procedure_gen_error": lambda: "", "_procedure_gen_info": lambda: "",
        "_procedure_gen_pending": lambda: False,
        "_procedure_gen_pending_prompt": lambda: "", "_procedure_gen_pending_name": lambda: "",
        "_manual_steps_table": list,
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


# --- Tool name helpers ---


def norm_tool(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def counts_for_recipe(counts_norm: dict[str, int], required_tools: list[dict]) -> dict[str, int]:
    return {r["tool"]: counts_norm.get(norm_tool(r["tool"]), 0) for r in required_tools if r.get("tool")}


# --- Camera ---


def open_camera() -> cv2.VideoCapture | None:
    for api in ((cv2.CAP_DSHOW, cv2.CAP_ANY) if hasattr(cv2, "CAP_DSHOW") else (cv2.CAP_ANY,)):
        try:
            cap = cv2.VideoCapture(0, api)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return None


def get_video_capture() -> cv2.VideoCapture | None:
    source = st.session_state.get(KEY_VIDEO_SOURCE, "Live Webcam")
    existing = st.session_state.get(KEY_CAMERA)
    if existing is not None:
        try:
            if existing.isOpened():
                return existing
        except Exception:
            pass
        st.session_state[KEY_CAMERA] = None

    cap = None
    if source == "Live Webcam":
        cap = open_camera()
    elif source == "Upload Video":
        data = st.session_state.get(KEY_UPLOADED_FILE)
        if not data:
            return None
        tmp_path = st.session_state.get("_upload_tmp_path")
        if tmp_path is None or not Path(tmp_path).exists():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(data)
            tmp.flush()
            tmp.close()
            tmp_path = tmp.name
            st.session_state["_upload_tmp_path"] = tmp_path
        cap = cv2.VideoCapture(tmp_path)

    if cap is not None and cap.isOpened():
        st.session_state[KEY_CAMERA] = cap
        return cap
    if cap is not None:
        cap.release()
    return None


@st.cache_resource
def cached_model():
    return get_model(MODEL_PATH)


# --- Recipe ---


def load_recipe_safe() -> dict:
    try:
        return load_recipe(RECIPE_PATH)
    except Exception:
        return {"preop_required": [], "params": {"conf_min": CONF_MIN_DEFAULT, "stable_seconds": 2.0}, "intraop_rules": []}


recipe = load_recipe_safe()
preop_required = recipe.get("preop_required", [])
params = recipe.get("params", {})
stable_seconds = float(params.get("stable_seconds", 2.0))
intraop_rules = recipe.get("intraop_rules", [])
known_tools = [r["tool"] for r in preop_required]
_alias_map: dict[str, str] = recipe.get("aliases", {})


def alias_tool(name: str) -> str:
    return _alias_map.get(norm_tool(name), norm_tool(name))


def remap_detections(detections: list[dict]) -> list[dict]:
    ignore = {"", "empty", "background", "none", "null"}
    out = []
    for d in detections:
        mapped = alias_tool(d.get("name", ""))
        if mapped in ignore:
            continue
        item = dict(d)
        item["name"] = mapped
        out.append(item)
    return out


# --- Wire recipe into video module ---

configure_video(preop_required, known_tools, norm_tool, counts_for_recipe, remap_detections)

# --- Init ---

init_session_state()

# =============================================================================
# Sidebar
# =============================================================================

NAV_ITEMS = ["Setup", "Practice", "Report"]

with st.sidebar:
    st.markdown(
        '<div class="dashboard-header"><h1>SurgiPath</h1>'
        "<p>AI Surgical Training &nbsp;·&nbsp; Skill Assessment</p></div>",
        unsafe_allow_html=True,
    )
    mode = get_mode()
    badge_map = {"PRE_OP": ("● Setup", "status-idle"), "INTRA_OP": ("● Practicing", "status-active"), "POST_OP": ("● Report Ready", "status-ready")}
    badge_text, badge_cls = badge_map.get(mode, badge_map["PRE_OP"])
    st.markdown(f'<span class="status-badge {badge_cls}">{badge_text}</span>', unsafe_allow_html=True)

    ai_enabled = st.session_state.get("_ai_reasoning_enabled", True)
    if st.button("👁 AI Reasoning ON" if ai_enabled else "🙈 AI Reasoning OFF", width="stretch", key="ai_eye_toggle_btn"):
        st.session_state["_ai_reasoning_enabled"] = not ai_enabled
        st.rerun()

    st.markdown("---")
    st.markdown("### Video Source")
    source = st.radio("Feed", ["Live Webcam", "Upload Video"],
                      index=["Live Webcam", "Upload Video"].index(st.session_state.get(KEY_VIDEO_SOURCE, "Live Webcam")),
                      label_visibility="collapsed", key="source_radio")
    st.session_state[KEY_VIDEO_SOURCE] = source
    if source == "Upload Video":
        uploaded = st.file_uploader("Upload a lab recording", type=["mp4", "avi", "mov"], key="vid_upload")
        if uploaded is not None:
            data = uploaded.getvalue()
            if data:
                st.session_state[KEY_UPLOADED_FILE] = data
                if st.session_state.get("_upload_tmp_path"):
                    old = Path(st.session_state["_upload_tmp_path"])
                    if old.exists():
                        try: old.unlink()
                        except OSError: pass
                    st.session_state["_upload_tmp_path"] = None

    st.markdown("---")
    demo = st.toggle("Demo Mode (no camera needed)", value=st.session_state.get(KEY_DEMO_MODE, False), key="demo_toggle")
    if get_mode() != "PRE_OP":
        st.session_state[KEY_DEMO_MODE] = False
    else:
        st.session_state[KEY_DEMO_MODE] = demo
    if st.session_state.get(KEY_DEMO_MODE, False):
        st.caption("Synthetic detections for Setup only. Practice always uses real input.")

    webrtc_on = st.toggle("WebRTC Mode (higher FPS for live webcam)", value=st.session_state.get(KEY_WEBRTC_ENABLED, True), key="webrtc_toggle")
    st.session_state[KEY_WEBRTC_ENABLED] = webrtc_on
    if webrtc_on:
        st.caption("WebRTC streams video in browser at 15-30 FPS. Click START in the player to begin.")

    if resolve_brain_import():
        st.caption("AI reasoning: ready")
    else:
        st.caption("AI reasoning: unavailable (install brain dependencies)")

    auto_trans = st.toggle("Auto-start Practice when ready", value=st.session_state.get(KEY_AUTO_TRANSITION, AUTO_TRANSITION_ENABLED), key="auto_trans_toggle")
    st.session_state[KEY_AUTO_TRANSITION] = auto_trans

    st.markdown("---")
    st.markdown("### Navigation")
    nav = st.radio("Section", NAV_ITEMS,
                   index=NAV_ITEMS.index(st.session_state[KEY_NAV]) if st.session_state[KEY_NAV] in NAV_ITEMS else 0,
                   label_visibility="collapsed", key="nav_radio")
    st.session_state[KEY_NAV] = nav

    st.markdown("---")
    with st.expander("Settings", expanded=False):
        st.session_state[KEY_CONFIG_CONF_MIN] = st.slider("Min confidence", 0.2, 0.9, st.session_state.get(KEY_CONFIG_CONF_MIN, CONF_MIN_DEFAULT), 0.05)
        with st.expander("Advanced / Dev", expanded=False):
            imgsz_opts = [320, 416, 640, 832]
            cur = st.session_state.get(KEY_CONFIG_IMGSZ, IMGSZ_DEFAULT)
            st.session_state[KEY_CONFIG_IMGSZ] = st.selectbox("Inference size", imgsz_opts, index=imgsz_opts.index(cur) if cur in imgsz_opts else 2)
            st.session_state[KEY_CONFIG_FRAME_SKIP] = st.number_input("Frame skip", min_value=1, max_value=10, value=FRAME_SKIP_DEFAULT, step=1)
            cfg = get_config()
            st.caption(f"conf={cfg['conf_min']} imgsz={cfg['imgsz']} skip={cfg['frame_skip']}")
        with st.expander("AI (optional)", expanded=False):
            key_val = st.text_input("Gemini API key", value=st.session_state.get("_gemini_key", ""), type="password", help="Only stored in memory for this run.")
            st.session_state["_gemini_key"] = key_val
            if key_val:
                os.environ["GOOGLE_API_KEY"] = key_val
                st.caption("Gemini connected for AI summary features.")
            elif os.getenv("GOOGLE_API_KEY", ""):
                st.caption("Gemini key loaded from .env")

# =============================================================================
# Main header
# =============================================================================

st.markdown(
    '<div class="dashboard-header"><h1>SurgiPath</h1>'
    "<p>AI-Guided Skill Assessment &nbsp;·&nbsp; Medical Training Lab System</p></div>",
    unsafe_allow_html=True,
)

# =============================================================================
# Video feed (WebRTC / OpenCV / Demo)
# =============================================================================

is_demo = st.session_state.get(KEY_DEMO_MODE, False)
run_feed = nav in ("Setup", "Practice") and mode in ("PRE_OP", "INTRA_OP")
use_webrtc = (
    run_feed and not is_demo
    and st.session_state.get(KEY_VIDEO_SOURCE) == "Live Webcam"
    and st.session_state.get(KEY_WEBRTC_ENABLED, True)
)

if (not run_feed or is_demo or use_webrtc) and KEY_CAMERA in st.session_state and st.session_state[KEY_CAMERA] is not None:
    try: st.session_state[KEY_CAMERA].release()
    except Exception: pass
    st.session_state[KEY_CAMERA] = None

if run_feed:
    if st.session_state.get(KEY_STREAM_START_TS) is None:
        st.session_state[KEY_STREAM_START_TS] = time.time()
    st_autorefresh(interval=2000, limit=None, key="checklist_sync")

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]}
MEDIA_CONSTRAINTS = {"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15, "max": 30}}, "audio": False}

if use_webrtc:
    try: model = cached_model()
    except FileNotFoundError: st.error(f"Model not found: {MODEL_PATH}"); model = None
    if model is not None:
        cfg = get_config()
        result_q: queue.Queue = st.session_state[KEY_WEBRTC_QUEUE]
        ctx = webrtc_streamer(
            key="surgipath-live", mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=make_webrtc_callback(result_q, model, cfg),
            media_stream_constraints=MEDIA_CONSTRAINTS, async_processing=True,
        )
        st.session_state[KEY_WEBRTC_ACTIVE] = ctx.state.playing if ctx else False
        drain_webrtc_queue(result_q)

elif run_feed:
    @st.fragment(run_every=timedelta(seconds=1.0))
    def video_feed_fragment():
        if st.session_state.get(KEY_NAV, "") not in ("Setup", "Practice"):
            return
        if get_mode() not in ("PRE_OP", "INTRA_OP"):
            return

        evidence: EvidenceState = st.session_state[KEY_EVIDENCE]
        smoother = st.session_state[KEY_PREOP_SMOOTHER]

        if st.session_state.get(KEY_DEMO_MODE, False):
            tick = st.session_state.get(KEY_DEMO_TICK, 0)
            st.session_state[KEY_DEMO_TICK] = tick + 1
            detections = generate_demo_detections(tick, preop_required, get_mode())
            counts_norm = {norm_tool(k): v for k, v in count_tools(detections).items()}
            st.session_state[KEY_LAST_DETECTIONS] = detections
            st.session_state[KEY_LAST_COUNTS] = counts_norm
            smoother.update(counts_for_recipe(counts_norm, preop_required), preop_required)
            evidence.update(detections, frame_bgr=None, known_tools=known_tools)
            display_frame(render_demo_frame(detections))
        else:
            cap = get_video_capture()
            if cap is None:
                src = st.session_state.get(KEY_VIDEO_SOURCE, "Live Webcam")
                st.warning("Camera not available." if src == "Live Webcam" else "Upload a video file using the sidebar.")
                return
            try: model = cached_model()
            except FileNotFoundError: st.error(f"Model not found: {MODEL_PATH}"); return
            cfg = get_config()
            last_frame = None
            for _ in range(max(cfg["frame_skip"], 2)):
                ret, f = cap.read()
                if ret:
                    last_frame = f
            if last_frame is not None:
                detections = remap_detections(infer_tools(last_frame, conf=cfg["conf_min"], imgsz=cfg["imgsz"], model=model))
                annotated, _, _ = process_frame(last_frame, detections, evidence, smoother)
                display_frame(annotated)
            else:
                try: cap.release()
                except Exception: pass
                st.session_state[KEY_CAMERA] = None
                st.warning("No frames — camera may have disconnected.")

    video_feed_fragment()

# =============================================================================
# Tab routing
# =============================================================================

if nav == "Setup":
    render_setup_tab(preop_required, stable_seconds, is_demo, resolve_brain_import, generate_dynamic_syllabus, SyllabusError)

if nav == "Practice":
    render_practice_tab(intraop_rules, use_webrtc, get_config, resolve_brain_import, generate_final_critique, BRAIN_IMPORT_ERROR)

if nav == "Report":
    render_report_tab()

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption(f"SurgiPath • {datetime.now().strftime('%Y-%m-%d %H:%M')}")
