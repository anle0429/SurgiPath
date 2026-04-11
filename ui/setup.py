"""Setup tab: procedure generation, camera calibration, tool checklist."""

import os
import time
from datetime import datetime

import streamlit as st

from src.constants import (
    PHASES, PHASE_LABELS,
    KEY_EVIDENCE, KEY_CALIBRATION_DONE, KEY_DEMO_MODE,
    KEY_PREOP_SMOOTHER, KEY_PREOP_STABLE_START,
    KEY_LAST_DETECTIONS, KEY_NAV, KEY_SESSION_START,
    KEY_STREAK_SECONDS, KEY_STREAK_BEST,
    KEY_ALERTS_LOG, KEY_COACH_PROMPTS, KEY_OVERRIDES,
    KEY_PROMPT_COUNTER, KEY_LAST_PROMPT_TS,
    KEY_PERF_SAMPLES, KEY_LAST_DISPLAY_TS,
    KEY_AUTO_TRANSITION, KEY_VIDEO_SOURCE, KEY_WEBRTC_ENABLED,
)
from src.state import get_mode, set_mode
from src.logger import log_event
from src.evidence import EvidenceState
from ui.components import section_header
from ui.helpers import normalize_procedure_steps


def render_setup_tab(
    preop_required: list[dict],
    stable_seconds: float,
    is_demo: bool,
) -> None:
    section_header("01 / SETUP", "Lab Setup Checklist")

    # --- Procedure generation ---
    st.markdown("### Upload Your Procedure")
    table_seed = st.session_state.get("_manual_steps_table", [])
    if not table_seed:
        table_seed = [{"step_name": "Step 1", "instruction": "Prepare sterile field", "time_limit_seconds": 60}]
    edited = st.data_editor(
        table_seed, key="manual_steps_editor", num_rows="dynamic", width="stretch",
        column_config={
            "step_name": st.column_config.TextColumn("Step"),
            "instruction": st.column_config.TextColumn("Instruction"),
            "time_limit_seconds": st.column_config.NumberColumn("Time (s)", min_value=10, max_value=600, step=5),
        },
        hide_index=True,
    )
    st.session_state["_manual_steps_table"] = edited
    if st.button("Use Manual Procedure Table", width="stretch"):
        steps = []
        for i, row in enumerate(edited):
            instr = str(row.get("instruction", "")).strip()
            if not instr:
                continue
            step_name = str(row.get("step_name", "")).strip() or f"Step {i+1}"
            tlim = int(row.get("time_limit_seconds", 60) or 60)
            steps.append({"step_name": step_name, "instruction": instr, "time_limit_seconds": max(10, min(600, tlim))})
        st.session_state["_procedure_name"] = "Manual procedure note"
        st.session_state["_procedure_steps"] = steps

    proc_steps = st.session_state.get("_procedure_steps", [])
    if proc_steps:
        st.caption(f"Procedure plan: {st.session_state.get('_procedure_name', 'Session plan')}")
        st.table(
            [("Step", "Instruction", "Time (s)")] +
            [(str(i+1), s.get("instruction", ""), str(s.get("time_limit_seconds", 60))) for i, s in enumerate(proc_steps[:8])]
        )

    # --- Calibration ---
    evidence: EvidenceState = st.session_state[KEY_EVIDENCE]
    cal = evidence.calibration_status()
    with st.expander("Camera Calibration", expanded=not cal["ok"]):
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Brightness", f"{cal['brightness']:.0f}")
        with c2: st.metric("Sharpness", f"{cal['blur_score']:.0f}")
        with c3: st.metric("Detection Rate", f"{cal['detection_rate']:.0%}")
        c4, c5 = st.columns(2)
        with c4: st.metric("Workspace Coverage", f"{cal.get('workspace_coverage', 0):.0%}")
        with c5: st.metric("Centroid Spread", f"{cal.get('centroid_spread', 0):.2f}")
        if cal.get("obstruction_warning"):
            st.error("Possible obstruction detected — check that nothing is blocking the camera.")
        if cal["ok"]:
            st.success("Calibration OK — camera conditions are good.")
            st.session_state[KEY_CALIBRATION_DONE] = True
        else:
            for issue in cal["issues"]:
                st.warning(issue)
            if is_demo:
                st.session_state[KEY_CALIBRATION_DONE] = True

    # --- Tool checklist ---
    st.caption("Ensure all required tools are visible to the camera before starting.")
    smoother = st.session_state[KEY_PREOP_SMOOTHER]
    readiness = smoother.readiness_counts(preop_required) if preop_required else {}
    all_present = smoother.all_present(preop_required)
    now_ts = time.time()

    if all_present:
        if st.session_state[KEY_PREOP_STABLE_START] is None:
            st.session_state[KEY_PREOP_STABLE_START] = now_ts
    else:
        st.session_state[KEY_PREOP_STABLE_START] = None

    stable_start = st.session_state[KEY_PREOP_STABLE_START]
    stable_elapsed = (now_ts - stable_start) if stable_start else 0
    effective_stable = 0.0 if st.session_state.get(KEY_DEMO_MODE, False) else stable_seconds
    checklist_pass = all_present and (stable_elapsed >= effective_stable)

    n_required = len(preop_required)
    n_ok = sum(1 for r in preop_required if readiness.get(r["tool"], (0, False))[1])
    readiness_pct = int(100 * n_ok / n_required) if n_required else 100
    st.progress(readiness_pct / 100.0)
    status_text = (
        "All tools detected — ready!" if checklist_pass
        else "Almost there — hold steady..." if all_present
        else "Some tools still missing"
    )
    st.caption(f"Readiness: {n_ok}/{n_required} tools ({readiness_pct}%) — {status_text}")

    if preop_required:
        header = ("Tool", "Required", "Seen (window)", "Confidence", "Status")
        rows = []
        for r in preop_required:
            tool = r["tool"]
            detected, is_ok = readiness.get(tool, (0, False))
            ts = evidence.tool_state(tool)
            conf_str = f"{ts['avg_conf']:.0%}" if ts["avg_conf"] > 0 else "—"
            rows.append((str(tool), str(r.get("min_count", 1)), str(detected), conf_str, "Ready" if is_ok else "Missing"))
        st.table([header] + rows)

    # --- Begin session ---
    def _begin():
        if get_mode() != "PRE_OP":
            return
        set_mode("INTRA_OP")
        st.session_state[KEY_NAV] = "Practice"
        st.session_state[KEY_SESSION_START] = datetime.now().isoformat()
        for k, v in [(KEY_STREAK_SECONDS, 0.0), (KEY_STREAK_BEST, 0.0),
                      (KEY_PROMPT_COUNTER, 0), (KEY_LAST_DISPLAY_TS, 0.0)]:
            st.session_state[k] = v
        for k in [KEY_ALERTS_LOG, KEY_COACH_PROMPTS, KEY_OVERRIDES, KEY_LAST_PROMPT_TS, KEY_PERF_SAMPLES]:
            st.session_state[k] = type(st.session_state.get(k, []))()
        st.session_state["_tts_queue"] = []
        st.session_state["_tts_busy_until"] = 0.0
        st.session_state["_brain_summary"] = ""
        if st.session_state.get(KEY_DEMO_MODE, False):
            st.session_state[KEY_DEMO_MODE] = False
            st.session_state[KEY_VIDEO_SOURCE] = "Live Webcam"
            st.session_state[KEY_WEBRTC_ENABLED] = False
        log_event("STATE_CHANGE", {"from": "PRE_OP", "to": "INTRA_OP"}, mode="INTRA_OP")
        log_event("CHECKLIST_STATUS", {"status": "PASS", "readiness_pct": readiness_pct}, mode="INTRA_OP")

    if checklist_pass:
        if st.session_state.get(KEY_AUTO_TRANSITION, False):
            _begin()
            st.rerun()
        elif st.button("Begin Lab Session", type="primary", width="stretch"):
            _begin()
            st.rerun()
    else:
        st.button("Begin Lab Session", disabled=True, width="stretch",
                   help="All tools must be detected and held steady.")
