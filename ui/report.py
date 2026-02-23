"""Report tab: post-session mastery score, assessment, error details."""

from datetime import datetime

import streamlit as st

from src.constants import KEY_ALERTS_LOG, KEY_OVERRIDES, PHASE_LABELS
from ui.components import section_header


def render_report_tab() -> None:
    section_header("03 / REPORT", "Post-Op Session Report")

    prompts = st.session_state.get(KEY_ALERTS_LOG, [])
    overrides = st.session_state.get(KEY_OVERRIDES, [])
    denied_overrides = [o for o in overrides if (o.get("decision") or "").lower() == "deny"]

    total_steps = len(st.session_state.get("_procedure_steps", []))
    if total_steps <= 0:
        total_steps = max(1, len(prompts))
    manual_overrides = len(denied_overrides)
    vision_verified = max(0, total_steps - manual_overrides)
    mastery_score = max(0, 100 - (manual_overrides * 20))

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Mastery Score", f"{mastery_score}/100")
    with c2: st.metric("Vision Verified", vision_verified)
    with c3: st.metric("Manual Overrides", manual_overrides)

    if manual_overrides > 0:
        st.warning("Session Status: Clinically Incomplete. Some critical prompts were manually resolved without vision confirmation.")
    elif prompts:
        st.info("Session Status: Coaching interventions were detected. Review assessment details below.")
    else:
        st.success("Session Status: No major coaching issues were recorded.")

    if prompts:
        st.markdown("**Assessment:**")
        st.markdown("- Focus on reducing repeated high-risk prompts in the same phase.")
        st.markdown("- Maintain stable hand posture before tool transitions.")
    else:
        st.markdown("**Assessment:** Stable session with no detected critical errors.")

    # Error details table
    rows = []
    for p in prompts:
        error_time = p.get("error_time", "")
        if not error_time:
            ts_str = p.get("ts", "")
            try:
                error_time = datetime.fromisoformat(ts_str).strftime("%H:%M:%S")
            except Exception:
                error_time = "—"
        rows.append({
            "time": error_time,
            "phase": PHASE_LABELS.get(p.get("phase", ""), p.get("phase", "")),
            "severity": (p.get("risk_tier", "high") or "high").upper(),
            "error": p.get("message", ""),
        })

    st.markdown("---")
    st.markdown("### Error Details")
    if not rows:
        st.caption("No coaching errors recorded in this session.")
    else:
        sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        rows.sort(key=lambda r: (sev_rank.get(r["severity"], 3), r["time"]))
        st.table(
            [("Time", "Phase", "Severity", "Error")] +
            [(r["time"], r["phase"], r["severity"], r["error"]) for r in rows]
        )
