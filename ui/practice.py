"""Practice tab: real-time coaching, technique monitor, coach prompt cards."""

import time
from datetime import datetime

import streamlit as st

from src.constants import (
    PHASES, PHASE_LABELS, FEED_LOOP_FRAMES,
    COACH_PROMPT_COOLDOWN_SEC,
    KEY_SESSION_START, KEY_EVIDENCE, KEY_HELD_TOOLS,
    KEY_LAST_COUNTS, KEY_STREAK_SECONDS, KEY_STREAK_BEST,
    KEY_COACH_PROMPTS, KEY_ALERTS_LOG, KEY_OVERRIDES,
    KEY_LAST_PROMPT_TS, KEY_TECHNIQUE, KEY_DEMO_MODE, KEY_DEMO_TICK,
    KEY_JERK_DATA, KEY_ECONOMY_DATA, KEY_BIMANUAL_HISTORY,
    KEY_NAV, KEY_STREAM_START_TS,
    KEY_VIDEO_SOURCE, KEY_WEBRTC_ENABLED, KEY_WEBRTC_ACTIVE,
)
from src.state import get_phase, set_phase, set_mode
from src.logger import log_event
from src.rules import evaluate_rules
from src.evidence import EvidenceState
from src.hands import (
    get_technique_feedback, IDEAL_GRIPS_BY_PHASE, GRIP_DISPLAY_NAMES,
)
from ui.components import section_header, technique_monitor, technique_monitor_empty
from ui.helpers import (
    next_prompt_id, format_coach_message, prompt_sort_key,
    compute_perf_summary, procedure_phase_from_elapsed,
)
from ui.tts import queue_tts, flush_tts_queue


def render_practice_tab(
    intraop_rules: list[dict],
    use_webrtc: bool,
    get_config,
) -> None:
    section_header("02 / PRACTICE", "Practice Session")

    # --- Phase selection ---
    proc_steps = st.session_state.get("_procedure_steps", [])
    phase = get_phase() if get_phase() in PHASES else PHASES[0]

    if proc_steps:
        sess_start = st.session_state.get(KEY_SESSION_START)
        try:
            elapsed_s = max(0.0, (datetime.now() - datetime.fromisoformat(sess_start)).total_seconds()) if sess_start else 0.0
        except Exception:
            elapsed_s = 0.0
        step_idx, fixed_phase = procedure_phase_from_elapsed(proc_steps, elapsed_s)
        phase = fixed_phase
        prev_phase = get_phase()
        set_phase(phase)
        if phase != prev_phase:
            log_event("PHASE_CHANGE", {"phase": phase}, mode="INTRA_OP", phase=phase)
        st.caption(
            f"Operation (fixed by procedure): Step {step_idx + 1}/{len(proc_steps)} · "
            f"{proc_steps[step_idx].get('step_name', 'Procedure step')} · "
            f"{PHASE_LABELS.get(phase, phase)}"
        )
    else:
        phase = st.selectbox(
            "Current Exercise", PHASES,
            index=PHASES.index(get_phase()) if get_phase() in PHASES else 0,
            format_func=lambda p: PHASE_LABELS.get(p, p),
            key="phase_select",
        )
        prev_phase = get_phase()
        set_phase(phase)
        if phase != prev_phase:
            log_event("PHASE_CHANGE", {"phase": phase}, mode="INTRA_OP", phase=phase)

    # --- WebRTC performance metrics ---
    perf = compute_perf_summary()
    show_perf = use_webrtc or st.session_state.get(KEY_WEBRTC_ACTIVE, False)
    if perf and show_perf:
        pf1, pf2, pf3, pf4, pf5 = st.columns(5)
        with pf1: st.metric("Capture FPS", f"{perf['capture_fps']:.1f}")
        with pf2: st.metric("Inference FPS", f"{perf['infer_fps']:.1f}")
        with pf3: st.metric("Display FPS", f"{perf['display_fps']:.1f}")
        with pf4: st.metric("E2E p50", f"{perf['e2e_p50_ms']:.0f} ms")
        with pf5: st.metric("E2E p95", f"{perf['e2e_p95_ms']:.0f} ms")
        st.caption(f"WebRTC telemetry: {perf['samples']} frames (drop-oldest queue active).")
    elif show_perf:
        st.caption("Collecting WebRTC telemetry...")

    if (
        st.session_state.get(KEY_VIDEO_SOURCE) == "Live Webcam"
        and st.session_state.get(KEY_WEBRTC_ENABLED, True)
        and not st.session_state.get(KEY_WEBRTC_ACTIVE, False)
    ):
        st.info("WebRTC is enabled. Click START in the video widget to begin camera stream.")
        if st.button("Use camera fallback (no START needed)", key="webrtc_fallback_btn"):
            st.session_state[KEY_WEBRTC_ENABLED] = False
            st.rerun()

    # --- Rule evaluation ---
    counts = st.session_state.get(KEY_LAST_COUNTS, {})
    evidence_obj: EvidenceState = st.session_state[KEY_EVIDENCE]
    held_tools = st.session_state.get(KEY_HELD_TOOLS, set())
    dt_seconds = (FEED_LOOP_FRAMES * 0.033) / max(1, get_config()["frame_skip"])
    tool_signal = sum(int(v) for v in counts.values()) > 0

    if tool_signal:
        alerts = evaluate_rules(phase, counts, intraop_rules, dt_seconds, st.session_state, evidence=evidence_obj, held_tools=held_tools)
    else:
        alerts = []
        st.caption("Tool signal is low — coaching is currently based on hand technique only.")

    # --- Technique-based alerts ---
    _run_technique_alerts(alerts, phase, dt_seconds)

    # --- Streak ---
    if alerts:
        st.session_state[KEY_STREAK_SECONDS] = 0.0
    else:
        st.session_state[KEY_STREAK_SECONDS] = st.session_state.get(KEY_STREAK_SECONDS, 0) + dt_seconds
    current_streak = st.session_state[KEY_STREAK_SECONDS]
    if current_streak > st.session_state.get(KEY_STREAK_BEST, 0):
        st.session_state[KEY_STREAK_BEST] = current_streak

    # --- Register new alerts as Coach Prompts + TTS ---
    new_prompt_messages = _register_alerts(alerts, phase)

    col_s1, col_s2 = st.columns(2)
    with col_s1: st.metric("Current Streak", f"{int(current_streak)}s", help="Consecutive seconds without coaching alerts")
    with col_s2: st.metric("Best Streak", f"{int(st.session_state.get(KEY_STREAK_BEST, 0))}s")

    # --- Technique Monitor ---
    _render_technique_section(phase)

    # --- TTS ---
    for msg in new_prompt_messages:
        queue_tts(msg)
    flush_tts_queue()

    # --- Coach Prompt Cards ---
    _render_coach_cards(phase)

    # --- End session ---
    if st.button("End Lab Session", type="primary", width="stretch"):
        set_mode("POST_OP")
        st.session_state[KEY_NAV] = "Report"
        st.session_state[KEY_STREAM_START_TS] = None
        log_event("STATE_CHANGE", {"from": "INTRA_OP", "to": "POST_OP"}, mode="POST_OP")
        st.rerun()


# ---------------------------------------------------------------------------
# Internal helpers (keep Practice tab render function readable)
# ---------------------------------------------------------------------------

_TECH_RULES = [
    {"id": "tech_tremor", "phases": ("suturing", "closing"),
     "check": lambda t, _p: t.get("smoothness") == "tremor", "hold": 3.0,
     "message": "Hand tremor detected — try resting your wrists on the table for stability."},
    {"id": "tech_palmar_scalpel", "phases": ("incision",),
     "check": lambda t, _p: any(g.get("grip_type") == "palmar_grip" and "scalpel" in g.get("near_tools", []) for g in t.get("grips", [])),
     "hold": 2.5, "message": "Palmar grip on scalpel — consider a pencil grip for finer incision control."},
    {"id": "tech_open_needle", "phases": ("suturing",),
     "check": lambda t, _p: any(g.get("grip_type") == "open_hand" and "needle_holder" in g.get("near_tools", []) for g in t.get("grips", [])),
     "hold": 2.5, "message": "Open hand near needle holder — use a palmar or precision grip for secure control."},
    {"id": "tech_steep_angle", "phases": ("incision",),
     "check": lambda t, _p: any(g.get("instrument_angle", 0) > 70 and "scalpel" in g.get("near_tools", []) for g in t.get("grips", [])),
     "hold": 3.0, "message": "Steep scalpel angle (>70°) — aim for 30-45° for a controlled incision."},
]


def _run_technique_alerts(alerts: list[dict], phase: str, dt_seconds: float) -> None:
    tech_summary = st.session_state.get(KEY_TECHNIQUE, {})
    tech_timers = st.session_state.setdefault("_tech_timers", {})
    tech_triggered = st.session_state.setdefault("_tech_triggered", set())

    for tr in _TECH_RULES:
        if phase not in tr["phases"]:
            tech_timers[tr["id"]] = 0
            tech_triggered.discard(tr["id"])
            continue
        if tr["check"](tech_summary, phase):
            tech_timers[tr["id"]] = tech_timers.get(tr["id"], 0) + dt_seconds
            if tr["id"] not in tech_triggered and tech_timers[tr["id"]] >= tr["hold"]:
                alerts.append({
                    "rule_id": tr["id"], "message": tr["message"], "phase": phase,
                    "risk_tier": "medium", "avg_conf": 0.0, "seen_ratio": 0.0,
                    "last_seen_ts": time.time(), "evidence_tools": {},
                })
                tech_triggered.add(tr["id"])
        else:
            tech_timers[tr["id"]] = 0
            tech_triggered.discard(tr["id"])


def _register_alerts(alerts: list[dict], phase: str) -> list[str]:
    new_msgs: list[str] = []
    last_prompt_ts = st.session_state.setdefault(KEY_LAST_PROMPT_TS, {})
    now_epoch = time.time()
    for a in alerts:
        rule_id = a.get("rule_id", "") or "unknown_rule"
        if (now_epoch - float(last_prompt_ts.get(rule_id, 0.0))) < COACH_PROMPT_COOLDOWN_SEC:
            continue
        last_prompt_ts[rule_id] = now_epoch
        tier = a.get("risk_tier", "high")
        coach_message = format_coach_message(a.get("message", ""), tier)
        pid = next_prompt_id()
        prompt = {
            "prompt_id": pid, "ts": datetime.now().isoformat(),
            "error_time": datetime.now().strftime("%H:%M:%S"),
            "phase": a.get("phase", phase), "rule_id": rule_id,
            "message": coach_message, "risk_tier": tier,
            "avg_conf": a.get("avg_conf", 1.0), "seen_ratio": a.get("seen_ratio", 1.0),
            "last_seen_ts": a.get("last_seen_ts", 0),
        }
        st.session_state[KEY_COACH_PROMPTS].append(prompt)
        st.session_state[KEY_ALERTS_LOG].append(prompt)
        log_event("COACH_PROMPT", {**prompt}, mode="INTRA_OP", phase=phase)
        if len(new_msgs) < 2:
            new_msgs.append(coach_message)
    return new_msgs


def _render_technique_section(phase: str) -> None:
    tech = st.session_state.get(KEY_TECHNIQUE, {})
    grips = tech.get("grips", [])
    smoothness = tech.get("smoothness", "steady")
    bimanual = tech.get("bimanual", {})

    display_grips = grips
    display_smooth = smoothness
    if st.session_state.get(KEY_DEMO_MODE, False) and not grips:
        demo_tick = st.session_state.get(KEY_DEMO_TICK, 0)
        display_grips = [{"grip_type": ["pencil_grip", "pencil_grip", "pencil_grip", "palmar_grip", "precision_grip"][demo_tick % 5],
                          "instrument_angle": 32 + (demo_tick % 7) * 3, "handedness": "Right", "near_tools": []}]
        display_smooth = ["steady", "steady", "moderate"][demo_tick % 3]
        st.session_state[KEY_JERK_DATA] = {"smoothness_score": [0.85, 0.80, 0.55, 0.82, 0.30][demo_tick % 5],
                                           "mean_jerk": [1.2, 1.5, 4.0, 1.3, 8.5][demo_tick % 5],
                                           "label": ["fluid", "fluid", "moderate", "fluid", "jerky"][demo_tick % 5]}
        st.session_state[KEY_ECONOMY_DATA] = {"path_length_px": 850 + demo_tick * 12,
                                              "directness": [0.78, 0.72, 0.65, 0.81][demo_tick % 4],
                                              "idle_ratio": [0.20, 0.25, 0.38, 0.18][demo_tick % 4],
                                              "label": ["efficient", "efficient", "moderate", "efficient"][demo_tick % 4]}

    if not display_grips:
        technique_monitor_empty()
        return

    g = display_grips[0]
    fb = get_technique_feedback(g["grip_type"], phase, display_smooth, g["instrument_angle"])

    _color = {"good": "#00E5A0", "warning": "#FFB703", "error": "#FF3D3D"}
    _icon = {"good": "\u2713", "warning": "\u26A0", "error": "\u2717"}
    _label = {"good": "GOOD", "warning": "REVIEW", "error": "CORRECT"}
    sc = _color.get(fb["status"], "#6A8CA8")
    si = _icon.get(fb["status"], "\u2014")
    sl = _label.get(fb["status"], "\u2014")

    smc = {"steady": "#00E5A0", "moderate": "#FFB703", "tremor": "#FF3D3D"}.get(display_smooth, "#6A8CA8")
    sm_icon = {"steady": "\u2713", "moderate": "\u26A0", "tremor": "\u2717"}.get(display_smooth, "\u2014")

    ang = fb["angle"]
    if ang > 70: anc, an_label = "#FF3D3D", "Steep"
    elif 25 <= ang <= 60: anc, an_label = "#00E5A0", "Ideal"
    else: anc, an_label = "#FFB703", "Low"

    ideal_names = [GRIP_DISPLAY_NAMES.get(g2, g2) for g2 in IDEAL_GRIPS_BY_PHASE.get(phase, [])]
    hands_count = bimanual.get("hands_count", len(display_grips))

    jerk_data = st.session_state.get(KEY_JERK_DATA, {})
    jk_label = jerk_data.get("label", "fluid")
    jk_score = jerk_data.get("smoothness_score", 1.0)
    jkc = {"fluid": "#00E5A0", "moderate": "#FFB703", "jerky": "#FF3D3D"}.get(jk_label, "#6A8CA8")

    econ_data = st.session_state.get(KEY_ECONOMY_DATA, {})
    ec_label = econ_data.get("label", "efficient")
    ec_direct = econ_data.get("directness", 1.0)
    ecc = {"efficient": "#00E5A0", "moderate": "#FFB703", "excessive": "#FF3D3D"}.get(ec_label, "#6A8CA8")

    technique_monitor(
        status_color=sc, status_icon=si, status_label=sl, grip_msg=fb["message"],
        smooth_color=smc, smooth_icon=sm_icon, smooth_label=display_smooth.title(),
        jerk_color=jkc, jerk_pct=f"{jk_score * 100:.0f}%", jerk_label=jk_label.title(),
        econ_color=ecc, econ_pct=f"{ec_direct * 100:.0f}%", econ_label=ec_label.title(),
        angle_color=anc, angle_deg=f"{ang:.0f}\u00b0", angle_label=an_label,
        hands_color="#00E5A0" if hands_count >= 2 else "#5E7D9A", hands_count=hands_count,
        tip_text=fb["tip"], ideal_str=" / ".join(ideal_names) if ideal_names else "Any",
    )

    # Technique details expander
    with st.expander("Technique Details", expanded=False):
        smooth_icons = {"steady": "Steady", "moderate": "Moderate", "tremor": "Tremor"}
        smooth_colors = {"steady": "\U0001f7e2", "moderate": "\U0001f7e1", "tremor": "\U0001f534"}
        tc1, tc2, tc3 = st.columns(3)
        with tc1: st.metric("Smoothness", f"{smooth_colors.get(smoothness, '\u26aa')} {smooth_icons.get(smoothness, smoothness)}")
        with tc2: st.metric("Hands Detected", bimanual.get("hands_count", 0))
        with tc3:
            if bimanual.get("detected"): st.metric("Inter-hand Dist", f"{bimanual['inter_hand_dist']:.0f} px")
            else: st.metric("Inter-hand Dist", "\u2014")

        if grips:
            grip_labels = {"pencil_grip": "Pencil Grip (pen-hold)", "palmar_grip": "Palmar Grip (power)",
                           "precision_grip": "Precision Grip (3-finger)", "open_hand": "Open Hand"}
            for g in grips:
                near = g.get("near_tools", [])
                st.caption(f"**{g.get('handedness', '?')}**: {grip_labels.get(g.get('grip_type', 'open_hand'), g.get('grip_type', ''))} · "
                           f"Angle: {g.get('instrument_angle', 0):.0f}\u00b0 · Near: {', '.join(near) if near else 'none'}")
        else:
            st.caption("No hands detected — technique analysis requires hand visibility.")

        bh = st.session_state.get(KEY_BIMANUAL_HISTORY, [])
        if len(bh) >= 5:
            avg_d = sum(bh[-20:]) / len(bh[-20:])
            std_d = (sum((x - avg_d) ** 2 for x in bh[-20:]) / len(bh[-20:])) ** 0.5
            sync = "Stable" if std_d < 25 else "Variable" if std_d < 50 else "Unstable"
            st.caption(f"Bimanual coordination: {sync} (avg {avg_d:.0f} px, \u03c3 {std_d:.0f} px)")


def _render_coach_cards(phase: str) -> None:
    prompts = st.session_state.get(KEY_COACH_PROMPTS, [])
    resolved_ids = {o["prompt_id"] for o in st.session_state.get(KEY_OVERRIDES, [])}
    active = sorted(
        [p for p in prompts if p["prompt_id"] not in resolved_ids],
        key=prompt_sort_key,
    )[:8]

    high_n = sum(1 for p in active if p.get("risk_tier") == "high")
    med_n = sum(1 for p in active if p.get("risk_tier") == "medium")
    low_n = sum(1 for p in active if p.get("risk_tier") == "low")

    st.markdown("**Coach Prompts** (priority view)")
    st.caption(f"Open notes: \U0001f534 High {high_n} \u00b7 \U0001f7e1 Medium {med_n} \u00b7 \U0001f7e2 Low {low_n} "
               f"(duplicate prompts are auto-suppressed for {int(COACH_PROMPT_COOLDOWN_SEC)}s)")

    if not active:
        st.success("Great work! No coaching notes so far.")
        return

    tier_icons = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f7e2"}
    for p in active:
        tier = p.get("risk_tier", "high")
        icon = tier_icons.get(tier, "\u26aa")
        error_time = p.get("error_time", "")
        if not error_time:
            try: error_time = datetime.fromisoformat(p.get("ts", "")).strftime("%H:%M:%S")
            except Exception: error_time = "\u2014"
        phase_label = PHASE_LABELS.get(p.get("phase", ""), p.get("phase", ""))

        with st.container(border=True):
            st.markdown(
                f"**{icon} {p.get('message', '')}**  \n"
                f"<small>Accuracy: {p.get('avg_conf', 0):.0%} ({tier}) \u00b7 "
                f"Time: {error_time} \u00b7 Phase: {phase_label}</small>",
                unsafe_allow_html=True,
            )
            if tier in ("low", "medium"):
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("Confirm issue", key=f"confirm_{p['prompt_id']}"):
                        st.session_state[KEY_OVERRIDES].append({"prompt_id": p["prompt_id"], "decision": "confirm", "ts": datetime.now().isoformat()})
                        log_event("USER_OVERRIDE", {"prompt_id": p["prompt_id"], "decision": "confirm"}, mode="INTRA_OP", phase=phase)
                        st.rerun()
                with bc2:
                    if st.button("Mark as resolved", key=f"deny_{p['prompt_id']}"):
                        st.session_state[KEY_OVERRIDES].append({"prompt_id": p["prompt_id"], "decision": "deny", "ts": datetime.now().isoformat()})
                        log_event("USER_OVERRIDE", {"prompt_id": p["prompt_id"], "decision": "deny"}, mode="INTRA_OP", phase=phase)
                        st.rerun()
