"""Pure helper functions used across multiple UI tabs."""

from datetime import datetime

import streamlit as st

from src.constants import (
    PHASES, KEY_PROMPT_COUNTER, KEY_PERF_SAMPLES, PERF_WINDOW_SIZE,
)
from src.state import get_phase


def next_prompt_id() -> str:
    c = st.session_state.get(KEY_PROMPT_COUNTER, 0) + 1
    st.session_state[KEY_PROMPT_COUNTER] = c
    return f"P{c:04d}"


def severity_rank(tier: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get((tier or "").lower(), 3)


def format_coach_message(message: str, tier: str) -> str:
    msg = (message or "").strip()
    if not msg:
        msg = "Please re-check your current step and tool setup."
    if msg[-1:] not in ".!?":
        msg += "."
    t = (tier or "").lower()
    if t == "high":
        return f"Action needed now: {msg}"
    if t == "medium":
        return f"Please review this step: {msg}"
    return f"Coaching tip: {msg}"


def prompt_sort_key(prompt: dict) -> tuple[int, float]:
    ts_iso = prompt.get("ts", "")
    try:
        ts_epoch = datetime.fromisoformat(ts_iso).timestamp()
    except Exception:
        ts_epoch = 0.0
    return (severity_rank(prompt.get("risk_tier", "high")), -ts_epoch)


def record_perf_sample(sample: dict) -> None:
    samples = st.session_state.setdefault(KEY_PERF_SAMPLES, [])
    samples.append(sample)
    if len(samples) > PERF_WINDOW_SIZE:
        del samples[:-PERF_WINDOW_SIZE]


def compute_perf_summary() -> dict:
    samples = st.session_state.get(KEY_PERF_SAMPLES, [])
    if not samples:
        return {}

    capture_fps_vals = [s["capture_fps"] for s in samples if s.get("capture_fps", 0) > 0]
    infer_fps_vals = [1000.0 / s["infer_ms"] for s in samples if s.get("infer_ms", 0) > 0]
    display_fps_vals = [s["display_fps"] for s in samples if s.get("display_fps", 0) > 0]
    e2e_vals = [s["e2e_ms"] for s in samples if s.get("e2e_ms", 0) > 0]

    def _avg(v):
        return sum(v) / len(v) if v else 0.0

    def _pct(v, q):
        if not v:
            return 0.0
        o = sorted(v)
        return o[int(max(0, min(len(o) - 1, round(q * (len(o) - 1)))))]

    return {
        "capture_fps": _avg(capture_fps_vals),
        "infer_fps": _avg(infer_fps_vals),
        "display_fps": _avg(display_fps_vals),
        "e2e_p50_ms": _pct(e2e_vals, 0.5),
        "e2e_p95_ms": _pct(e2e_vals, 0.95),
        "samples": len(samples),
    }


def build_brain_event_log(prompts: list[dict], overrides: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in prompts:
        out.append({
            "time": p.get("error_time", ""),
            "type": "coach_prompt",
            "tool": p.get("rule_id", ""),
            "detail": p.get("message", ""),
        })
    for o in overrides:
        ts = o.get("ts", "")
        out.append({
            "time": ts[11:19] if isinstance(ts, str) and len(ts) >= 19 else "",
            "type": "override",
            "tool": o.get("prompt_id", ""),
            "detail": o.get("decision", ""),
        })
    return out


def normalize_procedure_steps(res: object) -> list[dict]:
    raw_steps = []
    if hasattr(res, "steps"):
        raw_steps = getattr(res, "steps") or []
    elif isinstance(res, dict):
        raw_steps = res.get("steps", []) or []

    steps: list[dict] = []
    for i, item in enumerate(raw_steps):
        if hasattr(item, "model_dump"):
            d = item.model_dump()
        elif hasattr(item, "dict"):
            d = item.dict()
        elif isinstance(item, dict):
            d = item
        else:
            continue
        instr = str(d.get("instruction", "")).strip()
        if not instr:
            continue
        step_name = str(d.get("step_name", "")).strip() or f"Step {i + 1}"
        tlim = int(d.get("time_limit_seconds", 60) or 60)
        steps.append({
            "step_name": step_name,
            "instruction": instr,
            "time_limit_seconds": max(10, min(600, tlim)),
        })
    return steps


def phase_from_step_text(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("incis", "cut", "scalpel")):
        return "incision"
    if any(k in t for k in ("sutur", "needle", "close", "stitch")):
        return "suturing"
    if any(k in t for k in ("irrig", "flush", "wash", "suction")):
        return "irrigation"
    return "closing"


def procedure_phase_from_elapsed(steps: list[dict], elapsed_s: float) -> tuple[int, str]:
    if not steps:
        return 0, get_phase() if get_phase() in PHASES else PHASES[0]
    cum = 0.0
    idx = len(steps) - 1
    for i, s in enumerate(steps):
        step_t = max(15, int(s.get("time_limit_seconds", 60)))
        cum += step_t
        if elapsed_s <= cum:
            idx = i
            break
    step_text = f"{steps[idx].get('step_name', '')} {steps[idx].get('instruction', '')}"
    return idx, phase_from_step_text(step_text)
