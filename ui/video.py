"""Video pipeline: frame display, demo rendering, WebRTC callback, state sync."""

import base64
import queue
import random
import time
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st

from src.constants import (
    KEY_STREAM_START_TS, KEY_PREV_HANDS, KEY_HAND_STABILITY,
    KEY_HELD_TOOLS, KEY_TECHNIQUE, KEY_BIMANUAL_HISTORY,
    KEY_TIP_HISTORY, KEY_WRIST_PATH, KEY_JERK_DATA, KEY_ECONOMY_DATA,
    KEY_LAST_DETECTIONS, KEY_LAST_COUNTS, KEY_EVIDENCE, KEY_PREOP_SMOOTHER,
    KEY_LAST_DISPLAY_TS,
)
from src.detector import draw_detections, infer_tools, count_tools
from src.hands import (
    detect_hands, compute_hand_jitter, draw_hands, get_held_tools,
    compute_technique_summary, compute_jerk_smoothness, compute_motion_economy,
    INDEX_TIP, WRIST,
)
from src.evidence import EvidenceState
from src.utils import ToolPresenceSmoother
from ui.helpers import record_perf_sample


# These are set by app.py after recipe loading so video.py can reference them.
_preop_required: list[dict] = []
_known_tools: list[str] = []
_norm_tool = None
_counts_for_recipe = None
_remap_detections = None


def configure(preop_required, known_tools, norm_tool_fn, counts_for_recipe_fn, remap_fn):
    """Called once from app.py to inject recipe-dependent references."""
    global _preop_required, _known_tools, _norm_tool, _counts_for_recipe, _remap_detections
    _preop_required = preop_required
    _known_tools = known_tools
    _norm_tool = norm_tool_fn
    _counts_for_recipe = counts_for_recipe_fn
    _remap_detections = remap_fn


def _update_hand_and_tool_state(
    hand_results: list[dict],
    detections: list[dict],
    held: set[str],
    evidence: EvidenceState,
    smoother: ToolPresenceSmoother,
    frame_bgr: np.ndarray | None = None,
) -> None:
    """Write hand-tracking and tool-detection results into session_state."""
    prev_hands = st.session_state.get(KEY_PREV_HANDS, [])
    jitter = compute_hand_jitter(hand_results, prev_hands)
    st.session_state[KEY_PREV_HANDS] = hand_results
    st.session_state[KEY_HAND_STABILITY].append(jitter)
    st.session_state[KEY_HELD_TOOLS] = held

    jitter_samples = st.session_state.get(KEY_HAND_STABILITY, [])
    tech = compute_technique_summary(hand_results, jitter_samples, detections)
    st.session_state[KEY_TECHNIQUE] = tech

    bimanual = tech.get("bimanual", {})
    if bimanual.get("detected"):
        bh = st.session_state.get(KEY_BIMANUAL_HISTORY, [])
        bh.append(bimanual["inter_hand_dist"])
        if len(bh) > 60:
            bh = bh[-60:]
        st.session_state[KEY_BIMANUAL_HISTORY] = bh

    if hand_results:
        first = hand_results[0]
        lms = first.get("landmarks", [])
        if len(lms) > INDEX_TIP:
            tip_h = st.session_state.get(KEY_TIP_HISTORY, [])
            tip_h.append((lms[INDEX_TIP][0], lms[INDEX_TIP][1]))
            if len(tip_h) > 50:
                tip_h = tip_h[-50:]
            st.session_state[KEY_TIP_HISTORY] = tip_h
            st.session_state[KEY_JERK_DATA] = compute_jerk_smoothness(tip_h)
        if len(lms) > WRIST:
            wrist_h = st.session_state.get(KEY_WRIST_PATH, [])
            wrist_h.append((lms[WRIST][0], lms[WRIST][1]))
            if len(wrist_h) > 300:
                wrist_h = wrist_h[-300:]
            st.session_state[KEY_WRIST_PATH] = wrist_h
            st.session_state[KEY_ECONOMY_DATA] = compute_motion_economy(wrist_h)

    counts_norm = {_norm_tool(k): v for k, v in count_tools(detections).items()}
    st.session_state[KEY_LAST_DETECTIONS] = detections
    st.session_state[KEY_LAST_COUNTS] = counts_norm
    smoother.update(_counts_for_recipe(counts_norm, _preop_required), _preop_required)
    evidence.update(detections, frame_bgr=frame_bgr, known_tools=_known_tools)


def process_frame(
    frame_bgr: np.ndarray,
    detections: list[dict],
    evidence: EvidenceState,
    smoother: ToolPresenceSmoother,
) -> tuple[np.ndarray, list[dict], set[str]]:
    """Run MediaPipe + state update + draw overlays. Returns (annotated, hands, held)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_results = detect_hands(frame_rgb)
    held = get_held_tools(hand_results, detections)
    _update_hand_and_tool_state(hand_results, detections, held, evidence, smoother, frame_bgr)
    annotated = draw_detections(frame_bgr, detections)
    annotated = draw_hands(annotated, hand_results)
    return annotated, hand_results, held


def display_frame(frame_bgr: np.ndarray) -> None:
    """Embed a BGR frame as a base64 JPEG with clock overlay."""
    now_str = datetime.now().strftime("%H:%M:%S")
    stream_start = st.session_state.get(KEY_STREAM_START_TS)
    if stream_start:
        elapsed = int(time.time() - stream_start)
        mins, secs = divmod(elapsed, 60)
        elapsed_str = f"{mins:02d}:{secs:02d}"
    else:
        elapsed_str = "00:00"

    overlay = frame_bgr.copy()
    cv2.putText(overlay, now_str, (frame_bgr.shape[1] - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (196, 115, 26), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Stream {elapsed_str}", (frame_bgr.shape[1] - 155, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (126, 106, 90), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    st.markdown(
        f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px;">',
        unsafe_allow_html=True,
    )


def generate_demo_detections(tick: int, required_tools: list[dict], mode: str) -> list[dict]:
    all_tools = [r["tool"] for r in required_tools]
    if not all_tools:
        return []
    if mode == "PRE_OP":
        visible = all_tools[:min(len(all_tools), tick + 1)]
    else:
        drop = random.randint(0, min(2, len(all_tools) - 1)) if tick % 5 == 0 else 0
        visible = all_tools if drop == 0 else random.sample(all_tools, max(1, len(all_tools) - drop))

    detections = []
    for i, tool in enumerate(visible):
        row, col = i // 4, i % 4
        x1, y1 = 30 + col * 150, 30 + row * 120
        detections.append({
            "name": tool,
            "conf": round(random.uniform(0.70, 0.98), 2),
            "xyxy": [float(x1), float(y1), float(x1 + 120), float(y1 + 90)],
        })
    return detections


def render_demo_frame(detections: list[dict]) -> np.ndarray:
    frame = np.full((480, 640, 3), (245, 242, 240), dtype=np.uint8)
    cv2.putText(frame, "DEMO MODE", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (196, 115, 26), 2)
    cv2.putText(frame, "Simulated detections - no camera needed", (120, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 110, 100), 1)
    return draw_detections(frame, detections)


def make_webrtc_callback(result_q: queue.Queue, model, cfg: dict):
    """Return a video_frame_callback closure for webrtc_streamer."""
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        capture_ts = time.time()
        prev_capture_ts = getattr(callback, "_prev_capture_ts", 0.0)
        if prev_capture_ts > 0:
            capture_fps = 1.0 / max(capture_ts - prev_capture_ts, 1e-6)
        else:
            capture_fps = 0.0
        callback._prev_capture_ts = capture_ts

        img = frame.to_ndarray(format="bgr24")
        t0 = time.perf_counter()
        detections = _remap_detections(infer_tools(img, conf=cfg["conf_min"], imgsz=cfg["imgsz"], model=model))
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_results = detect_hands(frame_rgb)
        held = get_held_tools(hand_results, detections)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        payload = {
            "detections": detections, "hands": hand_results, "held_tools": held,
            "frame_bgr": img, "capture_ts": capture_ts,
            "infer_ms": infer_ms, "capture_fps": capture_fps,
        }
        try:
            result_q.put_nowait(payload)
        except queue.Full:
            try:
                result_q.get_nowait()
            except queue.Empty:
                pass
            try:
                result_q.put_nowait(payload)
            except queue.Full:
                pass

        annotated = draw_detections(img, detections)
        annotated = draw_hands(annotated, hand_results)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    return callback


def drain_webrtc_queue(result_q: queue.Queue) -> None:
    """Read the latest frame from the WebRTC thread and update session state."""
    latest = None
    while True:
        try:
            latest = result_q.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return

    _update_hand_and_tool_state(
        hand_results=latest["hands"],
        detections=latest["detections"],
        held=latest["held_tools"],
        evidence=st.session_state[KEY_EVIDENCE],
        smoother=st.session_state[KEY_PREOP_SMOOTHER],
        frame_bgr=latest.get("frame_bgr"),
    )

    now_ts = time.time()
    prev_display_ts = float(st.session_state.get(KEY_LAST_DISPLAY_TS, 0.0))
    display_fps = (1.0 / max(now_ts - prev_display_ts, 1e-6)) if prev_display_ts > 0 else 0.0
    st.session_state[KEY_LAST_DISPLAY_TS] = now_ts

    capture_ts = float(latest.get("capture_ts", 0.0))
    e2e_ms = (now_ts - capture_ts) * 1000.0 if capture_ts > 0 else 0.0
    record_perf_sample({
        "capture_fps": float(latest.get("capture_fps", 0.0)),
        "infer_ms": float(latest.get("infer_ms", 0.0)),
        "display_fps": display_fps,
        "e2e_ms": e2e_ms,
        "ts": now_ts,
    })
