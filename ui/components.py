"""Reusable HTML components rendered via st.markdown."""

import streamlit as st


def section_header(num: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-num">{num}</span>'
        f'<span class="section-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def technique_monitor(
    status_color: str, status_icon: str, status_label: str,
    grip_msg: str,
    smooth_color: str, smooth_icon: str, smooth_label: str,
    jerk_color: str, jerk_pct: str, jerk_label: str,
    econ_color: str, econ_pct: str, econ_label: str,
    angle_color: str, angle_deg: str, angle_label: str,
    hands_color: str, hands_count: int,
    tip_text: str, ideal_str: str,
) -> None:
    sc = status_color
    bm_label = "Bimanual" if hands_count >= 2 else "1 Hand"
    st.markdown(f"""
<div style="background:var(--bg-card);border:1px solid {sc}44;border-left:3px solid {sc};
            border-radius:8px;padding:1rem 1.25rem 0.9rem;margin-bottom:1rem;
            box-shadow:0 0 16px rgba(0,0,0,0.4);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:var(--text-3);
                 text-transform:uppercase;letter-spacing:0.12em;">Live Technique Monitor</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.60rem;font-weight:700;
                 color:{sc};background:{sc}18;border:1px solid {sc}44;border-radius:3px;
                 padding:0.15rem 0.5rem;letter-spacing:0.08em;">{status_label}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.6rem;margin-bottom:0.8rem;">
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {sc}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{sc};line-height:1;">{status_icon}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{sc};margin-top:0.2rem;">{grip_msg}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Grip (D)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {smooth_color}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{smooth_color};line-height:1;">{smooth_icon}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{smooth_color};margin-top:0.2rem;">{smooth_label}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Stability (A)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {jerk_color}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{jerk_color};line-height:1;">{jerk_pct}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{jerk_color};margin-top:0.2rem;">{jerk_label}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Smoothness (B)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {econ_color}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{econ_color};line-height:1;">{econ_pct}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{econ_color};margin-top:0.2rem;">{econ_label}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Economy (C)</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {angle_color}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{angle_color};line-height:1;">{angle_deg}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{angle_color};margin-top:0.2rem;">{angle_label}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Angle</div>
    </div>
    <div style="background:var(--bg-elevated);border-radius:6px;padding:0.65rem 0.5rem;text-align:center;border:1px solid {hands_color}30;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{hands_color};line-height:1;">{hands_count}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.70rem;font-weight:600;color:{hands_color};margin-top:0.2rem;">{bm_label}</div>
      <div style="font-size:0.57rem;color:var(--text-3);margin-top:0.25rem;text-transform:uppercase;letter-spacing:0.08em;">Hands</div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
              border-top:1px solid var(--border);padding-top:0.6rem;">
    <div style="font-size:0.80rem;color:var(--text-2);font-family:'IBM Plex Sans',sans-serif;">
      <span style="color:{sc};font-weight:700;">{status_icon}</span>&nbsp; {tip_text}
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.60rem;color:var(--text-3);
                white-space:nowrap;margin-left:1rem;">Ideal: {ideal_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def technique_monitor_empty() -> None:
    st.markdown("""
<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
            padding:0.85rem 1.25rem;margin-bottom:1rem;display:flex;align-items:center;gap:0.75rem;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:var(--text-3);
               text-transform:uppercase;letter-spacing:0.12em;">Live Technique Monitor</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.76rem;color:var(--text-3);">
    — No hands detected. Position hands in frame.</span>
</div>
""", unsafe_allow_html=True)
