"""App state helpers — mode (PRE_OP/INTRA_OP/POST_OP) and phase."""
from typing import Literal

AppMode = Literal["PRE_OP", "INTRA_OP", "POST_OP"]

MODE_KEY = "mode"
PHASE_KEY = "phase"
INITIAL_MODE: AppMode = "PRE_OP"
DEFAULT_PHASE = "incision"


def init_state() -> None:
    """Set default mode and phase if not already in session_state."""
    try:
        import streamlit as st
        if MODE_KEY not in st.session_state:
            st.session_state[MODE_KEY] = INITIAL_MODE
        if PHASE_KEY not in st.session_state:
            st.session_state[PHASE_KEY] = DEFAULT_PHASE
    except Exception:
        pass


def set_mode(mode: AppMode) -> None:
    try:
        import streamlit as st
        st.session_state[MODE_KEY] = mode
    except Exception:
        pass


def get_mode() -> AppMode:
    try:
        import streamlit as st
        return st.session_state.get(MODE_KEY, INITIAL_MODE)
    except Exception:
        return INITIAL_MODE


def set_phase(phase: str) -> None:
    try:
        import streamlit as st
        st.session_state[PHASE_KEY] = phase
    except Exception:
        pass


def get_phase() -> str:
    try:
        import streamlit as st
        return st.session_state.get(PHASE_KEY, DEFAULT_PHASE)
    except Exception:
        return DEFAULT_PHASE
