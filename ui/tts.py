"""Text-to-speech helpers: edge-tts -> gTTS -> pyttsx3 cascade."""

import base64
import io
import time

import streamlit as st


def _autoplay_audio_b64(audio_bytes: bytes, mime: str = "audio/mp3") -> None:
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    st.markdown(
        f'<audio autoplay><source src="data:{mime};base64,{b64}" type="{mime}"></audio>',
        unsafe_allow_html=True,
    )


def speak_prompt(text: str) -> bool:
    """Play TTS for a coaching prompt. Returns True if audio was produced."""
    if not text or not text.strip():
        return False

    # 1) edge-tts — natural Microsoft Edge voices, free, no API key
    try:
        import asyncio
        import edge_tts

        async def _generate():
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                audio_data = pool.submit(lambda: asyncio.run(_generate())).result(timeout=15)
        else:
            audio_data = asyncio.run(_generate())

        if audio_data and len(audio_data) > 100:
            _autoplay_audio_b64(audio_data, "audio/mp3")
            return True
    except Exception:
        pass

    # 2) gTTS — Google TTS, needs internet
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        audio_buf = io.BytesIO()
        tts.write_to_fp(audio_buf)
        audio_bytes = audio_buf.getvalue()
        if audio_bytes and len(audio_bytes) > 100:
            _autoplay_audio_b64(audio_bytes, "audio/mp3")
            return True
    except Exception:
        pass

    # 3) pyttsx3 — offline, server-side
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False


def _estimate_tts_seconds(text: str) -> float:
    words = max(1, len((text or "").split()))
    return max(2.5, words / 2.2 + 0.7)


def queue_tts(text: str) -> None:
    """Add text to TTS queue (avoids interrupting prior utterances)."""
    msg = (text or "").strip()
    if not msg:
        return
    q = st.session_state.setdefault("_tts_queue", [])
    if not q or q[-1] != msg:
        q.append(msg)
        if len(q) > 6:
            del q[:-6]


def flush_tts_queue() -> None:
    """Play one queued utterance if the previous one should be done."""
    busy_until = float(st.session_state.get("_tts_busy_until", 0.0))
    now = time.time()
    if now < busy_until:
        return
    q = st.session_state.setdefault("_tts_queue", [])
    if not q:
        return
    msg = q.pop(0)
    ok = speak_prompt(msg)
    if ok:
        st.session_state["_tts_busy_until"] = now + _estimate_tts_seconds(msg)
