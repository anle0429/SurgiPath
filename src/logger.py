"""Event logging — append to logs/events.jsonl and read back."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "events.jsonl"


def _log_path() -> Path:
    p = Path(LOG_FILE)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _ensure_log_dir() -> None:
    p = _log_path().parent
    p.mkdir(parents=True, exist_ok=True)


def log_event(
    event_type: str,
    payload: dict[str, Any],
    mode: str | None = None,
    phase: str | None = None,
) -> None:
    """Append one event record to the JSONL log file."""
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "type": event_type,
        "mode": mode,
        "phase": phase,
        **payload,
    }
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_events(limit: int | None = None) -> list[dict]:
    """Read events from the log; optionally return only the last N."""
    p = _log_path()
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    lines = [ln for ln in lines if ln]
    if limit is not None:
        lines = lines[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
