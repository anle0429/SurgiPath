"""Utilities: load recipe JSON and smooth tool presence over a sliding window."""
import json
from pathlib import Path
from collections import deque
from typing import Any


def load_recipe(path: str | Path = "recipes/trauma_room.json") -> dict[str, Any]:
    """Load and return recipe JSON from path."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class ToolPresenceSmoother:
    """Sliding window smoother to reduce detection flicker."""

    def __init__(self, window_size: int = 20):
        self.window_size = max(1, window_size)
        self._history: dict[str, deque[bool]] = {}

    def update(self, counts: dict[str, int], required_tools: list[dict]) -> None:
        """Record one detection sample."""
        for req in required_tools:
            tool = req.get("tool", "")
            min_count = req.get("min_count", 1)
            if not tool:
                continue
            present = counts.get(tool, 0) >= min_count
            if tool not in self._history:
                self._history[tool] = deque(maxlen=self.window_size)
            self._history[tool].append(present)

    def is_present(self, tool: str) -> bool:
        if tool not in self._history or len(self._history[tool]) == 0:
            return False
        return any(self._history[tool])

    def all_present(self, required_tools: list[dict]) -> bool:
        for req in required_tools:
            tool = req.get("tool", "")
            if tool and not self.is_present(tool):
                return False
        return True

    def readiness_counts(
        self, required_tools: list[dict]
    ) -> dict[str, tuple[int, bool]]:
        """Return per-tool detection counts for display."""
        out = {}
        for req in required_tools:
            tool = req.get("tool", "")
            if not tool:
                continue
            q = self._history.get(tool, deque())
            detected = sum(1 for x in q if x)
            out[tool] = (detected, self.is_present(tool))
        return out
