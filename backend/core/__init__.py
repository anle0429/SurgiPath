"""
SurgiPath backend modules.

- constants: paths, config defaults, session state keys
- detector: YOLO model load, inference, drawing
- evidence: per-tool confidence/stability tracker, calibration
- hands: MediaPipe hand detection, grip classification, technique analysis
- state: PRE_OP / INTRA_OP / POST_OP mode + phase
- rules: debounced rule evaluation with evidence gating
- logger: event logging to logs/events.jsonl
- utils: recipe loader, ToolPresenceSmoother
"""
