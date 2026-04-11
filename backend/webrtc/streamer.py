import asyncio
import json
import logging
import cv2
import numpy as np

from aiortc import MediaStreamTrack
from fastapi import WebSocket

# Try to import core CV models
try:
    from backend.core.detector import infer_tools, count_tools
    from backend.core.hands import detect_hands
    CV_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CV modules could not be loaded: {e}")
    CV_AVAILABLE = False


class VideoAnalyzerTrack(MediaStreamTrack):
    """
    A video track that receives frames from the client and runs YOLO + MediaPipe.
    Does NOT return modified frames (one-way video).
    Sends JSON metadata to the connected WebSocket.
    """
    kind = "video"

    def __init__(self, track: MediaStreamTrack, websocket: WebSocket):
        super().__init__()
        self.track = track
        self.websocket = websocket

    async def recv(self):
        # Read frame from the incoming WebRTC connection
        frame = await self.track.recv()
        
        # Convert frame to numpy array for CV
        img = frame.to_ndarray(format="bgr24")
        
        payload = {"tools": [], "hands": []}
        
        if CV_AVAILABLE:
            # 1. Run inference on tools
            detections = infer_tools(img, conf=0.5, imgsz=640)
            payload["tools"] = detections
            
            # 2. Run inference on hands
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hands_data = detect_hands(img_rgb)
            payload["hands"] = hands_data
        else:
            # Mock data if CV is structurally absent
            payload["tools"] = [{"name": "mock_tool", "conf": 0.99, "xyxy": [10, 10, 50, 50]}]
        
        # Send metadata asynchronously
        try:
            await self.websocket.send_json(payload)
        except Exception as e:
            # The websocket might have been closed
            pass
            
        # Return the untouched original frame to keep aiortc pipeline happy
        return frame
