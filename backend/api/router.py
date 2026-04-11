from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from backend.webrtc.streamer import VideoAnalyzerTrack

router = APIRouter()

# Store active websockets per client ID
active_websockets: dict[str, WebSocket] = {}
peer_connections: dict[str, RTCPeerConnection] = {}

class OfferRequest(BaseModel):
    sdp: str
    type: str
    client_id: str

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_websockets[client_id] = websocket
    try:
        while True:
            # We can receive configs here if needed
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        del active_websockets[client_id]
        if client_id in peer_connections:
            await peer_connections[client_id].close()
            del peer_connections[client_id]

@router.post("/offer")
async def offer(params: OfferRequest):
    client_id = params.client_id
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    pc = RTCPeerConnection()
    peer_connections[client_id] = pc

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # Assign the track to our analyzer
            # Get the connected websocket for this client
            ws = active_websockets.get(client_id)
            if ws:
                local_video = VideoAnalyzerTrack(track, ws)
                # Next.js WebRTC pipeline expects a transceiver/track back if it asked for sendrecv.
                # However, aiortc handles this. We shouldn't need to add it back if we don't display it,
                # but to satisfy the WebRTC negotiation, we can just add the dummy track.
                # Since user wants 1-way video, the client should create offer with `direction: "sendonly"`.
                # If they do, we don't need to add anything back!
            else:
                print(f"Warning: No WebSocket found for client {client_id}")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
