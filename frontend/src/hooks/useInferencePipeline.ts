import { useEffect, useRef, useState } from 'react';

type Box = [number, number, number, number];

export interface ToolDetection {
  name: string;
  conf: number;
  xyxy: Box;
}

export interface InferencePayload {
  tools: ToolDetection[];
  hands?: any[];
}

export function useInferencePipeline(videoElementRef: React.RefObject<HTMLVideoElement | null>) {
  const [active, setActive] = useState(false);
  const [metadata, setMetadata] = useState<InferencePayload | null>(null);

  const pcRef = useRef<RTCPeerConnection | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Cleanup on unmount
    return () => stopPipeline();
  }, []);

  const startPipeline = async () => {
    if (!videoElementRef.current) return;

    try {
      // 1. Get user media (video only)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: 15 },
        audio: false,
      });
      videoElementRef.current.srcObject = stream;
      videoElementRef.current.play();

      // 2. Setup WebRTC PeerConnection
      const pc = new RTCPeerConnection();
      pcRef.current = pc;

      // Add track to peer connection (sendonly)
      stream.getTracks().forEach((track) => {
        pc.addTransceiver(track, { direction: 'sendonly' });
      });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Unique client ID to pair WebSocket with WebRTC session
      const clientId = Math.random().toString(36).substring(7);

      // 3. Setup WebSocket for receiving metadata
      const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const payload: InferencePayload = JSON.parse(event.data);
          setMetadata(payload);
        } catch (err) {
          console.error("Failed to parse metadata", err);
        }
      };

      // 4. Send offer to backend
      const response = await fetch('http://localhost:8000/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription?.sdp,
          type: pc.localDescription?.type,
          client_id: clientId,
        }),
      });

      const answer = await response.json();
      await pc.setRemoteDescription(answer);

      setActive(true);
    } catch (err) {
      console.error('Failed to start pipeline:', err);
    }
  };

  const stopPipeline = () => {
    pcRef.current?.close();
    wsRef.current?.close();
    
    if (videoElementRef.current?.srcObject) {
      const stream = videoElementRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoElementRef.current.srcObject = null;
    }
    
    setActive(false);
    setMetadata(null);
  };

  return { active, startPipeline, stopPipeline, metadata };
}
