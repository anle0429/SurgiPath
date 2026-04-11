import { useState, useEffect, useRef } from "react";
import { BackendPayload } from "../types";

export function useWebSocket(mockMode: boolean = false) {
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const [payload, setPayload] = useState<BackendPayload | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (mockMode) {
      setWsStatus("connected");
      // Simulate fake incoming data from loop iteration
      const interval = setInterval(() => {
        setPayload({
          tools: [{ name: "scalpel", conf: 0.95, xyxy: [100, 100, 200, 200] }],
          hands: [],
          technique: {
            smoothness: Math.random() > 0.5 ? "steady" : "tremor",
            grips: [{ handedness: "Right", grip_type: "precision_grip", instrument_angle: 45, near_tools: ["scalpel"] }],
            bimanual: { detected: false, inter_hand_dist: 0, hands_count: 1 },
          },
        });
      }, 500);
      return () => {
        clearInterval(interval);
        setWsStatus("disconnected");
      };
    }

    let unmounted = false;
    let ws: WebSocket;

    const connect = () => {
      if (unmounted) return;
      setWsStatus("connecting");
      ws = new WebSocket("ws://localhost:8000/ws/test_client");
      wsRef.current = ws;

      ws.onopen = () => {
        if (!unmounted) setWsStatus("connected");
      };

      ws.onmessage = (e) => {
        if (unmounted) return;
        try {
          const data = JSON.parse(e.data);
          setPayload(data);
        } catch (err) {
          console.error("Invalid JSON from WS:", err);
        }
      };

      ws.onclose = () => {
        if (!unmounted) {
          setWsStatus("disconnected");
          // Reconnect logic
          setTimeout(connect, 3000);
        }
      };

      ws.onerror = (err) => {
        console.error("WS error:", err);
        ws.close();
      };
    };

    connect();

    return () => {
      unmounted = true;
      if (ws) ws.close();
    };
  }, [mockMode]);

  return { wsStatus, payload };
}
