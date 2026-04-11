import { useState, useEffect, useRef } from "react";

export function useWebRTC() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [rtcStatus, setRtcStatus] = useState<"disconnected" | "connecting" | "connected" | "failed">("disconnected");
  const pcRef = useRef<RTCPeerConnection | null>(null);

  useEffect(() => {
    let unmounted = false;
    let localStream: MediaStream;

    const startWebcam = async () => {
      try {
        localStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });

        if (unmounted) {
          localStream.getTracks().forEach((t) => t.stop());
          return;
        }

        setStream(localStream);
        await initWebRTC(localStream);
      } catch (e) {
        console.error("Camera error:", e);
        if (!unmounted) setRtcStatus("failed");
      }
    };

    const initWebRTC = async (strm: MediaStream) => {
      setRtcStatus("connecting");
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      pcRef.current = pc;

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "connected") setRtcStatus("connected");
        else if (pc.connectionState === "failed" || pc.connectionState === "disconnected") setRtcStatus("failed");
      };

      // Add track to peer connection (sendonly since we don't expect returning video frame right now)
      strm.getTracks().forEach((track) => {
        pc.addTransceiver(track, { direction: "sendonly" });
      });

      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // We use a fixed client_id for this testing harness
        const res = await fetch("http://localhost:8000/offer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sdp: offer.sdp,
            type: offer.type,
            client_id: "test_client",
          }),
        });

        if (!res.ok) {
          throw new Error("Failed to send WebRTC offer");
        }

        const answer = await res.json();
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
      } catch (e) {
        console.error("WebRTC Negotiation failed:", e);
        if (!unmounted) setRtcStatus("failed");
      }
    };

    startWebcam();

    return () => {
      unmounted = true;
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
      }
      if (pcRef.current) {
        pcRef.current.close();
      }
    };
  }, []);

  return { stream, rtcStatus };
}
