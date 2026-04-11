import { useEffect, useRef } from "react";
import { Alert } from "../types";

export function useTTS(alerts?: Alert[]) {
  const lastSpokenRef = useRef<string | null>(null);

  useEffect(() => {
    if (!alerts || alerts.length === 0) return;

    // Get the most recent alert
    const latestAlert = alerts[alerts.length - 1];
    
    // Prevent spamming the exact same message back to back
    const uniqueKey = `${latestAlert.timestamp}-${latestAlert.tip}`;
    if (lastSpokenRef.current === uniqueKey) return;

    // If it's a critical error/warning, we can cancel ongoing speech
    if (latestAlert.status === "error" || latestAlert.status === "warning") {
      window.speechSynthesis.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(latestAlert.tip);
    window.speechSynthesis.speak(utterance);
    
    lastSpokenRef.current = uniqueKey;

  }, [alerts]);
}
