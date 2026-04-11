import { useEffect, useRef } from "react";
import { OverlayCanvas } from "./OverlayCanvas";
import { HandData, ToolData } from "../types";

interface VideoStageProps {
  stream: MediaStream | null;
  hands?: HandData[];
  tools?: ToolData[];
  showLandmarks: boolean;
  showBoxes: boolean;
}

export function VideoStage({ stream, hands, tools, showLandmarks, showBoxes }: VideoStageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      if (videoRef.current.srcObject !== stream) {
        videoRef.current.srcObject = stream;
      }
    }
  }, [stream]);

  return (
    <div className="relative w-[640px] h-[480px] bg-black border border-gray-700">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute top-0 left-0 w-full h-full object-cover"
      />
      <OverlayCanvas
        hands={hands}
        tools={tools}
        showLandmarks={showLandmarks}
        showBoxes={showBoxes}
      />
    </div>
  );
}
