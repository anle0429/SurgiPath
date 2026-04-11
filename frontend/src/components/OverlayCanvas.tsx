import { useRef, useEffect } from "react";
import { HandData, ToolData } from "../types";

interface OverlayCanvasProps {
  hands?: HandData[];
  tools?: ToolData[];
  showLandmarks: boolean;
  showBoxes: boolean;
}

export function OverlayCanvas({ hands, tools, showLandmarks, showBoxes }: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animFrame: number;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (showBoxes && tools) {
        tools.forEach((tool) => {
          const [x1, y1, x2, y2] = tool.xyxy;
          ctx.strokeStyle = "blue";
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = "blue";
          ctx.font = "12px monospace";
          ctx.fillText(`${tool.name} (${tool.conf.toFixed(2)})`, x1, y1 - 4);
        });
      }

      if (showLandmarks && hands) {
        hands.forEach((hand) => {
          hand.landmarks.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = hand.handedness === "Left" ? "orange" : "green";
            ctx.fill();
          });
        });
      }

      animFrame = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animFrame);
  }, [hands, tools, showLandmarks, showBoxes]);

  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={480}
      className="absolute top-0 left-0 w-full h-full pointer-events-none"
    />
  );
}
