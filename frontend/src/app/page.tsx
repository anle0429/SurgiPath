"use client";

import { useState } from "react";
import { useWebRTC } from "../hooks/useWebRTC";
import { useWebSocket } from "../hooks/useWebSocket";
import { useTTS } from "../hooks/useTTS";

import { VideoStage } from "../components/VideoStage";
import { MetricsPanel } from "../components/MetricsPanel";
import { AlertsPanel } from "../components/AlertsPanel";
import { ConnectionStatus } from "../components/ConnectionStatus";

export default function Home() {
  const [mockMode, setMockMode] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showRawJson, setShowRawJson] = useState(false);

  const { stream, rtcStatus } = useWebRTC();
  const { wsStatus, payload } = useWebSocket(mockMode);

  useTTS(payload?.alerts);

  return (
    <div className="min-h-screen bg-black text-gray-200 p-8 font-sans flex flex-col gap-6">
      
      <header className="flex items-center justify-between border-b border-gray-800 pb-4">
        <div>
          <h1 className="text-xl font-bold text-white">SurgiPath Prototype Harness</h1>
          <p className="text-sm text-gray-500">Minimal testing UI for WebRTC & AI components</p>
        </div>
        <ConnectionStatus rtcStatus={rtcStatus} wsStatus={wsStatus} />
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* Left Column: Visuals & Controls */}
        <div className="flex flex-col gap-4">
          
          <div className="flex items-center gap-4 bg-gray-900 p-2 rounded border border-gray-700">
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input type="checkbox" checked={mockMode} onChange={e => setMockMode(e.target.checked)} />
              Mock Backend
            </label>
            <div className="w-px h-4 bg-gray-700 mx-1"></div>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input type="checkbox" checked={showLandmarks} onChange={e => setShowLandmarks(e.target.checked)} />
              Show Landmarks
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input type="checkbox" checked={showBoxes} onChange={e => setShowBoxes(e.target.checked)} />
              Show Boxes
            </label>
          </div>

          <VideoStage 
            stream={stream} 
            hands={payload?.hands} 
            tools={payload?.tools} 
            showLandmarks={showLandmarks}
            showBoxes={showBoxes}
          />
          
        </div>

        {/* Right Column: Data & Metrics */}
        <div className="flex flex-col gap-4">
          <MetricsPanel payload={payload} />
          <AlertsPanel alerts={payload?.alerts} />
          
          <div className="flex flex-col mt-4">
             <button 
               onClick={() => setShowRawJson(!showRawJson)} 
               className="bg-gray-800 text-xs text-left p-2 hover:bg-gray-700 border-t border-x border-gray-700"
             >
               {showRawJson ? "▼ Hide Raw JSON" : "▶ Show Raw JSON"}
             </button>
             {showRawJson && (
                <div className="bg-black p-4 border border-gray-700 h-64 overflow-auto">
                   <pre className="text-xs text-green-300">
                      {JSON.stringify(payload, null, 2)}
                   </pre>
                </div>
             )}
          </div>
        </div>
      </div>
    </div>
  );
}
