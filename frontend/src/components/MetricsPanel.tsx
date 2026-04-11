import { BackendPayload } from "../types";

export function MetricsPanel({ payload }: { payload: BackendPayload | null }) {
  if (!payload || !payload.technique) return <div className="p-4 border border-gray-600">Waiting for data...</div>;

  const { technique } = payload;
  
  return (
    <div className="p-4 font-mono text-sm border border-gray-600 bg-gray-900 text-green-400">
      <h3 className="mb-2 font-bold text-white border-b border-gray-700 pb-1">Real-time Metrics</h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <span className="text-gray-400">Smoothness: </span>
          <span className={technique.smoothness === "steady" ? "text-green-500" : "text-yellow-500"}>
            {technique.smoothness || "N/A"}
          </span>
        </div>
        
        <div>
          <span className="text-gray-400">Bimanual: </span>
          <span>{technique.bimanual?.detected ? "Yes" : "No"} ({technique.bimanual?.inter_hand_dist || 0}px)</span>
        </div>

        <div className="col-span-2 mt-2">
          <span className="text-gray-400 block mb-1">Grips detected:</span>
          {technique.grips.length > 0 ? (
            technique.grips.map((grip, idx) => (
              <div key={idx} className="pl-2 border-l border-gray-700 mb-1">
                <div>Hand: {grip.handedness}</div>
                <div>Type: {grip.grip_type}</div>
                <div>Angle: {grip.instrument_angle}°</div>
              </div>
            ))
          ) : (
            <div className="text-gray-500 italic">None</div>
          )}
        </div>
      </div>
    </div>
  );
}
