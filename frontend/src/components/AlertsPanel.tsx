import { Alert } from "../types";

export function AlertsPanel({ alerts }: { alerts?: Alert[] }) {
  return (
    <div className="p-4 font-mono text-sm border border-gray-600 bg-gray-900 h-64 overflow-y-auto w-full">
      <h3 className="mb-2 font-bold text-white border-b border-gray-700 pb-1 sticky top-0 bg-gray-900">TTS / Alerts Log</h3>
      
      {!alerts || alerts.length === 0 ? (
        <div className="text-gray-500 italic">No alerts recorded yet.</div>
      ) : (
        <div className="flex flex-col gap-2">
          {defaultsAlerts(alerts).slice().reverse().map((alert, idx) => (
             <div 
               key={idx} 
               className={`p-2 border-l-2 ${statusColor(alert.status)} bg-black rounded`}
             >
               <div className="text-xs text-gray-500">{alert.timestamp}</div>
               <div className="font-semibold text-gray-200">{alert.message}</div>
               <div className="text-gray-400 mt-1">{alert.tip}</div>
             </div>
          ))}
        </div>
      )}
    </div>
  );
}

function statusColor(status: string) {
  switch (status) {
    case "good": return "border-green-500";
    case "warning": return "border-yellow-500";
    case "error": return "border-red-500";
    default: return "border-gray-500";
  }
}

function defaultsAlerts(alerts: Alert[]): Alert[] {
  // Return last 20 elements maximum to prevent DOM bloat
  return alerts.slice(-20);
}
