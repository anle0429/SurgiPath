export function ConnectionStatus({ rtcStatus, wsStatus }: { rtcStatus: string; wsStatus: string }) {
  const getColor = (status: string) => {
    switch (status) {
      case "connected":
        return "bg-green-500";
      case "connecting":
        return "bg-yellow-500 cursor-wait";
      case "failed":
      case "disconnected":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <div className="flex gap-4 p-2 bg-gray-900 rounded text-white text-xs font-mono">
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${getColor(rtcStatus)}`} />
        WebRTC: {rtcStatus}
      </div>
      <div className="flex items-center gap-2">
        <div className={`w-3 h-3 rounded-full ${getColor(wsStatus)}`} />
        WebSocket: {wsStatus}
      </div>
    </div>
  );
}
