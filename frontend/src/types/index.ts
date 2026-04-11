export interface Landmark {
  0: number; // x
  1: number; // y
  2?: number; // z
}

export interface HandData {
  handedness: string;
  landmarks: Landmark[];
  fingertips: Landmark[];
  wrist: Landmark;
  centroid: Landmark;
}

export interface ToolData {
  name: string;
  conf: number;
  xyxy: [number, number, number, number];
}

export interface GripData {
  handedness: string;
  grip_type: string;
  instrument_angle: number;
  near_tools: string[];
}

export interface Bimanual {
  detected: boolean;
  inter_hand_dist: number;
  hands_count: number;
}

export interface BackendPayload {
  tools?: ToolData[];
  hands?: HandData[];
  alerts?: Alert[];
  technique?: {
    grips: GripData[];
    smoothness: string;
    bimanual: Bimanual;
  };
}

export interface Alert {
  id?: string;
  timestamp: string;
  status: "good" | "warning" | "error";
  message: string;
  tip: string;
}
