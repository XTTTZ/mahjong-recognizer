export interface TilePrediction {
  id: string;
  classId: string;
  label: string;
  bbox: [number, number, number, number];
  score: number;
  uncertain: boolean;
}

export interface RecognitionResult {
  requestId: string;
  modelVersion: string;
  image: { width: number; height: number };
  tiles: TilePrediction[];
  quality: { status: 'ok' | 'needs_retake'; reasons: string[] };
  timings: { totalMs: number };
  recognitionMode?: RecognitionMode;
  gatewayMs?: number;
  clientWaitMs?: number;
}

export type CloudRecognitionMode = 'fast' | 'precision';
export type RecognitionMode = CloudRecognitionMode | 'local';
export interface ServiceAvailability { configured: boolean; fastConfigured: boolean }

export interface PreparedPhoto {
  file: File;
  fastFile?: File;
  url: string;
  width: number;
  height: number;
}

export interface TileCorrection { classId: string; label: string }
