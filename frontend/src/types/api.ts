// Clinical Curator — API types

export interface HealthResponse {
  status: string;
  models_loaded: number;
  gpu: boolean;
  version: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  review: string;
  description: string;
  version_tag: string;
  auc_score: number;
  inference_ms: number;
  is_temporal: boolean;
}

export type RiskLevel = 'CRITICAL' | 'CAUTION' | 'NORMAL';

export interface Prediction {
  label: string;
  confidence_pct: number;  // integer 0–100, no leading zeros
  risk_level: RiskLevel;
}

export interface PredictResponse {
  study_id: string;
  predictions: Prediction[];
  grad_cam_base64: string;
  model_used: string;
  model_version: string;
  inference_ms: number;
  top_finding: string;
  clinical_insight: string;
  timestamp: string;
}

export type VizType = 'GRAD-CAM' | 'SALIENCY MAP' | 'LAYER ATTN';

export interface BatchPredictItem {
  model_name: string;
  model_version: string;
  predictions: Prediction[];
  grad_cam_base64: string;
  viz_type: VizType;
  inference_ms: number;
}

export interface LatentResponse {
  pca: [number, number];
  tsne: [number, number];
  cluster_label: number;
  nearest_5: Array<{ id: string; dist: number }>;
}
