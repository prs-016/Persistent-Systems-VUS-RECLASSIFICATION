export type ClinvarStatus = "Pathogenic" | "Benign" | "VUS";
export type OriginCall = "Germline" | "Somatic";
export type Stage2Band =
  | "Watch closely"
  | "Modest signal"
  | "No distinguishing signal"
  | "Below baseline";
export type Stage2Source = "clinvar_v12" | "generalizable";

export interface VariantRow {
  chrom: string;
  pos: number;
  ref: string;
  alt: string;
  gene: string | null;
  vaf: number | null;
  gnomad_af: number | null;
  cosmic_hotspot: boolean | null;
  dbsnp_id: string | null;
  hpa_expression_level: string | null;
  low_tissue_expression_flag: boolean | null;
  clinvar_status: ClinvarStatus;
  predicted_class: OriginCall | null;
  predicted_class_confidence: number | null;
  germline_probability: number | null;
  has_mave_coverage: number | null;
  stage2_score: number | null;
  stage2_source: Stage2Source | null;
  stage2_band: Stage2Band | null;
  direction_pathogenic_probability: number | null;
  p_resolved_by_10y: number | null;
  reclassification_flag: boolean;
}

export interface RunSummary {
  total_variants: number;
  resolved_pathogenic: number;
  resolved_benign: number;
  vus_count: number;
  vus_predicted_germline: number;
  vus_predicted_somatic: number;
  predicted_germline_total: number;
  predicted_somatic_total: number;
  flagged_for_reclassification_review: number;
  vus_matched_clinvar_watchlist: number;
}

export interface ClassifyResult {
  summary: RunSummary;
  variants: VariantRow[];
}

export type JobState = "queued" | "running" | "done" | "error";

export interface JobStatusResponse {
  status: JobState;
  message?: string;
  result?: ClassifyResult;
}

export type ResultFilter =
  | "all"
  | "flagged"
  | "vus"
  | "pathogenic"
  | "benign"
  | "germline"
  | "somatic";

export interface AiExplainResponse {
  explanation: string;
}
