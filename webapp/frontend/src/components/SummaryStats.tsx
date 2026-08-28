import type { RunSummary } from "../types";

export function SummaryStats({ summary }: { summary: RunSummary }) {
  const cells: { value: number; label: string; emphasis?: boolean }[] = [
    { value: summary.total_variants, label: "Total variants" },
    { value: summary.resolved_pathogenic, label: "ClinVar pathogenic" },
    { value: summary.resolved_benign, label: "ClinVar benign" },
    { value: summary.vus_count, label: "VUS, unresolved" },
    { value: summary.vus_predicted_germline, label: "VUS, germline origin" },
    { value: summary.vus_predicted_somatic, label: "VUS, somatic origin" },
    { value: summary.vus_matched_clinvar_watchlist, label: "Matched ClinVar watchlist" },
    { value: summary.flagged_for_reclassification_review, label: "Flagged for review", emphasis: true },
  ];

  return (
    <div className="stat-grid">
      {cells.map((c) => (
        <div className="stat-cell" key={c.label} data-emphasis={c.emphasis}>
          <div className="stat-value tabular">{c.value}</div>
          <div className="stat-label">{c.label}</div>
        </div>
      ))}
    </div>
  );
}
