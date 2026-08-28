import { Fragment, useState } from "react";
import { AiSummary } from "./AiSummary";
import type { ResultFilter, RunSummary, VariantRow } from "../types";

const FILTERS: { key: ResultFilter; label: string }[] = [
  { key: "all", label: "All variants" },
  { key: "vus", label: "VUS (main focus)" },
  { key: "flagged", label: "Flagged for review" },
  { key: "germline", label: "Germline" },
  { key: "somatic", label: "Somatic" },
  { key: "pathogenic", label: "Pathogenic" },
  { key: "benign", label: "Benign" },
];

function applyFilter(rows: VariantRow[], filter: ResultFilter): VariantRow[] {
  switch (filter) {
    case "flagged":
      return rows.filter((r) => r.reclassification_flag);
    case "vus":
      return rows.filter((r) => r.clinvar_status === "VUS");
    case "pathogenic":
      return rows.filter((r) => r.clinvar_status === "Pathogenic");
    case "benign":
      return rows.filter((r) => r.clinvar_status === "Benign");
    case "germline":
      return rows.filter((r) => r.predicted_class === "Germline");
    case "somatic":
      return rows.filter((r) => r.predicted_class === "Somatic");
    default:
      return rows;
  }
}

function StatusPill({ value }: { value: string | null }) {
  if (!value) return <span className="cell-dim">&mdash;</span>;
  const key = value.toLowerCase();
  return <span className={`pill pill-${key}`}>{value}</span>;
}

function bandClass(band: string | null): string {
  switch (band) {
    case "Watch closely":
      return "band-watch";
    case "Modest signal":
      return "band-modest";
    case "No distinguishing signal":
      return "band-none";
    case "Below baseline":
      return "band-below";
    default:
      return "";
  }
}

function SourceTag({ source }: { source: string | null }) {
  if (!source) return <span className="cell-dim">&mdash;</span>;
  if (source === "clinvar_v12") {
    return <span className="source-tag source-tag-watchlist">ClinVar match</span>;
  }
  return <span className="source-tag source-tag-generalizable">Generalizable</span>;
}

function directionLabel(p: number | null): { text: string; className: string } {
  if (p == null) return { text: "—", className: "cell-dim" };
  if (p >= 0.65) return { text: `${(p * 100).toFixed(0)}% pathogenic-leaning`, className: "direction-pathogenic" };
  if (p <= 0.35) return { text: `${(100 - p * 100).toFixed(0)}% benign-leaning`, className: "direction-benign" };
  return { text: `${(p * 100).toFixed(0)}% pathogenic, unclear`, className: "direction-unclear" };
}

interface Props {
  rows: VariantRow[];
  summary: RunSummary;
  tissue: string;
  filter: ResultFilter;
  onFilterChange: (f: ResultFilter) => void;
}

export function ResultsTable({ rows, summary, tissue, filter, onFilterChange }: Props) {
  const visible = applyFilter(rows, filter);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Results</h2>
        <p>
          Every variant, its ClinVar resolution, and Stage 1's germline/somatic
          origin call, which runs on the whole file, not just VUS. VUS rows are
          the main focus, use the filter below, and expand a row for its
          reclassification detail.
        </p>
      </div>

      <AiSummary summary={summary} variants={rows} tissue={tissue} />

      <div className="filter-tabs" role="tablist">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            role="tab"
            aria-selected={filter === f.key}
            data-active={filter === f.key}
            onClick={() => onFilterChange(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th aria-hidden="true"></th>
              <th>Gene</th>
              <th>Position</th>
              <th>Change</th>
              <th>ClinVar</th>
              <th>Stage 1 call</th>
              <th>Reclass. band</th>
              <th>Flag</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((v, i) => {
              const key = `${v.chrom}-${v.pos}-${v.alt}-${i}`;
              const isOpen = expanded.has(key);
              const direction = directionLabel(v.direction_pathogenic_probability);
              return (
                <Fragment key={key}>
                  <tr data-flagged={v.reclassification_flag}>
                    <td>
                      <button
                        type="button"
                        className="expand-btn"
                        onClick={() => toggle(key)}
                        aria-expanded={isOpen}
                        aria-label={isOpen ? "Hide detail" : "Show detail"}
                      >
                        {isOpen ? "−" : "+"}
                      </button>
                    </td>
                    <td className="cell-strong">{v.gene ?? "—"}</td>
                    <td className="mono tabular">
                      {v.chrom}:{v.pos.toLocaleString()}
                    </td>
                    <td className="mono">
                      {v.ref} &rarr; {v.alt}
                    </td>
                    <td>
                      <StatusPill value={v.clinvar_status} />
                    </td>
                    <td>
                      <StatusPill value={v.predicted_class} />
                    </td>
                    <td className={bandClass(v.stage2_band)}>{v.stage2_band ?? "—"}</td>
                    <td>{v.reclassification_flag ? <span className="flag-badge">Review</span> : null}</td>
                  </tr>
                  {isOpen && (
                    <tr className="detail-row">
                      <td colSpan={8}>
                        <div className="detail-grid">
                          <div className="detail-item">
                            <span className="detail-label">VAF</span>
                            <span className="detail-value tabular">
                              {v.vaf != null ? v.vaf.toFixed(3) : "—"}
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">gnomAD AF</span>
                            <span className="detail-value tabular">
                              {v.gnomad_af != null ? v.gnomad_af.toExponential(2) : "—"}
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Stage 1 confidence</span>
                            <span className="detail-value tabular">
                              {v.predicted_class_confidence != null
                                ? `${(v.predicted_class_confidence * 100).toFixed(1)}%`
                                : "—"}
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Reclass. score</span>
                            <span className="detail-value tabular">
                              {v.stage2_score != null ? v.stage2_score.toFixed(3) : "—"}
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Score source</span>
                            <span className="detail-value">
                              <SourceTag source={v.stage2_source} />
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Direction if resolved</span>
                            <span className={`detail-value ${direction.className}`}>{direction.text}</span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Resolved within 10y</span>
                            <span className="detail-value tabular">
                              {v.p_resolved_by_10y != null ? `${(v.p_resolved_by_10y * 100).toFixed(1)}%` : "—"}
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">Tissue expression (HPA)</span>
                            <span className="detail-value">
                              {v.hpa_expression_level ?? "—"}
                              {v.low_tissue_expression_flag ? " · low breadth" : ""}
                            </span>
                          </div>
                          <div className="detail-item">
                            <span className="detail-label">dbSNP</span>
                            <span className="detail-value mono">{v.dbsnp_id ?? "—"}</span>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
        {visible.length === 0 && <p className="empty-state">No variants match this filter.</p>}
      </div>

      <p className="footnote">
        Bands read differently by source. For a generalizable-sourced VUS, watch
        closely means a score of 0.60 or above; modest signal sits above the
        model's flat baseline. For a ClinVar-matched VUS, the band instead
        reflects the real probability of resolving within 10 years. Stage 1 does
        not assign pathogenicity, only germline-versus-somatic origin, and it
        runs on every variant, ClinVar-resolved or not.
      </p>
    </section>
  );
}
