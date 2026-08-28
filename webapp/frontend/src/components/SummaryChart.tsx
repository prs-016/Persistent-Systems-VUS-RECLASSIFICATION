import type { RunSummary } from "../types";

const R = 42;
const CIRCUMFERENCE = 2 * Math.PI * R;

interface Segment {
  key: string;
  label: string;
  value: number;
  cls: string;
}

function Donut({ segments }: { segments: Segment[] }) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  let offset = 0;

  return (
    <svg viewBox="0 0 100 100" className="donut-svg" role="img" aria-label="Breakdown of variants by ClinVar resolution">
      <circle cx="50" cy="50" r={R} className="donut-track" />
      {segments.map((s) => {
        const frac = s.value / total;
        const dash = frac * CIRCUMFERENCE;
        const circle = (
          <circle
            key={s.key}
            cx="50"
            cy="50"
            r={R}
            className={`donut-seg ${s.cls}`}
            strokeDasharray={`${dash} ${CIRCUMFERENCE - dash}`}
            strokeDashoffset={-offset}
          />
        );
        offset += dash;
        return circle;
      })}
      <text x="50" y="47" textAnchor="middle" className="donut-total">
        {total}
      </text>
      <text x="50" y="61" textAnchor="middle" className="donut-total-label">
        variants
      </text>
    </svg>
  );
}

function Bar({ segments }: { segments: Segment[] }) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  return (
    <div className="origin-bar" role="img" aria-label="Split of all variants by Stage 1 predicted origin">
      {segments.map((s) => (
        <div
          key={s.key}
          className={`origin-bar-seg ${s.cls}`}
          style={{ width: `${(s.value / total) * 100}%` }}
          title={`${s.label}: ${s.value}`}
        />
      ))}
    </div>
  );
}

export function SummaryChart({ summary }: { summary: RunSummary }) {
  const resolutionSegments: Segment[] = [
    { key: "pathogenic", label: "ClinVar pathogenic", value: summary.resolved_pathogenic, cls: "dot-pathogenic" },
    { key: "benign", label: "ClinVar benign", value: summary.resolved_benign, cls: "dot-benign" },
    { key: "vus", label: "VUS, unresolved", value: summary.vus_count, cls: "dot-vus" },
  ];

  const originSegments: Segment[] = [
    { key: "germline", label: "Germline", value: summary.predicted_germline_total, cls: "dot-germline" },
    { key: "somatic", label: "Somatic", value: summary.predicted_somatic_total, cls: "dot-somatic" },
  ];

  return (
    <div className="summary-charts">
      <div className="chart-card">
        <h3>ClinVar resolution</h3>
        <div className="chart-body">
          <Donut segments={resolutionSegments} />
          <ul className="chart-legend">
            {resolutionSegments.map((s) => (
              <li key={s.key}>
                <i className={`legend-dot ${s.cls}`} />
                {s.label}
                <span className="chart-legend-value tabular">{s.value}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="chart-card">
        <h3>Predicted origin, all variants (Stage 1)</h3>
        <Bar segments={originSegments} />
        <ul className="chart-legend chart-legend-row">
          {originSegments.map((s) => (
            <li key={s.key}>
              <i className={`legend-dot ${s.cls}`} />
              {s.label}
              <span className="chart-legend-value tabular">{s.value}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
