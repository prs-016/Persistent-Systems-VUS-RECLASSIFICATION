// A second custom illustration, distinct from the hero's scatter plot,
// showing the actual four stages a file goes through. Real information
// (matches PipelineProgress's own step list), not decoration.
const STEPS = [
  {
    label: "Upload",
    detail: "MAF or VCF, parsed and normalized to one variant schema.",
    icon: (
      <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
        <path d="M16 22V8M16 8l-6 6M16 8l6 6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M7 24v2a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-2" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    label: "Annotate",
    detail: "Ensembl VEP and local ANNOVAR resolve ClinVar, gnomAD, COSMIC, dbSNP.",
    icon: (
      <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
        <circle cx="16" cy="16" r="10" fill="none" stroke="currentColor" strokeWidth="2.2" />
        <path d="M16 11v5l3.5 2" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    label: "Stage 1",
    detail: "XGBoost calls germline vs. somatic origin on unresolved variants.",
    icon: (
      <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
        <path d="M9 20c0-5 3-9 7-9s7 4 7 9" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
        <circle cx="16" cy="21" r="2.2" fill="currentColor" />
        <path d="M16 21 20 13" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    label: "Stage 2",
    detail: "Scores every VUS for reclassification likelihood, direction, timing.",
    icon: (
      <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
        <rect x="7" y="16" width="4" height="9" rx="1" fill="currentColor" />
        <rect x="14" y="10" width="4" height="15" rx="1" fill="currentColor" />
        <rect x="21" y="13" width="4" height="12" rx="1" fill="currentColor" />
      </svg>
    ),
  },
];

export function ProcessSteps() {
  return (
    <section className="process-band" aria-label="How the pipeline processes an upload">
      <ol className="process-steps">
        {STEPS.map((step, i) => (
          <li key={step.label}>
            <div className="process-icon">{step.icon}</div>
            <div className="process-text">
              <span className="process-index">
                {String(i + 1).padStart(2, "0")}
              </span>
              <h3>{step.label}</h3>
              <p>{step.detail}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
