const STEPS: { key: string; label: string; match: string; note?: string }[] = [
  { key: "parse", label: "Parsing uploaded file", match: "Parsing uploaded" },
  { key: "vep", label: "Querying ClinVar, gnomAD, COSMIC, dbSNP (Ensembl VEP)", match: "Querying Ensembl VEP" },
  {
    key: "annovar",
    label: "Running local ANNOVAR annotation",
    match: "Running local ANNOVAR",
    note: "Usually the slowest step, ANNOVAR scans its reference databases on every run regardless of file size. A minute or two here is normal, not stuck.",
  },
  { key: "hpa", label: "Looking up tissue expression (Human Protein Atlas)", match: "tissue expression breadth" },
  { key: "features", label: "Computing classification features", match: "Computing classification features" },
  { key: "exonic", label: "Deriving exonic-consequence features", match: "exonic-consequence" },
  { key: "stage1", label: "Scoring Stage 1: Germline vs. Somatic origin", match: "Stage 1 Germline" },
  { key: "stage2", label: "Scoring Stage 2: reclassification likelihood", match: "Stage 2 generalizable" },
];

function currentStepIndex(message: string | undefined): number {
  if (!message) return -1;
  if (message.toLowerCase().startsWith("done")) return STEPS.length;
  const idx = STEPS.findIndex((s) => message.includes(s.match));
  return idx;
}

export function PipelineProgress({ message }: { message?: string }) {
  const activeIdx = currentStepIndex(message);

  return (
    <div className="pipeline-progress">
      <ol>
        {STEPS.map((step, i) => {
          const state = i < activeIdx ? "done" : i === activeIdx ? "active" : "pending";
          return (
            <li key={step.key} data-state={state}>
              <span className="marker" aria-hidden="true">
                {state === "done" ? (
                  <svg width="10" height="10" viewBox="0 0 10 10">
                    <path d="M1 5.2 L3.8 8 L9 1.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : state === "active" ? (
                  <span className="spinner" />
                ) : null}
              </span>
              <span className="label">
                {step.label}
                {state === "active" && step.note && <span className="step-note">{step.note}</span>}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
