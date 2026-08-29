import { useCallback, useEffect, useState } from "react";
import { ApiError, fetchAiExplanation } from "../api";
import type { AiVariantExplanation, RunSummary, VariantRow } from "../types";

interface Props {
  summary: RunSummary;
  variants: VariantRow[];
  tissue: string;
}

// Sends only the flagged rows, and only the fields relevant to why they
// were flagged, to keep the request small and avoid shipping the whole
// table to a third-party API.
function flaggedPayload(variants: VariantRow[]): VariantRow[] {
  return variants.filter((v) => v.reclassification_flag).slice(0, 40);
}

export function AiSummary({ summary, variants, tissue }: Props) {
  const [items, setItems] = useState<AiVariantExplanation[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const flagged = flaggedPayload(variants);

  const run = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchAiExplanation(summary, flagged, tissue)
      .then((res) => setItems(res.explanations))
      .catch((e: unknown) => {
        setError(
          e instanceof ApiError
            ? e.message
            : "Could not reach the AI explanation service.",
        );
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary, tissue, variants]);

  useEffect(() => {
    setItems(null);
    setError(null);
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary]);

  return (
    <div className="ai-summary">
      <div className="ai-summary-head">
        <h3>What to look at first</h3>
        <button type="button" className="ai-refresh-btn" onClick={run} disabled={loading}>
          {loading ? "Generating..." : "Regenerate"}
        </button>
      </div>

      {loading && (
        <div className="ai-summary-loading">
          <span className="spinner" />
          Reading the flagged variants and tissue expression data...
        </div>
      )}

      {!loading && error && (
        <p className="ai-summary-error">
          {error} Falling back to the run summary above and the column
          definitions below the table.
        </p>
      )}

      {!loading && !error && items && items.length > 0 && (
        <>
          <div className="ai-summary-items">
            {items.map((item) => (
              <div className="ai-summary-item" key={`${item.chrom}:${item.pos}:${item.ref}:${item.alt}`}>
                <div className="ai-summary-item-head">
                  <span className="ai-summary-item-variant">{item.variant_label}</span>
                  <span className="ai-summary-item-band">{item.stage2_band}</span>
                </div>
                <p className="ai-summary-body">{item.explanation}</p>
              </div>
            ))}
          </div>
          <p className="ai-summary-meta">
            Generated from this run's {flagged.length} flagged variant
            {flagged.length === 1 ? "" : "s"} by Gemini ({items.length} explained
            individually). This is a plain-language reading aid, not a clinical
            interpretation, verify against the scores and bands in the table.
          </p>
        </>
      )}

      {!loading && !error && items && items.length === 0 && (
        <p className="ai-summary-error">
          No flagged variants to explain for this run.
        </p>
      )}
    </div>
  );
}
