// A custom, thematically-relevant illustration instead of a stock photo:
// a stylized variant plot, dots colored the same way the results table
// colors calls, so it previews what the app actually does rather than
// decorating the page with something generic.
function VariantPlot() {
  const points: { x: number; y: number; r: number; cls: string }[] = [
    { x: 28, y: 150, r: 5, cls: "dot-benign" },
    { x: 52, y: 168, r: 4, cls: "dot-benign" },
    { x: 74, y: 140, r: 5, cls: "dot-vus" },
    { x: 98, y: 96, r: 6, cls: "dot-vus" },
    { x: 122, y: 60, r: 7, cls: "dot-pathogenic" },
    { x: 146, y: 112, r: 5, cls: "dot-vus" },
    { x: 168, y: 176, r: 4, cls: "dot-benign" },
    { x: 192, y: 150, r: 5, cls: "dot-vus" },
    { x: 214, y: 44, r: 7, cls: "dot-pathogenic" },
    { x: 238, y: 128, r: 5, cls: "dot-vus" },
    { x: 260, y: 168, r: 4, cls: "dot-benign" },
    { x: 284, y: 80, r: 6, cls: "dot-pathogenic" },
    { x: 306, y: 156, r: 5, cls: "dot-vus" },
    { x: 330, y: 184, r: 4, cls: "dot-benign" },
    { x: 352, y: 132, r: 5, cls: "dot-vus" },
    { x: 40, y: 108, r: 4, cls: "dot-vus" },
    { x: 86, y: 190, r: 4, cls: "dot-benign" },
    { x: 110, y: 152, r: 5, cls: "dot-vus" },
    { x: 134, y: 176, r: 4, cls: "dot-benign" },
    { x: 178, y: 68, r: 6, cls: "dot-pathogenic" },
    { x: 202, y: 100, r: 5, cls: "dot-vus" },
    { x: 250, y: 56, r: 6, cls: "dot-pathogenic" },
    { x: 272, y: 190, r: 4, cls: "dot-benign" },
    { x: 318, y: 108, r: 5, cls: "dot-vus" },
    { x: 342, y: 72, r: 6, cls: "dot-pathogenic" },
  ];

  return (
    <svg viewBox="0 0 380 220" className="hero-plot" role="img" aria-label="Illustrative plot of classified variants by chromosome position">
      <line x1="10" y1="200" x2="370" y2="200" className="plot-axis" />
      <line x1="10" y1="20" x2="10" y2="200" className="plot-axis" />
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <line key={i} x1={10 + i * 60} y1="20" x2={10 + i * 60} y2="200" className="plot-gridline" />
      ))}
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={p.r} className={`plot-dot ${p.cls}`} />
      ))}
    </svg>
  );
}

interface Props {
  onGetStarted: () => void;
}

export function Hero({ onGetStarted }: Props) {
  return (
    <section className="hero">
      <div className="hero-copy">
        <p className="hero-eyebrow">Persistent Systems</p>
        <h1>
          Turn a raw variant file into a
          <span className="hero-highlight"> reviewed reclassification list</span>
        </h1>
        <p className="hero-lede">
          Upload a MAF or VCF. ClinVar resolves what it already knows, the
          project's own trained Stage 1 model calls germline versus somatic
          origin on what's left, and Stage 2 scores every VUS for
          reclassification likelihood, direction, and timing, then flags
          the ones worth a second look.
        </p>
        <div className="hero-actions">
          <button type="button" className="primary-btn hero-cta" onClick={onGetStarted}>
            Upload a file
          </button>
        </div>
        <dl className="hero-stats">
          <div>
            <dt>121,736</dt>
            <dd>ClinVar VUS in the watchlist</dd>
          </div>
          <div>
            <dt>0.9218</dt>
            <dd>Stage 1 production threshold</dd>
          </div>
          <div>
            <dt>3</dt>
            <dd>Stage 2 signal tiers</dd>
          </div>
        </dl>
      </div>
      <div className="hero-visual">
        <VariantPlot />
        <ul className="hero-legend">
          <li><i className="legend-dot dot-pathogenic" />Pathogenic-leaning</li>
          <li><i className="legend-dot dot-vus" />VUS, under review</li>
          <li><i className="legend-dot dot-benign" />Benign-leaning</li>
        </ul>
      </div>
    </section>
  );
}
