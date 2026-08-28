import { forwardRef, useCallback, useImperativeHandle, useRef, useState } from "react";

const TISSUES = [
  ["breast", "Breast"],
  ["lung", "Lung"],
  ["colon", "Colon"],
  ["skin", "Skin"],
  ["stomach", "Stomach"],
  ["ovary", "Ovary"],
  ["prostate", "Prostate"],
  ["other", "Other / unspecified"],
] as const;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

interface Props {
  disabled: boolean;
  onRun: (file: File, tissue: string) => void;
}

export interface UploadPanelHandle {
  openFilePicker: () => void;
}

export const UploadPanel = forwardRef<UploadPanelHandle, Props>(function UploadPanel(
  { disabled, onRun },
  ref,
) {
  const [file, setFile] = useState<File | null>(null);
  const [tissue, setTissue] = useState("breast");
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((files: FileList | null) => {
    if (files && files.length > 0) setFile(files[0]);
  }, []);

  useImperativeHandle(ref, () => ({
    openFilePicker: () => inputRef.current?.click(),
  }));

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Upload a variant file</h2>
        <p>
          MAF, gzipped MAF (.maf.gz), or VCF/TCF. Runs the trained Stage 1 origin
          classifier and Stage 2 reclassification-likelihood model against your file.
        </p>
      </div>

      <div
        className="dropzone"
        data-active={dragActive}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".maf,.gz,.vcf,.tcf,.txt"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
        <p className="dz-primary">Drop a file here, or click to browse</p>
        <p className="dz-secondary">.maf &middot; .maf.gz &middot; .vcf &middot; .tcf</p>
      </div>

      {file && (
        <div className="filerow">
          <span className="filerow-name">
            {file.name} <span className="filerow-size">{formatSize(file.size)}</span>
          </span>
          <button type="button" className="link-btn" onClick={() => setFile(null)}>
            Remove
          </button>
        </div>
      )}

      <div className="controls-row">
        <label htmlFor="tissue">
          Tissue context
          <select id="tissue" value={tissue} onChange={(e) => setTissue(e.target.value)}>
            {TISSUES.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          className="primary-btn"
          disabled={!file || disabled}
          onClick={() => file && onRun(file, tissue)}
        >
          {disabled ? "Running" : "Run classification"}
        </button>
      </div>
    </section>
  );
});
