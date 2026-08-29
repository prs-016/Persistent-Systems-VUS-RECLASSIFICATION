import { useCallback, useEffect, useRef, useState } from "react";
import "./app.css";
import { ApiError, fetchJobStatus, submitClassificationJob } from "./api";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { ProcessSteps } from "./components/ProcessSteps";
import { SummaryChart } from "./components/SummaryChart";
import { UploadPanel, type UploadPanelHandle } from "./components/UploadPanel";
import { PipelineProgress } from "./components/PipelineProgress";
import { SummaryStats } from "./components/SummaryStats";
import { ResultsTable } from "./components/ResultsTable";
import { TableSkeleton } from "./components/TableSkeleton";
import type { ClassifyResult, JobState, ResultFilter } from "./types";

const POLL_INTERVAL_MS = 1200;
// Render's free-tier wake-from-idle can take 50+ seconds; at the 1.2s poll
// interval this gives a woken-up instance roughly a minute of retries
// before the UI gives up and reports a real error.
const MAX_CONSECUTIVE_POLL_FAILURES = 40;
type Theme = "light" | "dark";

function initialTheme(): Theme {
  const fromDom = document.documentElement.dataset.theme;
  if (fromDom === "light" || fromDom === "dark") return fromDom;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState | "idle">("idle");
  const [progressMessage, setProgressMessage] = useState<string>();
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<ResultFilter>("vus");
  const [tissue, setTissue] = useState("breast");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const pollRef = useRef<number | null>(null);
  const uploadPanelRef = useRef<UploadPanelHandle>(null);
  // How many *consecutive* status-check failures we've seen. Render's free
  // tier spins down after 15 minutes idle and can take 50+ seconds to wake
  // back up (see Render's own banner on the dashboard) -- a request landing
  // in that window comes back as a transient 502/timeout even though the
  // classify job is queued and running fine server-side. Treating the very
  // first failed poll as fatal was showing users "Could not complete this
  // run" for jobs that were actually still in progress. Only give up after
  // several failures in a row; any successful poll resets the counter.
  const pollFailuresRef = useRef(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("vt-theme", theme);
    } catch {
      // private browsing / storage disabled — theme just won't persist
    }
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearTimeout(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback((id: string) => {
    fetchJobStatus(id)
      .then((status) => {
        pollFailuresRef.current = 0;
        setProgressMessage(status.message);
        if (status.status === "error") {
          setJobState("error");
          setError(status.message ?? "The pipeline failed.");
          return;
        }
        if (status.status === "done" && status.result) {
          setJobState("done");
          setResult(status.result);
          return;
        }
        setJobState(status.status);
        pollRef.current = window.setTimeout(() => poll(id), POLL_INTERVAL_MS);
      })
      .catch((e: unknown) => {
        pollFailuresRef.current += 1;
        if (pollFailuresRef.current < MAX_CONSECUTIVE_POLL_FAILURES) {
          // Likely the free instance waking up from an idle spin-down
          // (Render warns this can take 50+ seconds) rather than a real
          // failure -- keep the "running" UI up and quietly retry instead
          // of reporting an error for a job that's still in progress.
          setProgressMessage("Reconnecting to the server...");
          pollRef.current = window.setTimeout(() => poll(id), POLL_INTERVAL_MS);
          return;
        }
        setJobState("error");
        setError(e instanceof Error ? e.message : "Lost connection while checking progress.");
      });
  }, []);

  const handleRun = useCallback(
    async (file: File, selectedTissue: string) => {
      setError(null);
      setResult(null);
      setTissue(selectedTissue);
      setJobState("queued");
      setProgressMessage("Uploading file...");
      try {
        const { job_id } = await submitClassificationJob(file, selectedTissue);
        setJobId(job_id);
        poll(job_id);
      } catch (e: unknown) {
        setJobState("error");
        setError(e instanceof ApiError ? e.message : "Could not reach the local server.");
      }
    },
    [poll],
  );

  const running = jobState === "queued" || jobState === "running";

  const scrollToUpload = useCallback(() => {
    document.getElementById("upload")?.scrollIntoView({ behavior: "smooth", block: "start" });
    // Give the scroll a beat to start before the native file picker opens,
    // since the picker is a modal dialog that would otherwise freeze the
    // scroll mid-animation.
    window.setTimeout(() => uploadPanelRef.current?.openFilePicker(), 350);
  }, []);

  return (
    <div className="app-shell">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <main>
        <Hero onGetStarted={scrollToUpload} />

        <ProcessSteps />

        <div id="upload">
          <UploadPanel ref={uploadPanelRef} disabled={running} onRun={handleRun} />
        </div>

        {(running || error) && (
          <section className="panel">
            <div className="panel-head">
              <h2>Run status</h2>
            </div>
            {running && <PipelineProgress message={progressMessage} />}
            {error && (
              <div className="error-banner" role="alert">
                <strong>Could not complete this run.</strong>
                <span>{error}</span>
              </div>
            )}
          </section>
        )}

        {running && (
          <section className="panel">
            <div className="panel-head">
              <h2>Results</h2>
              <p>Populating as the pipeline finishes each stage.</p>
            </div>
            <TableSkeleton />
          </section>
        )}

        {result && !running && (
          <>
            <section className="panel">
              <div className="panel-head">
                <h2>Summary</h2>
              </div>
              <SummaryStats summary={result.summary} />
              <SummaryChart summary={result.summary} />
            </section>
            <ResultsTable
              rows={result.variants}
              summary={result.summary}
              tissue={tissue}
              filter={filter}
              onFilterChange={setFilter}
            />
          </>
        )}
      </main>

      <footer id="about" className="site-footer">
        <p>VUS Reclassification &middot; Persistent Systems.</p>
        {jobId && <p className="run-id mono">Run {jobId}</p>}
      </footer>
    </div>
  );
}
