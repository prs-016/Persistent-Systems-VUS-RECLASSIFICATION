"""
VUS Classifier API.

FastAPI backend for the VUS Reclassification app. Wraps pipeline.py (the real
Stage 1 + Stage 2 inference pipeline, using the project's own trained
model artifacts) behind a small job-queue API so the React frontend can
upload a file, then poll for progress and the final result.

Run in development (frontend served separately by Vite, proxied to this
process):

    uvicorn main:app --reload --port 8765

Run in production (this process also serves the built frontend from
../frontend/dist):

    uvicorn main:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import os
import threading
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import ai_explain
import pipeline

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend", "dist"))


def _load_dotenv(path: str) -> None:
    """Tiny .env loader so GEMINI_API_KEY doesn't need to live in the shell
    profile or get typed on the command line. Deliberately minimal, no new
    dependency, just KEY=VALUE lines; existing environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(os.path.join(BACKEND_DIR, ".env"))

app = FastAPI(title="VUS Reclassification API", version="1.0.0")

# Only needed for local dev, where Vite runs on :5173 and this API on
# :8765 as separate origins. The production build is served by this same
# process, so no cross-origin requests happen there.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobStatus(BaseModel):
    status: str  # queued | running | done | error
    message: Optional[str] = None
    result: Optional[dict] = None


JOBS: dict[str, JobStatus] = {}
JOBS_LOCK = threading.Lock()


def _set_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        current = JOBS[job_id].model_dump()
        current.update(kwargs)
        JOBS[job_id] = JobStatus(**current)


def _run_job(job_id: str, filename: str, raw_bytes: bytes, tissue: str) -> None:
    def progress(msg: str) -> None:
        _set_job(job_id, status="running", message=msg)

    try:
        result = pipeline.run_pipeline(filename, raw_bytes, tissue=tissue, progress=progress)
        _set_job(job_id, status="done", message="Done.", result=result)
    except pipeline.PipelineError as e:
        _set_job(job_id, status="error", message=str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set_job(job_id, status="error", message=f"{type(e).__name__}: {e}")


@app.post("/api/classify")
async def classify(file: UploadFile = File(...), tissue: str = Form("breast")) -> dict:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = JobStatus(status="queued", message="Queued.")

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, file.filename or "upload.maf", raw_bytes, tissue),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


class ExplainRequest(BaseModel):
    summary: dict
    flagged: list[dict]
    tissue: str = "breast"


@app.post("/api/explain")
def explain(body: ExplainRequest) -> dict:
    try:
        explanations = ai_explain.generate_explanation(body.summary, body.flagged, body.tissue)
    except ai_explain.AiExplainError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"explanations": explanations}


# ---------------------------------------------------------------------------
# Serve the built frontend, if present, so `uvicorn main:app` is the only
# process needed in production. In development, run the Vite dev server
# separately and this block is simply inert (dist/ won't exist yet).
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/favicon.svg")
    def favicon() -> FileResponse:
        return FileResponse(os.path.join(FRONTEND_DIST, "favicon.svg"))

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
