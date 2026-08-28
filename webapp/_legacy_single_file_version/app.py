#!/usr/bin/env python3
"""
VUS Classifier web app — local server, stdlib only (no Flask/FastAPI needed;
none is installed in stage1/venv). Serves the frontend in static/ and a
JSON API at /api/classify that runs the real Stage 1 + Stage 2 pipeline
(pipeline.py) against an uploaded MAF/VCF file, using the project's own
trained model weights in ../data/.

Run with the project's own virtualenv, from inside this webapp/ folder:

    ../venv/bin/python app.py

Then open http://localhost:8765 in a browser. Everything runs locally;
the only outbound network call the pipeline makes is to the public Ensembl
VEP REST API (rest.ensembl.org) for ClinVar/gnomAD/COSMIC/dbSNP annotation.
"""
import json
import os
import re
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(WEBAPP_DIR, "static")
PORT = int(os.environ.get("PORT", "8765"))

sys.path.insert(0, WEBAPP_DIR)
import pipeline  # noqa: E402

# In-memory job store: uploads run in a background thread so the browser can
# poll progress instead of sitting on one long blocking request.
JOBS = {}
JOBS_LOCK = threading.Lock()


def _set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _run_job(job_id, filename, raw_bytes, tissue):
    def progress(msg):
        _set_job(job_id, status="running", message=msg)

    try:
        result = pipeline.run_pipeline(filename, raw_bytes, tissue=tissue, progress=progress)
        _set_job(job_id, status="done", result=result, message="Done.")
    except pipeline.PipelineError as e:
        _set_job(job_id, status="error", message=str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _set_job(job_id, status="error", message=f"{type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    server_version = "VUSClassifier/1.0"

    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        if not os.path.exists(path):
            self.send_error(404, "Not found")
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        elif self.path.startswith("/api/status/"):
            job_id = self.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "unknown job id"}, status=404)
            else:
                self._send_json(job)
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if self.path != "/api/classify":
            self.send_error(404, "Not found")
            return
        content_type = self.headers.get("Content-Type", "")
        m = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
        if "multipart/form-data" not in content_type or not m:
            self._send_json({"error": "expected multipart/form-data upload"}, status=400)
            return
        boundary = (m.group(1) or m.group(2)).strip().encode("utf-8")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            fields, files = _parse_multipart(body, boundary)
        except Exception:
            self._send_json({"error": "could not parse upload"}, status=400)
            return

        if "file" not in files:
            self._send_json({"error": "no file uploaded"}, status=400)
            return
        filename, raw_bytes = files["file"]
        tissue = fields.get("tissue", "breast") or "breast"

        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "queued", "message": "Queued."}
        t = threading.Thread(target=_run_job, args=(job_id, filename, raw_bytes, tissue), daemon=True)
        t.start()
        self._send_json({"job_id": job_id})


def _parse_multipart(body: bytes, boundary: bytes):
    """Minimal multipart/form-data parser (stdlib `cgi` module was removed
    in Python 3.13, so this is hand-rolled rather than relying on it)."""
    delimiter = b"--" + boundary
    parts = body.split(delimiter)
    fields = {}
    files = {}
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        data = data.rstrip(b"\r\n")
        headers = head.decode("utf-8", errors="replace")
        name_match = re.search(r'name="([^"]+)"', headers)
        if not name_match:
            continue
        field_name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', headers)
        if filename_match:
            files[field_name] = (filename_match.group(1) or "upload.maf", data)
        else:
            fields[field_name] = data.decode("utf-8", errors="replace")
    return fields, files


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"VUS Classifier running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
