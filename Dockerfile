# Docker image for the VUS Reclassification webapp — built for Render's free
# tier (or anywhere else that'll take a plain Docker image). Bakes in
# everything the app needs to run standalone: the built React frontend, the
# FastAPI backend, the ANNOVAR install + hg38 reference databases, and the
# trained model artifacts. Nothing is downloaded at container start, so a
# cold start after Render's free-tier idle spin-down doesn't depend on the
# network being fast or even up — the only thing that gets rebuilt on wake
# is the small SQLite lookup cache pipeline_app.py builds itself on first
# request, straight from the watchlist CSVs already sitting in the image.
#
# Layer order below is deliberate: everything that almost never changes
# (system deps, Python deps, the multi-GB reference data) comes first, and
# the actual application code — the thing you're actually iterating on —
# comes last. Docker's build cache is a straight chain: invalidate one
# layer and every layer after it rebuilds too, even if its own contents
# didn't change. Put a frequently-edited file early and you pay to
# re-copy 4GB of ANNOVAR data on every single code tweak. Put it last and
# editing pipeline_app.py only rebuilds that one small layer.
#
# Build from the project root (the folder that contains this file, code/,
# webapp/, final_output_csv/):
#
#   docker build --platform linux/amd64 -t <your-dockerhub-username>/vus-reclassification:latest .
#
# See DEPLOY.md for the full walkthrough (build, push, Render setup).

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY webapp/frontend/package*.json ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build

# ---- Stage 2: the actual runtime image ----
FROM python:3.12-slim

# perl is the one system dependency: table_annovar.pl is a real Perl script,
# not something pip can install. This basically never changes, so it stays
# first.
RUN apt-get update \
    && apt-get install -y --no-install-recommends perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies next — only changes when requirements-docker.txt does.
COPY webapp/backend/requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Trained model artifacts + lookups. code/data/stage2_deploy/ is the curated
# subset of code/data/stage2/ that the deployed app actually reads at
# runtime (the full folder also holds several GB of raw research CSVs and
# old model versions that have no business being in a deploy image — see
# code/data/stage2_deploy/ itself, or ask Claude, for how it was built).
# These essentially never change between rebuilds, so they go early too.
COPY code/data/tcga_training/ ./code/data/tcga_training/
COPY code/data/hpa/proteinatlas.tsv ./code/data/hpa/proteinatlas.tsv
COPY code/data/stage2_deploy/ ./code/data/stage2/

# The ClinVar reclassification watchlist (the ~2.3M-row source that
# pipeline_app.py's SQLite index gets built from on first request).
COPY final_output_csv/ ./final_output_csv/

# ANNOVAR itself (perl scripts) plus the hg38 humandb/ reference databases
# (refGeneWithVer, clinvar_20221231, gnomad211_exome). This is the biggest
# thing in the image by far, a few GB, there's no way around that, it's
# what the annotation step actually needs on disk to run. Also essentially
# static, so it stays ahead of the application code below.
COPY code/annovar/ ./code/annovar/

# The frontend build from stage 1. Changes only when you touch the React
# app, which is much rarer than backend Python edits — still goes ahead of
# the backend code for that reason.
COPY --from=frontend-build /app/frontend/dist ./webapp/frontend/dist

# The real scoring/annotation code pipeline_app.py imports by name. Changes
# occasionally, more often than the above, less often than backend/.
COPY code/scripts/ ./code/scripts/
COPY code/config/ ./code/config/

# Backend application code (pipeline_app.py, main.py, ai_explain.py, ...) —
# the thing you're actually iterating on right now. Last on purpose: this
# is the layer (and only this layer) that gets invalidated when you edit
# these files, so a rebuild after a code change reuses every cached layer
# above and just re-copies ~250KB instead of ~4GB.
COPY webapp/backend/ ./webapp/backend/

WORKDIR /app/webapp/backend
ENV PYTHONUNBUFFERED=1
EXPOSE 8765

# Render (and most PaaS Docker runners) inject a $PORT env var and expect
# the container to bind to it rather than a fixed port, so this falls back
# to 8765 for a plain `docker run` locally but honors $PORT when it's set.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8765}"]
