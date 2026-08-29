# Deploying to Render (free tier)

This builds one Docker image containing the built frontend, the FastAPI
backend, ANNOVAR + its hg38 reference databases, and the trained model
artifacts — everything the app needs, nothing downloaded at runtime.

Render's Docker builder pulls from a git repo, and this image bakes in a
few GB of reference data that has no business living in git (it's already
excluded in `.gitignore`, for good reason). So instead of connecting
Render to GitHub, you build the image yourself on your Mac and push it to
Docker Hub as a plain public image, then point Render at that image
directly ("Deploy an existing image" instead of "Build from a repo").
That sidesteps GitHub entirely for the big files.

Everything below runs in your own Mac Terminal, not through Claude — Docker
Desktop and your Docker Hub login live there, not in this session.

## 1. Install Docker Desktop, if you don't have it

https://www.docker.com/products/docker-desktop/ — install, open it once so
the daemon is running, sign in (or create a free Docker Hub account at the
same time — you'll need one for step 3).

## 2. Build the image

From the project root (the folder with `Dockerfile` in it):

```
cd "/Users/guest2/Desktop/Persistent Systems"
docker build -t YOUR_DOCKERHUB_USERNAME/vus-reclassification:latest .
```

This will take a while the first time — it's copying a few GB of ANNOVAR
reference data into the image and building the React frontend. Grab a
coffee. Subsequent builds are much faster since Docker caches layers that
haven't changed.

## 3. Push it to Docker Hub

```
docker login
docker push YOUR_DOCKERHUB_USERNAME/vus-reclassification:latest
```

This uploads the whole image (~4GB), so it'll take a while depending on
your upload speed. It's a one-time cost per version — Render only pulls
the image when you deploy or redeploy, not every time the free instance
wakes from being idle.

## 4. Create the Render service

1. Go to https://dashboard.render.com → New → Web Service
2. Choose "Deploy an existing image" (not "Build and deploy from a Git
   repository")
3. Image URL: `docker.io/YOUR_DOCKERHUB_USERNAME/vus-reclassification:latest`
4. Instance type: Free
5. Under Environment, add: `GEMINI_API_KEY` = (the value from your local
   `webapp/backend/.env` — it's gitignored and wasn't baked into the
   image on purpose, so this is the only place it needs to be set)
6. Create Web Service

Render will pull the image and start the container. First boot will take
a bit longer than usual — the app builds its SQLite watchlist/HPA lookup
caches from the CSVs baked into the image the first time anything queries
them (this replaces the old "load 2.3M rows into a Python dict at
startup" approach, see pipeline_app.py). After that first request, it's
fast.

## What to expect on the free tier

- The service spins down after 15 minutes with no incoming requests, and
  takes roughly a minute to wake back up on the next request. It's not
  instant-always-on, but it never depends on your laptop being on either.
- Free RAM is 512MB. The SQLite optimization brings steady-state usage
  down a lot, but it's genuinely tight once pandas/numpy/xgboost/ANNOVAR
  are all loaded — if you see the service repeatedly restarting/crashing
  under real use, that's very likely an out-of-memory kill, and the fix
  at that point is either Render's paid Starter tier ($7/mo, same 512MB
  actually — you'd want Standard) or moving to something with more
  headroom (AWS EC2's 12-month free tier gives you 1GB with real room to
  spare, at the cost of a 12-month free clock and requiring a card).
- No persistent disk on the free tier, which sounds worse than it is
  here: everything the app needs is already baked into the image, so
  nothing needs to persist. The only thing that's rebuilt on every
  restart/wake is the small SQLite cache, and that rebuild reads from
  data that's already sitting in the image — no network round-trip, no
  re-download.

## Redeploying after a code change

Rebuild, push, then in the Render dashboard hit "Manual Deploy" → "Deploy
latest image" (or just re-push the same tag and Render can pick it up
depending on your deploy settings). No need to touch anything else.
