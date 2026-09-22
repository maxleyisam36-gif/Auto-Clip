# autoclip — web app

A simple web UI around the clipping pipeline: paste a link, watch progress,
preview and download clips when ready. Single-user by design (jobs are kept
in memory) — good for personal use.

## Run it locally first

You need **Python 3.10+** and **ffmpeg** installed (see the main README in
the parent folder for ffmpeg install commands).

```bash
cd web
pip install -r requirements.txt

# optional, for better clip selection:
export ANTHROPIC_API_KEY=sk-ant-...

uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, paste a video URL, set your options, hit
"Create clips." Status updates live; finished clips preview inline with a
download link.

## Deploying so it's reachable from anywhere

This app does real video/audio processing, so it needs a host that allows
**long-running background work** — not a typical serverless/edge platform.

**Recommended: Render or Railway, using the included Dockerfile**
1. Push this `web/` folder to a GitHub repo
2. On Render or Railway: "New Web Service" → connect the repo → it will
   detect the `Dockerfile` and build automatically
3. Set the `ANTHROPIC_API_KEY` environment variable in the platform's
   dashboard (optional but recommended)
4. Deploy — you'll get a public URL

The Dockerfile is there specifically because most platforms' plain Python
buildpacks don't include `ffmpeg`; Docker guarantees it's present.

**Note on cost/sizing:** transcription and video encoding are CPU-heavy.
Start with a mid-tier instance (not the free/smallest tier) or jobs will be
slow or time out. `--whisper-model tiny` is fastest if you need to keep
compute costs down.

## Known limitations of this version

- Jobs live in memory — restarting the server loses job history (finished
  clip *files* stay on disk in `job_output/`, but the UI won't know about
  them until you build a job-listing endpoint)
- Single server instance only (no multi-worker/queue system yet — fine for
  personal use, would need Redis/Celery to scale to many concurrent users)
- No auth — anyone with the URL can submit jobs. Fine for personal/private
  use; add a login step before sharing the link publicly
- No auto-posting to social platforms yet (separate, later step — each
  platform requires its own API approval)
