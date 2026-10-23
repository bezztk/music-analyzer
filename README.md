# Music Analyzer (extended prototype)

This is a modular prototype for Essentia-based music analysis. It is intended
for local/internal use, not for untrusted public upload traffic.

## Setup on Windows (PowerShell)

```powershell
python download_models.py
docker compose up -d --build
docker compose logs -f
```

Visit `http://localhost:7200/docs` and upload an audio file to `POST /analyze`.

The API returns `features`, `predictions`, `timings` (seconds) and `errors`.
An individual optional model or feature failure is reported in `errors`, without
hiding successful calculations. Model loading errors also appear there.
The `total` timing covers the Python analysis, **not** the upload/waiting time.
The audio is capped at the first 180 seconds during decoding.

## Scope / limitations

* 3 shared embeddings, up to 23 classification heads, signal features.
* Assumes a single Uvicorn worker. Shared ML model objects are used sequentially.
* 1 active analysis plus 2 queued; uploads may be parsed by FastAPI first.
* Many head configurations depend on the Essentia package version. Exact model
  availability and output tensor names must be tested with the installed version.
* Some Essentia features are version-dependent. They fail independently and are
  logged under `errors`. Do not interpret feature errors as zero values.
* Frame-based spectra are sampled at intervals for speed; these means are
  estimates, not exhaustive full-resolution analyses.
* Regression model scores are not calibrated probabilities.
* No file size limit, user authentication, or upload pre-validation is implemented.

## What is not implemented yet

Per-beat loudness, full chord sequence/changes, HPCP aggregation, LUFS/LRA,
GFCC, Bark bands, pitch salience and detailed dissonance are not included in
this prototype. They need additional verified Essentia pipelines to avoid
misleading values. No OpenL3/VGGish embedding is introduced.
