# Music Analyzer

Self-hosted audio analysis API built with Python, FastAPI and Essentia.

Analyzes the first **180 seconds** of an audio file.

### Features

* **Genres:** Discogs519, MTG-Jamendo, Electronic
* **Mood:** MIREX, DEAM, emoMusic and additional classifiers
* **Instruments:** Instrument and vocal detection
* **Audio:** BPM, key, loudness, dynamics and spectral features
* **Diagnostics:** Individual execution times and error reporting

### Setup

```bash
git clone https://github.com/bezztk/music-analyzer.git
cd music-analyzer
python3 download_models.py
docker compose up -d --build
```

### API

`POST /analyze` — Multipart audio upload (`file`)

```bash
curl -F "file=@track.mp3" http://localhost:7200/analyze
```

API documentation: `http://localhost:7200/docs`

### Stack

Python 3.11 · FastAPI · Essentia · TensorFlow · Docker