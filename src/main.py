"""Single-worker upload API; 1 active + up to 2 waiting analyses."""
import asyncio
import os
import tempfile
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from analyzer import analyze

app = FastAPI(title="Music Analyzer")
MAX_ACTIVE_REQUESTS = 3
_request_guard = threading.Lock()
_requests = 0
_analysis_lock = asyncio.Lock()

@app.post("/analyze")
async def analyze_track(file: UploadFile = File(...)):
    global _requests
    with _request_guard:
        if _requests >= MAX_ACTIVE_REQUESTS:
            raise HTTPException(status_code=503, detail="Analyzer busy; try again later")
        _requests += 1
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as target:
            path = target.name
            while chunk := await file.read(1024 * 1024):
                target.write(chunk)
        async with _analysis_lock:
            try:
                result = await asyncio.to_thread(analyze, path)
            except Exception as exc:
                # Return generic error to clients; exception details in logs.
                import logging
                logging.exception("Analysis failed")
                raise HTTPException(status_code=422, detail="Audio could not be analyzed") from exc
        return {"filename": file.filename, "analysis": result}
    finally:
        await file.close()
        if path and os.path.exists(path):
            os.remove(path)
        with _request_guard:
            _requests -= 1
