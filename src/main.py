import os
import tempfile
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from analyzer import analyze

app = FastAPI()

analysis_lock = asyncio.Lock()
request_slots = asyncio.Semaphore(5)


@app.post("/analyze")
async def analyze_track(file: UploadFile = File(...)):
    if request_slots.locked():
        raise HTTPException(
            status_code=503,
            detail="Analyzer is busy. Try again later."
        )

    await request_slots.acquire()
    path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as temp:
            path = temp.name

            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        async with analysis_lock:
            result = await asyncio.to_thread(analyze, path)

        return {
            "filename": file.filename,
            "analysis": result
        }

    finally:
        request_slots.release()
        await file.close()

        if path and os.path.exists(path):
            os.remove(path)