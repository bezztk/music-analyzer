import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from analyzer import analyze

app = FastAPI()


@app.post("/analyze")
async def analyze_track(file: UploadFile = File(...)):
    path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as temp:
            path = temp.name

            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        result = analyze(path)

        return {
            "filename": file.filename,
            "analysis": result
        }

    finally:
        await file.close()

        if path and os.path.exists(path):
            os.remove(path)