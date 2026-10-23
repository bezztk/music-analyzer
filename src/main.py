from fastapi import FastAPI, UploadFile, File
from analyzer import analyze

app = FastAPI()

@app.post("/analyze")
async def analyze_track(file: UploadFile = File(...)):
    content = await file.read()
    result = analyze(content)

    return {
        "filename": file.filename,
        "analysis": result
    }