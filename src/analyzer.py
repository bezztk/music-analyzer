import os
import tempfile

import essentia.standard as es

def analyze(audio: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as file:
        file.write(audio)
        path = file.name

    try:
        signal = es.MonoLoader(filename=path, sampleRate=44100)()

        rhythm = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, confidence, estimates, intervals = rhythm(signal)

        return {"bpm": round(float(bpm),2)}
    finally:
        os.remove(path)
