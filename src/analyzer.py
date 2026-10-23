"""Orchestrate one 180-second analysis with timings for every step."""
import time
import numpy as np
import essentia.standard as es
from embeddings import Embeddings
from classifiers import Classifiers, HEADS
from features import computations

MAX_DURATION = 180
SAMPLE_RATE = 44100

# Eagerly initialized once. One-Uvicorn-worker and lock in main.py are required
# for safe sequential reuse of these model instances.
EMBEDDINGS = Embeddings()
CLASSIFIERS = Classifiers()


def analyze(path: str):
    started = time.perf_counter()
    timings, errors, features, ml = {}, {}, {}, {}

    def measure(name, fn, destination, *, fatal=False):
        t = time.perf_counter()
        try:
            value = fn()
            destination[name] = value
            return value
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            if fatal:
                raise
            return None
        finally:
            timings[name] = round(time.perf_counter() - t, 3)

    signal = measure("loading", lambda: es.EasyLoader(
        filename=path, sampleRate=SAMPLE_RATE, startTime=0,
        endTime=MAX_DURATION)(), {}, fatal=True)
    if signal is None or len(signal) < 2048:
        raise ValueError("Audio is too short or could not be decoded")
    duration = round(len(signal) / SAMPLE_RATE, 3)

    audio_ml = measure("resampling", lambda: es.Resample(
        inputSampleRate=SAMPLE_RATE, outputSampleRate=16000)(signal), {}, fatal=True)

    for name, fn in computations(signal).items():
        measure(f"feature_{name}", fn, features)

    shared = {}
    for name in ("maest", "musicnn", "effnet"):
        vector = measure(
            f"embedding_{name}",
            lambda n=name: EMBEDDINGS.compute(n, audio_ml),
            {}
        )
        if vector is not None:
            shared[name] = vector
            ml[f"{name}_shape"] = list(np.asarray(vector).shape)

    for key, _group, _filename, embedding_name, _top_k in HEADS:
        if key not in CLASSIFIERS.models:
            errors[f"classifier_{key}"] = CLASSIFIERS.errors.get(key, "not initialized")
            continue
        if embedding_name not in shared:
            errors[f"classifier_{key}"] = f"Embedding {embedding_name} missing"
            continue
        measure(f"classifier_{key}", lambda k=key, n=embedding_name:
                CLASSIFIERS.evaluate(k, shared[n]), ml)

    timings["total"] = round(time.perf_counter() - started, 3)
    return {"duration": duration, "features": features, "predictions": ml,
            "timings": timings, "errors": errors}
