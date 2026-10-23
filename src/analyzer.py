import time
from pathlib import Path
from tracemalloc import start
import json
import numpy as np

from essentia import Pool
import essentia.standard as es

MAX_DURATION = 180

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Modelle einmalig initialisieren

maest = es.TensorflowPredictMAEST(
    graphFilename=str(MODEL_DIR / "discogs-maest-30s-pw-519l-2.pb"),
    output="PartitionedCall/Identity_12"
)

musicnn = es.TensorflowPredictMusiCNN(
    graphFilename=str(MODEL_DIR / "msd-musicnn-1.pb"),
    output="model/dense/BiasAdd"
)

effnet = es.TensorflowPredictEffnetDiscogs(
    graphFilename=str(MODEL_DIR / "discogs-effnet-bs64-1.pb"),
    output="PartitionedCall:1"
)

GENRE_MODEL = "genre_discogs519-discogs-maest-30s-pw-519l-1"

genre_classifier = es.TensorflowPredict(
    graphFilename=str(MODEL_DIR / f"{GENRE_MODEL}.pb"),
    inputs=["embeddings"],
    outputs=["PartitionedCall/Identity_1"]
)

with open(MODEL_DIR / f"{GENRE_MODEL}.json", encoding="utf-8") as file:
    genre_labels = json.load(file)["classes"]

def analyze(path: str):
    timings = {}
    start = time.perf_counter()

    signal = es.EasyLoader(
        filename=path,
        sampleRate=44100,
        startTime=0,
        endTime=180
    )()

    timings["loading"] = round(time.perf_counter() - start, 3)

    start = time.perf_counter()

    audio_ml = es.Resample(
        inputSampleRate=44100,
        outputSampleRate=16000
    )(signal)

    timings["resampling"] = round(time.perf_counter() - start, 3)
    start = time.perf_counter()

    # BPM
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, confidence, estimates, intervals = rhythm(signal)
    
    timings["bpm"] = round(time.perf_counter() - start, 3)
    start = time.perf_counter()
    
    # Tonart
    key_extractor = es.KeyExtractor(sampleRate=44100)
    key, scale, strength = key_extractor(signal)

    timings["key"] = round(time.perf_counter() - start, 3)
    start = time.perf_counter()

    # MAEST
    start = time.perf_counter()
    maest_embeddings = maest(audio_ml)
    timings["maest"] = round(time.perf_counter() - start, 3)

    # Discogs-Genre-Klassifikation
    start = time.perf_counter()
    pool = Pool()
    pool.set("embeddings", maest_embeddings)

    predictions = genre_classifier(pool)["PartitionedCall/Identity_1"]

    # Ergebnisse über alle analysierten Abschnitte mitteln
    predictions = np.asarray(predictions)
    print(f"Predictions shape: {predictions.shape}", flush=True)
    scores = predictions.reshape(-1, len(genre_labels)).mean(axis=0)

    genres = {
        label: round(float(score), 4)
        for label, score in zip(genre_labels, scores)
    }
    genres = dict(
        sorted(genres.items(), key=lambda item: item[1], reverse=True)[:20]
    )
    timings["discogs519"] = round(time.perf_counter() - start, 3)

    # MusiCNN
    start = time.perf_counter()
    musicnn_embeddings = musicnn(audio_ml)
    timings["musicnn"] = round(time.perf_counter() - start, 3)

    # Discogs-EffNet
    start = time.perf_counter()
    effnet_embeddings = effnet(audio_ml)
    timings["effnet"] = round(time.perf_counter() - start, 3)

    return {
        "duration": round(len(signal) / 44100, 2),
        "bpm": round(float(bpm), 2),
        "key": key,
        "scale": scale,
        "key_strength": round(float(strength), 3),

        "embeddings": {
            "maest": list(maest_embeddings.shape),
            "musicnn": list(musicnn_embeddings.shape),
            "effnet": list(effnet_embeddings.shape)
        },

        "genres": genres,
        "timings": timings
    }