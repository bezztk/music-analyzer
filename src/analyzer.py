import time
from pathlib import Path
from tracemalloc import start

import essentia.standard as es

MAX_DURATION = 180

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


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
    maest = es.TensorflowPredictMAEST(
        graphFilename=str(MODEL_DIR / "discogs-maest-30s-pw-519l-2.pb"),
        output="PartitionedCall/Identity_7"
    )

    maest_embeddings = maest(audio_ml)

    timings["maest"] = round(time.perf_counter() - start, 3)
    start = time.perf_counter()

    # MusiCNN
    musicnn = es.TensorflowPredictMusiCNN(
        graphFilename=str(MODEL_DIR / "msd-musicnn-1.pb"),
        output="model/dense/BiasAdd"
    )

    musicnn_embeddings = musicnn(audio_ml)

    timings["musicnn"] = round(time.perf_counter() - start, 3)
    start = time.perf_counter()
    # Discogs-EffNet
    effnet = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(MODEL_DIR / "discogs-effnet-bs64-1.pb"),
        output="PartitionedCall:1"
    )

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

        "timings": timings
    }