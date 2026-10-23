import time
import essentia.standard as es

MAX_DURATION = 180
SAMPLE_RATE = 44100


def analyze(path: str):
    start = time.perf_counter()

    signal = es.EasyLoader(
        filename=path,
        sampleRate=SAMPLE_RATE,
        startTime=0,
        endTime=MAX_DURATION
    )()

    load_time = time.perf_counter() - start

    print(f"Geladene Dauer: {len(signal) / SAMPLE_RATE:.2f}s", flush=True)
    print(f"Ladezeit: {load_time:.2f}s", flush=True)

    start = time.perf_counter()

    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, confidence, estimates, intervals = rhythm(signal)

    analysis_time = time.perf_counter() - start

    print(f"Analysezeit: {analysis_time:.2f}s", flush=True)

    return {
        "duration": round(len(signal) / SAMPLE_RATE, 2),
        "bpm": round(float(bpm), 2)
    }