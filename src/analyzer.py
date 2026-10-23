import essentia.standard as es

MAX_DURATION = 180
SAMPLE_RATE = 44100


def analyze(path: str):
    signal = es.MonoLoader(
        filename=path,
        sampleRate=SAMPLE_RATE,
        endTime=MAX_DURATION
    )()

    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, confidence, estimates, intervals = rhythm(signal)

    return {
        "duration": round(len(signal) / SAMPLE_RATE, 2),
        "bpm": round(float(bpm), 2)
    }