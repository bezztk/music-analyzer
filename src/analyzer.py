import essentia.standard as es

MAX_DURATION = 180
SAMPLE_RATE = 44100


def analyze(path: str):
    signal = es.EasyLoader(
        filename=path,
        sampleRate=SAMPLE_RATE,
        startTime=0,
        endTime=MAX_DURATION
    )()

    # BPM erkennen
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, confidence, estimates, intervals = rhythm(signal)

    # Tonart erkennen
    key_extractor = es.KeyExtractor(sampleRate=SAMPLE_RATE)
    key, scale, strength = key_extractor(signal)

    return {
        "duration": round(len(signal) / SAMPLE_RATE, 2),
        "bpm": round(float(bpm), 2),
        "key": key,
        "scale": scale,
        "key_strength": round(float(strength), 3)
    }