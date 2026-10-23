from pathlib import Path

import essentia.standard as es

MAX_DURATION = 180

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def analyze(path: str):

    # Audio für klassische Signalverarbeitung
    signal = es.EasyLoader(
        filename=path,
        sampleRate=44100,
        startTime=0,
        endTime=MAX_DURATION
    )()

    # Audio für neuronale Netzwerke
    audio_ml = es.EasyLoader(
        filename=path,
        sampleRate=16000,
        startTime=0,
        endTime=MAX_DURATION
    )()

    # BPM
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, confidence, estimates, intervals = rhythm(signal)

    # Tonart
    key_extractor = es.KeyExtractor(sampleRate=44100)
    key, scale, strength = key_extractor(signal)

    # MAEST
    maest = es.TensorflowPredictMAEST(
        graphFilename=str(MODEL_DIR / "discogs-maest-30s-pw-519l-2.pb"),
        output="PartitionedCall/Identity_7"
    )

    maest_embeddings = maest(audio_ml)

    # MusiCNN
    musicnn = es.TensorflowPredictMusiCNN(
        graphFilename=str(MODEL_DIR / "msd-musicnn-1.pb"),
        output="model/dense/BiasAdd"
    )

    musicnn_embeddings = musicnn(audio_ml)

    # Discogs-EffNet
    effnet = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(MODEL_DIR / "discogs-effnet-bs64-1.pb"),
        output="PartitionedCall:1"
    )

    effnet_embeddings = effnet(audio_ml)

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
        }
    }