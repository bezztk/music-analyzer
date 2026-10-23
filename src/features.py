"""Classical audio descriptors. Computations are timed individually by analyzer.py."""
import numpy as np
import essentia.standard as es

SAMPLE_RATE = 44100


def _round(x):
    v = float(x)
    return round(v, 5) if np.isfinite(v) else None


def computations(signal):
    """Return name->callable to keep each calculation independently measurable.

    Some descriptors are frame-level and use the mean across frames. These are
    *descriptors*, not model-estimated emotional scores.
    """
    def beats():
        bpm, _, _, _, _ = es.RhythmExtractor2013(method="multifeature")(signal)
        return _round(bpm)

    def key():
        tonic, scale, strength = es.KeyExtractor(sampleRate=SAMPLE_RATE)(signal)
        return {"key": tonic, "scale": scale, "strength": _round(strength)}

    def rms():
        return _round(es.RMS()(signal))

    def peak():
        return _round(np.max(np.abs(signal)))

    def zero_crossing_rate():
        return _round(es.ZeroCrossingRate()(signal))

    def dynamic_complexity():
        value, loudness = es.DynamicComplexity(sampleRate=SAMPLE_RATE)(signal)
        return {"complexity": _round(value), "loudness": _round(loudness)}

    def silence_rates():
        detector = es.SilenceRate(thresholds=[10 ** (-20 / 10), 10 ** (-40 / 10)])
        rates = [detector(frame) for frame in es.FrameGenerator(signal, frameSize=2048, hopSize=1024)]
        if not rates:
            return {"below_-20db": None, "below_-40db": None}
        values = np.mean(np.asarray(rates), axis=0)
        return {"below_-20db": _round(values[0]), "below_-40db": _round(values[1])}

    def onset_rate():
        return _round(es.OnsetRate()(signal)[1])

    def spectral():
        window = es.Windowing(type="hann")
        spectrum = es.Spectrum()
        centroid = es.Centroid(range=SAMPLE_RATE / 2)
        rolloff = es.RollOff(sampleRate=SAMPLE_RATE)
        complexity = es.SpectralComplexity(sampleRate=SAMPLE_RATE)
        flux = es.Flux()
        flatness = es.Flatness()
        entropy = es.Entropy()
        mfcc = es.MFCC(numberCoefficients=13)
        melbands = es.MelBands(numberBands=24, sampleRate=SAMPLE_RATE)
        results = {x: [] for x in ("centroid_hz", "rolloff_hz", "complexity", "flux", "flatness", "entropy")}
        mfccs, mels = [], []
        # 2048 window, 1024 hop; limit computation cost by examining at most ~2000 frames.
        n = max(1, (len(signal) - 2048) // 1024)
        skip = max(1, n // 2000)
        for frame in es.FrameGenerator(signal, frameSize=2048, hopSize=1024 * skip, startFromZero=True):
            spec = spectrum(window(frame))
            results["centroid_hz"].append(centroid(spec))
            results["rolloff_hz"].append(rolloff(spec))
            results["complexity"].append(complexity(spec))
            results["flux"].append(flux(spec))
            results["flatness"].append(flatness(spec))
            results["entropy"].append(entropy(spec))
            _, coeffs = mfcc(spec)
            mfccs.append(coeffs)
            mels.append(melbands(spec))
        out = {k: _round(np.mean(v)) for k, v in results.items()}
        out["mfcc_mean"] = [_round(x) for x in np.mean(mfccs, axis=0)]
        out["melbands_mean"] = [_round(x) for x in np.mean(mels, axis=0)]
        out["frames_analyzed"] = len(mfccs)
        return out

    return {
        "bpm": beats,
        "key": key,
        "rms": rms,
        "peak_amplitude": peak,
        "zero_crossing_rate": zero_crossing_rate,
        "dynamic_complexity": dynamic_complexity,
        "silence_rate": silence_rates,
        "onset_rate": onset_rate,
        "spectrum": spectral,
    }
