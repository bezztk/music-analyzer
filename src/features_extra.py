"""Additional signal descriptors. Each callable is independently timed in analyzer.py.

All measurements refer to the first MAX_DURATION seconds of decoded mono audio.
"""
from collections import Counter
import numpy as np
import essentia.standard as es

SAMPLE_RATE = 44100


def _num(value):
    x = float(value)
    return round(x, 5) if np.isfinite(x) else None


def _vector(values):
    return [_num(x) for x in values]


def _frames(signal, frame_size=4096, hop_size=2048, max_frames=1200):
    count = max(1, (len(signal) - frame_size) // hop_size + 1)
    stride = max(1, int(np.ceil(count / max_frames)))
    return es.FrameGenerator(signal, frameSize=frame_size,
                             hopSize=hop_size * stride, startFromZero=True)


def computations(signal):
    def loudness_ebu_r128():
        # Essentia's EBU R128 algorithm requires stereo. The existing input
        # has already been downmixed to mono; duplicating it approximates
        # stereo but MUST NOT be mistaken for original-stereo measurement.
        stereo = np.column_stack((signal, signal)).astype(np.float32)
        _momentary, _short, integrated, loudness_range = es.LoudnessEBUR128(
            sampleRate=SAMPLE_RATE)(stereo)
        return {"integrated_lufs": _num(integrated),
                "loudness_range_lu": _num(loudness_range),
                "method": "dual_mono_proxy_not_original_stereo"}

    # def tuning():
    #     peaks = es.SpectralPeaks(sampleRate=SAMPLE_RATE, minFrequency=40,
    #                              maxFrequency=5000, maxPeaks=100)
    #     spectrum = es.Spectrum(size=4096)
    #     window = es.Windowing(type="hann")
    #     detector = es.TuningFrequency()
    #     hz, cents = [], []
    #     for frame in _frames(signal):
    #         freqs, mags = peaks(spectrum(window(frame)))
    #         if len(freqs) < 3:
    #             continue
    #         frequency, offset = detector(freqs, mags)
    #         if np.isfinite(frequency) and 400 <= frequency <= 480:
    #             hz.append(float(frequency))
    #             cents.append(float(offset))
    #     return {"reference_frequency_hz": _num(np.median(hz)) if hz else None,
    #             "offset_from_440_cents": _num(np.median(cents)) if cents else None,
    #             "frames_used": len(hz)}

    # def harmonic_features():
    #     window = es.Windowing(type="hann")
    #     spectrum = es.Spectrum(size=4096)
    #     peaks = es.SpectralPeaks(sampleRate=SAMPLE_RATE, minFrequency=40,
    #                              maxFrequency=5000, maxPeaks=100, orderBy="frequency")
    #     chroma = es.HPCP(size=12, sampleRate=SAMPLE_RATE)
    #     dissonance = es.Dissonance()
    #     hpcps, roughness = [], []
    #     for frame in _frames(signal):
    #         freqs, mags = peaks(spectrum(window(frame)))
    #         if len(freqs) < 2:
    #             continue
    #         hpcps.append(np.asarray(chroma(freqs, mags), dtype=np.float32))
    #         roughness.append(float(dissonance(freqs, mags)))
    #     if not hpcps:
    #         return {"hpcp_mean": None, "sensory_dissonance": None,
    #                 "chords": None, "frames_used": 0}
    #     pcp = np.asarray(hpcps, dtype=np.float32)
    #     # A chord estimate per subsampled frame; these are rough hypotheses.
    #     chords, strengths = es.ChordsDetection(sampleRate=SAMPLE_RATE,
    #                                            hopSize=2048)(pcp)
    #     common = Counter(str(x) for x in chords).most_common(12)
    #     switches = sum(a != b for a, b in zip(chords[:-1], chords[1:]))
    #     return {"hpcp_mean": _vector(np.mean(pcp, axis=0)),
    #             "sensory_dissonance": _num(np.mean(roughness)),
    #             "chords": {"top": [{"chord": label, "count": count}
    #                                for label, count in common],
    #                        "changes_count": switches,
    #                        "changes_per_frame": _num(switches / max(1, len(chords)-1)),
    #                        "mean_strength": _num(np.mean(strengths)),
    #                        "method": "experimental_triad_detection"},
    #             "frames_used": len(hpcps)}

    def spectral_extra():
        window = es.Windowing(type="hann")
        spectrum = es.Spectrum(size=2048)
        gfcc = es.GFCC(inputSize=1025, highFrequencyBound=20000)
        contrast = es.SpectralContrast(sampleRate=SAMPLE_RATE, frameSize=2048)
        features_gfcc, features_contrast, features_valleys = [], [], []
        for frame in _frames(signal, frame_size=2048, hop_size=1024, max_frames=1800):
            spec = spectrum(window(frame))
            _, coeffs = gfcc(spec)
            contrasts, valleys = contrast(spec)
            features_gfcc.append(coeffs)
            features_contrast.append(contrasts)
            features_valleys.append(valleys)
        return {"gfcc_mean": _vector(np.mean(features_gfcc, axis=0)),
                "spectral_contrast_mean": _vector(np.mean(features_contrast, axis=0)),
                "spectral_valleys_mean": _vector(np.mean(features_valleys, axis=0)),
                "frames_used": len(features_gfcc)}

    return {"loudness_ebu_r128": loudness_ebu_r128,
            # "tuning_frequency": tuning,
            # "harmony": harmonic_features,
            "spectral_extra": spectral_extra}
