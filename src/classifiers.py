"""Small classification heads operating on already-computed embeddings."""
import json
from pathlib import Path
import numpy as np
import essentia.standard as es
from essentia import Pool
from embeddings import MODEL_DIR

# (API key, model directory, model filename, shared embedding, maximum returned tags)
# These are specifically the *classification-head* variants, not the older
# stand-alone /models/classifiers/ networks.
HEADS = [
    ("discogs519", "genre_discogs519", "genre_discogs519-discogs-maest-30s-pw-519l-1", "maest", 20),
    ("jamendo_genre", "mtg_jamendo_genre", "mtg_jamendo_genre-discogs-effnet-1", "effnet", 20),
    ("electronic_genre", "genre_electronic", "genre_electronic-discogs-effnet-1", "effnet", 20),
    ("mirex", "moods_mirex", "moods_mirex-msd-musicnn-1", "musicnn", None),
    ("deam", "deam", "deam-msd-musicnn-2", "musicnn", None),
    ("emomusic", "emomusic", "emomusic-msd-musicnn-2", "musicnn", None),
    ("jamendo_moodtheme", "mtg_jamendo_moodtheme", "mtg_jamendo_moodtheme-discogs-effnet-1", "effnet", 20),
    ("jamendo_instrument", "mtg_jamendo_instrument", "mtg_jamendo_instrument-discogs-effnet-1", "effnet", 20),
    ("jamendo_top50", "mtg_jamendo_top50tags", "mtg_jamendo_top50tags-discogs-effnet-1", "effnet", 20),
    ("happy", "mood_happy", "mood_happy-msd-musicnn-1", "musicnn", None),
    ("sad", "mood_sad", "mood_sad-msd-musicnn-1", "musicnn", None),
    ("aggressive", "mood_aggressive", "mood_aggressive-msd-musicnn-1", "musicnn", None),
    ("relaxed", "mood_relaxed", "mood_relaxed-msd-musicnn-1", "musicnn", None),
    ("party", "mood_party", "mood_party-msd-musicnn-1", "musicnn", None),
    ("acoustic", "mood_acoustic", "mood_acoustic-msd-musicnn-1", "musicnn", None),
    ("electronic", "mood_electronic", "mood_electronic-msd-musicnn-1", "musicnn", None),
    ("voice_instrumental", "voice_instrumental", "voice_instrumental-discogs-effnet-1", "effnet", None),
    ("danceability", "danceability", "danceability-discogs-effnet-1", "effnet", None),
    ("tonal_atonal", "tonal_atonal", "tonal_atonal-discogs-effnet-1", "effnet", None),
    ("engagement", "engagement", "engagement_regression-discogs-effnet-1", "effnet", None),
    ("approachability", "approachability", "approachability_regression-discogs-effnet-1", "effnet", None),
    ("timbre", "timbre", "timbre-discogs-effnet-1", "effnet", None),
    ("gender", "gender", "gender-discogs-effnet-1", "effnet", None),
]


def _prediction_output(metadata):
    outputs = metadata.get("schema", {}).get("outputs", [])
    for item in outputs:
        if item.get("output_purpose") == "predictions":
            return item["name"]
    # Regression metadata occasionally uses a different description.
    for item in outputs:
        if item.get("name", "").endswith(("Identity", "Softmax", "Sigmoid")):
            return item["name"]
    return outputs[0]["name"] if outputs else None


class Classifiers:
    def __init__(self):
        self.models = {}
        self.errors = {}
        for key, group, filename, embedding, top_k in HEADS:
            base = MODEL_DIR / filename
            if not base.with_suffix(".pb").is_file() or not base.with_suffix(".json").is_file():
                self.errors[key] = "Model files missing"
                continue
            try:
                with base.with_suffix(".json").open(encoding="utf-8") as handle:
                    meta = json.load(handle)
                labels = meta.get("classes")
                output = _prediction_output(meta)
                if key == "discogs519":
                    model = es.TensorflowPredict(
                        graphFilename=str(base.with_suffix(".pb")),
                        inputs=["embeddings"], outputs=["PartitionedCall/Identity_1"],
                    )
                elif key == "mirex":
                    model = es.TensorflowPredict2D(
                        graphFilename=str(base.with_suffix(".pb")),
                        input="serving_default_model_Placeholder",
                        output="PartitionedCall",
                    )
                else:
                    # Official TensorflowPredict2D model metadata specifies node names.
                    # Default inputs are left to Essentia when not provided.
                    kw = {"graphFilename": str(base.with_suffix(".pb"))}
                    if output:
                        kw["output"] = output
                    model = es.TensorflowPredict2D(**kw)
                self.models[key] = (model, embedding, labels, top_k)
            except Exception as exc:
                self.errors[key] = f"{type(exc).__name__}: {exc}"

    def evaluate(self, key, embeddings):
        model, _, labels, top_k = self.models[key]
        if key == "discogs519":
            pool = Pool()
            pool.set("embeddings", embeddings)
            pred = model(pool)["PartitionedCall/Identity_1"]
        else:
            pred = model(embeddings)
        values = np.asarray(pred, dtype=float)
        if labels:
            if values.size % len(labels):
                raise ValueError(f"Incompatible prediction shape {values.shape} for {len(labels)} labels")
            scores = values.reshape(-1, len(labels)).mean(axis=0)
            paired = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
            if top_k:
                paired = paired[:top_k]
            return {str(label): round(float(score), 5) for label, score in paired}
        # Regression: preserve the original order and include the mean values.
        # DEAM/emoMusic have valence, arousal as documented by Essentia.
        scores = values.reshape(-1, values.shape[-1]).mean(axis=0)
        if key in ("deam", "emomusic") and len(scores) == 2:
            return {"valence": round(float(scores[0]), 5), "arousal": round(float(scores[1]), 5)}
        return [round(float(x), 5) for x in scores]
