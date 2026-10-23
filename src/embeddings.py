"""Shared feature extractors (created once per server process)."""
from pathlib import Path
import essentia.standard as es

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

class Embeddings:
    def __init__(self):
        self.maest = es.TensorflowPredictMAEST(
            graphFilename=str(MODEL_DIR / "discogs-maest-30s-pw-519l-2.pb"),
            output="PartitionedCall/Identity_12",
        )
        self.musicnn = es.TensorflowPredictMusiCNN(
            graphFilename=str(MODEL_DIR / "msd-musicnn-1.pb"),
            output="model/dense/BiasAdd",
        )
        self.effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODEL_DIR / "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1",
        )

    def compute(self, name, audio):
        return getattr(self, name)(audio)
