"""Download official Essentia TensorFlow weights and matching JSON metadata.
Run on the HOST from project root: python download_models.py
"""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# This manifest is deliberately explicit and self-contained.
EMBEDDING_MODELS = [
    ("feature-extractors/maest", "discogs-maest-30s-pw-519l-2"),
    ("feature-extractors/musicnn", "msd-musicnn-1"),
    ("feature-extractors/discogs-effnet", "discogs-effnet-bs64-1"),
]
HEAD_MODELS = [
    ("genre_discogs519", "genre_discogs519-discogs-maest-30s-pw-519l-1"),
    ("mtg_jamendo_genre", "mtg_jamendo_genre-discogs-effnet-1"),
    ("genre_electronic", "genre_electronic-discogs-effnet-1"),
    ("moods_mirex", "moods_mirex-msd-musicnn-1"),
    ("deam", "deam-msd-musicnn-2"),
    ("emomusic", "emomusic-msd-musicnn-2"),
    ("mtg_jamendo_moodtheme", "mtg_jamendo_moodtheme-discogs-effnet-1"),
    ("mtg_jamendo_instrument", "mtg_jamendo_instrument-discogs-effnet-1"),
    ("mtg_jamendo_top50tags", "mtg_jamendo_top50tags-discogs-effnet-1"),
    *[(name, f"{name}-msd-musicnn-1") for name in
      ("mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party", "mood_acoustic", "mood_electronic")],
    *[(name, f"{name}-discogs-effnet-1") for name in
      ("voice_instrumental", "danceability", "tonal_atonal", "timbre", "gender")],
    ("engagement", "engagement_regression-discogs-effnet-1"),
    ("approachability", "approachability_regression-discogs-effnet-1"),
]


def download(url, filename):
    path = MODEL_DIR / filename
    if path.is_file() and path.stat().st_size > 0:
        print("[exists]", filename)
        return
    print("[download]", filename, flush=True)
    try:
        urlretrieve(url, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main():
    for group, filename in EMBEDDING_MODELS:
        download(f"https://essentia.upf.edu/models/{group}/{filename}.pb", f"{filename}.pb")
    for group, filename in HEAD_MODELS:
        for extension in ("pb", "json"):
            try:
                download(f"https://essentia.upf.edu/models/classification-heads/{group}/{filename}.{extension}", f"{filename}.{extension}")
            except Exception as exc:
                print(f"[warning] {filename}.{extension}: {exc}")
    print("Finished. Missing optional heads are reported under 'errors' by the API.")

if __name__ == "__main__":
    main()
