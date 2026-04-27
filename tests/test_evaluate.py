"""
Pytest-runnable quality-threshold tests for the scoring engine.

These assert minimum acceptable behavior — they fail loudly if the system
degrades. For the human-readable evaluation report run:

    python tests/evaluate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import load_songs, recommend_songs, confidence_label

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

PROFILES: dict[str, dict] = {
    "Late-Night Study (Lofi/Chill)": {
        "favorite_genre": "lofi", "favorite_mood": "chill",
        "target_energy": 0.40, "target_tempo": 80, "target_valence": 0.60,
        "likes_acoustic": True,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "High-Energy Pop Fan": {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.90, "target_tempo": 128, "target_valence": 0.85,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "Deep Intense Rock": {
        "favorite_genre": "rock", "favorite_mood": "intense",
        "target_energy": 0.92, "target_tempo": 150, "target_valence": 0.40,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "Conflicting: High Energy + Sad": {
        "favorite_genre": "metal", "favorite_mood": "sad",
        "target_energy": 0.95, "target_tempo": 160, "target_valence": 0.15,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "Unknown Genre (k-pop)": {
        "favorite_genre": "k-pop", "favorite_mood": "happy",
        "target_energy": 0.80, "target_tempo": 120, "target_valence": 0.85,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "Extreme Acoustic Minimalist": {
        "favorite_genre": "classical", "favorite_mood": "melancholy",
        "target_energy": 0.10, "target_tempo": 50, "target_valence": 0.10,
        "likes_acoustic": True,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
}


def _genre_in_catalog(genre: str, songs: list[dict]) -> bool:
    return any(s["genre"] == genre for s in songs)


def test_genre_precision_for_covered_profiles() -> None:
    """Top-1 must match the requested genre whenever that genre exists in the catalog."""
    songs = load_songs(DATA_PATH)
    for name, prefs in PROFILES.items():
        if not _genre_in_catalog(prefs["favorite_genre"], songs):
            continue
        top_song, _, _ = recommend_songs(prefs, songs, k=1)[0]
        assert top_song["genre"] == prefs["favorite_genre"], (
            f"'{name}': expected '{prefs['favorite_genre']}', got '{top_song['genre']}'"
        )


def test_top1_score_high_confidence_for_covered_profiles() -> None:
    """Top-1 score must be >= 7.0 (High confidence) when the genre is in the catalog."""
    songs = load_songs(DATA_PATH)
    for name, prefs in PROFILES.items():
        if not _genre_in_catalog(prefs["favorite_genre"], songs):
            continue
        _, score, _ = recommend_songs(prefs, songs, k=1)[0]
        assert score >= 7.0, (
            f"'{name}': score {score:.2f} is below the 7.0 High-confidence threshold"
        )


def test_scoring_is_deterministic() -> None:
    """Three consecutive runs on the same profile must produce the same top-1 score."""
    songs = load_songs(DATA_PATH)
    prefs = PROFILES["High-Energy Pop Fan"]
    scores = [recommend_songs(prefs, songs, k=1)[0][1] for _ in range(3)]
    assert len(set(scores)) == 1, f"Non-deterministic scores across runs: {scores}"


def test_missing_genre_score_below_high_confidence() -> None:
    """When the genre is absent from the catalog, top-1 must score < 7.0."""
    songs = load_songs(DATA_PATH)
    prefs = PROFILES["Unknown Genre (k-pop)"]
    _, score, _ = recommend_songs(prefs, songs, k=1)[0]
    assert score < 7.0, (
        f"Expected Medium/Low confidence for missing genre, got {score:.2f}"
    )


def test_confidence_label_thresholds() -> None:
    """confidence_label() must map scores to the correct tier."""
    assert confidence_label(9.5)  == "High"
    assert confidence_label(7.0)  == "High"
    assert confidence_label(6.99) == "Medium"
    assert confidence_label(4.0)  == "Medium"
    assert confidence_label(3.99) == "Low"
    assert confidence_label(0.0)  == "Low"
