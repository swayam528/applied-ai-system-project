import csv
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_tempo: float = 80.0
    target_valence: float = 0.60
    weights: Dict = None

    def __post_init__(self):
        """Set default feature weights if none were provided by the caller."""
        if self.weights is None:
            # Max possible score = 3.0 + 2.0 + 2.0 + 1.5 + 1.0 + 0.5 = 10.0
            self.weights = {
                "genre": 3.0,        # categorical: strongest signal (~8% random match rate)
                "mood": 2.0,         # categorical: important but less precise than genre
                "energy": 2.0,       # continuous: widest catalog range (0.22–0.97)
                "acousticness": 1.5, # continuous: separates organic vs produced
                "valence": 1.0,      # continuous: users tolerate more variation here
                "tempo": 0.5,        # continuous: normalized /200; partially redundant with energy
            }

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Score every song against user preferences and return the top-k sorted by score."""
        prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood":  user.favorite_mood,
            "target_energy":  user.target_energy,
            "target_tempo":   user.target_tempo,
            "target_valence": user.target_valence,
            "likes_acoustic": user.likes_acoustic,
            "weights":        user.weights,
        }
        scored = sorted(
            self.songs,
            key=lambda s: score_song(prefs, {
                "genre": s.genre, "mood": s.mood, "energy": s.energy,
                "tempo_bpm": s.tempo_bpm, "valence": s.valence,
                "acousticness": s.acousticness,
            })[0],
            reverse=True,
        )
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why this song was recommended."""
        prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood":  user.favorite_mood,
            "target_energy":  user.target_energy,
            "target_tempo":   user.target_tempo,
            "target_valence": user.target_valence,
            "likes_acoustic": user.likes_acoustic,
            "weights":        user.weights,
        }
        _, reasons = score_song(prefs, {
            "genre": song.genre, "mood": song.mood, "energy": song.energy,
            "tempo_bpm": song.tempo_bpm, "valence": song.valence,
            "acousticness": song.acousticness,
        })
        return " | ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with typed numeric fields."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    logger.info("Loaded %d songs from %s", len(songs), csv_path)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences; return (score_out_of_10, reasons_list)."""
    weights = user_prefs.get("weights", {
        "genre": 3.0,
        "mood": 2.0,
        "energy": 2.0,
        "acousticness": 1.5,
        "valence": 1.0,
        "tempo": 0.5,
    })

    score = 0.0
    reasons = []

    # --- Step 1: Genre match (categorical) ---
    w = weights["genre"]
    if song["genre"] == user_prefs["favorite_genre"]:
        score += w
        reasons.append(f"genre match (+{w})")
    else:
        reasons.append(f"genre mismatch — '{song['genre']}' vs '{user_prefs['favorite_genre']}' (+0.0)")

    # --- Step 2: Mood match (categorical) ---
    w = weights["mood"]
    if song["mood"] == user_prefs["favorite_mood"]:
        score += w
        reasons.append(f"mood match (+{w})")
    else:
        reasons.append(f"mood mismatch — '{song['mood']}' vs '{user_prefs['favorite_mood']}' (+0.0)")

    # --- Step 3: Energy similarity (continuous, 0–1 scale) ---
    w = weights["energy"]
    energy_diff = abs(song["energy"] - user_prefs["target_energy"])
    energy_pts = round(w * (1 - energy_diff), 2)
    score += energy_pts
    reasons.append(f"energy similarity (+{energy_pts})")

    # --- Step 4: Acousticness similarity (continuous, 0–1 scale) ---
    w = weights["acousticness"]
    target_acousticness = 1.0 if user_prefs.get("likes_acoustic") else 0.0
    acoustic_diff = abs(song["acousticness"] - target_acousticness)
    acoustic_pts = round(w * (1 - acoustic_diff), 2)
    score += acoustic_pts
    reasons.append(f"acousticness similarity (+{acoustic_pts})")

    # --- Step 5: Valence similarity (continuous, 0–1 scale) ---
    w = weights["valence"]
    valence_diff = abs(song["valence"] - user_prefs["target_valence"])
    valence_pts = round(w * (1 - valence_diff), 2)
    score += valence_pts
    reasons.append(f"valence similarity (+{valence_pts})")

    # --- Step 6: Tempo similarity (normalized to 0–1 by dividing by 200 BPM) ---
    w = weights["tempo"]
    norm_song_tempo = song["tempo_bpm"] / 200
    norm_user_tempo = user_prefs["target_tempo"] / 200
    tempo_diff = abs(norm_song_tempo - norm_user_tempo)
    tempo_pts = round(w * (1 - tempo_diff), 2)
    score += tempo_pts
    reasons.append(f"tempo similarity (+{tempo_pts})")

    return round(score, 2), reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, then return the top-k as (song, score, explanation) sorted highest first."""
    # Score every song in one list comprehension, then sort and slice.
    # score_song returns (score, reasons); join reasons into a single explanation string.
    scored = [
        (song, score, " | ".join(reasons))
        for song in songs
        for score, reasons in (score_song(user_prefs, song),)
    ]

    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
