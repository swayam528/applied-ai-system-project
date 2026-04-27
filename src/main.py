"""
Command line runner for the Music Recommender.

Modes
-----
Classic (default)
    Runs the original weighted-scoring recommender against all built-in
    profiles and prints a formatted leaderboard to stdout.

    python -m src.main

AI mode  (requires ANTHROPIC_API_KEY environment variable)
    Starts an interactive loop powered by Claude.  Type a natural-language
    music request and receive ranked recommendations with explanations.

    python -m src.main --ai

Options
-------
  --ai        Use the Claude-powered interactive mode.
  -v, --verbose   Enable DEBUG-level logging.
"""

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "..", "data", "songs.csv")

# Ensure src/ is importable when this module is run via -m src.main or directly.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Built-in user profiles (classic mode)
# ---------------------------------------------------------------------------
PROFILES: dict = {
    "Late-Night Study (Lofi / Chill)": {
        "favorite_genre": "lofi",
        "favorite_mood":  "chill",
        "target_energy":  0.40,
        "target_tempo":   80,
        "target_valence": 0.60,
        "likes_acoustic": True,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "High-Energy Pop Fan": {
        "favorite_genre": "pop",
        "favorite_mood":  "happy",
        "target_energy":  0.90,
        "target_tempo":   128,
        "target_valence": 0.85,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "Deep Intense Rock": {
        "favorite_genre": "rock",
        "favorite_mood":  "intense",
        "target_energy":  0.92,
        "target_tempo":   150,
        "target_valence": 0.40,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    # -------------------------------------------------------------------
    # Adversarial / edge-case profiles
    # -------------------------------------------------------------------
    "Conflicting: High Energy + Sad Mood": {
        "favorite_genre": "metal",
        "favorite_mood":  "sad",
        "target_energy":  0.95,
        "target_tempo":   160,
        "target_valence": 0.15,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "Unknown Genre (k-pop)": {
        "favorite_genre": "k-pop",
        "favorite_mood":  "happy",
        "target_energy":  0.80,
        "target_tempo":   120,
        "target_valence": 0.85,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "Extreme Acoustic Minimalist": {
        "favorite_genre": "classical",
        "favorite_mood":  "melancholy",
        "target_energy":  0.10,
        "target_tempo":   50,
        "target_valence": 0.10,
        "likes_acoustic": True,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Classic mode helpers
# ---------------------------------------------------------------------------
def print_recommendations(profile_name: str, user_prefs: dict, songs: list) -> None:
    """Print a formatted leaderboard for one user profile."""
    from recommender import recommend_songs

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print()
    print("=" * 60)
    print(f"  PROFILE: {profile_name}")
    print(f"  Genre: {user_prefs['favorite_genre']}  |  "
          f"Mood: {user_prefs['favorite_mood']}  |  "
          f"Energy: {user_prefs['target_energy']}")
    print("=" * 60)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n  #{rank}  {song['title']}  --  {song['artist']}")
        print(f"       Score : {score:.2f} / 10.0")
        print(f"       Genre : {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
        print("       Why   :")
        for reason in explanation.split(" | "):
            print(f"         * {reason}")
        print("  " + "-" * 58)

    print()


def run_classic_mode() -> None:
    """Run the original batch recommender against all built-in profiles."""
    from recommender import load_songs

    log = logging.getLogger(__name__)
    songs = load_songs(DATA_PATH)
    log.info("Classic mode — running %d profiles", len(PROFILES))

    for profile_name, user_prefs in PROFILES.items():
        print_recommendations(profile_name, user_prefs, songs)


# ---------------------------------------------------------------------------
# AI interactive mode
# ---------------------------------------------------------------------------
def run_ai_mode() -> None:
    """Run the interactive Claude-powered recommender."""
    import sys

    log = logging.getLogger(__name__)

    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "Error: GOOGLE_API_KEY is not set.\n"
            "Get a free key at https://aistudio.google.com (sign in with Google).\n"
            "Then export it:\n"
            "  export GOOGLE_API_KEY=AIza...\n"
            "Or copy .env.example to .env and fill it in.\n"
        )
        sys.exit(1)

    # Load dotenv if available (optional convenience)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        log.debug("Loaded .env file")
    except ImportError:
        pass

    from recommender import load_songs
    from ai_recommender import AIRecommender

    songs = load_songs(DATA_PATH)
    rec = AIRecommender(songs)

    print("\nAI Music Recommender  (powered by Claude)")
    print("Describe what you want in plain English.  Type 'quit' to exit.\n")
    print("Examples:")
    print('  "I need something chill and acoustic for late-night studying"')
    print('  "Pump me up for a workout — high energy, no ballads"')
    print('  "Sad indie vibes, something melancholy but beautiful"\n')

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q", "bye"):
            print("Goodbye!")
            break

        log.debug("Sending query to AIRecommender: %r", query)
        response = rec.recommend(query)
        print(f"\nAssistant:\n{response}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Music Recommender — classic scoring or Claude-powered AI mode."
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Use the interactive Claude-powered AI mode (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.ai:
        run_ai_mode()
    else:
        run_classic_mode()


if __name__ == "__main__":
    main()
