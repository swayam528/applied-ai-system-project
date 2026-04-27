"""
AI-powered music recommender using Google Gemini with agentic tool use and RAG.

Architecture
------------
RAG (Retrieval-Augmented Generation)
    The song catalog is the retrieval store.  A catalog summary (genres, moods,
    total count) is injected into the system prompt as retrieved context before
    Gemini generates anything.  Gemini then calls tools to fetch additional song
    rows on demand.  Every recommendation is grounded in actual retrieved catalog
    facts — the model cannot invent songs that are not in the CSV.

Agentic Workflow
    Gemini drives the loop autonomously via function calling:
      1. ``get_catalog_info`` — understand available genres/moods.
      2. ``search_songs``     — filter candidates by genre or mood.
      3. ``score_and_rank``   — run the weighted scoring formula for objective scores.
    The loop runs until Gemini returns plain text (no more function calls) or
    the safety cap (MAX_ITERATIONS) is reached.

Setup
-----
    1. Get a free API key at https://aistudio.google.com  (sign in with Google)
    2. Set:  export GOOGLE_API_KEY=AIza...
       Or add it to a .env file (copy .env.example).
    3. pip install -r requirements.txt
       Model used: gemini-2.0-flash (free tier — 1,500 req/day)

Usage
-----
    from ai_recommender import AIRecommender
    from recommender import load_songs

    songs = load_songs("data/songs.csv")
    rec   = AIRecommender(songs)
    print(rec.recommend("I want something chill for late-night studying"))
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Ensure sibling module ``recommender`` is importable regardless of how this
# module is imported (from tests/, from src/, or via -m src.main).
# ---------------------------------------------------------------------------
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from recommender import score_song  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (Gemini function-declaration format)
# ---------------------------------------------------------------------------
_TOOL_DECLARATIONS = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_catalog_info",
            description=(
                "Returns a summary of the music catalog: total song count, list of "
                "available genres, and list of available moods. Call this first to "
                "understand what the catalog contains."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={},
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="search_songs",
            description=(
                "Retrieve songs from the catalog matching the given genre and/or mood. "
                "Omit a field to skip that filter. Returns title, artist, genre, mood, energy."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "genre": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Genre to filter by (e.g. 'lofi', 'pop'). Optional.",
                    ),
                    "mood": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Mood to filter by (e.g. 'chill', 'happy'). Optional.",
                    ),
                },
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="score_and_rank",
            description=(
                "Score every song against explicit user-preference values and return the "
                "top-k results sorted by score out of 10.0. Call this once you have "
                "inferred the user's numeric preferences from their request."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "favorite_genre": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="User's preferred genre.",
                    ),
                    "favorite_mood": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="User's preferred mood.",
                    ),
                    "target_energy": genai.protos.Schema(
                        type=genai.protos.Type.NUMBER,
                        description="Target energy 0.0 (very calm) to 1.0 (very intense).",
                    ),
                    "target_tempo": genai.protos.Schema(
                        type=genai.protos.Type.NUMBER,
                        description="Target tempo in BPM (e.g. 70=slow, 120=upbeat, 150=fast).",
                    ),
                    "target_valence": genai.protos.Schema(
                        type=genai.protos.Type.NUMBER,
                        description="Target valence 0.0 (dark/sad) to 1.0 (bright/happy).",
                    ),
                    "likes_acoustic": genai.protos.Schema(
                        type=genai.protos.Type.BOOLEAN,
                        description="True = prefers organic/acoustic; False = electronic/produced.",
                    ),
                    "k": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Number of top songs to return (default 5).",
                    ),
                },
                required=["favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"],
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# AIRecommender
# ---------------------------------------------------------------------------
class AIRecommender:
    """
    Wraps the music catalog with a Gemini-powered agentic recommendation layer.

    Parameters
    ----------
    songs:
        List of song dicts as returned by ``recommender.load_songs``.
    model_name:
        Gemini model to use. Defaults to ``gemini-flash-latest`` (free tier).
    """

    MAX_ITERATIONS = 6  # safety cap on agentic tool-call rounds

    def __init__(
        self,
        songs: list[dict[str, Any]],
        model_name: str = "gemini-flash-latest",
    ) -> None:
        self.songs = songs
        self.model_name = model_name

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com and export it:\n"
                "  export GOOGLE_API_KEY=AIza..."
            )
        genai.configure(api_key=api_key)
        self._build_index()
        logger.info(
            "AIRecommender ready — %d songs | %d genres | %d moods | model=%s",
            len(self.songs),
            len(self.genre_index),
            len(self.mood_index),
            self.model_name,
        )

    # ------------------------------------------------------------------
    # Index construction (RAG retrieval store)
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Build fast look-up indexes for the search_songs tool."""
        self.genre_index: dict[str, list[dict]] = {}
        self.mood_index:  dict[str, list[dict]] = {}
        for song in self.songs:
            self.genre_index.setdefault(song["genre"], []).append(song)
            self.mood_index.setdefault(song["mood"],  []).append(song)

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------
    def _run_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return a JSON-serialised result string."""
        logger.debug("Tool %r called  input=%s", name, tool_input)

        if name == "get_catalog_info":
            payload: Any = {
                "total_songs": len(self.songs),
                "genres": sorted(self.genre_index.keys()),
                "moods":  sorted(self.mood_index.keys()),
            }

        elif name == "search_songs":
            results = self.songs
            if tool_input.get("genre"):
                results = [s for s in results if s["genre"] == tool_input["genre"]]
            if tool_input.get("mood"):
                results = [s for s in results if s["mood"] == tool_input["mood"]]
            payload = [
                {k: s[k] for k in ("title", "artist", "genre", "mood", "energy")}
                for s in results
            ]
            logger.debug("search_songs → %d results", len(payload))

        elif name == "score_and_rank":
            prefs: dict[str, Any] = {
                "favorite_genre":  tool_input["favorite_genre"],
                "favorite_mood":   tool_input["favorite_mood"],
                "target_energy":   float(tool_input.get("target_energy",  0.5)),
                "target_tempo":    float(tool_input.get("target_tempo",   100)),
                "target_valence":  float(tool_input.get("target_valence", 0.5)),
                "likes_acoustic":  bool(tool_input["likes_acoustic"]),
            }
            k = int(tool_input.get("k", 5))
            scored = []
            for song in self.songs:
                sc, reasons = score_song(prefs, song)
                scored.append(
                    {
                        "title":   song["title"],
                        "artist":  song["artist"],
                        "genre":   song["genre"],
                        "mood":    song["mood"],
                        "energy":  song["energy"],
                        "score":   sc,
                        "reasons": reasons,
                    }
                )
            scored.sort(key=lambda x: x["score"], reverse=True)
            payload = scored[:k]
            logger.debug(
                "score_and_rank → top=%.2f  bottom=%.2f",
                payload[0]["score"] if payload else 0,
                payload[-1]["score"] if payload else 0,
            )

        else:
            logger.warning("Unknown tool: %r", name)
            payload = {"error": f"Unknown tool: {name!r}"}

        result = json.dumps(payload, indent=2)
        logger.debug("Tool %r → %d chars", name, len(result))
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def recommend(self, user_query: str) -> str:
        """
        Process a natural-language music request with Gemini.

        RAG step
        --------
        A catalog summary (genres, moods, total count) is injected into the
        system instruction as retrieved context before Gemini generates anything.

        Agentic loop
        ------------
        Gemini calls tools iteratively until it returns plain text with no
        function calls remaining:
          - ``get_catalog_info`` → refresh metadata
          - ``search_songs``     → filter candidates
          - ``score_and_rank``   → compute objective scores

        Parameters
        ----------
        user_query:
            Natural-language music request from the user.

        Returns
        -------
        str
            Gemini's final recommendation text.
        """
        logger.info("New request: %r", user_query)

        # --- RAG: inject catalog summary as retrieved context ---
        catalog_context = json.dumps(
            {
                "total_songs": len(self.songs),
                "genres": sorted(self.genre_index.keys()),
                "moods":  sorted(self.mood_index.keys()),
            },
            indent=2,
        )

        system_instruction = (
            "You are a music recommendation assistant backed by a real song catalog.\n\n"
            "RETRIEVED CATALOG CONTEXT:\n"
            f"{catalog_context}\n\n"
            "INSTRUCTIONS:\n"
            "1. Use get_catalog_info if you need to check catalog metadata.\n"
            "2. Use search_songs to retrieve candidate songs by genre or mood.\n"
            "3. Use score_and_rank once you have inferred the user's numeric preferences "
            "(energy, tempo, valence, acousticness) from their request. "
            "This returns objective scores out of 10.0.\n"
            "4. Respond with 3–5 ranked recommendations. For each song include:\n"
            "   - Title and artist\n"
            "   - Score out of 10.0\n"
            "   - One sentence explaining why it fits the user's request\n\n"
            "Every recommendation MUST be a real song from the tool results. "
            "Do not invent songs."
        )

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                tools=[_TOOL_DECLARATIONS],
                system_instruction=system_instruction,
            )
            chat = model.start_chat()
        except Exception as exc:
            logger.error("Failed to create Gemini model: %s", exc)
            return (
                "Sorry — there was a problem initialising the AI model. "
                "Please check your GOOGLE_API_KEY and try again."
            )

        try:
            response = chat.send_message(user_query)
        except Exception as exc:
            logger.error("Gemini API error on first call: %s", exc)
            return (
                "Sorry — there was a problem contacting the AI service. "
                "Please check your GOOGLE_API_KEY and try again."
            )

        for iteration in range(self.MAX_ITERATIONS):
            logger.debug("Agentic iteration %d/%d", iteration + 1, self.MAX_ITERATIONS)

            # Collect any function calls from this response
            fn_calls = []
            for part in response.parts:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    fn_calls.append(fc)

            if not fn_calls:
                # No tool calls — this is the final text answer
                text = getattr(response, "text", None) or ""
                if text.strip():
                    logger.info("Recommendation complete after %d iteration(s)", iteration + 1)
                    return text
                # Edge case: empty response with no tool calls
                break

            # Execute every function call and build response parts
            response_parts = []
            for fc in fn_calls:
                result_str = self._run_tool(fc.name, dict(fc.args))
                response_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fc.name,
                            response={"result": json.loads(result_str)},
                        )
                    )
                )

            try:
                response = chat.send_message(response_parts)
            except Exception as exc:
                logger.error("Gemini API error during tool loop: %s", exc)
                return (
                    "Sorry — there was a problem contacting the AI service. "
                    "Please check your GOOGLE_API_KEY and try again."
                )

        logger.error("Agentic loop exhausted after %d iterations", self.MAX_ITERATIONS)
        return (
            "I was unable to generate recommendations after several attempts. "
            "Please try rephrasing your request."
        )
