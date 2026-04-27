"""
Tests for the AI recommendation layer (src/ai_recommender.py).

Tool-execution tests do NOT call the Gemini API — they run fully offline
without a GOOGLE_API_KEY.

The integration test (test_recommend_returns_non_empty_string) patches the
Gemini client so it also runs offline.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable when pytest is run from the project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Patch google.generativeai at import time so the module loads without a key.
# ---------------------------------------------------------------------------
import types

_fake_genai = types.ModuleType("google.generativeai")
_fake_protos = types.ModuleType("google.generativeai.protos")

# Minimal stubs needed for module-level code in ai_recommender.py
_fake_protos.Tool = MagicMock(return_value=MagicMock())
_fake_protos.FunctionDeclaration = MagicMock(return_value=MagicMock())
_fake_protos.Schema = MagicMock(return_value=MagicMock())
_fake_protos.Type = MagicMock()
_fake_protos.Type.OBJECT = "OBJECT"
_fake_protos.Type.STRING = "STRING"
_fake_protos.Type.NUMBER = "NUMBER"
_fake_protos.Type.BOOLEAN = "BOOLEAN"
_fake_protos.Type.INTEGER = "INTEGER"
_fake_protos.Part = MagicMock(return_value=MagicMock())
_fake_protos.FunctionResponse = MagicMock(return_value=MagicMock())

_fake_genai.configure = MagicMock()
_fake_genai.GenerativeModel = MagicMock()
_fake_genai.protos = _fake_protos

# Register in sys.modules before importing ai_recommender
_fake_google = types.ModuleType("google")
_fake_google.generativeai = _fake_genai
sys.modules.setdefault("google", _fake_google)
sys.modules["google.generativeai"] = _fake_genai
sys.modules["google.generativeai.protos"] = _fake_protos

from src.ai_recommender import AIRecommender  # noqa: E402

# ---------------------------------------------------------------------------
# Shared sample catalog
# ---------------------------------------------------------------------------
SAMPLE_SONGS: list[dict] = [
    {
        "id": 1, "title": "Sunrise City", "artist": "Neon Echo",
        "genre": "pop", "mood": "happy",
        "energy": 0.82, "tempo_bpm": 118, "valence": 0.84,
        "danceability": 0.79, "acousticness": 0.18,
    },
    {
        "id": 2, "title": "Midnight Coding", "artist": "LoRoom",
        "genre": "lofi", "mood": "chill",
        "energy": 0.42, "tempo_bpm": 78, "valence": 0.56,
        "danceability": 0.62, "acousticness": 0.71,
    },
    {
        "id": 3, "title": "Storm Runner", "artist": "Voltline",
        "genre": "rock", "mood": "intense",
        "energy": 0.91, "tempo_bpm": 152, "valence": 0.48,
        "danceability": 0.66, "acousticness": 0.10,
    },
]


# ---------------------------------------------------------------------------
# Fixture — AIRecommender with mocked Gemini (no API key needed)
# ---------------------------------------------------------------------------
@pytest.fixture
def rec(monkeypatch) -> AIRecommender:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    return AIRecommender(SAMPLE_SONGS)


# ---------------------------------------------------------------------------
# Tool: get_catalog_info
# ---------------------------------------------------------------------------
class TestGetCatalogInfo:
    def test_total_songs(self, rec):
        result = json.loads(rec._run_tool("get_catalog_info", {}))
        assert result["total_songs"] == 3

    def test_genres_present(self, rec):
        result = json.loads(rec._run_tool("get_catalog_info", {}))
        assert "pop" in result["genres"]
        assert "lofi" in result["genres"]
        assert "rock" in result["genres"]

    def test_moods_present(self, rec):
        result = json.loads(rec._run_tool("get_catalog_info", {}))
        assert "happy" in result["moods"]
        assert "chill" in result["moods"]
        assert "intense" in result["moods"]

    def test_genres_sorted(self, rec):
        result = json.loads(rec._run_tool("get_catalog_info", {}))
        assert result["genres"] == sorted(result["genres"])


# ---------------------------------------------------------------------------
# Tool: search_songs
# ---------------------------------------------------------------------------
class TestSearchSongs:
    def test_filter_by_genre(self, rec):
        result = json.loads(rec._run_tool("search_songs", {"genre": "lofi"}))
        assert len(result) == 1
        assert result[0]["genre"] == "lofi"

    def test_filter_by_mood(self, rec):
        result = json.loads(rec._run_tool("search_songs", {"mood": "intense"}))
        assert len(result) == 1
        assert result[0]["mood"] == "intense"

    def test_no_filter_returns_all(self, rec):
        result = json.loads(rec._run_tool("search_songs", {}))
        assert len(result) == 3

    def test_genre_and_mood_combined(self, rec):
        result = json.loads(rec._run_tool("search_songs", {"genre": "pop", "mood": "happy"}))
        assert len(result) == 1
        assert result[0]["title"] == "Sunrise City"

    def test_nonexistent_genre_returns_empty(self, rec):
        result = json.loads(rec._run_tool("search_songs", {"genre": "k-pop"}))
        assert result == []

    def test_result_fields(self, rec):
        result = json.loads(rec._run_tool("search_songs", {"genre": "pop"}))
        for field in ("title", "artist", "genre", "mood", "energy"):
            assert field in result[0]


# ---------------------------------------------------------------------------
# Tool: score_and_rank
# ---------------------------------------------------------------------------
class TestScoreAndRank:
    def test_returns_k_results(self, rec):
        result = json.loads(rec._run_tool("score_and_rank", {
            "favorite_genre": "pop", "favorite_mood": "happy",
            "target_energy": 0.82, "likes_acoustic": False, "k": 2,
        }))
        assert len(result) == 2

    def test_sorted_by_score_descending(self, rec):
        result = json.loads(rec._run_tool("score_and_rank", {
            "favorite_genre": "pop", "favorite_mood": "happy",
            "target_energy": 0.82, "likes_acoustic": False, "k": 3,
        }))
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_genre_match_wins(self, rec):
        result = json.loads(rec._run_tool("score_and_rank", {
            "favorite_genre": "pop", "favorite_mood": "happy",
            "target_energy": 0.82, "likes_acoustic": False,
        }))
        assert result[0]["genre"] == "pop"

    def test_result_has_score_and_reasons(self, rec):
        result = json.loads(rec._run_tool("score_and_rank", {
            "favorite_genre": "lofi", "favorite_mood": "chill",
            "target_energy": 0.40, "likes_acoustic": True,
        }))
        assert "score" in result[0]
        assert "reasons" in result[0]

    def test_scores_bounded_0_to_10(self, rec):
        result = json.loads(rec._run_tool("score_and_rank", {
            "favorite_genre": "rock", "favorite_mood": "intense",
            "target_energy": 0.91, "likes_acoustic": False,
        }))
        for entry in result:
            assert 0.0 <= entry["score"] <= 10.0


# ---------------------------------------------------------------------------
# Tool: unknown
# ---------------------------------------------------------------------------
def test_unknown_tool_returns_error_key(rec):
    result = json.loads(rec._run_tool("no_such_tool", {}))
    assert "error" in result


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
def test_genre_index_covers_all_songs(rec):
    assert sum(len(v) for v in rec.genre_index.values()) == len(SAMPLE_SONGS)


def test_mood_index_covers_all_songs(rec):
    assert sum(len(v) for v in rec.mood_index.values()) == len(SAMPLE_SONGS)


# ---------------------------------------------------------------------------
# Integration test — full recommend() with mocked Gemini chat
# ---------------------------------------------------------------------------
def test_recommend_returns_non_empty_string(monkeypatch):
    """
    Simulates two agentic iterations:
      1. Gemini calls score_and_rank  (response with function_call part)
      2. Gemini returns final text    (response with no function_call parts)
    No real API call is made.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    # --- Build fake parts ---
    # Iteration 1: function call part
    fake_fc = MagicMock()
    fake_fc.name = "score_and_rank"
    fake_fc.args = {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.40,
        "likes_acoustic": True,
        "k": 3,
    }

    tool_part = MagicMock()
    tool_part.function_call = fake_fc

    tool_response = MagicMock()
    tool_response.parts = [tool_part]
    tool_response.text = ""

    # Iteration 2: plain text part (no function_call)
    text_part = MagicMock()
    text_part.function_call = None

    final_response = MagicMock()
    final_response.parts = [text_part]
    final_response.text = "1. Midnight Coding by LoRoom (9.67/10) — perfect chill lofi."

    # Wire up chat mock
    mock_chat = MagicMock()
    mock_chat.send_message.side_effect = [tool_response, final_response]

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_chat

    _fake_genai.GenerativeModel.return_value = mock_model

    rec = AIRecommender(SAMPLE_SONGS)
    result = rec.recommend("I want something chill for studying")

    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_recommend_handles_api_error(monkeypatch):
    """If Gemini raises on the first call, recommend() returns a safe message."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    mock_chat = MagicMock()
    mock_chat.send_message.side_effect = Exception("network error")

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_chat

    _fake_genai.GenerativeModel.return_value = mock_model

    rec = AIRecommender(SAMPLE_SONGS)
    result = rec.recommend("anything")

    assert isinstance(result, str)
    assert "sorry" in result.lower() or "error" in result.lower() or "problem" in result.lower()
