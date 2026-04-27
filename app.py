"""
Streamlit UI for the Music Recommender.

Run with:
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "songs.csv")

PROFILES = {
    "🌙 Late-Night Study (Lofi / Chill)": {
        "favorite_genre": "lofi", "favorite_mood": "chill",
        "target_energy": 0.40, "target_tempo": 80, "target_valence": 0.60,
        "likes_acoustic": True,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "🔥 High-Energy Pop Fan": {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.90, "target_tempo": 128, "target_valence": 0.85,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "🎸 Deep Intense Rock": {
        "favorite_genre": "rock", "favorite_mood": "intense",
        "target_energy": 0.92, "target_tempo": 150, "target_valence": 0.40,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "⚡ Conflicting: High Energy + Sad Mood": {
        "favorite_genre": "metal", "favorite_mood": "sad",
        "target_energy": 0.95, "target_tempo": 160, "target_valence": 0.15,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "🌐 Unknown Genre (k-pop)": {
        "favorite_genre": "k-pop", "favorite_mood": "happy",
        "target_energy": 0.80, "target_tempo": 120, "target_valence": 0.85,
        "likes_acoustic": False,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
    "🎻 Extreme Acoustic Minimalist": {
        "favorite_genre": "classical", "favorite_mood": "melancholy",
        "target_energy": 0.10, "target_tempo": 50, "target_valence": 0.10,
        "likes_acoustic": True,
        "weights": {"genre": 3.0, "mood": 2.0, "energy": 2.0,
                    "acousticness": 1.5, "valence": 1.0, "tempo": 0.5},
    },
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0f0f1a; }
[data-testid="stHeader"] { background: transparent; }
h1 { color: #e2e8f0 !important; letter-spacing: -1px; }
p, label, div { color: #cbd5e1; }

/* ── Song card ── */
.song-card {
    background: #1e1e2e;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 10px 0;
    border-left: 4px solid #7c3aed;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.rank-badge {
    display: inline-block;
    background: #7c3aed;
    color: #fff;
    font-weight: 700;
    font-size: 0.75em;
    border-radius: 6px;
    padding: 2px 8px;
    margin-bottom: 6px;
}
.song-title {
    font-size: 1.15em;
    font-weight: 700;
    color: #f1f5f9;
    margin: 2px 0;
}
.song-artist {
    color: #94a3b8;
    font-size: 0.9em;
    margin-bottom: 10px;
}
.tag {
    display: inline-block;
    background: #2d2d44;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78em;
    color: #a5b4fc;
    margin-right: 4px;
}
.score-label {
    font-size: 0.8em;
    color: #64748b;
    margin-top: 10px;
    margin-bottom: 3px;
}
.score-track {
    background: #2d2d44;
    border-radius: 6px;
    height: 8px;
    width: 100%;
}
.score-fill {
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    border-radius: 6px;
    height: 8px;
}
.reason-list {
    margin-top: 10px;
    padding-left: 0;
    list-style: none;
}
.reason-list li {
    font-size: 0.82em;
    color: #94a3b8;
    padding: 2px 0;
}
.reason-list li::before { content: "▸ "; color: #7c3aed; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #7c3aed !important;
}

/* ── Chat ── */
[data-testid="stChatMessage"] { background: #1e1e2e !important; border-radius: 12px; }
[data-testid="stChatInput"] textarea {
    background: #1e1e2e !important;
    color: #e2e8f0 !important;
    border-color: #374151 !important;
}

/* ── Buttons / selects ── */
[data-testid="stSelectbox"] > div { background: #1e1e2e !important; border-color: #374151 !important; }
[data-testid="baseButton-primary"] {
    background: #7c3aed !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 🎵 Music Recommender")
st.markdown("<p style='color:#64748b;margin-top:-12px;'>Content-based filtering · RAG + Agentic AI · Powered by Gemini</p>",
            unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_songs():
    return load_songs(DATA_PATH)


@st.cache_resource
def get_ai_recommender():
    from ai_recommender import AIRecommender
    return AIRecommender(get_songs())


# ---------------------------------------------------------------------------
# Helper: render one song card
# ---------------------------------------------------------------------------
def song_card(rank: int, song: dict, score: float, explanation: str) -> None:
    score_pct = int(score / 10 * 100)
    reasons_html = "".join(
        f"<li>{r.strip()}</li>"
        for r in explanation.split("|")
        if r.strip()
    )
    st.markdown(f"""
    <div class="song-card">
        <span class="rank-badge">#{rank}</span>
        <div class="song-title">{song['title']}</div>
        <div class="song-artist">{song['artist']}</div>
        <span class="tag">{song['genre']}</span>
        <span class="tag">{song['mood']}</span>
        <span class="tag">energy {song['energy']}</span>
        <div class="score-label">Score: <strong style="color:#a78bfa">{score:.2f} / 10.0</strong></div>
        <div class="score-track"><div class="score-fill" style="width:{score_pct}%"></div></div>
        <ul class="reason-list">{reasons_html}</ul>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_ai, tab_classic = st.tabs(["🤖  AI Chat", "📋  Classic Profiles"])

# ── AI Chat ─────────────────────────────────────────────────────────────────
with tab_ai:
    has_key = bool(os.environ.get("GOOGLE_API_KEY"))

    if not has_key:
        st.warning(
            "**GOOGLE_API_KEY not found.**  "
            "Add it to a `.env` file in the project root and restart the app.\n\n"
            "```\nGOOGLE_API_KEY=AIzaSy...\n```"
        )
    else:
        st.markdown(
            "<p style='color:#64748b;font-size:0.9em'>"
            "Describe what you want in plain English — Gemini searches and scores the catalog for you."
            "</p>",
            unsafe_allow_html=True,
        )

        # Example chips
        examples = [
            "Chill lofi for late-night studying",
            "High energy workout banger",
            "Something sad and acoustic",
            "Upbeat pop to start my morning",
        ]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state["prefill"] = ex

        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input (handle both typed and example-chip)
        prefill = st.session_state.pop("prefill", None)
        prompt = st.chat_input("Describe what you want to hear…")
        if prefill and not prompt:
            prompt = prefill

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching catalog…"):
                    try:
                        rec = get_ai_recommender()
                        response = rec.recommend(prompt)
                    except Exception as e:
                        response = f"Error: {e}"
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.get("messages"):
            if st.button("Clear chat", key="clear"):
                st.session_state.messages = []
                st.rerun()

# ── Classic Profiles ─────────────────────────────────────────────────────────
with tab_classic:
    st.markdown(
        "<p style='color:#64748b;font-size:0.9em'>"
        "Pick a pre-built user profile and see the top-5 scored songs."
        "</p>",
        unsafe_allow_html=True,
    )

    profile_name = st.selectbox("Select a profile", list(PROFILES.keys()), label_visibility="collapsed")
    prefs = PROFILES[profile_name]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Genre", prefs["favorite_genre"])
    col_b.metric("Mood", prefs["favorite_mood"])
    col_c.metric("Energy target", prefs["target_energy"])

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        songs = get_songs()
        with st.spinner("Scoring catalog…"):
            results = recommend_songs(prefs, songs, k=5)

        st.markdown(f"#### Top 5 for **{profile_name}**")
        for rank, (song, score, explanation) in enumerate(results, 1):
            song_card(rank, song, score, explanation)
