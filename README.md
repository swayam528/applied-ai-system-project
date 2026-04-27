# 🎵 AI Music Recommender

> Content-based music recommendations powered by Google Gemini — featuring RAG retrieval, an agentic tool loop, a transparent weighted scoring engine, and a Streamlit UI.

---

## Original Project (Modules 1–3)

This project began as the **Music Recommender Simulation** built across Modules 1–3. The original goal was to represent songs and user taste profiles as structured data, then design a transparent weighted scoring rule that could rank 20 catalog songs against any user profile and explain every decision. It demonstrated the foundations of content-based filtering — how genre, mood, energy, tempo, valence, and acousticness can be combined into a single explainable score — without relying on any external AI model or API.

---

## What This Project Does and Why It Matters

This project takes that scoring foundation and wraps it with a real AI layer: a **Retrieval-Augmented Generation (RAG)** pipeline and an **agentic tool-use workflow** powered by Google Gemini. Instead of requiring a user to fill out a preference form, they can now type plain English — _"I need something chill for late-night studying"_ — and the AI interprets that request, searches the catalog, runs the scoring formula, and returns ranked recommendations with explanations.

This matters because it closes the gap between academic recommender-system theory and the kind of natural-language AI interfaces that users actually expect today. Every recommendation is still fully explainable (the scoring formula is unchanged), but the front door is now a conversation instead of a spreadsheet.

---

## System Architecture

```mermaid
flowchart TD
    %% ── User Input ──────────────────────────────────────────────────────
    subgraph INPUT["👤 Human Input"]
        Q["Natural-language query\ne.g. 'chill music for studying'"]
        P["Profile selection\ne.g. Late-Night Lofi"]
    end

    %% ── Data Layer ───────────────────────────────────────────────────────
    subgraph DATA["📄 Data Layer"]
        CSV[("songs.csv\n20 songs")]
        LOAD["load_songs()"]
        IDX["In-Memory Catalog Index\ngenre_index · mood_index\n── RAG Retrieval Store ──"]
        CSV --> LOAD --> IDX
    end

    %% ── UI ───────────────────────────────────────────────────────────────
    subgraph UI["🖥️ Streamlit UI  (app.py)"]
        AITAB["AI Chat tab"]
        CTAB["Classic Profiles tab"]
    end

    %% ── AI Path ──────────────────────────────────────────────────────────
    subgraph AI["🤖 AI Layer  (ai_recommender.py)"]
        CTX["RAG Injection\nCatalog summary → system prompt"]
        GEMINI["Gemini API\ngemini-flash-latest"]

        subgraph LOOP["⟳ Agentic Tool Loop"]
            T1["get_catalog_info\nreturns genres · moods"]
            T2["search_songs\nfilters catalog rows"]
            T3["score_and_rank\ncalls score_song()"]
        end
    end

    %% ── Scoring Engine ───────────────────────────────────────────────────
    subgraph SCORE["🧮 Scoring Engine  (recommender.py)"]
        SF["score_song()\n6 weighted features → 0–10 pts\ngenre · mood · energy\nvalence · tempo · acousticness"]
    end

    %% ── Testing ──────────────────────────────────────────────────────────
    subgraph TEST["🧪 Reliability Layer  (pytest)"]
        direction LR
        TT["Tool tests\nget_catalog_info ✓\nsearch_songs ✓\nscore_and_rank ✓"]
        IT["Integration test\nfull agentic loop ✓\nerror handling ✓"]
        UT["Unit tests\nRecommender · score_song ✓"]
        MOCK["Mocked Gemini client\n(runs fully offline)"]
        MOCK --> TT & IT
    end

    %% ── Output ───────────────────────────────────────────────────────────
    OUT["📋 Ranked Recommendations\ntitle · artist · score · reasons"]

    %% ── Connections ──────────────────────────────────────────────────────
    Q --> AITAB --> CTX
    IDX --> CTX --> GEMINI
    GEMINI -- "tool_use" --> T1 & T2 & T3
    T1 & T2 --> IDX
    T3 --> SF --> T3
    GEMINI -- "end_turn" --> OUT

    P --> CTAB --> SF --> OUT
    OUT --> UI

    %% ── Human-in-the-loop ────────────────────────────────────────────────
    OUT -. "Human reviews &\nre-prompts if needed" .-> Q

    %% ── Testing hooks ────────────────────────────────────────────────────
    TT -.-> LOOP
    IT -.-> GEMINI
    UT -.-> SF
```

### Architecture in plain English

There are two parallel paths through the system.

**AI path (RAG + Agentic):** The user's natural-language query enters the Streamlit chat tab. `AIRecommender` builds a system prompt that includes a pre-retrieved catalog summary (genres, moods, song count) — this is the RAG injection step. Gemini receives that context and then autonomously decides which tools to call. It typically calls `search_songs` to narrow candidates, then `score_and_rank` to get objective scores from the same weighted formula used in the classic path. When Gemini is satisfied it returns a final text answer (`end_turn`), which is displayed in the chat. The loop is capped at 6 iterations as a safety guard.

**Classic path (deterministic scoring):** The user selects one of six pre-built profiles from a dropdown. `recommend_songs()` scores every song directly using `score_song()` and returns the top 5. No API call is made. This path is instant and fully reproducible.

**Testing layer:** All 22 tests run without a live API key. The Gemini client is replaced by a `MagicMock` that returns controlled responses, so the tool-execution logic, the agentic loop, and the error-handling path are all verified offline.

---

## Project Structure

```
applied-ai-system-project/
├── app.py                      # Streamlit UI (run this for the demo)
├── data/
│   └── songs.csv               # 20-song catalog with 8 features per song
├── src/
│   ├── main.py                 # CLI runner (--ai flag for interactive mode)
│   ├── recommender.py          # Scoring engine: load_songs, score_song, recommend_songs
│   └── ai_recommender.py       # Gemini-powered RAG + agentic recommender
├── tests/
│   ├── test_recommender.py     # Unit tests for scoring engine
│   └── test_ai_recommender.py  # Tests for AI layer (mocked, offline)
├── .env.example                # API key template
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd applied-ai-system-project
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Google API key

Get a **free** key at [aistudio.google.com](https://aistudio.google.com) — sign in with Google, click **Get API key**, then **Create API key**. No credit card required.

```bash
cp .env.example .env
# Open .env and paste your key:
# GOOGLE_API_KEY=AIzaSy...
```

> The classic mode and all tests work without a key. Only the AI Chat tab requires it.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Alternative: CLI mode

```bash
python -m src.main            # classic batch mode (no key needed)
python -m src.main --ai       # interactive AI chat in the terminal
python -m src.main --verbose  # enable debug logging
```

### Run the tests

```bash
pytest                        # 22 tests, no API key needed
pytest -v                     # verbose output
```

---

## Sample Interactions

### 1. AI Chat — late-night study session

**Input:** `I need something chill for late-night studying`

**AI output:**

```
For your late-night study session, I recommend these chill, low-energy tracks
to help you stay focused:

1. Library Rain — Paper Lanterns  (9.57 / 10)
   With its very low energy and chill lofi vibes, this track perfectly mimics
   the peaceful atmosphere of a quiet library.

2. Midnight Coding — LoRoom  (9.25 / 10)
   This lofi track is specifically designed for late-night productivity,
   offering a steady, unobtrusive rhythm.

3. Focus Flow — LoRoom  (7.38 / 10)
   While slightly more driven than the others, its "focused" mood is tailored
   for maintaining deep concentration during long study blocks.

4. Spacewalk Thoughts — Orbit Bloom  (6.64 / 10)
   If you need something more atmospheric, this ambient piece provides a calm,
   spacey backdrop that won't distract you from your work.
```

Behind the scenes Gemini called `score_and_rank` with inferred preferences (`genre: lofi`, `energy: ~0.35`, `likes_acoustic: true`) before writing this response. The scores come directly from the weighted formula — the AI did not invent them.

---

### 2. Classic Profile — High-Energy Pop Fan

**Profile settings:** genre=pop · mood=happy · energy=0.90 · acoustic=false

| Rank | Song              | Artist        | Genre      | Score |
| ---- | ----------------- | ------------- | ---------- | ----- |
| #1   | Sunrise City      | Neon Echo     | pop        | 9.53  |
| #2   | Gym Hero          | Max Pulse     | pop        | 7.77  |
| #3   | Rooftop Lights    | Indigo Parade | indie pop  | 6.15  |
| #4   | Drop the Signal   | Flux Circuit  | electronic | 4.67  |
| #5   | Bailando en Fuego | La Tormenta   | latin      | 4.66  |

The 2.78-point gap between #1 and #2 is entirely explained by mood: Sunrise City is tagged `happy` (matching the profile) while Gym Hero is `intense`. Genre matches both songs equally (+3.0 pts each). This makes the scoring logic easy to audit.

---

### 3. Classic Profile — Extreme Acoustic Minimalist (edge case)

**Profile settings:** genre=classical · mood=melancholy · energy=0.10 · acoustic=true

| Rank | Song               | Artist         | Genre     | Score |
| ---- | ------------------ | -------------- | --------- | ----- |
| #1   | Sonata in Grey     | Clara Voss     | classical | 9.63  |
| #2   | Empty Porch        | River Hen      | folk      | 4.31  |
| #3   | Spacewalk Thoughts | Orbit Bloom    | ambient   | 3.94  |
| #4   | Dust and Rain      | Hound Freely   | blues     | 3.81  |
| #5   | Library Rain       | Paper Lanterns | lofi      | 3.74  |

This is an intentional stress test. The catalog has only one classical song, so #1 nearly maxes out (9.63) and everything else scores below 4.5 — a 5-point cliff that exposes a real limitation: single-genre catalogs fail users beyond the top result.

---

## Design Decisions

### Why content-based filtering instead of collaborative filtering?

Collaborative filtering requires behavioral data — play counts, skips, ratings — that does not exist for a new catalog. Content-based filtering works from song features alone and produces fully explainable results: every recommendation can be traced back to a specific feature match and weight. For a 20-song demo catalog this is the only viable approach, and the explainability is actually a feature for a portfolio project.

### Why RAG rather than asking Gemini to recommend from memory?

Gemini has no knowledge of the specific 20 songs in `songs.csv` — they are not in its training data. Without retrieval, the model would hallucinate song names. By indexing the catalog in memory and injecting a summary into the system prompt before the model speaks, every recommendation is anchored to real retrieved data. This is the same principle used in enterprise RAG systems, just at a smaller scale.

### Why agentic tool use instead of a single prompt?

A single-shot prompt would require hand-crafting all 20 song descriptions into the context window on every query. Tool use lets the model pull only what it needs: it calls `search_songs` to narrow to a genre, then `score_and_rank` to get objective scores, then writes its final answer. This mirrors production RAG architectures and keeps the prompt lean. The trade-off is latency — two or three API round-trips add ~2 seconds compared to a single call.

### Why a 6-iteration safety cap on the agentic loop?

Without a cap, a misbehaving model could loop indefinitely and exhaust the API quota. Six iterations is more than enough for any realistic query (observed maximum in testing: 2 iterations) while preventing runaway behavior. This is a standard guardrail in agentic systems.

### Why `gemini-flash-latest` instead of a heavier model?

Speed and cost. The task — parsing a music preference, calling two tools, writing a short list — does not require the reasoning depth of a frontier model. Flash-tier models handle it in under 2 seconds. For a class project demo, responsiveness matters more than marginal quality gains.

### Trade-offs accepted

| Decision              | Benefit                           | Cost                                                 |
| --------------------- | --------------------------------- | ---------------------------------------------------- |
| 20-song CSV catalog   | Simple, reproducible, no scraping | Genre gaps make edge-case profiles unreliable        |
| Binary genre matching | Transparent, zero false positives | Rock ≠ Metal even though they are adjacent           |
| Fixed weight ordering | Deterministic, auditable          | Users cannot reorder priorities without editing code |
| In-memory index       | Zero latency on retrieval         | Does not scale beyond a few thousand songs           |

---

## Scoring Formula

Each song receives a score out of **10.0** — the sum of six weighted terms:

```
score = genre_pts + mood_pts + energy_pts + acousticness_pts + valence_pts + tempo_pts
```

| Feature      | Type                  | Max pts | Rationale                                                |
| ------------ | --------------------- | ------- | -------------------------------------------------------- |
| Genre        | categorical match     | 3.0     | ~8% random match rate; defines the entire sonic world    |
| Mood         | categorical match     | 2.0     | Strong signal, but less precise than genre               |
| Energy       | continuous similarity | 2.0     | Widest numeric range (0.22–0.97); best single feel proxy |
| Acousticness | continuous similarity | 1.5     | Cleanly separates organic vs produced                    |
| Valence      | continuous similarity | 1.0     | Users tolerate wider variation here                      |
| Tempo        | continuous similarity | 0.5     | Partially redundant with energy; needs BPM normalization |

Continuous features use a proximity formula: `weight × (1 − |song_value − user_target|)`.

---

## Reliability and Evaluation

### Test suite summary

**27 tests, 27 passed — all run offline without an API key.**

```
pytest tests/            # runs all 27 tests in ~0.1 seconds
python tests/evaluate.py # prints the human-readable evaluation report
```

| Test file                | Tests | What it covers                                                                                                                             |
| ------------------------ | ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_ai_recommender.py` | 20    | Tool execution (get_catalog_info, search_songs, score_and_rank), index construction, full agentic loop (mocked Gemini), API error handling |
| `test_evaluate.py`       | 5     | Genre precision, High-confidence score floor (≥7.0), determinism, missing-genre graceful degradation, confidence label thresholds          |
| `test_recommender.py`    | 2     | Recommender sort order, explanation non-empty                                                                                              |

### Automated evaluation results

The evaluation script (`tests/evaluate.py`) runs all 6 profiles through the scoring engine and measures four metrics:

```
Profile                          Top-1 Result              Score   Conf    Genre Match
─────────────────────────────────────────────────────────────────────────────────────
Late-Night Study (Lofi/Chill)    Library Rain (lofi)        9.67   High       ✓
High-Energy Pop Fan              Sunrise City (pop)         9.53   High       ✓
Deep Intense Rock                Storm Runner (rock)        9.74   High       ✓
Conflicting: High Energy + Sad   Iron Cathedral (metal)     7.81   High       ✓
Unknown Genre (k-pop)            Sunrise City (pop)         6.67   Medium     ✗  ⚠
Extreme Acoustic Minimalist      Sonata in Grey (classical) 9.63   High       ✓
─────────────────────────────────────────────────────────────────────────────────────
Genre coverage       : 5/6 profiles have their genre in catalog  (83%)
Top-1 genre precision: 5/6 top results matched requested genre   (83%)
Average top-1 score  : 8.84 / 10.0
Determinism          : 6/6 identical results on 3 consecutive runs (100%)
```

**In plain English:** 5 out of 6 profiles returned a perfect genre match at #1. The 1 miss (`k-pop`) is a known catalog gap — no k-pop songs exist in the CSV, so the system falls back to the best numeric match and signals this with a **Medium confidence** badge in the UI. The scoring pipeline is 100% deterministic.

### Confidence scoring

Every song card in the UI displays a confidence label derived from its score:

| Score     | Label              | Meaning                                                |
| --------- | ------------------ | ------------------------------------------------------ |
| ≥ 7.0     | **High** (green)   | Genre matched (+3 pts guaranteed); features align      |
| 4.0 – 6.9 | **Medium** (amber) | Genre absent from catalog; best numeric fit            |
| < 4.0     | **Low** (red)      | Catalog gap; system is extrapolating from weak signals |

A score below 7.0 is only possible when the genre bonus (3.0 pts) is missed, which always indicates a catalog gap. This makes the confidence tier a reliable, automatically-derived guardrail that tells users when the system is guessing.

### Logging and error handling

- Every layer (`recommender.py`, `ai_recommender.py`, `main.py`) uses Python's `logging` module. Run `python -m src.main --verbose` to see DEBUG-level trace including tool names, input arguments, result sizes, and iteration counts.
- API errors from Gemini are caught with a `try/except` block and return a human-readable message (`"Sorry — there was a problem contacting the AI service"`) instead of an uncaught exception.
- The agentic loop is capped at 6 iterations to prevent runaway API usage.

### What worked

- Mocked integration tests simulate the full two-iteration agentic loop (tool call → final answer) without hitting the API — the entire suite runs in 0.09 seconds.
- Type coercions in the tool dispatcher (`bool()`, `float()`) caught a real class of bug: Gemini sometimes returns `"true"` (string) instead of `true` (boolean) for `likes_acoustic`, which would silently produce wrong scores without the coercion.
- The `test_recommend_handles_api_error` test confirmed that network failures surface a safe message rather than a Python traceback.

### What didn't work / limitations found

- **Model name drift:** The original default (`gemini-1.5-flash`) was unavailable on the free-tier endpoint; the app silently returned an error until the model was changed to `gemini-flash-latest`. A startup check that calls `list_models()` and warns on mismatch would prevent this in production.
- **Catalog gaps in AI mode:** Gemini returns plausible-sounding results even when no genre match exists, with no warning. The confidence badge addresses this in classic mode; AI mode relies on Gemini's own phrasing to signal uncertainty.
- **gRPC shutdown warning:** A cosmetic `grpc_wait_for_shutdown_with_timeout() timed out` warning appears on exit — a known upstream issue in `google-generativeai` that does not affect results.

---

## Reflection and Ethics

### Limitations and biases in the system

**Catalog representation bias** is the most significant structural problem. The 20-song catalog was hand-curated with a clear Western pop-music worldview: it includes lofi, rock, jazz, and classical but no K-pop, Afrobeats, bhangra, or Latin genres beyond one token entry. A user whose entire musical identity sits outside this narrow slice gets a Medium or Low confidence result by default — not because their taste is unusual, but because the dataset does not represent them. This is a common and serious issue in real recommender systems: the catalog reflects whoever built it, and underrepresented users get systematically worse results.

**Genre binary matching** treats adjacent genres as completely unrelated. A rock listener gets zero genre points for a metal song, even though the two genres share instrumentation, production style, and audience overlap. The scoring formula cannot express "this is close but not exactly right" for categorical features — it is either a match (3.0 pts) or a total miss (0.0 pts). This penalizes users who have nuanced genre preferences.

**The popularity trap** is a subtler bias. Songs with broad numerical appeal — high energy, low acousticness, positive valence — float to the top of many profiles even when they do not match the requested genre or mood. In the experiments, _Gym Hero_ appeared in the top 3 for both pop and rock profiles despite being pop-tagged, because its numerical features matched the rock profile's energy target closely. In a real-world system this means a small number of songs get over-recommended, crowding out catalog diversity.

**Weight ordering is fixed and opinionated.** The system always treats genre as more important than mood, mood as more important than energy, and so on. A user who genuinely cares more about mood than genre has no way to express that preference without editing Python source code. The weights encode the designer's assumptions about what listeners value, not the actual listener's preferences.

---

### Could this AI be misused?

A music recommender seems harmless, but three misuse vectors are worth naming honestly.

**Mood manipulation.** The system can be deliberately configured to serve emotionally vulnerable users a steady stream of sad, low-valence music. If the `target_valence` and `mood` fields were set by a third party rather than the user, the recommender could function as a mood-suppression tool. Real streaming platforms have faced criticism for exactly this — algorithmic depression spirals where the recommendation engine keeps surfacing darker content because the user keeps engaging with it. This system has no session memory and no feedback loop, which limits this risk, but adding either feature without safeguards would recreate it.

**Payola-style catalog bias.** In a commercial deployment, the scoring weights could be quietly adjusted to favor songs from labels or artists who pay for placement. Because the weights are tuned by the developer and invisible to the user, the manipulation would be undetectable from the outside. The defense is transparency: publishing the weights and making the scoring formula auditable (as this project does) makes gaming visible.

**API key exposure and data privacy.** The AI Chat mode sends user queries to Google's Gemini API. If a user types something like _"recommend music for my breakup"_ or _"something for my anxiety"_, that personal context is transmitted to an external service and subject to Google's data retention policies. A responsible deployment would include a privacy notice before the first query, offer an opt-out, and avoid logging query content beyond what is necessary.

**Prevention in this project:** The `.env` file is git-ignored to prevent accidental key commits, the weights are fully documented in the README so anyone can audit them, and the scoring formula is plain readable Python with no hidden adjustments.

---

### What surprised me while testing reliability

Two things genuinely surprised me.

**The missing-genre fallback was more coherent than expected.** I designed the k-pop edge case to expose a hard failure — zero genre points, total collapse. What actually happened was a score of 6.67 with _Sunrise City_ (pop, happy, high energy) at #1. The numeric features found a reasonable stylistic neighbor even with no genre match. This was encouraging for robustness but also a little alarming: the system returned a confident-sounding result without any signal to the user that it was guessing. That is what motivated adding the confidence label — the raw score alone was not enough to distinguish "great match" from "best we could do."

**The model name failed silently in a way that looked like a bug.** When the default model was set to `gemini-1.5-flash`, the app returned the string `"Sorry — there was a problem contacting the AI service"` with no indication of why. The actual cause — a 404 from the API because that model name did not exist on the free-tier endpoint — was buried in a log line at DEBUG level. I spent time checking the API key, the network, and the tool dispatcher before finding it. The lesson: when an AI API fails, the error message is almost never self-explanatory, and catching the raw exception and logging its full text is far more important than I initially thought.

---

### Collaboration with AI during this project

**One instance where the AI suggestion was genuinely helpful.**

While building the tool dispatcher in `ai_recommender.py`, the AI pointed out that Gemini sometimes returns `"true"` (a string) instead of `true` (a boolean) for the `likes_acoustic` parameter in tool calls, depending on how the model formats its JSON output. Without the suggested fix — wrapping inputs in `bool()` and `float()` coercions before passing them to `score_song()` — the acousticness calculation would receive a string and silently produce a wrong score. This was not something I would have caught without running the live API dozens of times to observe the model's inconsistent output format. The suggestion reflected genuine knowledge of how LLM tool-call outputs behave in practice, and it protected a real failure mode that the tests alone would not have caught.

**One instance where the AI suggestion was flawed.**

The AI initially recommended `gemini-1.5-flash` as the default model — described as widely available, free-tier compatible, and stable. This turned out to be incorrect for this API key and endpoint configuration: the model returned a 404 error with `limit: 0` on the free tier. The AI had suggested it with confidence and without qualification, and I only discovered the problem after deploying and testing the live key. The correct model (`gemini-flash-latest`) was found by calling `list_models()` at runtime — something the AI should have suggested as a discovery step rather than assuming a specific version string would work. This is a reminder that AI suggestions about external service configurations require independent verification, because the model's training data may not reflect current API availability.

---

### What this project taught me about AI and problem-solving

Building this project from a scoring formula all the way to a live AI chat interface made three things concrete that were abstract before.

**Retrieval is what makes generation trustworthy.** Gemini cannot know what songs are in a custom CSV file. Without RAG injection the model hallucinated artist names and song titles with complete confidence. Adding retrieval did not just improve the output — it changed what the model was fundamentally doing: from pattern-matching on training data to reasoning about a specific dataset I control. That distinction is the difference between a demo and a reliable system.

**Agentic autonomy requires explicit limits.** Watching the model decide to call `search_songs`, then `score_and_rank`, then write its answer — without being told the sequence — was impressive. But the same autonomy that made it useful made it unpredictable. Without the 6-iteration cap and the `try/except` around every API call, a misbehaving model response could loop indefinitely or expose a raw stack trace to a user. Every degree of autonomy you give an AI system needs a corresponding guardrail.

**Explainability is infrastructure, not decoration.** The weighted scoring formula was a Module 1 teaching exercise. By the final system it was doing real work: Gemini used the `reasons` list from `score_and_rank` to write its explanations. A black-box similarity score would have made the AI's output less trustworthy and harder to debug. Building interpretable systems from the beginning is not just good ethics — it makes every downstream component easier to build and test.

---

## Limitations and Known Issues

- **Small catalog (20 songs):** Most genres have only 1–3 representatives. Niche genre requests degrade gracefully but cannot be truly satisfied.
- **No personalization over time:** The system has no memory between sessions. It cannot learn that a user always skips metal or consistently replays jazz.
- **Binary genre matching:** Rock and metal share no points despite being adjacent genres. A genre-similarity matrix would address this.
- **gRPC shutdown warning:** Cosmetic only — does not affect results. Upstream `google-generativeai` issue.

---

## Portfolio

### Video Walkthrough

> 🎬 **Loom walkthrough:** [(https://www.loom.com/share/bf0b02975e4e4b3ba93929199c747481)]
>
> The walkthrough demonstrates:
>
> - ✅ End-to-end system run (3 different inputs)
> - ✅ AI feature behavior — RAG injection + agentic tool loop in action
> - ✅ Reliability behavior — confidence labels and evaluation metrics
> - ✅ Clear outputs for each case

---

### What This Project Says About Me as an AI Engineer

I built this project in layers — starting from a transparent, hand-crafted scoring formula and ending with a natural-language AI interface — which reflects how I think about engineering: understand the fundamentals before adding abstraction. The scoring engine was fully working and auditable before a single API call was written, and that discipline paid off directly: when Gemini calls `score_and_rank`, it gets back objective numbers it can trust and explain, instead of a black box. I gravitate toward explainability as infrastructure rather than decoration.

The harder lesson was about the gap between "the AI said it would work" and "it actually works." The model confidently recommended a Gemini model name that did not exist on the free-tier endpoint, and the failure was silent — no crash, just a polite error string. Debugging it meant reading raw API responses, not trusting the AI assistant. That experience shaped how I now treat all AI-generated suggestions about external services: verify at runtime, log everything, degrade gracefully. Building reliable AI systems is less about picking the right model and more about what you do around it — retrieval, guardrails, error handling, and tests that run without the API at all.

---

## References

- [Google Gemini API — Function Calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Retrieval-Augmented Generation — Lewis et al. 2020](https://arxiv.org/abs/2005.11401)
- [Spotify Audio Features Reference](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)
