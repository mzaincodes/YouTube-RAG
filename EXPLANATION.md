# Engineering deep-dive

Three parts:

1. **[How it works](#part-1--how-it-works)** — the data path, module by module.
2. **[Problems faced](#part-2--problems-faced-and-how-they-were-solved)** — the real bugs and surprises hit while building this, and how each was fixed.
3. **[Scalability](#part-3--what-makes-this-scalable)** — the specific design decisions that let this grow past a demo.

---

# Part 1 — How it works

There are two independent pipelines: **indexing** (runs when you add videos) and **querying** (runs on every question).

```
INDEXING   URLs → validate → fetch transcript → timed units → semantic chunks → embed → ChromaDB
QUERYING   question → condense → retrieve → MMR → compress → context → Gemini → streamed answer + citations
```

## 1.1 URL parsing (`utils.py`)

A YouTube video ID is always 11 characters of URL-safe base64, which makes validation exact rather than heuristic. `extract_video_id()` handles `watch?v=`, `youtu.be/`, `/embed/`, `/shorts/`, `/live/`, `/v/`, `music.youtube.com`, `youtube-nocookie.com`, and bare IDs — then verifies the hostname against an allow-list so a lookalike domain cannot slip through.

`parse_url_input()` splits free-form text on whitespace/commas/semicolons and returns `(video_ids, invalid_entries)`, de-duplicating while preserving order. The sidebar calls this on every keystroke to show the live "✅ 2 valid · ⚠️ 1 invalid" counter.

## 1.2 Transcript extraction (`transcript.py`)

Two sources are combined:

- **`youtube-transcript-api`** for timed caption cues (`text`, `start`, `duration`).
- **YouTube's oEmbed endpoint** for title, channel and thumbnail — the transcript library deliberately does not expose them, and the official Data API would need a second key and quota.

Transcript selection tries, in order: manual captions in your preferred languages → auto-generated in those languages → any manual captions (translated if possible) → anything at all. Human-written captions beat ASR because they are punctuated, which materially improves chunking (see §1.3).

Every library exception is translated by `_friendly_error()` into a sentence a non-technical user can act on. `TranscriptsDisabled` becomes *"Subtitles are disabled for this video"*, `IpBlocked` becomes *"YouTube is temporarily blocking transcript requests from this IP"*, and so on. Failures are collected per video, so **one bad link never fails the batch**.

Videos are fetched through a `ThreadPoolExecutor` — the work is pure network latency, so a small pool cuts wall-clock time on a playlist-sized batch roughly linearly. Results are re-sorted into the user's original order afterwards, because completion order is nondeterministic.

## 1.3 Chunking — the interesting part (`chunking.py`)

Naive fixed-length splitting is the single biggest quality killer in transcript RAG. Caption cues are 1–5 second fragments with no relationship to meaning; cutting every 1000 characters reliably slices sentences and topics in half, producing chunks that retrieve poorly and read badly when quoted.

**Pass 1 — caption cues → timed units.** Cues are merged into the largest natural unit available:

- If the transcript **is punctuated** (human captions), units are *sentences*. A cue may contain several sentences, so complete ones are emitted and the remainder carried forward. Each sentence's timestamp is interpolated across the cue's span proportionally to text length.
- If the transcript **is not punctuated** — YouTube ASR output contains no full stops whatsoever — sentence splitting would return one giant blob. The `_is_punctuated()` heuristic (sentence terminators per word) detects this and falls back to ~24-second *time windows*.

Either way, every unit carries a real start and end timestamp. **This is what makes citations deep-link accurately**, and it survives every later transformation.

**Pass 2 — units → chunks.** With an embedding model available, all units are embedded and the cosine distance between *consecutive* units is computed. A large distance means the speaker changed subject. Breakpoints go where distance exceeds a high percentile of the distribution, so chunk boundaries land on topic boundaries. Two guards then apply:

- Groups exceeding `chunk_size × 2` are split at unit boundaries.
- Groups below `SEMANTIC_MIN_CHUNK_CHARS` are folded into the previous group — but only while the result stays under `chunk_size`, so a run of short groups cannot cascade into one oversized, topic-mixed chunk.

Overlap is applied as a word-boundary-aligned tail of the previous chunk, so context is never cut mid-word.

Each chunk gets a deterministic ID: `{video_id}:{index:05d}:{sha256(text)[:16]}`. Same video + same content = same ID, which is what makes re-indexing idempotent (§1.5).

Chunk metadata: `video_id`, `title`, `author`, `url`, `thumbnail`, `chunk_index`, `total_chunks`, `start_seconds`, `end_seconds`, `timestamp`, `timestamp_url`, `language`, `auto_generated`, `char_count`, `source`.

## 1.4 Embeddings (`embeddings.py`)

`GeminiEmbeddings` wraps `GoogleGenerativeAIEmbeddings` and implements LangChain's `Embeddings` protocol, so it drops straight into Chroma. It adds:

- **Bounded batching** — Google rejects requests over 100 inputs.
- **Exponential backoff with full jitter**, applied *only* to transient failures. Errors are classified: a 429 or 503 is retried; an invalid API key or missing model fails immediately rather than burning five retries on a certain failure.
- **A per-process query cache** — repeated questions cost nothing.
- **Model fallback** — `gemini-embedding-001` → `text-embedding-004` → `embedding-001`, so a model retirement degrades instead of breaking.
- **Zero-vector handling** for empty strings, which the API rejects.

One subtlety: the instance-level `task_type` is deliberately left unset. The underlying class then defaults to `RETRIEVAL_DOCUMENT` when embedding documents and `RETRIEVAL_QUERY` when embedding queries — the asymmetry that retrieval quality depends on. Setting `task_type` explicitly would override *both* and silently degrade every search.

## 1.5 Vector store (`vector_store.py`)

`VectorStoreManager` owns the single Chroma handle and its lifecycle.

**Idempotent indexing.** Deterministic chunk IDs mean re-adding identical content upserts rather than duplicating. When a video is *re-indexed with different settings*, old vectors are deleted first (`where={"video_id": …}`) so changing `CHUNK_SIZE` cannot leave orphans behind.

**The video registry.** Chroma can answer "how many vectors?" cheaply but not "which videos, with what titles?" — that needs scanning every metadata row. Since the sidebar renders that list on every rerun, an O(N) scan per keystroke would be unacceptable. So the manager maintains `chroma_db/video_registry.json`: a small, atomically-written sidecar (temp file + `os.replace`) holding one record per video. If it is ever missing or corrupt, it is transparently rebuilt from vector metadata.

**Embedding-space guard.** Vectors from different embedding models are mathematically incomparable — mixing them returns confident nonsense rather than an error. The manager writes an `embedding_fingerprint` onto the collection metadata and compares it on startup, surfacing a visible warning if you changed `EMBEDDING_MODEL` under an existing index.

**Metadata sanitisation.** All chunk metadata is coerced to `str`/`int`/`float`/`bool` before writing, matching Chroma's documented contract.

## 1.6 Retrieval (`retriever.py`)

Over-fetch → diversify → threshold → dedupe → compress.

- **MMR** balances relevance against novelty so the top-k is not five near-identical chunks of the same passage. It is implemented in NumPy over candidate vectors read back from Chroma (see [Problem 7](#7-chroma-gives-you-scores-or-mmr-never-both)).
- **Score thresholding** (optional) drops weak matches.
- **Deduplication** collapses chunks whose first 400 characters hash identically — overlap makes this common.
- **Context compression** trims long chunks to their most query-relevant sentences using lexical overlap scoring. This is deliberately *not* an LLM-based compressor: compression runs on every chunk of every turn, so it must cost zero tokens and add zero latency. Sentence order is preserved so excerpts still read naturally.

## 1.7 The LangGraph workflow (`graph.py`)

```
START → condense_query → retrieve ─┬─ chunks found ──→ format_context → generate → END
                                   └─ nothing found ──→ no_context ──────────────→ END
```

- **`condense_query`** rewrites follow-ups ("what about that?") into standalone search queries using chat history. It uses a *separate, non-streaming, temperature-0* model so its tokens can never leak into the streamed answer. If rewriting fails, the raw question is used — it is an optimisation, never a hard dependency.
- **`retrieve`** runs the retrieval pipeline.
- **`format_context`** renders chunks into a numbered `CONTEXT` block under a hard character budget.
- **`generate`** streams from Gemini.
- **`no_context`** returns the exact refusal sentence **without calling the model at all**. This is the strongest anti-hallucination guarantee in the system: the fixed string is a code path, not a request the model might decline to honour, and it costs nothing.

Every node is a small function over a `TypedDict` state, so adding a reranker or a guardrail means adding a node and an edge.

**Streaming** uses `stream_mode=["messages", "values"]`, which yields token deltas *and* the terminal state. Tokens are filtered by `metadata["langgraph_node"] == "generate"`. The public API yields `("token", str)` events and exactly one `("final", RagResponse)`.

## 1.8 Prompting (`prompts.py`)

The system prompt fixes grounding rules, the exact refusal sentence, and citation format. Two deliberate choices:

- **Context precedes the question.** The final tokens before generation restate the task, which measurably improves instruction adherence.
- **No template engine.** Prompts are assembled by plain concatenation — see [Problem 11](#11-transcripts-contain-curly-braces).

## 1.9 UI (`app.py` + `ui.py`)

`app.py` owns flow only: session state, sidebar workflow, chat loop. `ui.py` owns markup. Service objects are built once behind `@st.cache_resource` keyed on a configuration fingerprint, because Streamlit re-executes the entire script on every interaction — without caching, the Chroma handle and compiled graph would be rebuilt on every keystroke. All third-party strings (titles, channel names) are HTML-escaped before interpolation.

---

# Part 2 — Problems faced, and how they were solved

These are the actual issues encountered while building this, in the order they appeared.

### 1. Python 3.14 had only just been released

The target machine had **only Python 3.14.6** — no pyenv, no conda, no Homebrew Python. Several of the heavier dependencies (`chromadb`, `onnxruntime`, `tokenizers`, `pyarrow`) ship compiled wheels, and a missing `cp314` wheel means a source build that usually fails.

**Solution:** resolve the dependency graph *before* writing a line of code, using `pip install --dry-run` on the full stack. Every package resolved with native `cp314` wheels, so 3.14 was safe. Had it not been, the alternative was pinning to 3.12. **Verifying the environment first turned a potential mid-project rewrite into a two-minute check.**

### 2. `youtube-transcript-api` 1.x removed the API every tutorial uses

Every guide online calls `YouTubeTranscriptApi.get_transcript(video_id)` — a static method. In 1.x that method **does not exist**; the library moved to an instance API:

```python
# Every tutorial (and pre-1.0 releases) — raises AttributeError today
transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

# Correct for 1.x
api = YouTubeTranscriptApi()
fetched = api.fetch(video_id, languages=["en"])
segments = fetched.snippets            # objects with .text / .start / .duration
```

Writing from memory would have produced code that fails on the first click.

**Solution:** introspect the installed library (`inspect.signature`, `dir()`) instead of trusting recall, and pin `>=1.0` in `requirements.txt` with a comment explaining why downgrading breaks. The same check caught that `FetchedTranscript` exposes `.snippets`, `.language`, `.language_code` and `.is_generated`, and that `TranscriptList` is iterable — all of which the fallback logic uses.

### 3. The transcript library returns no video title

`youtube-transcript-api` returns captions and nothing else. Citations showing "Video dQw4w9WgXcQ" are useless. The obvious fix — the YouTube Data API — needs a second API key, OAuth setup and quota, which contradicts the "one key, zero config" goal.

**Solution:** YouTube's public **oEmbed** endpoint returns title, channel and thumbnail with no key, no quota and no auth:

```
GET https://www.youtube.com/oembed?url=<video_url>&format=json
```

It returns HTTP 400 for invalid/private videos, which doubles as a liveness check. Metadata is cosmetic, so any failure degrades to a placeholder rather than aborting indexing.

### 4. `CERTIFICATE_VERIFY_FAILED` on macOS

The first oEmbed call using `urllib` died with:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate
```

This is the classic python.org-installer-on-macOS problem: the bundled Python does not use the system keychain, and its certificate store is empty until you run `Install Certificates.command`. Shipping an app that requires users to run a setup script first is a bad experience.

**Solution:** use `requests` instead of `urllib`. It bundles `certifi` and carries its own CA store, so it works on a stock install with no user action. `requests` was already in the tree (both `chromadb` and `youtube-transcript-api` depend on it), so this cost nothing.

### 5. ChromaDB rejects short collection names

A test using `collection_name="t"` failed with:

```
InvalidArgumentError: Expected a name containing 3-512 characters from
[a-zA-Z0-9._-], starting and ending with a character in [a-zA-Z0-9]
```

A user setting `CHROMA_COLLECTION=yt` in `.env` would hit an unhandled crash at startup.

**Solution:** `sanitize_collection_name()` coerces any input into a legal name — replacing illegal characters, padding short names, trimming to 512 — and falls back to `youtube_rag` with a warning if the result is still invalid.

### 6. Chroma defaults to L2, which broke relevance scores

Relevance percentages in the UI were **negative**, with Chroma emitting:

```
UserWarning: Relevance scores must be between 0 and 1, got [-0.022, -0.041]
```

Two causes. First, Chroma's default distance metric is **L2, not cosine** — confirmed by probing `configuration_json` across three construction styles:

| Construction | Resulting space |
|---|---|
| `collection_metadata={"hnsw:space": "cosine"}` | `cosine` |
| `collection_configuration={"hnsw": {"space": "cosine"}}` | `cosine` |
| *(default)* | **`l2`** |

Second, even with cosine, LangChain's default relevance function is `1 - distance`, and cosine distance ranges over `[0, 2]` — so anything past orthogonal goes negative.

**Solution:** set `hnsw:space: cosine` explicitly *and* supply a correct normalisation, `1 - distance/2`, which maps identical→1.0, orthogonal→0.5, opposite→0.0. Scores now land in `[0, 1]` and the warning is gone. `collection_metadata` was chosen over `collection_configuration` because it also carries the embedding fingerprint in the same dict.

### 7. Chroma gives you scores **or** MMR, never both

The UI shows a relevance percentage on every citation, and MMR is wanted for diversity. But `langchain-chroma` exposes them through different methods: `similarity_search_with_relevance_scores()` returns scores without diversification, and `max_marginal_relevance_search()` returns diversified documents **with the scores discarded**.

**Solution:** implement MMR directly. Candidates are over-fetched *with* scores, their vectors are read back via `store.get(include=["embeddings"])` — a local disk read, no API call — and the maximal-marginal-relevance selection runs in NumPy. This keeps both properties, and made the trade-off tunable via `MMR_LAMBDA`. Because `get()` does not preserve query ordering, results are realigned by ID before selection.

### 8. Semantic chunking missed real topic boundaries (found by a test)

An assertion that no chunk should mix three unrelated topics **failed**. Investigation showed the percentile threshold was the culprit. With nine sentence gaps — seven near-zero, two large (0.977 and 0.981) — `np.percentile(distances, 90)` interpolates linearly and returns **0.9778**, which lands *between the two real topic shifts*. Only the larger one became a breakpoint; the other topic change was silently missed.

```python
np.percentile(d, 90)                   # 0.9778 → 1 of 2 breakpoints  ✗
np.percentile(d, 90, method="lower")   # 0.977  → 2 of 2 breakpoints  ✓
```

**Solution:** use `method="lower"`, which snaps the threshold to an *observed* distance instead of inventing a value between two data points. The topic-coherence test then passed. This bug would have quietly degraded retrieval quality on every video — it was only visible because the test asserted on semantic behaviour rather than just "did it return chunks".

A second, related fix: `_merge_undersized()` folded small groups into their predecessor unconditionally, so a run of short groups could cascade into one giant chunk spanning every topic — undoing the semantic split. It now refuses a merge that would exceed `chunk_size`.

### 9. Auto-generated captions have no punctuation at all

Sentence-based chunking assumes sentences exist. YouTube ASR output looks like:

```
so today we're talking about langchain and how it works it's really useful for
building applications you can use it with many different models and providers
```

No full stops anywhere. Sentence splitting returns **one unit** for the entire video, which then gets brute-force split by character count — exactly the naive behaviour being avoided.

**Solution:** detect it. `_is_punctuated()` measures sentence terminators per word; below a 1% ratio the transcript is treated as ASR output and units become ~24-second time windows instead. Semantic grouping then runs normally on top of those units, so auto-captioned videos still get topic-aware chunks.

### 10. Setting `task_type` would have silently degraded every search

`GoogleGenerativeAIEmbeddings` accepts a `task_type` field, and setting it to `RETRIEVAL_DOCUMENT` looks correct for an indexing pipeline. Reading the installed source showed it would be a bug:

```python
effective_task_type = task_type or self.task_type or "RETRIEVAL_DOCUMENT"  # embed_documents
effective_task_type = task_type or self.task_type or "RETRIEVAL_QUERY"     # embed_query
```

The class already does the right asymmetric thing **only while `self.task_type` is `None`**. Setting it at instance level overrides *both* paths, so queries would be embedded as documents — degrading retrieval quality with no error, no warning and no visible symptom.

**Solution:** deliberately leave `task_type` unset, with a comment explaining why, so nobody "fixes" it later.

### 11. Transcripts contain curly braces

`ChatPromptTemplate` and `str.format()` treat `{` and `}` as variable syntax. Transcripts of programming talks are full of them — `{"key": "value"}`, `${VAR}`, f-string examples. Passing retrieved text through a template engine raises `KeyError` on a stray brace, or worse, lets transcript content act as a template variable.

**Solution:** assemble every prompt by plain string concatenation. Only values under application control are ever interpolated; retrieved text is appended, never formatted. The integration test deliberately includes a chunk containing `{braces}`.

### 12. Streamlit crashes when a widget has both a `key` and a default

The first sidebar implementation seeded widget state as `None` and passed `value=`/`index=`:

```python
st.session_state.setdefault("top_k", None)
st.select_slider("k", options=range(2, 21), value=st.session_state.top_k or 6, key="top_k")
```

Streamlit raises when a widget's `key` already exists in session state *and* an explicit default is supplied — and `None` is not a valid option anyway.

**Solution:** seed widget-backed keys with **real defaults** from settings, then declare the widgets key-only with no `value`/`index`. A related crash was fixed at the same time: deleting a video left its ID in the `video_filter` multiselect, and Streamlit rejects a default that is not in `options`. Stale IDs are now pruned before the widget is created.

### 13. Verifying the UI without a browser

A Streamlit app that imports cleanly can still crash on first render. `curl` returns HTTP 200 for the SPA shell whether or not the Python script works, so it proves nothing.

**Solution:** use `streamlit.testing.v1.AppTest`, which executes the script headlessly and exposes rendered elements and exceptions. This caught the widget-key crash above and verified the no-API-key path renders a clean error instead of a traceback.

Better still, the full app was tested **end-to-end offline** by patching only `src.embeddings.build_embeddings` and `src.graph.build_llm` before the run. Because `from X import y` resolves at execution time, the patches are picked up by the freshly-executed script — so the test drives a **real YouTube fetch, real chunking, a real ChromaDB write and the real LangGraph workflow**, faking only the paid API calls. It clicks *Process videos*, asks a question, sends a follow-up, clears the chat and deletes the video, asserting no exceptions throughout.

### 14. Two models, one stream

Query condensation and answer generation both call Gemini. Naively, the condensing model's tokens appear in the streamed answer, so the user watches a rewritten search query get typed out before the real answer starts.

**Solution:** filter streamed chunks by `metadata["langgraph_node"] == "generate"`, and give the condenser a separate non-streaming instance. LangGraph's `stream_mode="messages"` tags every chunk with its originating node, which makes this exact.

### 15. A model that the API lists but refuses to run (found in production)

After the app was deployed with a real free-tier key, indexing worked but every
question failed with *"The configured Gemini model was not found."* The raw
exception told the real story:

```
404 NOT_FOUND: This model models/gemini-2.5-flash is no longer available to new
users. Please update your code to use a newer model.
```

Two things made this hard to diagnose:

- **The models endpoint still lists it.** `client.models.list()` returns
  `models/gemini-2.5-flash` for that key, so availability genuinely cannot be
  determined from configuration or from the catalogue — only by *calling* it.
  Google grandfathered existing users and blocks keys created after the
  deprecation date.
- **My error message pointed at the broken model.** The 404 branch of
  `friendly_llm_error()` recommended "gemini-2.5-flash or gemini-2.5-pro" — the
  exact models that do not work for a new key. Technically-correct classification,
  actively misleading advice.

Probing the key's whole catalogue revealed a third issue: `gemini-2.5-pro`,
`gemini-pro-latest` and `gemini-2.0-flash` all returned **429 RESOURCE_EXHAUSTED**
— Pro-tier models have almost no free-tier quota, so recommending them as a
fallback would have swapped one failure for another.

**Solution — three changes:**

1. **Default to an alias.** `MODEL_NAME` is now `gemini-flash-latest`, which
   tracks whatever the current Flash model is. Concrete version numbers rot;
   aliases do not.
2. **Automatic fallback, matching the embedding layer.** `resolve_chat_model()`
   probes the configured model with one tiny request at startup and walks a
   candidate list if it fails, treating 404/403/429 as "unusable, try the next"
   and a network blip as "keep the choice". The chain ends with the older 2.x
   names so grandfathered keys keep working. When a substitution happens the UI
   shows a notice naming the model actually in use.

   This was the real lesson: **the same fallback already existed for embeddings
   and was simply missing for the chat model** — an inconsistency that only a
   live key exposed.
3. **Honest error messages.** "no longer available" now gets its own branch,
   distinct from a mistyped model name, and every suggestion points at the alias.

A fourth issue surfaced while verifying the fix. Probing with
`max_output_tokens=16` returned **empty content** from `gemini-flash-latest` and
`gemini-3.5-flash`: these are *thinking* models, and reasoning tokens are drawn
from the same budget as the visible answer, so a small budget can be consumed
entirely before any text is produced. `MAX_OUTPUT_TOKENS` was raised from 2048 to
4096 so long summaries cannot be truncated to nothing. The probe itself ignores
content and only checks that no exception was raised.

---

# Part 3 — What makes this scalable

Scalability here means four things: **more videos**, **more users**, **more question volume**, and **more source types**. Each got specific attention.

## 3.1 Indexing scales sub-linearly in cost

| Technique | Effect |
|---|---|
| **Deterministic chunk IDs** (`{video_id}:{index}:{sha256}`) | Re-indexing identical content upserts instead of duplicating. Re-running the same batch is free. |
| **Incremental indexing** | Already-indexed videos are skipped by a registry lookup, not a vector scan. Adding video #500 costs the same as adding video #2. |
| **Delete-before-replace** | Re-indexing with new chunk settings removes stale vectors first, so the collection cannot accumulate orphans over time. |
| **Parallel transcript fetch** | Thread pool over pure-I/O work; a 20-video batch finishes in roughly the time of the slowest video, not the sum. |
| **Batched embeddings** | Bounded at Google's 100-input limit, default 64. One request per 64 chunks instead of one per chunk. |
| **Semantic-chunking cost guard** | `SEMANTIC_MAX_SENTENCES_PROBE` (1500 units) caps the extra embedding pass. Past that, a multi-hour transcript falls back to structural chunking automatically — cost stays bounded without operator intervention. |
| **Retry only what's retryable** | Errors are classified; a bad API key fails in 1 attempt, not 5 × exponential backoff. |

## 3.2 Query cost is bounded, not proportional to the index

This is the property that matters most. **A question against 500 videos costs the same as a question against 5:**

- **ANN search** — Chroma's HNSW index is sub-linear in collection size.
- **Fixed `k`** — exactly `RETRIEVAL_K` chunks reach the model regardless of index size.
- **Hard context ceiling** — `MAX_CONTEXT_CHARS` caps assembled context; `build_context_block()` stops adding blocks at the budget, so the prompt cannot outgrow the context window no matter how much was retrieved.
- **Bounded history** — `HISTORY_TURNS` caps conversation growth, so turn 50 costs what turn 5 did.
- **Per-document budget** — compression trims each chunk to `MAX_CONTEXT_CHARS / k`.
- **Zero-cost compression** — lexical rather than LLM-based, so it adds no tokens and no latency per turn.
- **Query embedding cache** — repeated or refreshed questions skip the API entirely.

## 3.3 The registry pattern — avoiding an O(N) scan on every rerun

Streamlit re-executes the whole script on **every keystroke**. The sidebar shows every indexed video with title, channel and chunk count. Reading that from Chroma means loading every metadata row — at 500 videos × 100 chunks that is 50,000 rows per keystroke.

The `video_registry.json` sidecar reduces this to one small file read (and in practice one in-memory dict lookup, since it is cached on the manager). It is written atomically (temp file + `os.replace`, so a crash mid-write cannot corrupt it) and is treated as a **cache, not a source of truth** — if it is missing or unparseable it is rebuilt from vector metadata. The expensive path exists but is only ever a recovery path.

## 3.4 Swappable components

Every external dependency sits behind an interface, so scaling up means changing one constructor:

| Component | Today | Scale-up path |
|---|---|---|
| Vector store | Embedded Chroma (`persist_directory`) | `chromadb.HttpClient` server mode, or Qdrant/pgvector — only `_open_store()` changes |
| Embeddings | `GeminiEmbeddings` | Anything implementing LangChain's `Embeddings` protocol; the fallback chain already proves the seam works |
| LLM | `ChatGoogleGenerativeAI` | Any `BaseChatModel`; `RagPipeline` accepts injected models |
| Chat history | Session state | LangGraph checkpointer (`InMemorySaver` → `SqliteSaver` → `PostgresSaver`) |
| Source type | YouTube transcripts | Any loader producing `(text, metadata)` — chunking and indexing are source-agnostic |

The dependency-injection seam on `RagPipeline(retriever, settings, llm=…, condense_llm=…)` exists for testing, but it is the same seam you would use to route different users to different models.

## 3.5 The graph is the extension point

Adding a capability means adding a node and an edge, not restructuring control flow:

```python
builder.add_node("rerank", self._node_rerank)       # cross-encoder reranking
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "format_context")
```

The conditional edge after `retrieve` already demonstrates branching, so a web-search fallback for out-of-corpus questions, a guardrail node after `generate`, or a multi-query fan-out all fit the existing shape. Because nodes are plain functions over a `TypedDict`, each is unit-testable in isolation.

## 3.6 Correctness safeguards that prevent silent scaling failures

Systems fail quietly at scale. Three guards catch the failure modes that produce *wrong answers* rather than errors:

- **Embedding fingerprint** — mixing vectors from two embedding models returns confident nonsense, not an error. The collection records which model built it and the app warns loudly on mismatch.
- **Metadata sanitisation** — all values coerced to Chroma's supported scalar types, so a metadata change cannot fail a write mid-batch.
- **Per-video error isolation** — one failed video is reported and skipped; the other nineteen still index.

## 3.7 Operational readiness

- **Configuration is environment-driven and validated** — every value is range-checked and clamped (`chunk_overlap` is forced below `chunk_size`, `fetch_k` below `k` is corrected, unknown `SEARCH_TYPE` falls back with a warning). Bad config produces a warning and a safe default, not a crash.
- **Structured logging** at every boundary, with chatty third-party loggers silenced, and handlers installed exactly once despite Streamlit's reruns.
- **No secrets in logs, vectors or UI** — only the last six characters of the key are used, as a cache-invalidation token.
- **Lazy initialisation** — the Chroma handle opens on first use; the registry loads on first read.

---

## Verification performed

Everything above was tested against the **real installed libraries**, not assumed:

- **API surface probes** — `youtube-transcript-api` 1.2.4, `langchain-google-genai` 4.3.2, `langgraph` 1.2.10, `langchain-chroma` 1.1.0 were each introspected before use.
- **55 integration checks** covering URL parsing (9 forms), chunking (both punctuated and ASR paths, reproducibility, topic coherence), the vector store (idempotency, persistence, registry rebuild, deletion, filters), retrieval (score ranges, MMR at three λ values, compression) and the graph (streaming, history, the empty-database short-circuit).
- **A live-API run against a real free-tier key** — indexing, a streamed multi-paragraph summary with six timestamped citations, a history-aware follow-up, and a correct refusal on an out-of-corpus question.
- **22 end-to-end app checks** driving the real Streamlit UI: boot, index a real YouTube video, ask a question, follow up, clear chat, delete video — asserting zero exceptions at every step.
- **Fresh-environment install** from `requirements.txt` to confirm the pinned set is complete and sufficient.

Every component, including Gemini, has now been exercised against the live API.
