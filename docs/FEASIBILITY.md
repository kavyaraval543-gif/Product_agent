# Phase 1 — Feasibility Assessment

_Everything marked **[verified]** was tested against the live service on **2026-10-17** from the development environment (4 vCPU, 16 GB RAM, no GPU). Anything I could not test is marked **[not verified]**. Nothing here is assumed to still be true later: sources change, so `pda doctor` re-checks robots.txt on every setup._

## 1. Verdict

**Feasible at ₹0, with one important finding: the two most obvious review sources — Apple App Store and Google Play — are not usable for an agent that respects robots.txt.** A genuine, permitted, key-less source exists: the **Mozilla Add-ons (AMO) public ratings API**, which returns star rating, review text, date, product version and a permalink. The agent is built on it. Because every source sits behind a small interface (`src/pda/sources/base.py`), adding another later is a ~100-line change.

## 2. Source evaluation

| Source | Technically reachable? | robots.txt | Free & key-less? | Verdict |
|---|---|---|---|---|
| **Apple App Store review RSS** (`itunes.apple.com/{cc}/rss/customerreviews/...`) | Yes — returned HTTP 200 JSON **[verified]** | `User-agent: *` → **`Disallow: /*/rss/*`** (also `/search*`, `/*/lookup?`) **[verified]** | Yes | **Rejected.** The server answers, but Apple's robots.txt forbids it, and respecting robots.txt is a hard requirement. The matching test is in `tests/test_robots.py`. |
| **Google Play** (`play.google.com`) | Not attempted | `Disallow: /_` and `Disallow: /store/getreviews` **[verified]**. As I understand it, open-source Play scrapers use an internal `/_/…/batchexecute` endpoint, which `/_` covers **[not verified]** | The official Play Developer API only covers apps you publish **[not verified]** | **Rejected.** |
| **Mozilla Add-ons (AMO) ratings API** (`addons.mozilla.org/api/v5/ratings/rating/`) | Yes **[verified]** | `Allow: /`; only `/firefox/downloads/` and locale search/collection paths are disallowed — **`/api/` is allowed** **[verified]** | Yes, no key | **Selected.** |
| Steam reviews (`store.steampowered.com/appreviews/<appid>`) | Yes — JSON with text, date, vote **[verified]** | `/appreviews/` not disallowed **[verified]** | Yes | Good **second source** (not implemented). Thumbs up/down only — no star ratings. |
| Hacker News (Algolia API) | Yes **[verified]** | none published (404) **[verified]** | Yes | Useful later for discussion threads; not product reviews. |
| GitHub Issues API | Blocked in the dev sandbox (403) **[verified]**; works on a normal machine (60 req/h unauthenticated) **[not verified here]** | n/a | Yes | Good later source for open-source products. |
| Reddit, Trustpilot, G2 etc. | Not investigated | — | Typically need OAuth / have restrictive terms | Not pursued. |

### Why AMO, for a PM portfolio
- It is a real consumer product category (browser extensions) with real, current, complaint-rich reviews.
- Official, documented API (Mozilla's `addons-server` docs) — not scraping HTML.
- Has what a product analysis needs: **1–5 stars, free text, ISO date, product version, permalink**.
- The `app_version` field lets later phases link complaint spikes to specific releases.

### Selected application
**Bitwarden Password Manager** (`bitwarden-password-manager`) — chosen by sampling 10 popular add-ons for text volume and substance **[verified]**: 3,144 text reviews, median 142 characters, 28 of the latest 50 rated ≤2★, newest review 2026-10-08. Others rejected for this purpose: uBlock Origin (reviews short and nearly all 5★, median 64 chars, 1/50 low-rated), LastPass (newest review in the sample dated 2026-08-13, i.e. stale). The app is one line in `config.toml`; switching is a config change.

## 3. What was verified about the AMO API

| Question | Result **[verified]** |
|---|---|
| Can I fetch only reviews that contain text? | Yes: `filter=without_empty_body` |
| Maximum page size? | **50** — larger values are silently capped (requested 51 and 100, got 50) |
| Ordering? | **Newest first**, confirmed across 150 consecutive reviews. This is **not documented**, so the collector re-checks it on every page and disables early-stopping if it ever fails |
| Pagination? | `page` + `next`; stops cleanly at the end |
| Rate-limit headers? | None sent; no limit published. We use **≥1 s between requests** and honour `Retry-After` |
| Caching? | `Cache-Control: max-age=360` → brand-new reviews can lag ~6 minutes |
| Source-wide totals? | `show_grouped_ratings=true` returns star counts for **all** ratings (text or not) |
| Permalink to a single review? | `/en-US/firefox/addon/<slug>/reviews/<id>/` → **200** for a real id, **404** for a fake id |
| Is the API stable? | Mozilla's docs say v5 "can change at any time without warning" |
| Reviewer identity in payload? | Present (name, user id, username). **We discard it on purpose** and never store it |

**Terms of use.** Mozilla's general terms page contains no clause on automated access. It incorporates the Acceptable Use Policy by reference; I read that policy and found no explicit automation clause, but two relevant ones: _"Collect or harvest personally identifiable information without permission"_ and _"interferes with or disrupts Mozilla's services"_. The design answers both: reviewer identity is dropped and e-mail addresses/phone numbers are redacted from review text; traffic is 1 request/second with a hard cap on pages per run. **I am not a lawyer and Mozilla publishes no explicit API usage policy; read the terms yourself before publishing anything built on this data.**

### Limitations of the data (these shape every conclusion)
- **Self-selected sample.** Of 9,369 ratings for Bitwarden, only 3,144 (33.6%) have text.
- **Not comparable to the headline rating.** Source-wide mean of all ratings: **4.58**. Mean of the 300 most recent text reviews I collected: **2.82** (49% rated ≤2★). These are different populations (all-time/all raters vs recent/text-only), so the gap says nothing by itself about product quality.
- Only each user's **latest** rating per add-on is returned.
- About **12% of collected reviews were non-English** (35/300) and are excluded from analysis (the English-only prompt is a stated limitation).
- 2 of 300 were exact-text duplicates posted minutes apart with different star ratings (flagged and excluded).
- Review counts are **not** a count of affected users and not business impact.

## 4. Local model feasibility

Ollama 0.40.5 installed and ran in the dev environment **[verified]**, CPU-only. Model tested: **`qwen2.5:3b`** (1.9 GB download), JSON-schema-constrained output.

| Measurement (this machine) | Result |
|---|---|
| Model load (first call) | ~10 s |
| Output speed | ~8–9 tokens/s |
| Time per review (≈100 output tokens + prompt) | **~13 s** |
| 250 reviews | **≈ 55 minutes** (one-off; later runs only process new reviews) |

Guidance (**only the 3B model was measured**; the rest is typical but **[not verified]**):

| Your machine | Suggested model |
|---|---|
| 8 GB RAM, CPU only | `qwen2.5:3b` or `llama3.2:3b` — works, expect ~10–15 s/review |
| 16 GB RAM | `qwen2.5:7b` is a candidate (larger models are usually better judges but slower on CPU — **neither accuracy nor speed measured here**; compare with `pda eval`) |
| Any NVIDIA/Apple-silicon GPU | Same models, much faster — Ollama uses the GPU automatically |

Small models make mistakes (see `docs/MVP_RESULTS.md`). The design assumes this: numbers are never produced by the model, and every quote is checked verbatim against the review.

## 5. Recommended architecture (smallest thing that works)

```
 config.toml ─► PoliteClient (robots.txt · 1 req/s · retries · timeouts)
                    │
        AmoSource ──┘  newest-first pages, verified ordering, reviewer identity dropped
                    ▼
        clean_review()   normalise · redact PII · language (lingua) · usability · content hash
                    ▼
        SQLite ◄── insert idempotently (UNIQUE source+id) · flag content duplicates · log run + events
                    ▼
        ReviewAnalyzer ─► Ollama (JSON-schema output) ─► validate ─► verify quote verbatim
                    ▼
        pandas aggregation (counts, shares, mean rating)  +  issue-label grouping (TF-IDF)
                    ▼
        Markdown report: [measured] vs [AI] labelled separately, every claim links to a real review
```

- **Python + SQLite + pandas + Pydantic + Ollama**, nothing else at runtime. No agent framework, no vector DB, no queue.
- **One agent**, not several: the work is a linear pipeline with a few decision points; Phase 4 adds an LLM tool-selection loop over the same functions.
- **Incremental by construction:** reviews are keyed `(source, source_review_id)`; analyses are keyed `(review, model, prompt_version)`. Re-running costs one HTTP page and zero LLM calls when nothing is new.

## 6. Exact setup steps

```bash
# 1. Python 3.11+ (check)
python3 --version

# 2. Install Ollama from https://ollama.com/download  (free), then:
ollama pull qwen2.5:3b            # ~1.9 GB, one-off
ollama serve                      # skip if the Ollama desktop app is already running

# 3. Project
cd product-discovery-agent        # the unzipped folder (already a git repository)
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp config.example.toml config.toml                      # then set collector.contact to your GitHub URL/e-mail

# 4. Verify the setup
pda doctor
pytest
```

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| AMO API v5 changes without notice | Strict response validation → clear `SourceFormatError`, run recorded as failed, nothing silently wrong |
| robots.txt later forbids the API | Checked before every run; agent stops with status `failed` (`robots_disallowed`) and does not retry |
| Small LLM misjudges reviews | Deterministic counts, verbatim-quote audit, versioned prompts, evaluation set (Phase 5) with measured accuracy |
| Lexical issue grouping misses synonyms | Documented limitation (`xfail` test); semantic embeddings via Ollama are the planned upgrade |
| Machine too slow for the model | Incremental analysis, `--limit`, smaller model, runs resume after interruption |
