# Product Discovery Agent

A **local, zero-cost** AI agent that discovers publicly available customer reviews, finds recurring problems, and links every finding back to the real review it came from.

- Runs entirely on your machine: **Python · SQLite · [Ollama](https://ollama.com) (local LLM) · pandas**. No paid API, no cloud, no payment details.
- Collects feedback **by itself** from a permitted public API. You never upload a CSV.
- Remembers what it has seen: re-running fetches only new reviews and analyses only unanalysed ones.
- Keeps **measurements** (counts, ratings, dates) strictly separate from **AI interpretations** (themes, severity), and verifies every quote it shows against the original review text.
- Ships with an **evaluation harness** and reports its own measured accuracy, including where it is wrong.

> **Status: Phase 1–2 complete (command-line MVP).** The dashboard, opportunity scoring, trend analysis, recommendations/PRD generation, the tool-using agent loop and the scheduler are _planned_, not built — see the [roadmap](#roadmap). This README only describes what exists and has been tested.

## Keeping this project locally with git (no GitHub or cloud needed)

```bash
cd product-discovery-agent
git log --oneline                       # the project arrives with its first commit
git config user.name  "Your Name"       # set your identity once
git config user.email "you@example.com"
git commit --amend --reset-author --no-edit    # optional: put your name on the first commit
# ... work, then:
git add -A && git commit -m "describe the change"
```

`data/`, `config.toml` and `reports/` are git-ignored on purpose (they hold other people's review text and your settings).
For a backup on another drive without any online service: `git init --bare /path/to/backup/pda.git && git remote add backup /path/to/backup/pda.git && git push backup main`.
Publishing to GitHub later is optional: create an empty repo there, then `git remote add origin <url> && git push -u origin main`.

## Why Mozilla Add-ons reviews?

The obvious sources are off-limits to a well-behaved agent. Apple's review feed answers requests but its `robots.txt` says `Disallow: /*/rss/*`; Google Play's says `Disallow: /_` (the path scrapers rely on) and `/store/getreviews`. The **Mozilla Add-ons public ratings API** is documented, key-less, allowed by robots.txt, and returns star rating, text, date, product version and a permalink. Full evidence and every limitation: [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).

The default target is **Bitwarden Password Manager** for Firefox; change one line of `config.toml` to analyse another add-on.

## Quickstart

Requires Python 3.11+ and ~2.5 GB of disk for the two small models.

```bash
# 1. Local LLM (free). Install Ollama from https://ollama.com/download, then:
ollama pull qwen2.5:3b          # ~1.9 GB, judges each review. The only model whose accuracy was measured.
ollama pull nomic-embed-text    # ~274 MB, groups similar complaints (optional: falls back to word-overlap grouping)
ollama serve                    # skip if the Ollama desktop app is already running

# 2. Project
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp config.example.toml config.toml                        # set collector.contact to your GitHub URL or e-mail

# 3. Check everything is wired up
pda doctor                      # every line should say [OK]
pytest                          # ~195 offline tests

# 4. Run it
pda collect                     # fetch reviews (first run: up to 300 new reviews, ~10 s)
pda analyze --limit 25          # LLM pass. ~13 s per review on a CPU-only machine, so start small
pda themes --out reports/themes.md
```

`pda run` does collect → analyse → report in one go. Everything is resumable: stop with Ctrl-C and re-run; finished work is kept.

## Commands

| Command | What it does |
|---|---|
| `pda doctor` | Checks Python, config, database, Ollama + models, and that robots.txt permits the API |
| `pda collect [--full] [--limit N]` | Fetches new reviews. Stops at the first page of already-known reviews (`--full` goes deeper into history) |
| `pda reclean` | Re-applies current cleaning rules to stored reviews, **no network** |
| `pda analyze [--limit N] [--retry-failed] [--gold FILE]` | Analyses reviews that have no result for the current model + prompt version (`--gold`: only the labelled ones — fast prompt iteration) |
| `pda themes [--out FILE]` | Writes the evidence-backed themes report (Markdown) |
| `pda eval [--gold FILE] [--no-details]` | Scores the model against labelled reviews, with confidence intervals |
| `pda status` | Review counts, analysis counts, recent runs |
| `pda run` | collect → analyse → report |

## How it works

```
AMO ratings API ─► PoliteClient ─► clean ─► SQLite ─► Ollama (JSON-schema output) ─► verify ─► aggregate ─► report
 (public, no key)  robots.txt      PII      dedupe     per-review analysis        quote must    pandas +    links to
                   1 req/s         lang     runs log   (prompt versioned)         be verbatim   embeddings  real reviews
```

**Guarantees that are enforced in code and covered by tests**

| Guarantee | Mechanism |
|---|---|
| Respects robots.txt | Own RFC 9309 parser (`*`, `$`, longest-match). Python 3.11/3.12's stdlib parser ignores wildcards and would call Apple's review feed "allowed" (measured; fixed in 3.13). Checked before every request **and every redirect hop**; unreachable robots.txt = fail closed |
| Polite | ≥1 s between requests (config cannot go lower), bounded retries with backoff, honours `Retry-After`, identifying User-Agent, page cap per run |
| No duplicates | `UNIQUE(source, source_review_id)`; identical long texts under different ids are flagged and excluded |
| Incremental | New reviews detected by id; analyses keyed by `(review, model, prompt_version)`; embeddings cached by text hash |
| No invented evidence | A model-written quote is only shown if it appears **verbatim** in the review; otherwise a plain excerpt of the real review is shown instead |
| Numbers are not AI-generated | Counts, shares and mean ratings are computed with pandas; the LLM only classifies |
| Structured output | Ollama is constrained to a JSON Schema, then re-validated with Pydantic; bounded retries; failures are recorded per review |
| Inconsistent output is corrected _and recorded_ | e.g. a review the model says reports no problem cannot carry a problem theme; the model's raw answer and every override are stored |
| Safe failure | Stops (does not loop) on robots block, Ollama down, missing model, or 5 consecutive bad outputs; every run and event is stored |
| Untrusted text stays inert | Reviews are passed to the model as delimited data (markers cannot be spoofed); quotes and labels are Markdown-escaped in reports |

## Measured quality (summary)

Full detail, protocol and caveats: [`docs/MVP_RESULTS.md`](docs/MVP_RESULTS.md). Model `qwen2.5:3b`, CPU only.

| Measured on real Bitwarden reviews | Result |
|---|---|
| Collection | 300 reviews in 11 s, 7 HTTP requests |
| Analysis | 248/248 succeeded, ≈13 s per review (CPU only) |
| Quotes found verbatim in the review | 91.5% (the rest are never shown) |
| Theme agreement with a labelled **held-out** sample (n=34) | **62% strict, 74% lenient** (95% CI 45–76% / 57–85%) |
| "Does this review report a problem?" (held-out) | precision 100%, recall 96% |

Theme **shares** are not stable: the same 248 reviews were 55% "bugs" under one prompt and 21% under another (per-item accuracy was about equal). The report therefore leads with specific recurring problems, grouped across themes.

**Read this honestly:** labels come from a single AI rater and have not yet been human-verified; the sample is small; the 3B model's theme choices are right roughly two times in three. Use it to triage and to find evidence quickly, not to make unsupervised decisions.

## Privacy and responsible use

- Reviewer **names and ids are never stored** (they are in the API payload and are dropped by the parser).
- E-mail addresses and phone-like numbers inside review text are redacted before analysis.
- `data/` (the database) and `config.toml` are git-ignored. Don't commit collected reviews: they are other people's words. The `eval/` gold files hold only ids and labels, never review text.
- This project reads a public API at 1 request/second. Read Mozilla's terms yourself before publishing anything derived from it ([what I could and couldn't verify](docs/FEASIBILITY.md#3-what-was-verified-about-the-amo-api)).

## Testing

```bash
pytest          # all tests run offline: HTTP against a local scripted server, the LLM against fakes
```

~195 tests cover robots.txt rules (including Apple's real file), retries/rate limits/redirects, the AMO parser, cleaning, dedupe, incremental collection, the analysis workflow with mocked failures, quote verification, aggregation arithmetic, embedding grouping, schema migration, report escaping and the evaluation maths. They pass on Python 3.11, 3.12 and 3.13. Tests prove the **logic**; they say nothing about how _accurate_ the LLM is — that is what `pda eval` measures.

## Known limitations

- Reviews are a **self-selected sample**; counts are not "number of affected users" and not business impact.
- A 3B local model makes real mistakes (documented with examples). Theme: ~62% strict agreement on held-out data. **Severity is unvalidated** (the model called 44–54% of reviews "high").
- Labels used for evaluation are provisional (single AI rater).
- English reviews only (≈12% of collected reviews were non-English and are excluded).
- Issue grouping is heuristic. With embeddings it is much better than word-overlap grouping on this data but some groups are loose (see results); the distance threshold was tuned by eye on one app and has no ground truth.
- The AMO v5 API is documented as "not frozen"; the collector validates responses and fails loudly if the shape changes.
- One source (AMO) is implemented. The interface is ready for more (Steam reviews was verified reachable and allowed).

## Project layout

```
src/pda/
  config.py      typed settings          robots.py     RFC 9309 parser
  http.py        polite HTTP client      cleaning.py   normalise · PII · language · usability
  sources/       base.py · amo.py        db.py         SQLite schema + migrations
  llm.py         Ollama chat + embed     analyzer.py   prompt · quote verification · consistency rules
  themes.py      aggregation + grouping  reporting.py  Markdown report
  evaluation.py  scoring vs labels       pipeline.py   collect / analyze / reclean runs
  cli.py         `pda` commands
tests/           offline test-suite
eval/            gold_dev.jsonl · gold_test.jsonl (ids + labels only)
docs/            FEASIBILITY.md · MVP_RESULTS.md
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Feasibility: sources, robots.txt, local-model check | ✅ done |
| 2 | Collector, SQLite, dedupe, cleaning, local LLM, evidence-backed themes, evaluation harness | ✅ done |
| 3 | Opportunity scoring (transparent formula), trend/emerging-issue analysis (by product version and time), recommendations, PRD export | ⬜ next |
| 4 | LLM tool-selection loop with validated arguments and step limits, local scheduler | ⬜ |
| 5 | Streamlit dashboard, human-verified evaluation set + report, product brief, case study, screenshots | ⬜ |
