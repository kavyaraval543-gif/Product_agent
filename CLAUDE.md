# Product Discovery Agent: project guide (read this first)

A local, zero-cost agent that collects public customer reviews, finds recurring problems, and links every finding to real reviews. See `README.md` for usage and `docs/` for evidence.

## People and working style
- Two people use this project: **Ayush and Kavya**. If it is unclear which of them is asking, or a requirement is ambiguous, **ask**.
- Be honest about what is measured vs assumed. Never claim accuracy that has not been tested. Never invent reviews, customers, users or business outcomes.

## Hard constraints (do not break these)
1. **Budget is ₹0.** No paid APIs, hosting or subscriptions. Only Python, SQLite, Ollama (local LLM), Streamlit, pandas, pytest, Git.
2. **Respect robots.txt on every request and every redirect hop.** Apple's review RSS and Google Play are rejected on this basis (`docs/FEASIBILITY.md`). If a source forbids access, stop and use another source; do not work around it.
3. **Every recommendation must link to real review records.** A model-written quote is shown only if it is found verbatim in the review (`analyzer.verify_quote`).
4. **Numbers are computed by code (pandas), never by the LLM.** Keep `[measured]` and `[AI]` visibly separate in every output.
5. **Do not store reviewer names or ids.** Do not commit `data/`, `config.toml` or `reports/` (they hold other people's words).

## Status
- Phase 1 (feasibility) and Phase 2 (collector, SQLite, cleaning, local LLM, evidence-backed problems, evaluation harness): **done**.
- Phase 3 (next): opportunity scoring, trends by product version/time, recommendations, PRD export.
- Phase 4: LLM tool-selection loop (validated args, step limit) + local scheduler. Phase 5: Streamlit dashboard, human-verified evaluation, product brief, case study, screenshots.

## Commands
```bash
pip install -e ".[dev]" && pytest          # ~200 offline tests; 1 expected failure (documented synonym limitation)
pda doctor                                  # checks Python, DB, Ollama + models, robots.txt
pda collect | analyze [--limit N] | themes --out FILE | eval --gold FILE [--no-details] | status | reclean | run
```
Needs Ollama running with `qwen2.5:3b` (judging) and `nomic-embed-text` (grouping). Only `qwen2.5:3b` has measured accuracy.

## Architecture (src/pda/)
`http.py` polite client (robots, 1 req/s, retries, redirects) → `sources/amo.py` (Mozilla Add-ons API) → `cleaning.py` (PII, language via lingua, usability, hashes) → `db.py` (SQLite, migrations via `PRAGMA user_version`, now v3) → `llm.py` (Ollama chat with JSON-schema output + embeddings) → `analyzer.py` (prompt **v3**, quote verification, defanged delimiters) → `models.py` (output schema; `normalised()` consistency rules, all overrides stored) → `themes.py` (counts; cross-theme problem groups via mean-centred embeddings, TF-IDF fallback) → `reporting.py` (Markdown, escapes untrusted text) → `pipeline.py` (recorded runs) → `cli.py`. `evaluation.py` scores the model against `eval/gold_*.jsonl`.

## Decisions that must not regress
- Analyses are keyed `(review, model, prompt_version)`. **Bump `PROMPT_VERSION` when the prompt or schema changes meaning.**
- Schema field order is generation order. `reports_problem` comes before problem fields (prevents invented problems for praise-only reviews).
- `ambiguous` reviews are kept as low-confidence, never dropped. Only `irrelevant` is excluded.
- The report leads with problems grouped across themes; theme shares are unstable (55% → 21% "bugs" from a prompt change alone).
- Embeddings must be mean-centred before clustering, or everything collapses into one cluster. Zero vectors form one group.
- Own robots.txt parser exists because Python 3.11/3.12's stdlib one ignores wildcards.

## Evaluation protocol
- `eval/gold_dev.jsonl`: iterate prompts against this. `eval/gold_test.jsonl`: held out; score once per candidate with `--no-details`.
- **Labels are provisional (single AI rater, not human-verified). Ayush and Kavya should review and correct them.**
- Gold files hold ids, labels and text hashes only, never review text.
- Last measured (held-out, n=34, qwen2.5:3b, prompt v3): theme 62% strict / 74% lenient; problem-detection precision 100%, recall 96%. Details: `docs/MVP_RESULTS.md`.

## Known limitations / open items
- Severity unvalidated (model says "high" for 44-54% of reviews). No 7B model measured. Cluster quality has no ground truth; the same root cause (e.g. self-hosted break after 2026.7.0) splits across several groups.
- New-review detection verified with mocks only; no new review appeared during the live test.
- Source is Mozilla Add-ons only. Steam reviews verified reachable and allowed; not implemented.
- `collector.contact` in `config.toml` must be set to your own GitHub URL or email before real use.

## Phase 3 plan (proposed, not built)
1. **Opportunity score** (transparent, in code): frequency (measured share of relevant reviews) × severity (AI, 0-4) × confidence (verified-quote rate, group size, rating consistency); effort only when information exists. Show the formula in the UI. Never present review counts as measured customer impact.
2. **Trends:** counts per group by product version and month (measured); flag emerging/worsening groups with a stated minimum-count rule.
3. **Recommendations + PRD:** LLM writes from supplied evidence ids only; verify every quote; refuse to output a recommendation with no linked reviews; include risks, assumptions, success metrics.
4. Add tests and an evaluation for each step before moving on.

## Windows
- The owner runs this on **Windows (PowerShell)**. It has NOT been run on Windows yet; only a Linux box was available. Use **Python 3.12 64-bit** (3.11 lacks Windows wheels for the newest numpy/scipy/lingua).
- Console output must not crash on legacy code pages: `cli.make_output_safe` (regression test in `tests/test_cli_console.py`). Always name `encoding="utf-8"` for text files.
- Not built yet: Streamlit dashboard, scoring/trends/recommendations/PRD, agent loop, scheduler (Windows Task Scheduler or APScheduler).

## Gotchas
- `pkill -f "<pattern>"` can kill your own shell if the pattern is in the command line; use `pkill -x name`.
- A first `pda analyze` of ~250 reviews takes about an hour on a CPU-only laptop; use `--limit`, it is resumable.
- lingua loads language models lazily on first use (a few seconds, ~200 MB).
