# Phase 2 — Measured results on real data

Everything below was produced by running the code in this repository against the live Mozilla Add-ons API and a local Ollama model on **2026-10-17**. Environment: 4 vCPU, 16 GB RAM, **no GPU**, Python 3.13 (tests also pass on 3.11 and 3.12), Ollama 0.40.5, `qwen2.5:3b` for judgement and `nomic-embed-text` for grouping. Target: _Bitwarden Password Manager_ (Firefox add-on).

Legend: **[measured]** = counted/timed by code or by comparison with labelled data. **[judgement]** = my reading of the output, not a metric.

## 1. Collection  [measured]

| | |
|---|---|
| Command | `pda collect` (first run, `initial_backfill_limit = 300`) |
| Duration | **11 s** |
| HTTP requests | **7** (1 robots.txt + 6 API pages of 50) |
| Reviews stored | 300 (dates 2025-05-08 → 2026-10-08) |
| Usable after cleaning | **248** (82.7%) |
| Not usable | 52: 35 non-English (ru 9, de 9, fr 6, es 5, pl 2, it 2, zh 1, ja 1) · 14 too short (< 4 words) · 2 duplicate-content · 1 no letters |
| Source-reported totals | 3,144 text reviews; 9,369 ratings in total (33.6% have text) |

**Incremental behaviour** — see §7 for the second-run demonstration.

### Cleaning audit (done by reading, because a language filter that silently drops English reviews would bias everything)
The first version used hand-written stopword lists to detect language. Auditing the 300 real reviews found it failing **in both directions**:
- false positive: _"Secure, no nonsense, no ads and professional password manager"_ flagged Spanish (the word "no");
- a bilingual English + Russian review flagged non-Latin;
- after patching those, a regression: genuine Spanish, French, Italian and Polish reviews passed as English.

Patching a tiny word list could not fix both at once, so it was replaced by the open-source `lingua` n-gram detector restricted to ~30 plausible languages. Re-audit of the same 300 reviews **[judgement, single reader]**: all 35 reviews flagged as foreign are foreign, except one borderline English + Portuguese review (#178); the 17 usable reviews with the fewest English function words are all English. The two flagged duplicate pairs have identical text, the same day and version, but different star ratings and adjacent ids — consistent with review spam, correctly excluded.

## 2. LLM analysis — operations  [measured]

### 2a. Prompt v2 over all 248 usable reviews
| | |
|---|---|
| Succeeded / failed | **248 / 0** |
| Time per review | mean 12.7 s · median 12.4 s · p90 16.6 s · max 27.9 s |
| Output | ≈108 tokens/review; ≈8.5 tokens/s end-to-end |
| Total wall-clock | ≈ 52 minutes (one-off; later runs only analyse new reviews) |
| Quotes found verbatim in the review | **233 / 248 (94%)** |

The 15 unverified quotes are the evidence check working: e.g. a review contained the typo "expexted" and the model silently _corrected_ it; another quote stitched two non-adjacent sentences together. Those quotes are never displayed; a plain excerpt of the real review is shown instead.

### 2b. Pilot: prompt v1 → v2 on the same 8 reviews  [judgement, n=8]
| Defect seen in v1 output | Cause | Fix in v2 |
|---|---|---|
| Two reviews got exactly my prompt's example label ("autofill fails on login pages") although neither said that | small models copy prompt examples | removed copyable examples; told the model not to reuse them |
| Functional failures filed under _usability_ux_ | `theme` was chosen _before_ the problem was described | schema order: problem statement and label first |
| `snake_case` labels; literal `"None"` in free-text fields | formatting noise | deterministic normalisation (tested) |
| A correct quote rejected because the model changed `"` to `'` | verification too literal about punctuation | quote-mark style ignored, words still must match exactly |
| A clear one-sentence bug report labelled `ambiguous` | relevance gate too strict | see §3 |

## 3. Accuracy against labelled samples  [measured]

### Protocol
- Two disjoint random samples of usable reviews: **dev** (seed 20261017, 40 reviews) and **held-out test** (seed 20261018, 40 reviews).
- I (Claude) labelled each sample **before looking at the model's output for it**. Items where I had already seen the model's answer are flagged and **excluded from scores** (1 in dev, 6 in test) → n = 39 and n = 34.
- Prompts were iterated **only** against dev. The test set was scored once per candidate, aggregate numbers only (`pda eval --no-details`), so its individual errors could not be used for tuning.
- Where several themes are defensible the label lists acceptable alternatives → _strict_ (primary label) and _lenient_ (any acceptable) scores.
- Labels are stored as ids + labels + a text hash, **not review text** (`eval/gold_dev.jsonl`, `eval/gold_test.jsonl`).

> **These labels are one rater's judgement (an AI's), not ground truth, and not human-verified.** Ayush and Kavya should review and correct them before anyone quotes these numbers as accuracy. Intervals are 95% Wilson intervals; at n≈35 they are wide.

### Results
| | Dev v2 (n=39) | Dev v3 (n=39) | **Test v2 (n=34)** | **Test v3 (n=34)** |
|---|---|---|---|---|
| Relevance (strict) | 90% | 97% | 94% | **97%** |
| Sentiment (4 classes) | 72% | 72% | 74% | **74%** |
| **Theme, strict** | 67% (51–79) | 62% (46–75) | 62% (45–76) | **62% (45–76)** |
| **Theme, lenient** | 82% (67–91) | 85% (70–93) | 74% (57–85) | **74% (57–85)** |
| "Reports a problem?" precision | 91% | 97% | 93% | **100%** |
| "Reports a problem?" recall | 100% | 100% | 100% | **96%** |
| Quote found verbatim | 95% | 92% | 97% | **97%** |
| Feature-request P / R | 67 / 67 | 67 / 67 | 100 / 100 | 100 / 50 (n=2 positives) |

### What this does and does not show
- **Theme assignment by a 3B model agrees with this rater roughly 60–67% of the time strictly and 74–85% leniently.** That is useful for triage, not for unsupervised decisions.
- **v3 reduced invented problems and improved relevance, but did not improve theme accuracy** (identical 62% on the held-out set).
- v1→v2→v3 traded one bias for another: on the dev set v2 over-filed things under `bugs_reliability` (13/13 recall but only 13 of 22 predictions correct, ~59% precision); v3's balanced definitions flipped five genuine bugs (mostly autofill and biometric failures) into `usability_ux`. Rewording alone did not fix it, which points at the model's size rather than the prompt.
- The `reports_problem` gate works through **schema order**, not through code overrides: the deterministic consistency rules fired on only 2 of 80 reviews, and every firing is recorded in the database.
- **Severity is not evaluated** and should not be trusted: under v2 the model called 54% of reviews `high`, which looks inflated.
- Not measured: other models (a larger model is the obvious next experiment: `ollama pull qwen2.5:7b`, set `llm.model`, run `pda analyze --gold eval/gold_dev.jsonl` then `pda eval`), inter-rater agreement, other apps.

## 4. A design lesson the data forced  [judgement, supported by counts]
Under v2, the model marked 23 of 248 reviews `ambiguous` and 2 `irrelevant`. Of the 10 non-genuine verdicts I read (excluding my labelled reviews), 8 were **short but specific bug reports or tips about the same broken release** ("Latest version breaks self-hosting."); one `irrelevant` verdict was about the product too. Excluding everything not marked `genuine_feedback` would have hidden the single largest problem in the data (self-hosted/Vaultwarden breakage after 2026.7.0). The aggregation now excludes only `irrelevant` and shows `ambiguous` reviews as **low-confidence** — safe because every quote is verified against the real text.

## 5. Final run under the shipped prompt (v3), all 248 usable reviews  [measured]
| | |
|---|---|
| Succeeded / failed | **248 / 0** |
| Time per review | mean 12.9 s · median 12.7 s · p90 16.3 s · max 28.1 s |
| Quotes found verbatim | **227 / 248 (91.5%)** |
| Deterministic consistency overrides | 11 / 248 (4.4%), all of the kind "model said _no problem_ but still filled in a problem theme/label"; each is stored with the model's raw answer |

**Theme shares are not stable measurements.** Same 248 reviews, same model, only the prompt changed:

| Theme | v2 | v3 |
|---|---:|---:|
| bugs_reliability | 55% | 21% |
| usability_ux | 6% | 21% |
| no_issue | 15% | 22% |
| performance | 13% | 17% |
| compatibility | 2% | 10% |
| account_sync | 3% | 6% |
| `high` severity | 54% | 44% |
| relevance: genuine / ambiguous / irrelevant | 223 / 23 / 2 | 241 / 7 / 0 |

Per-item agreement with the labels was about the same (62–67% strict) while the _distribution_ moved this much. So the report **leads with specific recurring problems** (§6) and shows theme shares only as a coarse, caveated table.

## 6. Grouping similar complaints  [measured counts, quality = judgement]
194 relevant reviews describe a problem. Grouped across themes:

| Method | Groups | Single-review groups | Largest groups |
|---|---:|---:|---|
| TF-IDF word overlap (fallback) | 141 | **119** | 10, 8, 7, 7, 5, 4 |
| Embeddings (`nomic-embed-text`), threshold 0.60 | 76 | 37 | 13, 12, 9, 9, 9, 8 |
| … 0.70 | 53 | 17 | 16, 13, 10, 10, 10, 9 |
| **… 0.75 (default)** | **41** | **9** | 19, 18, 14, 13, 10, 10 |
| … 0.80 | 31 | 4 | 19, 18, 16, 15, 14, 10 |
| … 0.85 | 21 | 1 | 24, 20, 19, 18, 16, 14 |

Why embeddings needed one trick: every complaint says roughly "Bitwarden stopped working", so raw embeddings were all ~0.75 similar and collapsed into a single 151-review cluster (measured). **Subtracting the mean vector** before clustering removed that shared component. Without it, no threshold I tried separated problems.

Quality of the default (reading the groups, **[judgement]**): the top groups are recognisable problems — extension freezes (19), page slowdowns (18, plus a second slowdown group of 10), login issues (14), autofill (13), sync (10), self-hosting (9), biometric unlock (8). **Known weaknesses:**
- Embeddings group by _symptom wording_, not root cause. The one big event in this data — the self-hosted/Vaultwarden break after 2026.7.0 — is spread across at least four groups ("extension freezes", "login issues", "no sync", "self hosting"). A human reading the quotes sees it; the headings alone do not.
- Group labels are the member description nearest the cluster centre; some are generic ("extension freezes") or are a clipped sentence.
- The threshold was chosen by eye on one app. There is **no ground truth for cluster quality**, so no accuracy is claimed.

## 7. Incremental behaviour  [measured, live]
Second `pda collect` right after the first: **4 s, 2 HTTP requests** (robots.txt + 1 page), 50 reviews seen, 50 already known, 0 stored, stop reason `caught_up`. Second `pda analyze`: 0 reviews attempted, 0 LLM calls. _Limit:_ no new review was published during the test window, so detection of genuinely new reviews is verified by the mocked test-suite only, not live.

## 8. Defects found by running on real data and by adversarial review
Most were invisible to the first round of unit tests. Each was found by running the system on real reviews, by an edge-case test, or by reviewing my own code against the requirements, and each now has a regression test.

| # | Defect | How it surfaced | Fix |
|---|---|---|---|
| 1 | The stdlib robots parser says Apple's review feed is _allowed_ on Python 3.11/3.12 | Compared against Apple's real robots.txt | Own RFC 9309 parser; Apple's real file is a test fixture |
| 2 | Redirects would bypass robots.txt of the target host | Code review against the "respect robots.txt" requirement | Manual redirect handling, every hop checked, bounded |
| 3 | Hand-written language lists failed both ways on real reviews | Auditing 300 flagged/unflagged reviews | `lingua` detector, re-audited |
| 4 | Model copied my prompt's example label into unrelated reviews | Pilot of 8 reviews | Removed copyable examples |
| 5 | Praise-only reviews received invented problems (3 of 7 in dev) | Dev-set confusion matrix | `reports_problem` gate first in the schema + audited overrides |
| 6 | Model's `ambiguous` verdict hid terse but real bug reports | Reading non-genuine verdicts | `ambiguous` kept as low-confidence evidence |
| 7 | Schema-field names used as issue labels ("bugs reliability", "no issue") | Reading group members | Generic labels ignored; fall back to the problem statement |
| 8 | Identical long reviews posted on the same day under adjacent ids with different star ratings | Duplicate flags | Content-hash duplicate detection (2 pairs) |
| 9 | A zero vector after mean-centring crashes scikit-learn | Edge-case test with identical texts | Zero vectors form one group explicitly |
| 10 | Markdown/HTML in review text would render live in shared reports | Security review of the report generator | Escaping, with a mutation-tested test |
| 11 | A review could contain our own data-block markers | Prompt-injection review | Markers defanged in untrusted text |
| 12 | Theme headline ("Usability & UX is #1") was misleading because autofill bugs were filed as UX | Reading the first full report | Report leads with cross-theme problems; themes demoted and caveated |
| 13 | My own docstring and README overstated that the stdlib parser ignores wildcards | Tested on 3.13: it does not | Corrected to what was measured per Python version |

## 9. Not measured, not claimed
- Accuracy of **severity**, **sentiment on other products**, or **any other model** (no 7B run was done).
- Agreement between raters (one rater, an AI; the labels need human review).
- Cluster quality (no ground truth).
- Detection of newly published reviews on the live service.
- Behaviour on a GPU, or on machines slower than the test box.
- Anything about other apps or sources.

## 10. Reproduce
```bash
pda collect && pda analyze && pda themes --out reports/themes.md
pda eval --gold eval/gold_dev.jsonl                      # dev: free to iterate against
pda eval --gold eval/gold_test.jsonl --no-details        # held-out: score once per candidate
```
Review ids in the gold files refer to Bitwarden's add-on reviews; reviews deleted since will be reported as "not in database".
