"""Command line interface:  python -m pda <command>   (or `pda <command>` after `pip install -e .`)"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pda import __version__
from pda.analyzer import PROMPT_VERSION
from pda.config import Settings, load_settings
from pda.db import Database
from pda.evaluation import evaluate, load_gold, render_markdown as render_eval_markdown
from pda.errors import LLMModelMissing, LLMUnavailable, PdaError, RobotsDisallowed
from pda.llm import OllamaClient
from pda.pipeline import analyze, collect, compute_embeddings, reclean
from pda.reporting import render_markdown, snapshot_to_grouped
from pda.sources import build_client
from pda.themes import overview, summarise_issues, summarise_themes

EXIT_FOR_STATUS = {"success": 0, "partial": 3, "failed": 1}


def _llm(settings: Settings) -> OllamaClient:
    c = settings.llm
    return OllamaClient(c.base_url, c.model, temperature=c.temperature, num_ctx=c.num_ctx, timeout_s=c.timeout_s, max_retries=c.max_retries)


def _open(settings: Settings) -> tuple[Database, int]:
    db = Database(settings.storage.db_path)
    a = settings.app
    return db, db.upsert_app(a.source, a.slug, a.name)


# ----------------------------------------------------------------------------- commands


def cmd_init(settings: Settings, args: argparse.Namespace) -> int:
    db, app_id = _open(settings)
    print(f"Database ready: {settings.storage.db_path}")
    print(f"App registered: {settings.app.name} ({settings.app.source}:{settings.app.slug}) id={app_id}")
    if not Path("config.toml").exists():
        print("Tip: copy config.example.toml to config.toml and set collector.contact to your GitHub URL or email.")
    return 0


def cmd_doctor(settings: Settings, args: argparse.Namespace) -> int:
    ok = True

    def line(good: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"[{'OK' if good else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))

    line(sys.version_info >= (3, 11), "Python >= 3.11", sys.version.split()[0])
    line(True, "Config", f"{settings.app.source}:{settings.app.slug}, model {settings.llm.model}")
    if "YOUR-USERNAME" in settings.collector.contact:
        print("[WARN] collector.contact is still the placeholder; set it in config.toml so the source can identify you.")
    try:
        db, _ = _open(settings)
        line(True, "Database", settings.storage.db_path)
    except Exception as exc:  # noqa: BLE001 - diagnostic command
        line(False, "Database", str(exc))
    try:
        version = _llm(settings).ensure_ready()
        line(True, "Ollama", f"v{version}, model '{settings.llm.model}' installed")
    except (LLMUnavailable, LLMModelMissing) as exc:
        line(False, "Ollama", str(exc))
    if settings.grouping.method == "embeddings":
        try:
            have = _llm(settings).installed_models()
            emb = settings.llm.embedding_model
            if (emb if ":" in emb else f"{emb}:latest") in have:
                line(True, "Embedding model", emb)
            else:
                print(f"[WARN] Embedding model '{emb}' not installed: reports will fall back to lexical grouping. Run: ollama pull {emb}")
        except LLMUnavailable:
            pass  # already reported by the Ollama check above
    if not args.offline:
        try:
            client = build_client(settings)
            client.check_allowed("https://addons.mozilla.org/api/v5/ratings/rating/?addon=" + settings.app.slug)
            line(True, "robots.txt permits the ratings API", f"({client.request_count} request made)")
        except RobotsDisallowed as exc:
            line(False, "robots.txt", str(exc))
        except PdaError as exc:
            line(False, "Source reachability", str(exc))
    return 0 if ok else 1


def cmd_collect(settings: Settings, args: argparse.Namespace) -> int:
    db, _ = _open(settings)
    res = collect(settings, db, trigger=args.trigger, full=args.full, limit=args.limit, max_pages=args.max_pages)
    i = res.inserts
    print(f"Collect run #{res.run_id}: {res.status.upper()} (stopped: {res.stop_reason})")
    print(f"  pages fetched: {res.pages_fetched} | HTTP requests: {res.requests_made} | reviews seen: {res.fetched_reviews}")
    print(f"  NEW stored: {i.inserted} (usable {i.usable}, unusable {i.unusable}, content-duplicates {i.duplicate_content}) | already known: {i.already_known}")
    if i.unusable_reasons:
        print(f"  unusable reasons: {json.dumps(i.unusable_reasons)}")
    if res.skipped_by_source:
        print(f"  skipped at source: {json.dumps(res.skipped_by_source)}")
    if res.error:
        print(f"  ERROR: {res.error}")
    if res.status != "success":
        print("  Note: newer reviews are always picked up by the next `pda collect`. If this was the first (backfill) run,\n"
              "  run `pda collect --full` to continue fetching older history; plain `collect` stops at already-known pages.")
    return EXIT_FOR_STATUS[res.status]


def cmd_reclean(settings: Settings, args: argparse.Namespace) -> int:
    db, _ = _open(settings)
    res = reclean(settings, db)
    print(f"Reclean run #{res.run_id}: {res.reviews} stored reviews re-checked, {res.changed} changed "
          f"(now usable: {res.became_usable}, now unusable: {res.became_unusable}). No network used.")
    return 0


def cmd_analyze(settings: Settings, args: argparse.Namespace) -> int:
    db, _ = _open(settings)

    def progress(i: int, n: int, info: dict) -> None:
        mark = "quote verified" if info["verified"] else "quote NOT verified"
        print(f"  [{i}/{n}] review {info['review_id']}: {info['theme']:<17} | {info['label'] or '-':<40} | {mark}", flush=True)

    only = None
    if args.gold:  # try a prompt/model on the labelled reviews only
        _, items = load_gold(args.gold)
        only = {g.source_review_id for g in items}
        print(f"Restricting to {len(only)} reviews listed in {args.gold}")
    res = analyze(settings, db, _llm(settings), trigger=args.trigger, limit=args.limit, retry_failed=args.retry_failed,
                  only_source_ids=only, progress=progress)
    print(f"Analyze run #{res.run_id}: {res.status.upper()} (stopped: {res.stop_reason})")
    print(f"  attempted {res.attempted} | succeeded {res.succeeded} | failed {res.failed} | quotes verified {res.quote_verified}/{res.succeeded} | consistency overrides applied to {res.overridden}")
    if res.mean_latency_s is not None:
        print(f"  mean latency {res.mean_latency_s}s/review | effective output speed {res.effective_tokens_per_s} tok/s")
    if res.error:
        print(f"  ERROR: {res.error}")
    return EXIT_FOR_STATUS[res.status]


def build_report(settings: Settings, db: Database, app_id: int) -> str:
    df = db.analysed_df(app_id, settings.llm.model, PROMPT_VERSION)
    counts = db.review_counts(app_id)
    total_text, grouped = snapshot_to_grouped(db.latest_snapshot(app_id))
    vectors, note = None, "TF-IDF lexical similarity (word overlap; synonyms are not merged)"
    g = settings.grouping
    if g.method == "embeddings":
        try:
            vectors = compute_embeddings(settings, db, _llm(settings), df)
            note = (f"semantic embeddings (`{settings.llm.embedding_model}`, mean-centred, average-linkage clustering, "
                    f"cosine distance threshold {g.embedding_distance_threshold})")
        except PdaError as exc:
            note = f"TF-IDF lexical similarity - FALLBACK because embeddings were unavailable ({exc})"
            print(f"Warning: {exc}\n  Falling back to lexical grouping. Fix with: ollama pull {settings.llm.embedding_model}", file=sys.stderr)
    issues = summarise_issues(df, vectors=vectors, vector_threshold=g.embedding_distance_threshold)
    return render_markdown(
        settings.app.name, overview(df), summarise_themes(df), issues, model=settings.llm.model, prompt_version=PROMPT_VERSION,
        snapshot_total_text=total_text, snapshot_grouped=grouped, collected_total=counts["total"], collected_usable=counts["usable"],
        grouping_note=note,
    )


def cmd_themes(settings: Settings, args: argparse.Namespace) -> int:
    db, app_id = _open(settings)
    if not db.analysis_counts(app_id, settings.llm.model, PROMPT_VERSION).get("ok"):
        print("No analysed reviews yet for this model. Run `pda collect` then `pda analyze` first.", file=sys.stderr)
        return 1
    md = build_report(settings, db, app_id)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(md)
    return 0


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    db, app_id = _open(settings)
    meta, gold = load_gold(args.gold)
    res = evaluate(db, app_id, settings.llm.model, PROMPT_VERSION, meta, gold)
    if not res.n_scored:
        print(f"Nothing to score: {res.skipped or 'no gold items'}. Analyse the labelled reviews first (pda analyze).", file=sys.stderr)
        return 1
    md = render_eval_markdown(res, details=not args.no_details)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(md)
    return 0


def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    db, app_id = _open(settings)
    c = db.review_counts(app_id)
    a = db.analysis_counts(app_id, settings.llm.model, PROMPT_VERSION)
    print(f"{settings.app.name} ({settings.app.source}:{settings.app.slug})")
    print(f"  reviews stored: {c['total']} (usable {c['usable']}) | dates {c['oldest']} .. {c['newest']}")
    if c["unusable_reasons"]:
        print(f"  unusable: {json.dumps(c['unusable_reasons'])}")
    print(f"  analysed with {settings.llm.model}/{PROMPT_VERSION}: ok {a.get('ok', 0)}, failed {a.get('failed', 0)}, pending {len(db.pending_for_analysis(app_id, settings.llm.model, PROMPT_VERSION))}")
    runs = db.recent_runs(8)
    print("  recent runs:")
    for r in runs:
        print(f"    #{r['id']:<3} {r['kind']:<8} {r['trigger']:<9} {r['status']:<8} {r['started_at']}")
    if not runs:
        print("    (none yet)")
    return 0


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    """One manual end-to-end pass: collect -> analyse new reviews -> write the themes report."""
    db, app_id = _open(settings)
    rc = cmd_collect(settings, argparse.Namespace(trigger=args.trigger, full=False, limit=None, max_pages=None))
    if rc == 1:
        print("Collection failed; not continuing.", file=sys.stderr)
        return rc
    rc2 = cmd_analyze(settings, argparse.Namespace(trigger=args.trigger, limit=args.analyze_limit, retry_failed=False))
    if db.analysis_counts(app_id, settings.llm.model, PROMPT_VERSION).get("ok"):
        out = Path("reports") / f"themes_{settings.app.slug}.md"
        cmd_themes(settings, argparse.Namespace(out=str(out)))
    return max(rc, rc2)


# ----------------------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pda", description="Product Discovery Agent (local, zero-cost)")
    p.add_argument("-c", "--config", help="path to config.toml (default: ./config.toml or $PDA_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"pda {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and register the configured app").set_defaults(func=cmd_init)

    d = sub.add_parser("doctor", help="check Python, config, database, Ollama and source access")
    d.add_argument("--offline", action="store_true", help="skip the network check of the source")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("collect", help="fetch new reviews from the configured source")
    c.add_argument("--full", action="store_true", help="do not stop at already-known pages (extends history)")
    c.add_argument("--limit", type=int, help="stop after this many NEW reviews")
    c.add_argument("--max-pages", type=int, help="override collector.max_pages_per_run")
    c.add_argument("--trigger", default="manual", choices=["manual", "scheduled", "agent"])
    c.set_defaults(func=cmd_collect)

    sub.add_parser("reclean", help="re-apply current cleaning rules to stored reviews (no network)").set_defaults(func=cmd_reclean)

    a = sub.add_parser("analyze", help="analyse reviews not yet analysed by the current model/prompt")
    a.add_argument("--limit", type=int, help="analyse at most N reviews (newest first)")
    a.add_argument("--retry-failed", action="store_true")
    a.add_argument("--gold", help="analyse only the reviews listed in this gold file (fast prompt iteration)")
    a.add_argument("--trigger", default="manual", choices=["manual", "scheduled", "agent"])
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("themes", help="print or write the evidence-backed themes report (Markdown)")
    t.add_argument("--out", help="write to this file instead of stdout")
    t.set_defaults(func=cmd_themes)

    e = sub.add_parser("eval", help="score the model against a labelled gold file (ids + labels only)")
    e.add_argument("--gold", default="eval/gold_dev.jsonl")
    e.add_argument("--no-details", action="store_true", help="aggregate scores only (use for the held-out test set)")
    e.add_argument("--out", help="write the report to this file instead of stdout")
    e.set_defaults(func=cmd_eval)

    sub.add_parser("status", help="show counts and recent runs").set_defaults(func=cmd_status)

    r = sub.add_parser("run", help="collect + analyse + write report in one go")
    r.add_argument("--analyze-limit", type=int, help="cap reviews analysed in this run")
    r.add_argument("--trigger", default="manual", choices=["manual", "scheduled", "agent"])
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    try:
        return args.func(settings, args)
    except PdaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Progress already saved is kept; re-run to continue.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
