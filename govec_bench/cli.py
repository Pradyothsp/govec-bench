"""The harness's commands over its results: `python -m govec_bench.cli <command>`.

import [PATH ...]   index results files (default: results/, the published ones included);
                    folders are searched, so `import ../govec-notes/bench` adds older sessions
runs                list indexed runs, newest first, with what each database ran as
run EXPERIMENT      run an experiment's rounds (benchmarks/experiment.py); --dry-run shows the plan
compare SESSION     what a session found, per metric: each arm's median and range, change, verdict
ui                  watch the session running (tui/live.py), or browse sessions when none is (tui/browse.py)

`run` opens the live view when the terminal can show one; --plain prints instead.
"""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from govec_bench.benchmarks.experiment import (
    add_run_arguments,
    find_experiment,
    find_session_folder,
    load_experiment,
    run_experiment,
    running_sessions,
    session_directory,
)
from govec_bench.compare import PUBLISHED, SessionKey, compare, resolve_session
from govec_bench.results import RESULTS_DB, RESULTS_DIR
from govec_bench.store import connect, import_paths


def _import(args: argparse.Namespace) -> None:
    imported = import_paths(args.db, args.paths or [RESULTS_DIR])
    new = sum(item.new for item in imported)
    print(f"{len(imported)} results files, {new} new, {len(imported) - new} already indexed, in {args.db}")


def _ran_as(image: str | None, commit: str | None, env: str | None) -> str:
    parts = [commit or (image or "image not recorded")]
    if env:
        parts.extend(f"{key.removeprefix('GOVEC_').lower()}={value}" for key, value in json.loads(env).items())
    return " ".join(parts)


def _runs(args: argparse.Namespace) -> None:
    query = "SELECT id, timestamp, benchmark, dataset, label FROM runs"
    conditions, params = [], []
    if args.benchmark:
        conditions.append("benchmark = ?")
        params.append(args.benchmark)
    if args.database:
        conditions.append("id IN (SELECT run_id FROM entries WHERE database = ?)")
        params.append(args.database)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(args.limit)

    with closing(connect(args.db)) as db:
        db.row_factory = sqlite3.Row
        for run in db.execute(query, params).fetchall():
            print(
                f"{run['id']}  {(run['timestamp'] or '')[:16]}  {run['benchmark']:<15} {run['dataset'] or '':<14} {run['label']}"
            )
            for entry in db.execute(
                "SELECT database, image, govec_commit, env, error FROM entries WHERE run_id = ? ORDER BY rowid",
                (run["id"],),
            ):
                failed = f"  FAILED: {entry['error']}" if entry["error"] else ""
                print(
                    f"    {entry['database']:<14} {_ran_as(entry['image'], entry['govec_commit'], entry['env'])}{failed}"
                )


def _run(args: argparse.Namespace) -> None:
    path = find_experiment(args.experiment)
    experiment = load_experiment(path)
    if args.plain or args.dry_run or not sys.stdout.isatty():
        # --into: the live view started this run and shows its run.log, so nothing to echo.
        run_experiment(experiment, args.rounds, dry_run=args.dry_run, directory=args.into, echo=args.into is None)
        return

    from govec_bench.tui.app import BenchApp, start_runner  # noqa: PLC0415 -- Textual only when a view opens

    directory = session_directory(experiment)
    process = start_runner(path, directory, args.rounds)
    with closing(connect(args.db)) as db:
        BenchApp(db, live=directory, baseline=args.baseline, process=process).run()


def _ui(args: argparse.Namespace) -> None:
    from govec_bench.tui.app import BenchApp  # noqa: PLC0415 -- Textual only when a view opens

    if args.live is not None:
        live: Path | None = find_session_folder(args.live)
    elif args.session is None:
        live = next(iter(running_sessions()), None)
    else:
        live = None

    with closing(connect(args.db)) as db:
        session: SessionKey | None = resolve_session(db, args.session) if args.session and live is None else None
        BenchApp(db, live=live, session=session, baseline=args.baseline).run()


def _compare(args: argparse.Namespace) -> None:
    with closing(connect(args.db)) as db:
        lines = compare(
            db,
            args.session,
            control=args.control,
            baseline=args.baseline,
            patterns=args.metric or (),
            every=args.all,
        )
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench")
    parser.add_argument("--db", type=Path, default=RESULTS_DB, help=f"the results index (default: {RESULTS_DB})")
    commands = parser.add_subparsers(dest="command", required=True)

    importing = commands.add_parser("import", help="index results files")
    importing.add_argument("paths", nargs="*", type=Path, help=f"files or folders (default: {RESULTS_DIR})")
    importing.set_defaults(handler=_import)

    listing = commands.add_parser("runs", help="list indexed runs, newest first")
    listing.add_argument("--benchmark", help="only this benchmark (pipeline, ef_sweep, ...)")
    listing.add_argument("--database", help="only runs that include this database")
    listing.add_argument("--limit", type=int, default=20)
    listing.set_defaults(handler=_runs)

    running = commands.add_parser("run", help="run an experiment's rounds")
    add_run_arguments(running)
    running.set_defaults(handler=_run)

    comparing = commands.add_parser("compare", help="what a session found, metric by metric")
    comparing.add_argument(
        "session", help="an experiment (its latest session), experiment/session, or a session's folder"
    )
    comparing.add_argument("--control", help="the arm the others are compared to (default: round 1's first)")
    comparing.add_argument(
        "--baseline", help=f"an earlier session to compare each arm with, or {PUBLISHED!r} for the README's runs"
    )
    comparing.add_argument(
        "--metric", action="append", help="only metrics matching this pattern (queries_grpc.*); repeat for several"
    )
    comparing.add_argument("--all", action="store_true", help="every metric stored, not only the headline ones")
    comparing.set_defaults(handler=_compare)

    browsing = commands.add_parser("ui", help="browse sessions, or watch one running")
    browsing.add_argument("session", nargs="?", help="the session to open (default: the newest)")
    browsing.add_argument(
        "--live",
        nargs="?",
        const="",
        metavar="SESSION",
        help="watch a session: the newest running (the default when one is), or an experiment, experiment/session or folder",
    )
    browsing.add_argument("--baseline", help=f"compare against this session, or {PUBLISHED!r}")
    browsing.set_defaults(handler=_ui)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
