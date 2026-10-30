"""The harness's commands over its results: `python -m govec_bench.cli <command>`.

import [PATH ...]   index results files (default: results/, the published ones included);
                    folders are searched, so `import ../govec-notes/bench` adds older sessions
runs                list indexed runs, newest first, with what each database ran as
"""

import argparse
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

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

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
