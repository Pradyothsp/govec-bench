import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import sqlite_utils

DB_PATH = Path("results") / "eval.db"


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    question_id: str
    question: str
    expected_source: str
    expected_heading: str | None = None
    gold_answer: str | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    run_id: int
    question_id: str
    retrieved: list[dict[str, object]]
    hit: bool
    rank: int | None = None
    answer_correct: bool | None = None
    citation_correct: bool | None = None


def get_db(path: Path = DB_PATH) -> sqlite_utils.Database:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_utils.Database(path)


def ensure_schema(db: sqlite_utils.Database) -> None:
    if "eval_questions" not in db.table_names():
        db.table("eval_questions").create(
            {
                "question_id": str,
                "question": str,
                "expected_source": str,
                "expected_heading": str,
                "gold_answer": str,
            },
            pk="question_id",
        )

    if "eval_runs" not in db.table_names():
        db.table("eval_runs").create(
            {"run_id": int, "timestamp": str, "stage": str, "git_commit": str},
            pk="run_id",
        )

    if "eval_results" not in db.table_names():
        db.table("eval_results").create(
            {
                "run_id": int,
                "question_id": str,
                "retrieved": str,
                "hit": bool,
                "rank": int,
                "answer_correct": bool,
                "citation_correct": bool,
            },
            foreign_keys=[
                ("run_id", "eval_runs", "run_id"),
                ("question_id", "eval_questions", "question_id"),
            ],
        )


def insert_questions(db: sqlite_utils.Database, questions: list[EvalQuestion]) -> None:
    db.table("eval_questions").upsert_all([asdict(q) for q in questions], pk="question_id")


def insert_run(db: sqlite_utils.Database, stage: str, git_commit: str | None = None) -> int:
    row = db.table("eval_runs").insert(
        {"timestamp": datetime.now(UTC).isoformat(), "stage": stage, "git_commit": git_commit}
    )
    if row.last_pk is None:
        msg = "insert into eval_runs did not return a row id"
        raise RuntimeError(msg)

    return cast(int, row.last_pk)


def insert_results(db: sqlite_utils.Database, results: list[EvalResult]) -> None:
    rows = []
    for result in results:
        row = asdict(result)
        row["retrieved"] = json.dumps(row["retrieved"])
        rows.append(row)

    db.table("eval_results").insert_all(rows)


def recall_at_k(db: sqlite_utils.Database, stage: str) -> float:
    # Questions with expected_source = "NONE" are true negative controls (no
    # correct source exists in the corpus) -- they have nothing to recall, so
    # they're excluded rather than counted as misses.
    row = next(
        db.query(
            """
            SELECT AVG(eval_results.hit) AS recall
            FROM eval_results
            JOIN eval_runs ON eval_runs.run_id = eval_results.run_id
            JOIN eval_questions ON eval_questions.question_id = eval_results.question_id
            WHERE eval_runs.stage = ? AND eval_questions.expected_source != 'NONE'
            """,
            [stage],
        )
    )
    recall = row["recall"]
    return float(recall) if recall is not None else 0.0
