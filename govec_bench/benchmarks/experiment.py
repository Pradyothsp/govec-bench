"""A comparison as a file, run in rounds: `bench run experiments/<name>.yaml`.

An experiment names its arms (a database, and for GoVec the code and settings it runs) and how many
rounds to run them. Each round runs every arm through the pipeline (all of it, or the steps it
names, with what they need), one at a time, in an
order that changes between rounds, so drift over a session (heat, background load) doesn't land
on one arm. One session's files go in results/sessions/<experiment>/<started>/: a results file per
arm per round, session.json (the plan, what ran, the machine and commits) and run.log.

    description: what the comparison is for
    dataset: dbpedia            # a key of datasets/registry.py's DATASETS
    size: large                 # small (10k) or large (100k)
    rounds: 3
    order: alternate            # alternate (file order, then reversed), shuffle, or fixed
    steps: [restart-loaded]     # optional: only these pipeline steps (pipeline.MEASUREMENTS)
    arms:
      - name: v0.2.1
        database: govec-mmap    # a database in databases.yaml, opt-in ones included
        govec: ghcr.io/pradyothsp/govec:0.2.1   # GoVec only: an image or a git ref (--govec)
        set: {hnsw_m: 32}       # GoVec only: settings, as for --set

It replaces the shell scripts a night of runs used to need: it starts Docker, builds every image
and the workload before the first round, keeps the Mac awake, and waits for Docker to give disk
back between 100k runs.
"""

import argparse
import contextlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple, TextIO

import yaml

from govec_bench.adapters.registry import DATABASES, Database, env_value
from govec_bench.benchmarks.common import (
    GOVEC_SERVICE,
    MIN_FREE_DISK_BYTES,
    STARTED,
    docker,
    free_space,
    setup_of,
    shortfalls,
)
from govec_bench.benchmarks.images import describe_govec_ref, resolve_govec_image
from govec_bench.benchmarks.pipeline import (
    PIPELINE,
    Stage,
    Workload,
    build_workload,
    measure_database,
    select_steps,
    steps_of,
)
from govec_bench.datasets.registry import DATASETS, SIZES, Size
from govec_bench.events import Event, EventLog
from govec_bench.results import RESULTS_DIR, write_results
from govec_bench.results_file import RunTag

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"

SESSIONS_DIR = RESULTS_DIR / "sessions"

ORDERS = ("alternate", "shuffle", "fixed")

# One GB above the pipeline's own guard, so a run doesn't start only to stop at it. Docker
# Desktop's disk grows into the Mac's while Qdrant optimizes and gives the space back minutes later.
DISK_WAIT_BYTES = MIN_FREE_DISK_BYTES + 1000**3
DISK_WAIT_S = 20 * 60
DISK_POLL_S = 30

DOCKER_START_S = 5 * 60
DOCKER_SETTLE_S = 30  # the VM is still busy for a while after `docker info` first answers

_EXPERIMENT_KEYS = {"description", "dataset", "size", "rounds", "order", "steps", "arms"}
_ARM_KEYS = {"name", "database", "govec", "set"}


class Arm(NamedTuple):
    name: str
    database: Database  # with the arm's image and settings applied


class Experiment(NamedTuple):
    name: str
    description: str
    dataset: str
    size: Size
    rounds: int
    order: str
    arms: tuple[Arm, ...]
    steps: tuple[str, ...] | None = None  # None: the whole pipeline

    def pipeline(self) -> list[Stage]:
        return select_steps(PIPELINE, self.steps) if self.steps else list(PIPELINE)


def _arm(index: int, entry: object) -> Arm:
    if not isinstance(entry, dict):
        msg = f"arms[{index}]: expected name, database and settings, got {entry!r}"
        raise TypeError(msg)

    unknown = set(entry) - _ARM_KEYS
    if unknown:
        msg = f"arms[{index}]: unknown keys {sorted(unknown)}; expected some of {sorted(_ARM_KEYS)}"
        raise ValueError(msg)

    name, database_name = entry.get("name"), entry.get("database")
    if not isinstance(name, str) or not name:
        msg = f"arms[{index}]: needs a name"
        raise ValueError(msg)
    if database_name not in DATABASES:
        msg = f"{name}: database {database_name!r} is not one of {sorted(DATABASES)}"
        raise ValueError(msg)

    database = DATABASES[database_name]
    image_ref, settings = entry.get("govec"), entry.get("set") or {}
    if (image_ref or settings) and database.service != GOVEC_SERVICE:
        msg = f"{name}: govec and set change GoVec, but {database_name} runs {database.service}"
        raise ValueError(msg)
    if not isinstance(settings, dict):
        msg = f"{name}: set should be settings by key, got {settings!r}"
        raise TypeError(msg)

    env = {**database.env, **{f"GOVEC_{key.upper()}": env_value(value) for key, value in settings.items()}}
    return Arm(name, database._replace(env=env, image_ref=str(image_ref) if image_ref else None))


def parse_experiment(name: str, text: str) -> Experiment:
    spec = yaml.safe_load(text) or {}
    if not isinstance(spec, dict):
        msg = f"{name}: expected an experiment's settings, got a {type(spec).__name__}"
        raise TypeError(msg)

    unknown = set(spec) - _EXPERIMENT_KEYS
    if unknown:
        msg = f"{name}: unknown keys {sorted(unknown)}; expected some of {sorted(_EXPERIMENT_KEYS)}"
        raise ValueError(msg)

    dataset, size = spec.get("dataset", "sift"), spec.get("size", "small")
    rounds, order = spec.get("rounds", 1), spec.get("order", "alternate")
    if dataset not in DATASETS:
        msg = f"{name}: dataset {dataset!r} is not one of {sorted(DATASETS)}"
        raise ValueError(msg)
    if size not in SIZES:
        msg = f"{name}: size {size!r} is not one of {list(SIZES)}"
        raise ValueError(msg)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        msg = f"{name}: rounds should be a positive number, got {rounds!r}"
        raise ValueError(msg)
    if order not in ORDERS:
        msg = f"{name}: order {order!r} is not one of {list(ORDERS)}"
        raise ValueError(msg)

    entries = spec.get("arms")
    if not isinstance(entries, list) or not entries:
        msg = f"{name}: needs a list of arms"
        raise ValueError(msg)

    arms = tuple(_arm(index, entry) for index, entry in enumerate(entries))
    names = [arm.name for arm in arms]
    duplicates = sorted({arm for arm in names if names.count(arm) > 1})
    if duplicates:
        msg = f"{name}: arm names must be unique, {duplicates} repeat"
        raise ValueError(msg)

    description = str(spec.get("description", ""))
    return Experiment(name, description, dataset, size, rounds, order, arms, _steps(name, spec.get("steps")))


def _steps(name: str, steps: object) -> tuple[str, ...] | None:
    if steps is None:
        return None
    names = tuple(str(step) for step in steps) if isinstance(steps, list) else ()
    if not names:
        msg = f"{name}: steps should be a list of pipeline steps, got {steps!r}"
        raise ValueError(msg)
    try:
        select_steps(PIPELINE, names)
    except ValueError as exc:
        msg = f"{name}: {exc}"
        raise ValueError(msg) from None
    return names


def find_experiment(path_or_name: str) -> Path:
    # A path, or the name of a file in experiments/.
    path = Path(path_or_name)
    if path.exists():
        return path

    named = EXPERIMENTS_DIR / f"{path_or_name}.yaml"
    if named.exists():
        return named

    known = sorted(file.stem for file in EXPERIMENTS_DIR.glob("*.yaml"))
    msg = f"no experiment {path_or_name!r}: not a file, and not one of {known}"
    raise SystemExit(msg)


def load_experiment(path: Path) -> Experiment:
    return parse_experiment(path.stem, path.read_text())


def round_order(names: Sequence[str], round_number: int, order: str, rng: random.Random) -> list[str]:
    # alternate: each arm runs first as often as last over an even number of rounds.
    if order == "fixed" or (order == "alternate" and round_number % 2 == 1):
        return list(names)
    if order == "alternate":
        return list(reversed(names))

    return rng.sample(list(names), k=len(names))


def plan(experiment: Experiment, rounds: int, rng: random.Random) -> list[list[str]]:
    names = [arm.name for arm in experiment.arms]
    return [round_order(names, number, experiment.order, rng) for number in range(1, rounds + 1)]


# --- The session's surroundings: output, Docker, disk, sleep ---------------------------------


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}")


class _Tee:
    # Writes to the session's run.log and, unless the live view is showing it, the terminal,
    # flushing each write, so a night's output is on disk as it happens.
    def __init__(self, terminal: TextIO, file: TextIO, *, echo: bool) -> None:
        self.terminal, self.file = terminal, file
        self.streams = (terminal, file) if echo else (file,)

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


@contextlib.contextmanager
def logged_to(path: Path, *, echo: bool = True) -> Iterator[None]:
    with path.open("a") as file:
        out, err = _Tee(sys.stdout, file, echo=echo), _Tee(sys.stderr, file, echo=echo)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield


@contextlib.contextmanager
def kept_awake() -> Iterator[None]:
    # caffeinate exits with this process, so a crash can't leave the Mac unable to sleep.
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        yield
        return

    process = subprocess.Popen([caffeinate, "-dims", "-w", str(os.getpid())])  # noqa: S603 -- fixed args
    try:
        yield
    finally:
        process.terminate()


def _docker_running() -> bool:
    try:
        docker("info")
    except subprocess.CalledProcessError, RuntimeError:
        return False
    return True


def ensure_docker() -> None:
    if _docker_running():
        return

    if sys.platform != "darwin":
        msg = "Docker isn't running"
        raise SystemExit(msg)

    log("starting Docker")
    subprocess.run(["open", "-a", "Docker"], check=True)  # noqa: S607 -- macOS's own open
    deadline = time.monotonic() + DOCKER_START_S
    while not _docker_running():
        if time.monotonic() > deadline:
            msg = f"Docker didn't start within {DOCKER_START_S // 60} minutes"
            raise SystemExit(msg)
        time.sleep(5)
    time.sleep(DOCKER_SETTLE_S)


def wait_for_disk(warn: Callable[[str], None] = log) -> bool:
    deadline = time.monotonic() + DISK_WAIT_S
    while short := shortfalls(free_space(), DISK_WAIT_BYTES):
        if time.monotonic() > deadline:
            return False
        details = "; ".join(f"{space.where} has {space.free_bytes / 1000**3:.1f} GB free" for space in short)
        warn(
            f"{details} (needs {DISK_WAIT_BYTES / 1000**3:.0f}); waiting {DISK_POLL_S} s for Docker to give space back"
        )
        time.sleep(DISK_POLL_S)
    return True


def _output(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603 -- fixed tools
    except OSError, subprocess.CalledProcessError:
        return ""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Machine:
    platform: str
    cpu: str
    cpus: int | None
    memory_bytes: int | None
    docker: str


def machine() -> Machine:
    return Machine(
        platform=platform.platform(),
        cpu=_output("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        cpus=os.cpu_count(),
        memory_bytes=int(_output("sysctl", "-n", "hw.memsize") or 0) or None,
        docker=_output("docker", "version", "--format", "{{.Server.Version}}"),
    )


@dataclass(frozen=True, slots=True)
class BenchCommit:
    # govec-bench's own code: the harness is part of what a session measured.
    commit: str
    uncommitted: list[str]  # git status lines of tracked files


def bench_commit() -> BenchCommit:
    root = EXPERIMENTS_DIR.parent
    return BenchCommit(
        commit=_output("git", "rev-parse", "--short=12", "HEAD", cwd=root),
        uncommitted=_output("git", "status", "--porcelain", "--untracked-files=no", cwd=root).splitlines(),
    )


# --- Running one ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmSetup:
    database: str
    govec: str | None  # the image or git ref it runs; None for the pin
    env: dict[str, str]


RunStatus = Literal["done", "failed", "skipped"]


@dataclass(slots=True)
class ArmRun:
    round: int
    arm: str
    position: int
    started: str
    status: RunStatus | None = None
    finished: str | None = None
    file: str | None = None  # the results file, in the session's directory
    error: str | None = None


SessionStatus = Literal["running", "done", "failed", "interrupted"]


@dataclass(slots=True)
class SessionRecord:
    experiment: str
    description: str
    session: str
    dataset: str
    size: Size
    arms: dict[str, ArmSetup]
    steps: list[str]  # the pipeline steps each run runs
    plan: list[list[str]]
    machine: Machine
    govec_bench: BenchCommit
    started: str = field(default_factory=_now)
    finished: str | None = None
    status: SessionStatus = "running"
    runs: list[ArmRun] = field(default_factory=list)
    pid: int = field(default_factory=os.getpid)  # the runner's, so a live view can stop it


class Session:
    # session.json, rewritten after every run, so an interrupted night still says what it did;
    # events.jsonl, appended as things happen, for the live view.
    def __init__(self, experiment: Experiment, directory: Path, rounds: list[list[str]]) -> None:
        self.experiment = experiment
        self.directory = directory
        self.path = directory / "session.json"
        self.events = EventLog(directory)
        self.record = SessionRecord(
            experiment=experiment.name,
            description=experiment.description,
            session=directory.name,
            dataset=experiment.dataset,
            size=experiment.size,
            arms={
                arm.name: ArmSetup(arm.database.name, arm.database.image_ref, arm.database.env)
                for arm in experiment.arms
            },
            steps=steps_of(experiment.pipeline()),
            plan=rounds,
            machine=machine(),
            govec_bench=bench_commit(),
        )

    def add(self, run: ArmRun) -> None:
        self.record.runs.append(run)
        self.save()

    def warn(self, text: str) -> None:
        log(f"  {text}")
        self.events.emit(Event("warning", text=text))

    def finish(self, status: SessionStatus) -> None:
        self.record.status = status
        self.record.finished = _now()
        self.save()
        self.events.emit(Event("end", status=status))

    def save(self) -> None:
        self.path.write_text(json.dumps(asdict(self.record), indent=2))


def run_arm(
    session: Session, arm: Arm, *, workload: Workload, dataset_name: str, round_number: int, position: int
) -> None:
    experiment = session.experiment
    database = arm.database
    run = ArmRun(round=round_number, arm=arm.name, position=position, started=_now())
    session.events.emit(Event("arm", round=round_number, arm=arm.name, position=position))

    if experiment.size == "large" and not wait_for_disk(session.warn):
        run.status = "skipped"
        run.error = f"the disk stayed below {DISK_WAIT_BYTES / 1000**3:.0f} GB free for {DISK_WAIT_S // 60} minutes"
        session.warn(f"skipped {arm.name}: {run.error}")
        session.add(run)
        session.events.emit(Event("result", round=round_number, arm=arm.name, status=run.status))
        return

    def progress(kind: Literal["stage", "step"], label: str) -> None:
        session.events.emit(Event(kind, round=round_number, arm=arm.name, text=label))

    # An arm that fails before its container starts mustn't be recorded as the last one that did.
    STARTED.pop(database.name, None)
    payload = measure_database(database, workload, position, progress, experiment.pipeline())

    path = write_results(
        benchmark="pipeline",
        dataset=dataset_name,
        results={database.name: payload},
        setup=setup_of({database.name: database}),
        experiment=RunTag(
            name=experiment.name, session=session.directory.name, round=round_number, arm=arm.name, position=position
        ),
        directory=session.directory,
        name=f"r{round_number}-{arm.name}",
        steps=list(experiment.steps) if experiment.steps else None,
    )

    run.file, run.finished = path.name, _now()
    error = payload.get("error")
    if error is None:
        run.status = "done"
        log(f"  {arm.name} done -> {path.name}")
    else:
        run.status, run.error = "failed", str(error)
        session.warn(f"{arm.name} failed partway ({error}) -> {path.name}")
    session.add(run)
    session.events.emit(Event("result", round=round_number, arm=arm.name, status=run.status, file=path.name))


def run_rounds(session: Session, rounds: list[list[str]], workload: Workload, dataset_name: str) -> None:
    arms = {arm.name: arm for arm in session.experiment.arms}
    for number, order in enumerate(rounds, start=1):
        log(f"round {number} of {len(rounds)}: {', '.join(order)}")
        session.events.emit(Event("round", round=number))
        for position, name in enumerate(order):
            log(f"  {name}:")
            run_arm(
                session,
                arms[name],
                workload=workload,
                dataset_name=dataset_name,
                round_number=number,
                position=position,
            )


def describe(experiment: Experiment, rounds: list[list[str]]) -> None:
    print(f"{experiment.name}: {experiment.description}".rstrip(": "))
    print(f"  {experiment.dataset} {experiment.size}, {len(rounds)} rounds, order {experiment.order}")
    if experiment.steps:
        print(f"  steps: {', '.join(steps_of(experiment.pipeline()))}")
    for arm in experiment.arms:
        database = arm.database
        image = describe_govec_ref(database.image_ref) if database.image_ref else "the image docker-compose.yml pins"
        settings = " ".join(f"{key.removeprefix('GOVEC_').lower()}={value}" for key, value in database.env.items())
        print(f"  {arm.name}: {database.name} on {image}{', ' + settings if settings else ''}")
    for number, order in enumerate(rounds, start=1):
        print(f"  round {number}: {', '.join(order)}")


def session_directory(experiment: Experiment) -> Path:
    return SESSIONS_DIR / experiment.name / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def runner_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _is_running(folder: Path) -> bool:
    try:
        record = json.loads((folder / "session.json").read_text())
    except OSError, ValueError:
        return False
    if record.get("status") != "running":
        return False
    pid = record.get("pid")
    return isinstance(pid, int) and runner_alive(pid)


def running_sessions(sessions: Path = SESSIONS_DIR) -> list[Path]:
    # Sessions whose runner is still going, newest first (a folder is named for when it started).
    folders = (path.parent for path in sessions.glob("*/*/session.json"))
    return sorted((folder for folder in folders if _is_running(folder)), key=lambda folder: folder.name, reverse=True)


def find_session_folder(text: str | None, sessions: Path = SESSIONS_DIR) -> Path:
    # The session to watch: none named, the newest one running; else an experiment (its newest
    # session), experiment/session, or a folder.
    if not text:
        running = running_sessions(sessions)
        if not running:
            msg = "no session is running; `task bench:ui` browses the finished ones"
            raise SystemExit(msg)
        return running[0]

    for folder in (Path(text), sessions / text):
        if (folder / "session.json").exists():
            return folder
    started = sorted((sessions / text).glob("*/session.json"), key=lambda path: path.parent.name)
    if started:
        return started[-1].parent

    msg = f"no session {text!r} in {sessions}"
    raise SystemExit(msg)


def run_experiment(
    experiment: Experiment,
    rounds: int | None = None,
    *,
    dry_run: bool = False,
    directory: Path | None = None,
    echo: bool = True,
) -> Path | None:
    # directory: where the session goes, when the live view picked it to watch; else a new one.
    # echo: print to the terminal as well as run.log; off when the live view shows run.log.
    rng = random.Random()  # noqa: S311 -- an order, not a secret
    schedule = plan(experiment, rounds or experiment.rounds, rng)
    if dry_run:
        describe(experiment, schedule)
        return None

    directory = directory or session_directory(experiment)
    directory.mkdir(parents=True, exist_ok=True)

    with logged_to(directory / "run.log", echo=echo), kept_awake():
        describe(experiment, schedule)
        session = Session(experiment, directory, schedule)
        session.save()
        try:
            # Everything that could stop the night, done before the first round.
            session.events.emit(Event("step", text="checking Docker"))
            ensure_docker()
            for arm in experiment.arms:
                if arm.database.image_ref:
                    session.events.emit(Event("step", text=f"getting {arm.name}'s image ({arm.database.image_ref})"))
                    image = resolve_govec_image(arm.database.image_ref)
                    log(f"{arm.name} runs {image.image}")
            log("building the workload")
            session.events.emit(Event("step", text="building the workload"))
            dataset_name, workload = build_workload(DATASETS[experiment.dataset], experiment.size)

            run_rounds(session, schedule, workload, dataset_name)
        except KeyboardInterrupt:
            session.finish("interrupted")
            log(f"interrupted; what ran is in {directory}")
            raise SystemExit(130) from None
        except BaseException as exc:
            session.warn(f"stopped: {type(exc).__name__}: {exc}")
            session.finish("failed")
            raise

        session.finish("done")
        log(f"done: {directory}")

    return directory


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment", help="an experiment file, or the name of one in experiments/")
    parser.add_argument("--rounds", type=int, help="run this many rounds instead of the file's")
    parser.add_argument("--dry-run", action="store_true", help="show the plan and what each arm runs; run nothing")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print to the terminal instead of opening the live view (always, without one)",
    )
    parser.add_argument("--baseline", help="for the live view: a session, or 'published', to show each metric against")
    # Where the session goes: set by the live view, which starts the run and watches this folder.
    parser.add_argument("--into", type=Path, help=argparse.SUPPRESS)
