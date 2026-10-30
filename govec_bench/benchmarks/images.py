"""Which GoVec image a run uses, when it isn't the one docker-compose.yml pins.

A ref with a colon is an image, used as it is: `ghcr.io/pradyothsp/govec:0.2.1`, `govec:local`.
Anything else is a git ref of the GoVec repo (`HEAD`, `v0.2.1`, a branch, a commit), built here
from a clean copy of that commit and tagged with it, so a commit is built once and every result
names the exact code it ran. Uncommitted changes are never in a built image.
"""

import os
import shutil
import subprocess
import tarfile
import tempfile
from functools import cache
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

GOVEC_REPO = Path(os.environ.get("GOVEC_REPO", Path(__file__).resolve().parents[2].parent / "govec"))

BUILT_IMAGE = "govec-bench/govec"


class GovecImage(NamedTuple):
    image: str
    commit: str | None  # the GoVec commit it was built from; None for an image used as it is


def is_image(ref: str) -> bool:
    return ":" in ref


def built_tag(commit: str) -> str:
    return f"{BUILT_IMAGE}:{commit}"


def _run(*args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, input=stdin, check=True, capture_output=True)  # noqa: S603 -- fixed tools, refs passed as single args


def _git(*args: str) -> str:
    return _run("git", "-C", str(GOVEC_REPO), *args).stdout.decode().strip()


def _docker() -> str:
    docker = shutil.which("docker")
    if docker is None:
        msg = "docker is required to build a GoVec image but was not found on PATH"
        raise RuntimeError(msg)

    return docker


def _image_exists(image: str) -> bool:
    return subprocess.run([_docker(), "image", "inspect", image], check=False, capture_output=True).returncode == 0  # noqa: S603 -- fixed args


def _build(commit: str, tag: str) -> None:
    archive = _run("git", "-C", str(GOVEC_REPO), "archive", "--format=tar", commit).stdout

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=BytesIO(archive)) as tar:
            tar.extractall(tmp, filter="data")

        _run(_docker(), "build", "--quiet", "--build-arg", f"VERSION={commit}", "--tag", tag, tmp)


def commit_of(ref: str) -> str:
    # Raises CalledProcessError when the GoVec repo has no such ref.
    return _git("rev-parse", "--short=12", f"{ref}^{{commit}}")


def describe_govec_ref(ref: str) -> str:
    # What a ref would run, without building anything: for bench run --dry-run.
    if is_image(ref):
        return f"image {ref}" + ("" if _image_exists(ref) else " (not pulled yet)")

    commit = commit_of(ref)
    tag = built_tag(commit)
    return f"{ref} = commit {commit}, " + (f"built as {tag}" if _image_exists(tag) else "to be built")


@cache
def resolve_govec_image(ref: str) -> GovecImage:
    if is_image(ref):
        return GovecImage(image=ref, commit=None)

    commit = commit_of(ref)
    tag = built_tag(commit)

    if _git("status", "--porcelain", "--untracked-files=no") and _git("rev-parse", "--short=12", "HEAD") == commit:
        print(f"note: {GOVEC_REPO} has uncommitted changes; the image is built from {commit} without them")

    if _image_exists(tag):
        print(f"GoVec {ref} is {commit}: using {tag}")
    else:
        print(f"GoVec {ref} is {commit}: building {tag} from a clean copy of {GOVEC_REPO}...")
        _build(commit, tag)

    return GovecImage(image=tag, commit=commit)
