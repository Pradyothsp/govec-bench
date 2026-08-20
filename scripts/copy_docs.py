#!/usr/bin/env python3
import shutil
from pathlib import Path

DOCS_DIR = Path("data/docs")
EXCLUDE_DIR_NAMES = {".venv", ".git", "node_modules", "data", "results"}

SOURCE_REPOS = [
    Path("../govec"),
    Path("../govec-python"),
    Path(),
]


def _is_excluded(path: Path, repo_root: Path) -> bool:
    relative_parts = path.relative_to(repo_root).parts
    return any(part in EXCLUDE_DIR_NAMES for part in relative_parts)


def copy_repo_docs(repo_root: Path) -> int:
    count = 0

    for path in sorted(repo_root.rglob("*.md")):
        if _is_excluded(path, repo_root):
            continue

        relative = path.relative_to(repo_root)
        dest = DOCS_DIR / repo_root.resolve().name / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        count += 1

    return count


def main() -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True)

    for repo_root in SOURCE_REPOS:
        count = copy_repo_docs(repo_root)
        print(f"{repo_root.resolve().name}: copied {count} files")

    print(f"Done. Files in {DOCS_DIR}/")


if __name__ == "__main__":
    main()
