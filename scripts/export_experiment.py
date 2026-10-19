"""Archive reproducible experiment inputs/results, excluding credentials and model caches."""

from __future__ import annotations

import argparse
import tarfile
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("eval/results/rebuild-20261019")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (
        args.output
        or Path("artifacts") / f"paperagent-{datetime.now():%Y%m%d-%H%M%S}.tar.gz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [
        Path("corpus"),
        Path("eval/data"),
        Path("eval/question_bank"),
        Path("eval/review/work"),
        args.results,
        Path("docs/experiments.md"),
        Path("docs/results.json"),
        Path("requirements.lock.txt"),
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path(".env.example"),
        Path("README.md"),
        Path("tests"),
    ]
    with tarfile.open(output, "w:gz") as archive:
        for source in sources:
            for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
                if (
                    not path.is_file()
                    or "__pycache__" in path.parts
                    or path.suffix == ".pyc"
                ):
                    continue
                archive.add(path, arcname=str(path), recursive=False)
        # Keep the implementation that generated these results alongside the data.
        for directory in [
            Path("."),
            Path("retrieval"),
            Path("pipeline"),
            Path("eval"),
            Path("scripts"),
        ]:
            for path in sorted(directory.glob("*.py")):
                if path.parts[0] != "corpus":
                    archive.add(path, arcname=str(path), recursive=False)
    print(output.resolve())


if __name__ == "__main__":
    main()
