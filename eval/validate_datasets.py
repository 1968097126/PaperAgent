"""CLI for strict evaluation-set validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.datasets import validate_datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("eval/data"))
    parser.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
    args = parser.parse_args(argv)
    try:
        result = validate_datasets(args.data_dir, args.chunks)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
