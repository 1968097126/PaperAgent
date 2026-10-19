"""Load local experiment secrets without printing or persisting them."""

import os
from pathlib import Path


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name = name.strip()
        if name in {"DEEPSEEK_API_KEY", "NANOCC_MODEL_CACHE_DIR"}:
            os.environ.setdefault(name, value.strip().strip("\"'"))
