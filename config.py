"""Configuration management for nano claude (multi-provider)."""
import json
import os
import warnings
from pathlib import Path

from local_env import load_local_env

CONFIG_DIR   = Path(os.environ.get("NANOCC_CONFIG_DIR", Path.home() / ".nano_claude"))
CONFIG_FILE  = CONFIG_DIR  / "config.json"
HISTORY_FILE = CONFIG_DIR  / "input_history.txt"
SESSIONS_DIR = CONFIG_DIR  / "sessions"

DEFAULTS = {
    "model":            "claude-opus-4-6",
    "max_tokens":       8192,
    "permission_mode":  "auto",   # auto | accept-all | manual
    "verbose":          False,
    "thinking":         False,
    "thinking_budget":  10000,
    "custom_base_url":  "",       # for "custom" provider
    # Per-provider API keys (optional; env vars take priority)
    # "anthropic_api_key": "sk-ant-..."
    # "openai_api_key":    "sk-..."
    # "gemini_api_key":    "..."
    # "kimi_api_key":      "..."
    # "qwen_api_key":      "..."
    # "zhipu_api_key":     "..."
    # "deepseek_api_key":  "..."
}


def load_config() -> dict:
    # Reading configuration must also work in read-only environments and offline evaluation.
    load_local_env()
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except (OSError, ValueError, TypeError):
            warnings.warn("Cannot read configuration; using defaults.", RuntimeWarning, stacklevel=2)
    # Backward-compat: legacy single api_key → anthropic_api_key
    if cfg.get("api_key") and not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = cfg.pop("api_key")
    # Also accept ANTHROPIC_API_KEY env for backward-compat
    if not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    return cfg


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(exist_ok=True)
    data = dict(cfg)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def current_provider(cfg: dict) -> str:
    from providers import detect_provider
    return detect_provider(cfg.get("model", "claude-opus-4-6"))


def has_api_key(cfg: dict) -> bool:
    """Check whether the active provider has an API key configured."""
    from providers import get_api_key
    pname = current_provider(cfg)
    key = get_api_key(pname, cfg)
    return bool(key)


def calc_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    from providers import calc_cost as _cc
    return _cc(model, in_tokens, out_tokens)
