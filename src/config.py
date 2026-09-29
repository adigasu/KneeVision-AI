"""
KneeVision-AI Central Configuration Loader
==========================================
Resolves all environment-specific settings from (in priority order):
  1. Environment variables  (KNEEVISION_DATA_DIR, KAGGLE_USERNAME, ...)
  2. configs/env.yaml       (local developer overrides, git-ignored)
  3. configs/base.yaml      (project defaults, committed to git)

Usage
-----
    from src.config import cfg, resolve_data_dir, get_python_bin

    data_dir = resolve_data_dir()          # auto-finds RSNA dataset
    python   = get_python_bin()            # active interpreter path
    ckpt_dir = cfg.checkpoints.local_dir   # from base/env yaml
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

# Optional yaml support (falls back gracefully if PyYAML not installed)
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ── Locate repo root ──────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    """Walk up from this file until we find configs/base.yaml."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "configs" / "base.yaml").exists():
            return candidate
    return here.parent   # fallback: parent of src/


REPO_ROOT = _find_repo_root()


# ── Simple dict-to-namespace helper ──────────────────────────────────────────
class _Namespace:
    """Recursive attribute access on a plain dict."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return f"Namespace({vars(self)})"

    def get(self, key, default=None):
        return getattr(self, key, default)


# ── Load YAML (with env-var override support) ─────────────────────────────────
def _load_yaml(path: Path) -> dict:
    if not _HAS_YAML or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config() -> _Namespace:
    base = _load_yaml(REPO_ROOT / "configs" / "base.yaml")
    env  = _load_yaml(REPO_ROOT / "configs" / "env.yaml")
    merged = _deep_merge(base, env)

    # Environment variable overrides
    _env_overrides = {
        "data.data_dir": os.environ.get("KNEEVISION_DATA_DIR", ""),
        "data.cache_dir": os.environ.get("KNEEVISION_CACHE_DIR", ""),
        "checkpoints.local_dir": os.environ.get("KNEEVISION_CKPT_DIR", ""),
        "kaggle.username": os.environ.get("KAGGLE_USERNAME", ""),
        "project.repo_root": os.environ.get("KNEEVISION_REPO_ROOT", ""),
    }
    for dotkey, val in _env_overrides.items():
        if not val:
            continue
        parts = dotkey.split(".")
        d = merged
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val

    return _Namespace(merged)


cfg = _load_config()


# ── Public helpers ────────────────────────────────────────────────────────────
def resolve_cache_dir(must_exist: bool = False) -> Path:
    """
    Return the preprocessed series cache directory.
    Priority:
      1. KNEEVISION_CACHE_DIR env var
      2. configs/env.yaml data.cache_dir
      3. configs/base.yaml data.cache_dir
      4. Auto-detect existing candidate paths:
         data/cached_series_384, data/cached_series_288px, data/preprocessed_256
    """
    # 1. Environment variable
    if os.environ.get("KNEEVISION_CACHE_DIR"):
        p = Path(os.environ["KNEEVISION_CACHE_DIR"]).expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        p = p.resolve()
        if not must_exist or p.exists():
            return p

    # 2. Config YAML
    data_cfg = getattr(cfg, "data", None)
    configured = getattr(data_cfg, "cache_dir", "") if data_cfg else ""
    if configured:
        p = Path(configured).expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        p = p.resolve()
        if not must_exist or p.exists():
            return p

    # 3. Candidate fallback detection if must_exist is True
    candidates = [
        REPO_ROOT / "data" / "cached_series_384",
        REPO_ROOT / "data" / "cached_series_288px",
        REPO_ROOT / "data" / "preprocessed_256",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    # Default fallback
    return (REPO_ROOT / "data" / "cached_series_384").resolve()

def resolve_data_dir() -> Path:
    """
    Return the RSNA dataset directory, searching candidate paths in order.
    Priority: KNEEVISION_DATA_DIR env var → env.yaml data.data_dir → search_paths.
    """
    # 1. Explicit env var
    if os.environ.get("KNEEVISION_DATA_DIR"):
        p = Path(os.environ["KNEEVISION_DATA_DIR"]).expanduser().resolve()
        if p.exists():
            return p

    # 2. env.yaml / base.yaml data_dir
    data_dir = getattr(cfg, "data", None)
    explicit = getattr(data_dir, "data_dir", "") if data_dir else ""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p

    # 3. search_paths list
    search = getattr(data_dir, "search_paths", []) if data_dir else []
    if isinstance(search, str):
        search = [search]
    for cand in search:
        p = Path(cand).expanduser().resolve()
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not locate the RSNA dataset directory.\n"
        "Set KNEEVISION_DATA_DIR env var or set data.data_dir in configs/env.yaml.\n"
        f"Searched: {search}"
    )


def resolve_checkpoint_dir() -> Path:
    """Return the local checkpoints directory (created if missing)."""
    ckpt = getattr(cfg, "checkpoints", None)
    local = getattr(ckpt, "local_dir", "./checkpoints") if ckpt else "./checkpoints"
    p = (REPO_ROOT / local).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_python_bin() -> str:
    """Return the path to the currently active Python interpreter."""
    return sys.executable


def get_conda_env_name() -> str:
    """Return conda env name from config (informational only)."""
    proj = getattr(cfg, "project", None)
    return getattr(proj, "conda_env", "rsna-knee") if proj else "rsna-knee"


def get_kaggle_config() -> dict:
    """Return Kaggle username / slugs from env.yaml or environment variables."""
    kg = getattr(cfg, "kaggle", None)
    username = (
        os.environ.get("KAGGLE_USERNAME")
        or (getattr(kg, "username", "") if kg else "")
    )
    dataset_slug = (
        os.environ.get("KNEEVISION_DATASET_SLUG")
        or (getattr(kg, "dataset_slug", "") if kg else "")
    )
    kernel_slug = (
        os.environ.get("KNEEVISION_KERNEL_SLUG")
        or (getattr(kg, "kernel_slug", "") if kg else "")
    )
    return {
        "username": username,
        "dataset_slug": dataset_slug,
        "kernel_slug": kernel_slug,
    }


def resolve_artifacts_dir() -> Path:
    """Return the base artifacts directory (created if missing)."""
    art = getattr(cfg, "artifacts", None)
    local = getattr(art, "root_dir", "./artifacts") if art else "./artifacts"
    p = (REPO_ROOT / local).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_experiment_artifacts(exp_name: str) -> dict[str, Path]:
    """
    Return organized directory paths for a specific experiment:
      - 'root'        : artifacts/experiments/<exp_name>/
      - 'checkpoints' : artifacts/experiments/<exp_name>/checkpoints/
      - 'logs'        : artifacts/experiments/<exp_name>/logs/
      - 'metrics'     : artifacts/experiments/<exp_name>/metrics/
      - 'figures'     : artifacts/experiments/<exp_name>/figures/
    All directories are automatically created.
    """
    base_art = resolve_artifacts_dir()
    exp_root = base_art / "experiments" / exp_name
    paths = {
        "root": exp_root,
        "checkpoints": exp_root / "checkpoints",
        "logs": exp_root / "logs",
        "metrics": exp_root / "metrics",
        "figures": exp_root / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
