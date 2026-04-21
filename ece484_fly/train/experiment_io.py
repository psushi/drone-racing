"""Helpers for reproducible experiment artifacts."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import toml
from ml_collections import ConfigDict


def checkpoint_config_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_suffix(".toml")


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_suffix(".json")


def resolve_config_path(repo_root: Path, config: str) -> Path:
    config_path = Path(config)
    if config_path.exists():
        return config_path.resolve()
    return (repo_root / "config" / config).resolve()


def choose_runtime_config_path(
    repo_root: Path,
    checkpoint_path: str | Path,
    config: str,
    *,
    prefer_checkpoint_sidecar: bool = True,
    default_config_name: str = "level1.toml",
) -> Path:
    """Resolve which config file a tool should use.

    If the caller keeps the default config name and a checkpoint-sidecar config exists,
    prefer that sidecar. This keeps train/watch/debug aligned by default.
    """

    checkpoint_cfg = checkpoint_config_path(checkpoint_path)
    if prefer_checkpoint_sidecar and config == default_config_name and checkpoint_cfg.exists():
        return checkpoint_cfg.resolve()
    return resolve_config_path(repo_root, config)


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def write_experiment_sidecar(
    *,
    checkpoint_path: str | Path,
    resolved_config_path: Path,
    cfg: ConfigDict,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Write config and metadata next to a checkpoint path."""

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_config_path(checkpoint)
    metadata_path = checkpoint_metadata_path(checkpoint)

    with config_path.open("w", encoding="utf-8") as f:
        toml.dump(cfg.to_dict(), f)

    payload = {
        "checkpoint_path": str(checkpoint.resolve()),
        "config_path": str(config_path.resolve()),
        "source_config_path": str(resolved_config_path.resolve()),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    return config_path, metadata_path


def default_metadata(
    *,
    repo_root: Path,
    seed: int,
    device: str,
    num_envs_effective: int,
    transitions_per_iter: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "device": device,
        "num_envs_effective": num_envs_effective,
        "transitions_per_iter": transitions_per_iter,
        "git_commit": _git_commit(repo_root),
    }
