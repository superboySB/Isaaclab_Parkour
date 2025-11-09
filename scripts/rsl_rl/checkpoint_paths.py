# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for resolving checkpoint locations used by the RSL-RL scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_ROOT = _PROJECT_ROOT / "assets"


def _iter_existing_dirs(directories: Iterable[Path]):
    """Yield directories that exist on disk."""
    for directory in directories:
        if directory.is_dir():
            yield directory


def _pick_latest_checkpoint(directory: Path) -> Optional[Path]:
    """Return the latest checkpoint inside ``directory`` according to the training iteration."""
    def _iteration(path: Path) -> int:
        stem = path.stem
        digits = "".join(filter(str.isdigit, stem))
        return int(digits) if digits else -1

    candidates = sorted(directory.glob("model_*.pt"), key=_iteration)
    if candidates:
        return candidates[-1]

    fallback = directory / "checkpoint.pt"
    if fallback.exists():
        return fallback

    return None


def get_local_pretrained_checkpoint(task_name: str) -> Optional[str]:
    """Resolve the checkpoint path cached under the repository's ``assets`` folder for a task.

    This helper looks for the latest ``model_*.pt`` file inside the corresponding assets sub-folder.
    Currently, tasks containing ``Teacher-Unitree-Go2`` map to ``assets/pretrained_teacher`` and
    tasks containing ``Student-Unitree-Go2`` map to ``assets/pretrained_student``.
    """
    task_to_asset_dirs: list[Path] = []

    if "Teacher-Unitree-Go2" in task_name:
        task_to_asset_dirs.append(_ASSETS_ROOT / "pretrained_teacher")
    if "Student-Unitree-Go2" in task_name:
        task_to_asset_dirs.append(_ASSETS_ROOT / "pretrained_student")

    for directory in _iter_existing_dirs(task_to_asset_dirs):
        checkpoint = _pick_latest_checkpoint(directory)
        if checkpoint:
            return str(checkpoint)

    return None
