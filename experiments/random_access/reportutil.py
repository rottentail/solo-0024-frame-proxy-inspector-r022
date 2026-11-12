"""Shared helpers for the random-access experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"


def write_report(name: str, payload: dict) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def summarize_costs(batch) -> dict:
    return {
        "cells": len(batch.outcomes),
        "batch_pictures_decoded": batch.batch_pictures_decoded,
        "batch_compressed_bytes_fed": batch.batch_bytes_fed,
        "seeks": batch.seeks,
        "worst_single_cell_pictures": max(
            item.cell_pictures_decoded for item in batch.outcomes
        ),
        "all_exact": batch.all_exact,
    }


def run_module() -> None:
    sys.path.insert(0, str(HERE))
