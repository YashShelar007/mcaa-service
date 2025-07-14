#!/usr/bin/env python3
# logger.py

import os
import csv
import subprocess
import warnings
from datetime import datetime

# Suppress PyTorch weight-only warning
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

LOG_PATH = "logs/experiment_log.csv"
FIELDNAMES = [
    "timestamp", "git_commit", "script", "step",
    "accuracy", "size_mb", "latency_ms"
]

# Ensure logs/ directory and CSV header exist
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

def _get_git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=1
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"

def log(entry: dict):
    """
    Append a row to logs/experiment_log.csv. Fills in timestamp & git_commit if missing.
    """
    # Fill defaults
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    if "git_commit" not in entry:
        entry["git_commit"] = _get_git_commit()

    # Keep only known fields, in order
    row = {k: entry.get(k, "") for k in FIELDNAMES}

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
