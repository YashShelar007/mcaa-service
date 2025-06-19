# logger.py
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

import os, csv, subprocess
from datetime import datetime

LOG_PATH   = "logs/experiment_log.csv"
FIELDNAMES = [
    "timestamp", "git_commit", "script", "step",
    "accuracy", "size_mb", "latency_ms"
]

# Ensure logs/ and header exist
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
    # Fill timestamp & git_commit if missing
    if "timestamp" not in entry:
        entry["timestamp"]  = datetime.utcnow().isoformat() + "Z"
    if "git_commit" not in entry:
        entry["git_commit"] = _get_git_commit()

    # Only keep known fields
    row = {k: entry.get(k, "") for k in FIELDNAMES}
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
