import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_metadata(models_dir, model_version, extra=None):
    meta = {
        "model_version": model_version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    if extra:
        meta.update(extra)

    path = Path(models_dir) / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta