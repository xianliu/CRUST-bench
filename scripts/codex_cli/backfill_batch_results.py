"""
Backfill `batch_results.csv` from an existing Codex CLI batch output directory.

Why this exists:
- Older versions of `verify_batch.py` could crash before writing CSV, while per-benchmark
  outputs (reward.txt, job.log, etc.) were still generated.

This script scans `<output-root>/*/reward.txt` and related log files and writes a fresh
`batch_results.csv` with the same columns used by the newer batch runner.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _norm(name: str) -> str:
    n = name.replace("-", "_")
    if n and n[0].isdigit():
        n = "proj_" + n
    if "." in n:
        n = n.split(".", 1)[0]
    return n


def _read_reward(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        return txt[0].strip() if txt else ""
    except Exception:
        return ""


def _parse_seconds(job_log: Path) -> str:
    # Not always present; keep best-effort.
    return ""


def _parse_verify_exit(job_log: Path) -> str:
    # Best-effort: if a job.log exists, assume verify script completed (exit=0)
    # unless it contains an obvious failure marker.
    if not job_log.exists():
        return ""
    try:
        txt = job_log.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    # If it logged ERROR lines, mark non-zero.
    if re.search(r"\bERROR\b", txt):
        return "1"
    return "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        required=True,
        help="Existing batch output directory (contains per-benchmark subdirs).",
    )
    ap.add_argument(
        "--benchmarks-dir",
        default=str(REPO_ROOT / "datasets" / "CRUST_bench" / "CBench"),
        help="CBench directory used to define the canonical benchmark list/order.",
    )
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    cbench = Path(args.benchmarks_dir).resolve()
    if not out_root.exists():
        raise SystemExit(f"output-root not found: {out_root}")
    if not cbench.exists():
        raise SystemExit(f"benchmarks-dir not found: {cbench}")

    benchmarks = sorted([p.name for p in cbench.iterdir() if p.is_dir() and not p.name.startswith(".")])

    rows = []
    for b in benchmarks:
        out_dir = out_root / _norm(b)
        reward_path = out_dir / "reward.txt"
        job_log = out_dir / "job.log"
        row = {
            "benchmark": b,
            "reward": _read_reward(reward_path) if reward_path.exists() else "",
            "verify_exit": _parse_verify_exit(job_log),
            "seconds": _parse_seconds(job_log),
            "attempts": "",
            "output_dir": str(out_dir),
            "job_log": str(job_log),
            "codex_log": str(out_dir / "codex_cli.log"),
            "run_log": str(out_dir / "run.log"),
            "error": "",
        }
        rows.append(row)

    out_csv = out_root / "batch_results.csv"
    fields = [
        "benchmark",
        "reward",
        "verify_exit",
        "seconds",
        "attempts",
        "output_dir",
        "job_log",
        "codex_log",
        "run_log",
        "error",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()


