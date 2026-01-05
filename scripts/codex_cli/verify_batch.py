"""
Batch-verify multiple CRUST-bench benchmarks using the official OpenAI Codex CLI agent.

Default batch = the 10 benchmarks visible in the user's screenshot:
  2DPartInt, 42-Kocaeli-Printf, aes128-SIMD, amp, approxidate,
  avalanche, bhshell, bigint, bitset, blt

Each benchmark run:
- scaffolds output crate via `src/run.py --mode scaffold`
- runs Codex CLI agent for up to --timeout-sec seconds
- writes reward.txt (1/0), job.log, run.log, codex_cli.log, error_report.csv, test_report.csv

Batch outputs:
- <output-root>/batch.log
- <output-root>/batch_results.csv
"""

from __future__ import annotations

import argparse
import csv
import threading
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_BENCHMARKS = [
    "2DPartInt",
    "42-Kocaeli-Printf",
    "aes128-SIMD",
    "amp",
    "approxidate",
    "avalanche",
    "bhshell",
    "bigint",
    "bitset",
    "blt",
]

def _normalize_model(model: str) -> str:
    # Codex CLI expects OpenAI model IDs like `gpt-5-mini` (no `openai/` prefix).
    # Keep backward compatibility for users who pass `openai/<model>`.
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _norm(name: str) -> str:
    # Safe-ish folder name, keep close to CRUST normalize rules.
    n = name.replace("-", "_")
    if n and n[0].isdigit():
        n = "proj_" + n
    if "." in n:
        n = n.split(".", 1)[0]
    return n


def _append(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")
    print(msg.rstrip("\n"))

def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _run_one(benchmark: str, timeout_sec: int, model: str, out_dir: Path) -> dict:
    t0 = time.time()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "codex_cli" / "verify_benchmark.py"),
        "--benchmark",
        benchmark,
        "--timeout-sec",
        str(timeout_sec),
        "--model",
        model,
        "--output-dir",
        str(out_dir),
    ]
    p = subprocess.run(cmd, cwd=str(REPO_ROOT))
    dt = round(time.time() - t0, 2)

    reward_path = out_dir / "reward.txt"
    reward = reward_path.read_text(encoding="utf-8").strip() if reward_path.exists() else ""
    job_log = out_dir / "job.log"
    return {
        "benchmark": benchmark,
        "output_dir": str(out_dir),
        "reward": reward,
        "verify_exit": str(p.returncode),
        "seconds": str(dt),
        "job_log": str(job_log),
        "codex_log": str(out_dir / "codex_cli.log"),
        "run_log": str(out_dir / "run.log"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmark list (defaults to the 10 in the screenshot).",
    )
    ap.add_argument("--timeout-sec", type=int, default=120, help="Seconds per benchmark for Codex CLI agent.")
    ap.add_argument("--model", default="gpt-5-nano", help="Codex agent model.")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="How many benchmarks to run in parallel. Use 1 for sequential (default).",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=0,
        help="How many times to retry a benchmark if verify_benchmark.py exits non-zero.",
    )
    ap.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "codex_cli_batch_10"),
        help="Directory to store per-benchmark outputs + batch summary.",
    )
    args = ap.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    benchmarks = _dedupe_keep_order(benchmarks)
    model = _normalize_model(args.model)
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    batch_log = out_root / "batch.log"
    results_csv = out_root / "batch_results.csv"

    jobs = max(1, int(args.jobs))
    retries = max(0, int(args.retries))
    _append(
        batch_log,
        f"[batch] start benchmarks={benchmarks} timeout={args.timeout_sec}s model={model} jobs={jobs} retries={retries}",
    )

    lock = threading.Lock()
    rows_by_benchmark: dict[str, dict] = {}

    _csv_fields = [
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

    def _write_csv_locked() -> None:
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_csv_fields, extrasaction="ignore")
            w.writeheader()
            # keep stable order in CSV
            for b in benchmarks:
                if b in rows_by_benchmark:
                    w.writerow(rows_by_benchmark[b])

    def _task(b: str) -> dict:
        out_dir = out_root / _norm(b)
        attempts = 0
        last_row: dict | None = None
        while True:
            attempts += 1
            last_row = _run_one(b, int(args.timeout_sec), model, out_dir)
            if last_row["verify_exit"] == "0" or attempts > retries + 1:
                last_row["attempts"] = str(attempts)
                return last_row
            with lock:
                _append(batch_log, f"[batch] retry {b} attempt={attempts}/{retries+1} exit={last_row['verify_exit']}")

    # Kick off tasks
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {}
        for idx, b in enumerate(benchmarks, start=1):
            out_dir = out_root / _norm(b)
            _append(batch_log, f"[batch] ({idx}/{len(benchmarks)}) queued {b} -> {out_dir}")
            futs[ex.submit(_task, b)] = (idx, b, out_dir)

        done_count = 0
        for fut in as_completed(futs):
            idx, b, out_dir = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "benchmark": b,
                    "output_dir": str(out_dir),
                    "reward": "",
                    "verify_exit": "1",
                    "seconds": "0",
                    "job_log": str(out_dir / "job.log"),
                    "codex_log": str(out_dir / "codex_cli.log"),
                    "run_log": str(out_dir / "run.log"),
                    "error": str(e),
                    "attempts": "1",
                }

            with lock:
                rows_by_benchmark[b] = row
                done_count += 1
                _append(
                    batch_log,
                    f"[batch] ({done_count}/{len(benchmarks)}) done {b} exit={row.get('verify_exit')} reward={row.get('reward')} seconds={row.get('seconds')} attempts={row.get('attempts')}",
                )
                _write_csv_locked()

    _append(batch_log, "[batch] done")


if __name__ == "__main__":
    main()


