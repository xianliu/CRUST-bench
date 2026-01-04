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
import subprocess
import sys
import time
from pathlib import Path


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
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "codex_cli_batch_10"),
        help="Directory to store per-benchmark outputs + batch summary.",
    )
    args = ap.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    model = _normalize_model(args.model)
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    batch_log = out_root / "batch.log"
    results_csv = out_root / "batch_results.csv"

    _append(batch_log, f"[batch] start benchmarks={benchmarks} timeout={args.timeout_sec}s model={model}")

    rows = []
    for idx, b in enumerate(benchmarks, start=1):
        out_dir = out_root / _norm(b)
        _append(batch_log, f"[batch] ({idx}/{len(benchmarks)}) start {b} -> {out_dir}")
        t0 = time.time()

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "codex_cli" / "verify_benchmark.py"),
            "--benchmark",
            b,
            "--timeout-sec",
            str(args.timeout_sec),
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

        rows.append(
            {
                "benchmark": b,
                "output_dir": str(out_dir),
                "reward": reward,
                "verify_exit": str(p.returncode),
                "seconds": str(dt),
                "job_log": str(job_log),
                "codex_log": str(out_dir / "codex_cli.log"),
                "run_log": str(out_dir / "run.log"),
            }
        )
        _append(batch_log, f"[batch] ({idx}/{len(benchmarks)}) done {b} exit={p.returncode} reward={reward} seconds={dt}")

        # Write/update CSV after each benchmark so progress is visible mid-run.
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["benchmark", "reward", "verify_exit", "seconds", "output_dir", "job_log", "codex_log", "run_log"],
            )
            w.writeheader()
            w.writerows(rows)

    _append(batch_log, "[batch] done")


if __name__ == "__main__":
    main()


