"""
Carefully verify the first two CRUST-bench "tasks" (benchmarks) in the same spirit as
`/Users/xianliu/Downloads/CRUST-bench-main/scripts/codex_cli/verify_benchmark.py`,
but WITHOUT running Codex CLI.

What this does per benchmark:
1) Scaffold an output Rust crate via `src/run.py --mode scaffold --single_benchmark <name>`.
2) Validate scaffold outputs (Cargo.toml, src/, metadata/c_src, interfaces/tests present).
3) Run `cargo build` and `cargo test --no-run` to ensure type-checking / test compilation.
4) Write a per-benchmark report JSON + a combined summary JSON.

Default benchmarks = the first two from the batch script in CRUST-bench-main:
  - 2DPartInt
  - 42-Kocaeli-Printf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FIRST_TWO = ["2DPartInt", "42-Kocaeli-Printf"]


def _now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _norm(name: str) -> str:
    # Keep close to CRUST normalize rules (benchmark.py normalize_project_name).
    n = name.replace("-", "_")
    if n and n[0].isdigit():
        n = "proj_" + n
    if "." in n:
        n = n.split(".", 1)[0]
    return n


def _ensure_cargo_on_path(env: Dict[str, str]) -> Dict[str, str]:
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists():
        env = dict(env)
        env["PATH"] = f"{cargo_bin}:{env.get('PATH','')}"
    return env


def _append_log(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = msg.rstrip("\n")
    print(line)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_last_lines(path: Path, n: int = 200) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def _resolve_dataset_dir(kind: str) -> Path:
    """
    kind: 'CBench' or 'RBench'
    Prefer datasets/CRUST_bench/{kind} (as in CRUST-bench-main), fallback to datasets/{kind}.
    """
    p1 = REPO_ROOT / "datasets" / "CRUST_bench" / kind
    if p1.exists():
        return p1
    p2 = REPO_ROOT / "datasets" / kind
    return p2


def _scaffold_one(
    benchmark: str,
    output_dir: Path,
    benchmark_dir: Path,
    rust_dir: Path,
    env: Dict[str, str],
    job_log: Path,
) -> Tuple[int, Path, Path]:
    """
    Returns: (exit_code, project_dir, run_log)
    """
    run_log = output_dir / "run.log"
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = REPO_ROOT / "src" / "prompts" / "transpilation_prompts" / "bullet_point" / "bullet_point_interface.prompt"
    repairer_prompt = REPO_ROOT / "src" / "prompts" / "repair_prompts" / "bullet_point" / "bullet_point.prompt"
    endpoint_cfg = REPO_ROOT / "src" / "endpoints" / "configs" / "o1.json"
    # NOTE: scaffold mode won't actually call the endpoint, but src/run.py requires args.

    cmd_run = [
        sys.executable,
        str(REPO_ROOT / "src" / "run.py"),
        "--benchmark_dir",
        str(benchmark_dir),
        "--rust_dir",
        str(rust_dir),
        "--output_dir",
        str(output_dir),
        "--prompt",
        str(prompt),
        "--prompt_format",
        "bullet_point_with_system_instructions",
        "--prompt_strategy",
        "all",
        "--repairer_prompt",
        str(repairer_prompt),
        "--repairer_format",
        "bullet_point_with_system_instructions",
        "--repairer_strategy",
        "all",
        "--iterations",
        "0",
        "--endpoint",
        "codex",
        "--config",
        str(endpoint_cfg),
        "--mode",
        "scaffold",
        "--single_benchmark",
        benchmark,
    ]

    _append_log(job_log, f"[{_now_stamp()}] scaffold: start benchmark={benchmark}")
    with run_log.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd_run, cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    _append_log(job_log, f"[{_now_stamp()}] scaffold: exit={p.returncode} log={run_log}")
    if p.returncode != 0:
        raise RuntimeError(f"src/run.py failed (exit={p.returncode}). Last lines:\n{_read_last_lines(run_log)}")

    children = [d for d in output_dir.iterdir() if d.is_dir() and (d / "Cargo.toml").exists()]
    if len(children) != 1:
        raise RuntimeError(f"Expected exactly one project under {output_dir}, found: {[c.name for c in children]}")
    return p.returncode, children[0], run_log


def _check_scaffold_layout(proj_dir: Path) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["project_dir"] = str(proj_dir)
    checks["has_cargo_toml"] = (proj_dir / "Cargo.toml").exists()
    checks["has_src_dir"] = (proj_dir / "src").exists()

    # scaffold mode copies C into metadata/c_src
    c_src_dir = proj_dir / "metadata" / "c_src"
    checks["has_metadata_c_src"] = c_src_dir.exists()
    if c_src_dir.exists():
        c_files = [p for p in c_src_dir.iterdir() if p.is_file()]
        checks["metadata_c_src_files"] = len(c_files)
        checks["metadata_c_src_nonempty_files"] = sum(1 for p in c_files if p.stat().st_size > 0)
    else:
        checks["metadata_c_src_files"] = 0
        checks["metadata_c_src_nonempty_files"] = 0

    # scaffold keeps interfaces in src/interfaces and writes include!("interfaces/*.rs") stubs to src/*.rs
    interfaces_dir = proj_dir / "src" / "interfaces"
    checks["has_interfaces_dir"] = interfaces_dir.exists()
    iface_files = sorted([p for p in interfaces_dir.glob("*.rs")]) if interfaces_dir.exists() else []
    checks["interfaces_rs_files"] = [p.name for p in iface_files]
    checks["interfaces_rs_count"] = len(iface_files)

    tests_dir = proj_dir / "src" / "bin"
    checks["has_tests_dir"] = tests_dir.exists()
    test_files = sorted([p for p in tests_dir.glob("*.rs")]) if tests_dir.exists() else []
    checks["tests_rs_files"] = [p.name for p in test_files]
    checks["tests_rs_count"] = len(test_files)

    lib_rs = proj_dir / "src" / "lib.rs"
    checks["has_lib_rs"] = lib_rs.exists()
    if lib_rs.exists():
        checks["lib_rs_size"] = lib_rs.stat().st_size
    else:
        checks["lib_rs_size"] = 0

    return checks


def _run_cargo(proj_dir: Path, env: Dict[str, str], job_log: Path) -> Dict[str, Any]:
    res: Dict[str, Any] = {}

    def _run(cmd: List[str], name: str, timeout: int = 900) -> None:
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(proj_dir), env=env, capture_output=True, text=True, timeout=timeout)
        res[name] = {
            "cmd": cmd,
            "exit": p.returncode,
            "seconds": round(time.time() - t0, 2),
            "stdout_tail": "\n".join(p.stdout.splitlines()[-120:]),
            "stderr_tail": "\n".join(p.stderr.splitlines()[-120:]),
        }
        _append_log(job_log, f"[{_now_stamp()}] {name}: exit={p.returncode} seconds={res[name]['seconds']}")

    _append_log(job_log, f"[{_now_stamp()}] cargo: build")
    _run(["cargo", "build", "-q"], "cargo_build")

    _append_log(job_log, f"[{_now_stamp()}] cargo: test --no-run")
    _run(["cargo", "test", "-q", "--no-run"], "cargo_test_no_run")

    return res


def _verify_one(
    benchmark: str,
    out_root: Path,
    benchmark_dir: Path,
    rust_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    out_dir = out_root / _norm(benchmark)
    job_log = out_dir / "job.log"
    report: Dict[str, Any] = {
        "benchmark": benchmark,
        "output_dir": str(out_dir),
        "started_at": _now_stamp(),
        "benchmark_dir": str(benchmark_dir),
        "rust_dir": str(rust_dir),
    }

    # Share target dir across re-runs for speed.
    env = dict(env)
    env["CARGO_TARGET_DIR"] = str((out_root / ".cargo_target" / _norm(benchmark)).resolve())
    os.environ["CARGO_TARGET_DIR"] = env["CARGO_TARGET_DIR"]
    os.environ["PATH"] = env.get("PATH", os.environ.get("PATH", ""))

    try:
        _, proj_dir, run_log = _scaffold_one(
            benchmark=benchmark,
            output_dir=out_dir,
            benchmark_dir=benchmark_dir,
            rust_dir=rust_dir,
            env=env,
            job_log=job_log,
        )
        report["project_dir"] = str(proj_dir)
        report["run_log"] = str(run_log)

        report["scaffold_checks"] = _check_scaffold_layout(proj_dir)
        report["cargo"] = _run_cargo(proj_dir, env=env, job_log=job_log)

        ok = (
            report["scaffold_checks"]["has_cargo_toml"]
            and report["scaffold_checks"]["has_metadata_c_src"]
            and report["scaffold_checks"]["metadata_c_src_nonempty_files"] > 0
            and report["scaffold_checks"]["interfaces_rs_count"] > 0
            and report["scaffold_checks"]["tests_rs_count"] > 0
            and report["cargo"]["cargo_build"]["exit"] == 0
            and report["cargo"]["cargo_test_no_run"]["exit"] == 0
        )
        report["ok"] = bool(ok)
    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)
        _append_log(job_log, f"[{_now_stamp()}] ERROR: {e}")

    report["finished_at"] = _now_stamp()
    (out_dir / "verify_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_FIRST_TWO),
        help="Comma-separated benchmark list. Default: first two (2DPartInt, 42-Kocaeli-Printf).",
    )
    ap.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "verify_first_two"),
        help="Output directory root for reports and scaffolded crates.",
    )
    args = ap.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    benchmark_dir = _resolve_dataset_dir("CBench")
    rust_dir = _resolve_dataset_dir("RBench")
    if not benchmark_dir.exists():
        raise SystemExit(f"CBench not found at: {benchmark_dir}")
    if not rust_dir.exists():
        raise SystemExit(f"RBench not found at: {rust_dir}")

    env = _ensure_cargo_on_path(os.environ.copy())
    # Keep cargo output quieter; warnings not important for this verification.
    env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -Awarnings"

    summary: Dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "benchmark_dir": str(benchmark_dir),
        "rust_dir": str(rust_dir),
        "benchmarks": benchmarks,
        "started_at": _now_stamp(),
        "results": [],
    }

    for b in benchmarks:
        summary["results"].append(_verify_one(b, out_root, benchmark_dir, rust_dir, env))

    summary["finished_at"] = _now_stamp()
    summary["ok_all"] = all(r.get("ok") for r in summary["results"])
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== verify_first_two_tasks summary ===")
    for r in summary["results"]:
        print(f"- {r['benchmark']}: ok={r.get('ok')} output_dir={r.get('output_dir')}")
        if not r.get("ok") and r.get("error"):
            print(f"  error: {r['error']}")
    print(f"ok_all: {summary['ok_all']}")
    print(f"summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()


