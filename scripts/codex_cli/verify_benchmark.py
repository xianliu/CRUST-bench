"""
Verify a single CRUST-bench benchmark using the official OpenAI Codex CLI agent.

What this does:
1) Uses src/run.py to generate an output Rust project for ONE benchmark (fast, local generation).
2) Runs `codex exec` as an agent inside that Rust project for a bounded time budget.
3) Re-runs `cargo test` and writes CRUST-style reports for the output directory.

Prereqs:
- Rust installed (`cargo` available)
- OpenAI Codex CLI installed (e.g. `npm i -g @openai/codex`)
- Logged in: `printenv OPENAI_API_KEY | codex login --with-api-key`

Example:
  python scripts/codex_cli/verify_benchmark.py --benchmark 2DPartInt --timeout-sec 120 --model gpt-5-nano
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "codex_cli_verify"


def _which_codex() -> Optional[str]:
    # Prefer the global codex CLI if installed via npm/brew.
    for p in ["/usr/local/bin/codex", "/opt/homebrew/bin/codex"]:
        if Path(p).exists():
            return p
    return shutil.which("codex")


def _ensure_cargo_on_path(env: dict) -> dict:
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.exists():
        env = dict(env)
        env["PATH"] = f"{cargo_bin}:{env.get('PATH','')}"
    return env


def _read_last_lines(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def _append_log(job_log: Path, msg: str) -> None:
    job_log.parent.mkdir(parents=True, exist_ok=True)
    line = msg.rstrip("\n")
    print(line)
    with job_log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _now_stamp() -> str:
    # Avoid adding extra deps; keep a simple monotonic-ish stamp.
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S")

def _mask_key(k: str) -> str:
    k = (k or "").strip()
    if not k:
        return "<empty>"
    if len(k) <= 12:
        return k[:3] + "..." + k[-2:]
    return k[:7] + "..." + k[-4:]


def _load_openai_key_from_dotenv(dotenv_path: Path) -> Optional[str]:
    if not dotenv_path.exists():
        return None
    for line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("OPENAI_API_KEY="):
            return s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _force_openai_key_from_dotenv(env: dict) -> tuple[dict, str]:
    """
    Force OPENAI_API_KEY from `<repo>/.env` if present and non-empty.
    Falls back to existing env OPENAI_API_KEY otherwise.
    Returns (env, source_str).
    """
    dotenv = REPO_ROOT / ".env"
    key = _load_openai_key_from_dotenv(dotenv)
    if key:
        env = dict(env)
        env["OPENAI_API_KEY"] = key
        return env, ".env"
    if env.get("OPENAI_API_KEY"):
        return env, "env"
    return env, "missing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, help="Benchmark name, e.g. 2DPartInt or file2str")
    ap.add_argument("--model", default="gpt-5-nano", help="Codex agent model, e.g. gpt-5-nano")
    ap.add_argument("--timeout-sec", type=int, default=120, help="Max seconds to let Codex agent run")
    ap.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to place the generated benchmark project (directory).",
    )
    ap.add_argument(
        "--run-iterations",
        type=int,
        default=0,
        help="How many CRUST repair iterations to run before Codex agent (0 recommended).",
    )
    ap.add_argument(
        "--endpoint-config",
        default=str(REPO_ROOT / "src" / "endpoints" / "configs" / "codex.json"),
        help="Config for src/run.py model endpoint (not Codex CLI agent model).",
    )
    ap.add_argument(
        "--prompt",
        default=str(
            REPO_ROOT
            / "src"
            / "prompts"
            / "transpilation_prompts"
            / "bullet_point"
            / "bullet_point_interface.prompt"
        ),
        help="Transpilation prompt file for src/run.py.",
    )
    ap.add_argument(
        "--repairer-prompt",
        default=str(
            REPO_ROOT / "src" / "prompts" / "repair_prompts" / "bullet_point" / "bullet_point.prompt"
        ),
        help="Repair prompt file for src/run.py.",
    )
    ap.add_argument(
        "--prompt-format",
        default="bullet_point_with_system_instructions",
        help="Prompt format for src/run.py.",
    )
    ap.add_argument(
        "--repairer-format",
        default="bullet_point_with_system_instructions",
        help="Repairer prompt format for src/run.py.",
    )
    ap.add_argument("--benchmark-dir", default=str(REPO_ROOT / "datasets" / "CRUST_bench" / "CBench"))
    ap.add_argument("--rust-dir", default=str(REPO_ROOT / "datasets" / "CRUST_bench" / "RBench"))
    args = ap.parse_args()

    codex_bin = _which_codex()
    if not codex_bin:
        raise SystemExit("Could not find `codex` CLI. Install with `npm i -g @openai/codex`.")
    try:
        codex_version = subprocess.check_output([codex_bin, "--version"], text=True).strip()
    except Exception:
        codex_version = "<unknown>"

    output_dir = Path(args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    job_log = output_dir / "job.log"

    env = _ensure_cargo_on_path(os.environ.copy())
    env, key_source = _force_openai_key_from_dotenv(env)
    # Speed up repeated builds/tests by sharing cargo build artifacts across runs.
    # (Especially important for short agent budgets.)
    env["CARGO_TARGET_DIR"] = str((output_dir.parent / ".cargo_target" / args.benchmark).resolve())
    # Some CRUST utilities call subprocess without an explicit env; make sure PATH is updated
    # for the current process too (so `cargo` is found).
    os.environ["PATH"] = env.get("PATH", os.environ.get("PATH", ""))
    os.environ["CARGO_TARGET_DIR"] = env["CARGO_TARGET_DIR"]
    _append_log(job_log, f"[{_now_stamp()}] start benchmark={args.benchmark} model={args.model} timeout={args.timeout_sec}s")
    _append_log(job_log, f"[{_now_stamp()}] codex_cli={codex_bin} ({codex_version})")
    _append_log(job_log, f"[{_now_stamp()}] output_dir={output_dir}")
    _append_log(job_log, f"[{_now_stamp()}] OPENAI_API_KEY source={key_source} value={_mask_key(env.get('OPENAI_API_KEY',''))}")

    # 1) Generate benchmark output project via src/run.py
    run_log = output_dir / "run.log"
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _append_log(job_log, f"[{_now_stamp()}] scaffold: running src/run.py (log={run_log})")

    cmd_run = [
        sys.executable,
        str(REPO_ROOT / "src" / "run.py"),
        "--benchmark_dir",
        args.benchmark_dir,
        "--rust_dir",
        args.rust_dir,
        "--output_dir",
        str(output_dir),
        "--prompt",
        args.prompt,
        "--prompt_format",
        args.prompt_format,
        "--prompt_strategy",
        "all",
        "--repairer_prompt",
        args.repairer_prompt,
        "--repairer_format",
        args.repairer_format,
        "--repairer_strategy",
        "all",
        "--iterations",
        str(args.run_iterations),
        "--endpoint",
        "codex",
        "--config",
        args.endpoint_config,
        "--mode",
        "scaffold",
        "--single_benchmark",
        args.benchmark,
    ]
    with run_log.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd_run, cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(
            f"src/run.py failed (exit={p.returncode}). See log: {run_log}\n\nLast lines:\n{_read_last_lines(run_log)}"
        )
    _append_log(job_log, f"[{_now_stamp()}] scaffold: done exit={p.returncode}")

    # Find generated project folder (normalized name)
    # Because output_dir contains the project folder directly.
    children = [d for d in output_dir.iterdir() if d.is_dir() and (d / "Cargo.toml").exists()]
    if len(children) != 1:
        raise SystemExit(f"Expected exactly one project under {output_dir}, found: {[c.name for c in children]}")
    proj_dir = children[0]
    _append_log(job_log, f"[{_now_stamp()}] project_dir={proj_dir}")

    # Baseline test
    _append_log(job_log, f"[{_now_stamp()}] baseline: cargo test -q")
    baseline = subprocess.run(
        ["cargo", "test", "-q"],
        cwd=str(proj_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    _append_log(job_log, f"[{_now_stamp()}] baseline_exit={baseline.returncode}")

    # 2) Run Codex CLI agent with bounded time budget
    codex_log = output_dir / "codex_cli.log"
    _append_log(job_log, f"[{_now_stamp()}] codex: starting (log={codex_log})")
    prompt_text = subprocess.check_output(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "codex_cli" / "build_codex_agent_prompt.py"),
            "--project",
            proj_dir.name,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
    )
    with codex_log.open("w", encoding="utf-8") as f:
        try:
            codex_proc = subprocess.run(
                [
                    codex_bin,
                    "exec",
                    "-C",
                    str(proj_dir),
                    "-m",
                    args.model,
                    "-c",
                    "shell_environment_policy.inherit=all",
                    "--skip-git-repo-check",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "-o",
                    str(proj_dir / "metadata" / "codex_cli_last.txt"),
                ],
                input=prompt_text,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            codex_proc = None
            f.write(f"\n[TIMEOUT] Stopped after {args.timeout_sec} seconds.\n")
    _append_log(job_log, f"[{_now_stamp()}] codex_exit={None if codex_proc is None else codex_proc.returncode}")

    # 3) Re-test and generate CRUST-style reports
    _append_log(job_log, f"[{_now_stamp()}] after: cargo test -q")
    after = subprocess.run(
        ["cargo", "test", "-q"],
        cwd=str(proj_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    _append_log(job_log, f"[{_now_stamp()}] after_exit={after.returncode}")

    # Reward signal for external evaluation tooling.
    # 1 if final tests pass, else 0.
    (output_dir / "reward.txt").write_text("1\n" if after.returncode == 0 else "0\n", encoding="utf-8")
    _append_log(job_log, f"[{_now_stamp()}] reward={(output_dir / 'reward.txt').read_text(encoding='utf-8').strip()}")

    # Generate reports using existing utilities (same format as src/run.py produces)
    # Import using the repo's `src/` as a module root.
    _append_log(job_log, f"[{_now_stamp()}] reports: generating error_report.csv/test_report.csv")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from utils.compile_rust_utils import final_error_report, final_test_report  # type: ignore
    final_error_report(output_dir)
    final_test_report(output_dir)
    _append_log(job_log, f"[{_now_stamp()}] reports: done")

    print("=== Codex CLI verify done ===")
    print(f"benchmark: {args.benchmark}")
    print(f"codex_cli: {codex_bin} ({codex_version})")
    print(f"project_dir: {proj_dir}")
    print(f"run_log: {run_log}")
    print(f"codex_log: {codex_log}")
    print(f"codex_exit: {None if codex_proc is None else codex_proc.returncode}")
    print(f"baseline_exit: {baseline.returncode}")
    print(f"after_exit: {after.returncode}")
    print(f"error_report: {output_dir / 'error_report.csv'}")
    print(f"test_report: {output_dir / 'test_report.csv'}")
    print(f"reward: {(output_dir / 'reward.txt')}")
    print(f"job_log: {job_log}")


if __name__ == "__main__":
    main()


