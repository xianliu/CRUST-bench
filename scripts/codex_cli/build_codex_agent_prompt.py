"""
Build a Codex CLI "agent" prompt that reuses CRUST-bench prompter instructions.

Why:
- `src/run.py` uses prompters that embed a system prompt + rule/format reminders.
- Codex CLI uses a single natural-language instruction prompt.
- This script stitches the same instruction text into the Codex CLI prompt so the
  agent follows the same constraints (no unsafe, no FFI, keep signatures, etc.).

Usage:
  python scripts/codex_cli/build_codex_agent_prompt.py
  python scripts/codex_cli/build_codex_agent_prompt.py --project proj_2DPartInt
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TRANSPILATION_PROMPT = (
    REPO_ROOT
    / "src"
    / "prompts"
    / "transpilation_prompts"
    / "bullet_point"
    / "bullet_point_interface.prompt"
)
TRANSPILATION_RULES = (
    REPO_ROOT
    / "src"
    / "prompts"
    / "transpilation_prompts"
    / "bullet_point"
    / "rule_reminder.prompt"
)
REPAIR_PROMPT = (
    REPO_ROOT
    / "src"
    / "prompts"
    / "repair_prompts"
    / "bullet_point"
    / "bullet_point.prompt"
)
REPAIR_RULES = (
    REPO_ROOT
    / "src"
    / "prompts"
    / "repair_prompts"
    / "bullet_point"
    / "rule_reminder.prompt"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        default=None,
        help="Optional project name for context in the prompt (e.g., proj_2DPartInt).",
    )
    args = ap.parse_args()

    project_hint = f"Target project: {args.project}\n" if args.project else ""

    prompt = f"""You are OpenAI Codex CLI running as an autonomous coding agent in a local Rust repository.
Your job is to iteratively run tests, inspect files, edit code, and re-run tests until they pass.

{project_hint}CRUST-bench instruction set (MUST FOLLOW)
======================================================

These are the *same instruction texts* used by CRUST-bench prompters for transpilation and repair.
Follow them as hard constraints while you work locally.

--- Transpilation system prompt (from `{TRANSPILATION_PROMPT}`) ---
{_read(TRANSPILATION_PROMPT)}

--- Transpilation rule reminder (from `{TRANSPILATION_RULES}`) ---
{_read(TRANSPILATION_RULES)}

--- Repair system prompt (from `{REPAIR_PROMPT}`) ---
{_read(REPAIR_PROMPT)}

--- Repair rule reminder (from `{REPAIR_RULES}`) ---
{_read(REPAIR_RULES)}

Agent workflow
==============
- Run `cargo test` early to reproduce failures.
- Do NOT edit files under `src/bin/` (tests). Fix the library code to satisfy tests.
- The C reference implementation is available under `metadata/c_src/` in this repo. Use it when implementing behavior.
- The Rust interface stubs are under `src/interfaces/*.rs`. In scaffold-only runs, `src/<module>.rs` will `include!()` the corresponding interface file, so editing the interface stub is enough.
- Pay attention to comments in the interface stubs (they often specify subtle behavior like how lengths are counted).
- Make minimal, idiomatic Rust changes. Keep public signatures stable unless tests demand otherwise.
- Re-run `cargo test` after each change until it exits with status 0.
- When done, summarize the changes and the final `cargo test` result.
"""
    print(prompt)


if __name__ == "__main__":
    main()


