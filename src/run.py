import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

from benchmark import load_benchmarks_filtered
from utils.rust_project_builder import create_top_level_cargo_toml


def _scaffold_mode(args) -> None:
    """
    Scaffold-only: prepare output Rust crate(s) for agent-based workflows (Codex CLI).

    IMPORTANT: Keep imports here minimal and avoid importing any endpoint clients
    (Anthropic/OpenAI/etc). This mode should work in restricted environments and
    doesn't call any model.
    """
    # Create workspace top-level Cargo.toml
    create_top_level_cargo_toml(Path(args.output_dir))

    benchmarks = load_benchmarks_filtered(
        Path(args.benchmark_dir), Path(args.output_dir), single_benchmark=args.single_benchmark
    )

    # If user provided an RBench directory, copy it into each scaffolded output crate.
    if args.rust_dir:
        rust_dir = Path(args.rust_dir)
        ignore_vcs = shutil.ignore_patterns(".git", ".vscode", ".DS_Store")
        for b in benchmarks:
            shutil.rmtree(b.rust_path, ignore_errors=True)
            src1 = rust_dir / b.project_name
            src2 = rust_dir / (b.project_name[0].upper() + b.project_name[1:]) if b.project_name else None
            if src1.exists():
                shutil.copytree(src1, b.rust_path, ignore=ignore_vcs)
            elif src2 and src2.exists():
                shutil.copytree(src2, b.rust_path, ignore=ignore_vcs)
            else:
                raise FileNotFoundError(f"Rust project not found for {b.project_name} under {rust_dir}")
        # Reload benchmarks so paths + rust_headers/tests are refreshed.
        benchmarks = load_benchmarks_filtered(
            Path(args.benchmark_dir), Path(args.output_dir), single_benchmark=args.single_benchmark
        )
        assert all(len(b.rust_headers) > 0 for b in benchmarks), "Rust path not copied correctly"

    for b in benchmarks:
        # Materialize interface stubs as module files under src/ so the crate builds.
        interfaces_dir = b.rust_path / "src" / "interfaces"
        if interfaces_dir.exists():
            lib_rs = b.rust_path / "src" / "lib.rs"
            lib_content = lib_rs.read_text(encoding="utf-8") if lib_rs.exists() else ""
            for iface in interfaces_dir.glob("*.rs"):
                module_dst = b.rust_path / "src" / iface.name
                if not module_dst.exists():
                    # Keep a single source of truth in src/interfaces/*.rs so agents
                    # can implement the stubs there without worrying about duplicates.
                    module_dst.write_text(f'include!("interfaces/{iface.name}");\n', encoding="utf-8")
                mod_line = f"pub mod {iface.stem};"
                if mod_line not in lib_content:
                    lib_content = (lib_content.rstrip() + "\n" + mod_line + "\n").lstrip("\n")
            if lib_rs.exists():
                lib_rs.write_text(lib_content, encoding="utf-8")
            else:
                lib_rs.write_text(lib_content + "\n", encoding="utf-8")

        # Copy C sources into metadata for agent reference.
        c_src_dir = b.rust_path / "metadata" / "c_src"
        c_src_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for f in (b.c_files + b.c_headers):
            name = f["file_name"]
            if name in seen:
                continue
            seen.add(name)
            (c_src_dir / name).write_text(f["content"], encoding="utf-8")

    print(f"Scaffolded {len(benchmarks)} benchmark(s) under: {args.output_dir}")


class Runner:
    def __init__(
        self,
        benchmark_dir,
        output_dir,
        prompt,
        prompt_format,
        prompt_strategy,
        repairer_prompt,
        repairer_format,
        repairer_strategy,
        iterations,
        endpoint,
        include_headers,
        config,
        n,
        single_benchmark,
        rust_dir,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)

        # Filter early so we don't instantiate / scaffold Rust projects for all benchmarks
        # when the user only wants a single one (Benchmark() can run `cargo new`).
        self.benchmarks = load_benchmarks_filtered(
            self.benchmark_dir, self.output_dir, single_benchmark=single_benchmark
        )
        self.prompt = prompt
        self.prompt_format = prompt_format
        self.prompt_strategy = prompt_strategy
        self.repairer_prompt = repairer_prompt
        self.repairer_format = repairer_format
        self.repairer_strategy = repairer_strategy
        self.iterations = int(iterations)
        self.endpoint = endpoint
        self.include_headers = include_headers
        self.config = config
        self.n = n
        if rust_dir:
            for benchmark in self.benchmarks:
                shutil.rmtree(benchmark.rust_path)
                if not Path(rust_dir + "/" + benchmark.project_name).exists():
                    # try capitalizing the first letter of the project name
                    shutil.copytree(
                        rust_dir + "/" + benchmark.project_name[0].upper() + benchmark.project_name[1:],
                        benchmark.rust_path,
                    )
                else:
                    shutil.copytree(
                        rust_dir + "/" + benchmark.project_name, benchmark.rust_path
                    )
            # we need to reload the benchmarks given that we have changed the rust path
            self.benchmarks = load_benchmarks_filtered(
                self.benchmark_dir, self.output_dir, single_benchmark=single_benchmark
            )
            assert all(
                len(benchmark.rust_headers) > 0 for benchmark in self.benchmarks
            ), "Rust path not copied correctly"
        # dump a config log
        print(f'''
        Number of benchmarks: {len(self.benchmarks)}
        Prompt: {self.prompt}
        Prompt format: {self.prompt_format}
        Prompt strategy: {self.prompt_strategy}
        Repairer prompt: {self.repairer_prompt}
        Repairer format: {self.repairer_format}
        Repairer strategy: {self.repairer_strategy}
        Iterations: {self.iterations}
        Endpoint: {self.endpoint}
        Include interfaces: {self.include_headers}
        Config: {self.config}
        ''')
 

    def transpile(self):
        # Heavy imports (endpoint clients, LLM plumbing) are deferred so scaffold mode
        # can run without extra deps / SSL requirements.
        from prompter import Prompter
        from transpiler import Transpiler
        from compile_projects import compile
        from utils.rust_project_builder import write_rust_files

        prompter = Prompter(
            self.prompt, self.prompt_format, self.prompt_strategy, self.include_headers
        )

        transpiler = Transpiler(
            self.benchmarks, prompter, endpoint=self.endpoint, config=self.config
        )
        results, benchmarks = transpiler.run()
        self.benchmarks = benchmarks
        transpiler.log_benchmarks(results)
        for benchmark in self.benchmarks:
            write_rust_files(benchmark)
        compile_results = compile(self.output_dir, 0)

    def repair(self):
        from prompter import RepairPrompter
        from repairer import Repairer
        from compile_projects import compile
        from utils.rust_project_builder import write_rust_files

        compile_results = []
        for i in range(self.iterations):
            repair_prompter = RepairPrompter(
                self.repairer_prompt, self.repairer_format, self.repairer_strategy
            )
            repairer = Repairer(
                self.benchmarks, repair_prompter, i, self.endpoint, config=self.config
            )
            results, benchmarks = repairer.run()
            self.benchmarks = benchmarks
            repairer.log_results(results)
            for benchmark in self.benchmarks:
                write_rust_files(benchmark)
            compile_results.append(compile(self.output_dir, i + 1))

    
        

    def multi_gen(self):
        from prompter import Prompter
        from transpiler import TranspilerN
        from compile_projects import compile
        from utils.rust_project_builder import write_rust_multi_files, create_top_level_cargo_toml
        from utils.compile_rust_utils import (
            aggregate_results,
            final_error_report,
            final_test_report,
            get_errors_for_iteration,
            performance_stats,
        )

        prompter = Prompter(
            self.prompt, self.prompt_format, self.prompt_strategy, self.include_headers
        )

        transpiler = TranspilerN(
            self.benchmarks, prompter, endpoint=self.endpoint, config=self.config, n=self.n
        )
        results, benchmarks_list = transpiler.run()
        for idx, benchmarks in enumerate(benchmarks_list):
            for benchmark in benchmarks:
                write_rust_multi_files(benchmark, idx)
            # create top level directory for each iteration
            create_top_level_cargo_toml(benchmarks[0][0].rust_path.parent / f"{idx}")
        # clean up directories that are not used
        for dir in self.output_dir.iterdir():
            # check if dir name is a number
            if dir.is_dir() and not dir.name.isdigit():
                shutil.rmtree(dir)
        
        # get ready for repair
        self.benchmarks = []
        for i in range(self.n):
            # multi_gen does not support per-run single_benchmark filtering here; each output
            # directory is already scoped to this run.
            self.benchmarks = load_benchmarks_filtered(
                self.benchmark_dir, self.output_dir / str(i)
            )
            compile_results = compile(self.output_dir / str(i), 0)
            self.repair()
            self.get_loc()
            final_error_report(self.output_dir / str(i))
            final_test_report(self.output_dir / str(i))
            get_errors_for_iteration(self.output_dir / str(i))
            performance_stats(self.output_dir / str(i))
            self.benchmarks = []
        # aggregate the results based on the benchmark name
        aggregate_results([self.output_dir / str(i) for i in range(self.n)], self.output_dir)

    def get_loc(self):
        with open(self.output_dir / "loc.csv", "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["project", "c_loc", "rust_loc", "builds"])
            for benchmark in self.benchmarks:
                res = benchmark.compile_rust()
                c_loc, rust_loc = benchmark.count_loc()
                if res == []:
                    if rust_loc == 3:
                        builds = "Empty Project"
                    else:
                        builds = "True"
                else:
                    builds = "False"

                csv_writer.writerow([benchmark.project_name, c_loc, rust_loc, builds])

    def test_perf(self):
        from utils.compile_rust_utils import (
            final_error_report,
            final_test_report,
            get_errors_for_iteration,
            performance_stats,
        )
        from understand_errors import get_numbers, process_proj

        self.transpile()
        self.repair()
        self.get_loc()
        # errors per iteration of repair
        get_errors_for_iteration(self.output_dir)
        # errors after all iterations have been run
        final_error_report(self.output_dir)
        # test based errors after all iterations have been run
        final_test_report(self.output_dir)
        # performance stats per project
        performance_stats(self.output_dir)
        # categorize errors based on error type
        process_proj(self.output_dir)
        # get the final numbers
        get_numbers(self.output_dir)


def main(args):
    print("mode:", args.mode)
    if args.mode == "scaffold":
        _scaffold_mode(args)
        return

    # Heavy-mode imports below (avoid endpoint client init in scaffold mode)
    from endpoint_config import endpoint_resolver

    config = endpoint_resolver(args.config, args.endpoint)
    runner = Runner(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        prompt_format=args.prompt_format,
        prompt_strategy=args.prompt_strategy,
        repairer_prompt=args.repairer_prompt,
        repairer_format=args.repairer_format,
        repairer_strategy=args.repairer_strategy,
        iterations=args.iterations,
        endpoint=args.endpoint,
        include_headers=args.include_headers,
        config=config,
        single_benchmark=args.single_benchmark,
        rust_dir=args.rust_dir,
        n=args.n,
    )
    if args.mode == "normal":
        runner.test_perf()
    elif args.mode == "multi_gen":
        print(f"Top-{args.n} generation, with temperature {config['temperature']}")
        runner.multi_gen()
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate your model with CRUST-bench")
    argparser.add_argument("--benchmark_dir", type=str, required=True, help="Path to the C project (CBench) directory")
    argparser.add_argument("--rust_dir", type=str, required=False, help="Path to the Rust project (RBench) directory")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    argparser.add_argument("--prompt", type=str, required=True, help="Prompt to use for the model")
    argparser.add_argument(
        "--mode",
        type=str,
        default="normal",
        help="The mode(normal, multi_gen, scaffold). scaffold only creates output crate(s) for agent-based repair.",
    )
    argparser.add_argument("--endpoint", type=str, required=True, help="Endpoint to use for the model. Look at the `endpoints/call_endpoint.py` for more information.")
    argparser.add_argument("--prompt_format", type=str, required=True, help="Format of the prompt (markdown, bullet_point)")
    argparser.add_argument("--prompt_strategy", type=str, required=False, default="all", help="Strategy to use for the prompt (all- all files are appended to the prompt)")
    argparser.add_argument("--repairer_prompt", type=str, required=True, help="Prompt to use for the repairer")
    argparser.add_argument("--repairer_format", type=str, required=True, help="Format of the repairer prompt(markdown, bullet_point)")
    argparser.add_argument("--repairer_strategy", type=str, required=True, help="Strategy to use for the repairer prompt(all- all files are appended to the prompt)")
    argparser.add_argument("--iterations", type=str, required=True, help="Number of iterations to run the repairer")
    argparser.add_argument("--include_headers", type=bool, default=True, help="Whether to include headers in the prompt")
    argparser.add_argument("--single_benchmark", type=str, default=None, help="Set this flag when you only want to run a single benchmark to run")
    argparser.add_argument("--config", type=str, required=False, default=None, help="Path to the endpoint config file")
    argparser.add_argument("--n", type=int, default=1, help="Number of generations to receive from the model during transpilation")
    args = argparser.parse_args()
    main(args)
