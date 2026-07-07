"""CLI entry point for ornith-mlx-eval.

Owns argument parsing, command dispatch, exit codes, and user-facing
stdout/stderr.  Delegates logic to domain modules.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="ornith-mlx-eval",
        description="Ornith MLX Evaluation Harness — evaluate Ornith MLX models locally on Apple Silicon.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
        help="Available commands",
    )

    # ---- profile ----------------------------------------------------------
    profile_p = subparsers.add_parser(
        "profile",
        help="Run preflight checks and report environment readiness",
    )
    profile_p.add_argument(
        "--model",
        help="Model ID for metadata resolution (no weights downloaded)",
    )
    profile_p.add_argument(
        "--output-root",
        default="benchmark_results",
        help="Output root directory for writability check (default: benchmark_results)",
    )

    # ---- list-suites ------------------------------------------------------
    list_suites_p = subparsers.add_parser(
        "list-suites",
        help="List discoverable public evaluation suites",
    )

    # ---- validate-suite ---------------------------------------------------
    validate_p = subparsers.add_parser(
        "validate-suite",
        help="Validate a suite JSON file against the harness schema",
    )
    validate_p.add_argument(
        "suite_path",
        help="Path to a suite JSON file",
    )

    # ---- smoke ------------------------------------------------------------
    smoke_p = subparsers.add_parser(
        "smoke",
        help="Run a gated real MLX smoke test (requires resource gates)",
    )
    smoke_p.add_argument(
        "--model",
        required=True,
        help="Model ID (e.g. mlx-community/Ornith-1.0-9B-4bit)",
    )
    smoke_p.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum generation tokens (default: 32)",
    )
    smoke_p.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature",
    )
    smoke_p.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling top-p",
    )
    smoke_p.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling",
    )
    smoke_p.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    smoke_p.add_argument(
        "--output-root",
        default="benchmark_results",
        help="Output root directory (default: benchmark_results)",
    )

    # ---- run --------------------------------------------------------------
    run_p = subparsers.add_parser(
        "run",
        help="Run an evaluation suite (defaults to mock, no model download)",
    )
    run_p.add_argument(
        "--model",
        help="Model ID for evaluation",
    )
    run_p.add_argument(
        "--suite",
        help="Suite identifier (from list-suites) or 'all'",
    )
    run_p.add_argument(
        "--limit",
        type=int,
        help="Limit scored cases (smoke-only when set)",
    )
    run_p.add_argument(
        "--runtime",
        default="mock",
        choices=["mock", "mlx"],
        help="Evaluation runtime: mock (default, no download) or mlx",
    )
    run_p.add_argument(
        "--output-root",
        default="benchmark_results",
        help="Output root directory (default: benchmark_results)",
    )
    # Decoding options
    run_p.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature",
    )
    run_p.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling top-p",
    )
    run_p.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    # Resource options
    run_p.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum generation tokens",
    )
    run_p.add_argument(
        "--max-prompt-tokens",
        type=int,
        help="Maximum prompt tokens before rejection",
    )
    run_p.add_argument(
        "--max-kv-size",
        type=int,
        help="Maximum KV cache size",
    )

    # ---- report -----------------------------------------------------------
    report_p = subparsers.add_parser(
        "report",
        help="Build a Markdown report from persisted run results",
    )
    report_p.add_argument(
        "run_dir",
        help="Path to the run directory containing manifest.json and results.jsonl",
    )

    # ---- compare ----------------------------------------------------------
    compare_p = subparsers.add_parser(
        "compare",
        help="Compare two completed run directories",
    )
    compare_p.add_argument(
        "run_a",
        help="First run directory",
    )
    compare_p.add_argument(
        "run_b",
        help="Second run directory",
    )
    compare_p.add_argument(
        "--output",
        help="Compare output path (default: benchmark_results/compare_<a>_vs_<b>.md)",
    )
    compare_p.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow qualitative comparison of runs with fixed-invariant mismatches",
    )

    return parser


# ---- Command handlers -------------------------------------------------------


def _cmd_profile(args: argparse.Namespace) -> int:
    """profile – run preflight checks and report environment readiness."""
    from ornith_mlx_eval.profile import format_profile_output, run_profile

    result = run_profile(model_id=args.model, output_root=args.output_root)
    output = format_profile_output(result)
    print(output)

    if result["status"] == "fail":
        return 1
    return 0


def _cmd_list_suites(args: argparse.Namespace) -> int:
    """list-suites – list discoverable public evaluation suites."""
    from ornith_mlx_eval.suites import list_suites_info

    suites = list_suites_info()
    if not suites:
        print("No suites discovered.", file=sys.stderr)
        return 0

    for info in suites:
        status = "valid" if info.get("valid") else "INVALID"
        sid = info.get("suite_id", "?")
        count = info.get("case_count", 0)
        desc = info.get("description", "")
        hash_str = info.get("suite_hash", "")
        print(f"{sid}  [{status}]  cases: {count}  hash: {hash_str}")
        if desc:
            print(f"  {desc}")
        if info.get("errors"):
            for err in info["errors"]:
                print(f"    error: {err}")
    return 0


def _cmd_validate_suite(args: argparse.Namespace) -> int:
    """validate-suite – validate a suite JSON file against the harness schema."""
    from pathlib import Path

    from ornith_mlx_eval.suites import (
        SuiteValidationError,
        compute_prompt_template_hash,
        compute_suite_hash,
        load_suite,
        validate_suite,
    )

    path = Path(args.suite_path)
    try:
        suite = load_suite(path)
    except SuiteValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    errors = validate_suite(suite, suite_path=str(path))
    if errors:
        print(f"Suite '{path}' is INVALID:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    suite_hash = compute_suite_hash(suite)
    prompt_hash = compute_prompt_template_hash(suite)
    case_count = len(suite.get("cases", []))
    suite_id = suite.get("suite_id", "?")

    print(f"Suite: {suite_id}")
    print(f"  Path: {path}")
    print(f"  Status: valid")
    print(f"  Cases: {case_count}")
    print(f"  Suite hash: {suite_hash}")
    print(f"  Prompt-template hash: {prompt_hash}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """smoke – placeholder for milestone 4."""
    print(f"smoke: model={args.model} (not yet implemented)", file=sys.stderr)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """run – placeholder for milestone 3 with suite revalidation."""
    from pathlib import Path

    if args.suite:
        suite_path = Path(args.suite)
        if suite_path.exists():
            from ornith_mlx_eval.suites import (
                SuiteValidationError,
                load_suite,
                validate_suite,
            )
            try:
                suite = load_suite(suite_path)
                errors = validate_suite(suite, suite_path=str(suite_path))
                if errors:
                    print(f"Suite validation failed for '{args.suite}':", file=sys.stderr)
                    for err in errors:
                        print(f"  - {err}", file=sys.stderr)
                    return 1
            except SuiteValidationError as exc:
                print(f"Suite error: {exc}", file=sys.stderr)
                return 1
        else:
            # Named suite (e.g. "smoke", "all") — not yet implemented
            print(f"run: suite '{args.suite}' (suite lookup not yet implemented)", file=sys.stderr)
            return 1

    print(
        f"run: runtime={args.runtime} suite={args.suite} (not yet implemented)",
        file=sys.stderr,
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """report – placeholder for milestone 3."""
    print(f"report: {args.run_dir} (not yet implemented)", file=sys.stderr)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """compare – placeholder for milestone 3."""
    print(
        f"compare: {args.run_a} vs {args.run_b} (not yet implemented)",
        file=sys.stderr,
    )
    return 0


# ---- main entry point -----------------------------------------------------


_DISPATCH: dict[str, callable] = {
    "profile": _cmd_profile,
    "list-suites": _cmd_list_suites,
    "validate-suite": _cmd_validate_suite,
    "smoke": _cmd_smoke,
    "run": _cmd_run,
    "report": _cmd_report,
    "compare": _cmd_compare,
}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, dispatch to the requested command, and exit."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handler = _DISPATCH.get(args.command)
    if handler is None:
        # Should be unreachable due to argparse subparser registration
        print(f"ornith-mlx-eval: unknown command '{args.command}'", file=sys.stderr)
        sys.exit(2)

    try:
        exit_code = handler(args)
    except Exception:
        print("ornith-mlx-eval: internal error", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        exit_code = 3

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
