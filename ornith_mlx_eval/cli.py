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


# ---- Command handlers (stubs for scaffold milestone) -----------------------


def _cmd_profile(args: argparse.Namespace) -> int:
    """profile – placeholder for milestone 2."""
    print("profile: preflight checks (not yet implemented)", file=sys.stderr)
    return 0


def _cmd_list_suites(args: argparse.Namespace) -> int:
    """list-suites – placeholder for milestone 2."""
    print("list-suites: suite discovery (not yet implemented)", file=sys.stderr)
    return 0


def _cmd_validate_suite(args: argparse.Namespace) -> int:
    """validate-suite – placeholder for milestone 2."""
    print(f"validate-suite: {args.suite_path} (not yet implemented)", file=sys.stderr)
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """smoke – placeholder for milestone 4."""
    print(f"smoke: model={args.model} (not yet implemented)", file=sys.stderr)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """run – placeholder for milestone 3."""
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
