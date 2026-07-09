# Ornith MLX Eval

Local, MLX-only evaluation harness for Ornith MLX models on Apple Silicon.

The default workflow is safe and no-download: it validates suites, runs the
deterministic mock runtime, writes reproducible artifacts, regenerates reports,
and compares persisted runs without loading model weights.

Inspired by the practical local-eval presentation in
[jikkuatwork/ornith-eval](https://github.com/jikkuatwork/ornith-eval), adapted
for direct MLX usage instead of Ollama.

The goal is not to publish a leaderboard score. It is to answer: can this local
MLX harness safely run Ornith evals on this machine, produce reproducible
artifacts, and make real model smoke tests explicit and resource-gated?

## Executive Verdict

This repository is ready as a public, local-first MLX evaluation harness.

Highlights:

- Default workflow is no-download and deterministic through the mock runtime.
- Public smoke suite has `5` authored cases and validates with stable hashes.
- Completed runs produce `manifest.json`, `results.jsonl`, `summary.json`, and
  `report.md` in isolated run directories.
- Reports and comparisons are regenerated from persisted artifacts only.
- Real MLX smoke is available but explicitly gated to avoid accidental GiB-scale
  downloads or unsafe memory pressure.
- 8bit and 35B variants are rejected for normal local work on the target 16 GB
  Apple Silicon machine.
- Public clean-clone verification passed before this presentation refresh.

## Target Model And Environment

| Dimension | Value |
|---|---|
| Runtime | Direct MLX / MLX-LM |
| Default run mode | `mock`, no model download |
| First real smoke target | `mlx-community/Ornith-1.0-9B-4bit` |
| Promotion target | `mlx-community/Ornith-1.0-9B-6bit` |
| 4bit pinned revision | `1e980b9742a9e554a4d57e90b4c597811fb2fc4e` |
| 6bit pinned revision | `a2800933352a607ffbb1f814295fc3ff8e10ad69` |
| Rejected normal-local variants | 9B 8bit and 35B variants |
| Target hardware | Apple Silicon Mac with 16 GB unified memory |
| Python | 3.10+, verified with Python 3.12 |
| Package status | Public clean-clone install and test gate passed |

## Current Public Results

These are harness-readiness results, not model-quality benchmark scores. Real
model results are intentionally generated locally and kept out of git unless
the user opts into a smoke run.

| Check | Result |
|---|---:|
| Full local test gate | `519 passed` |
| Clean-clone test gate | `519 passed` |
| CLI help | passed |
| `pip check` | clean |
| Public smoke suite | valid, `5` cases |
| Mock run/report/compare workflow | passed |
| Private/generated path audit | no tracked private paths |
| README presentation refresh | local full gate passed |

## Reports And Artifacts

Generated reports are local outputs, not committed benchmark data.

| Artifact | Description |
|---|---|
| `manifest.json` | Run identity, suite/model/runtime metadata, settings, and environment. |
| `results.jsonl` | Case-level parsed responses, grades, hashes, and resource fields. |
| `summary.json` | Aggregate totals, pass rate, classification, and smoke-only flag. |
| `report.md` | Human-readable Markdown report regenerated from persisted files. |
| `compare_*.md` | Optional persisted-run comparison with invariant checks. |

Raw generated outputs belong under `benchmark_results/`, which is intentionally
ignored.

## Requirements

- Apple Silicon Mac for real MLX runtime work.
- Python 3.10 or newer; Python 3.12 is preferred.
- Project virtual environment. The `profile` command intentionally fails
  outside `.venv`.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ornith-mlx-eval --help
.venv/bin/python -m pip check
```

The package pins the verified local stack:

- `mlx==0.31.2`
- `mlx-lm==0.31.3`
- `transformers==5.0.0`
- `huggingface_hub==1.22.0`
- `numpy==2.5.1`
- `pytest==9.1.1`
- `jsonschema==4.26.0`

## CLI Workflow

List and validate authored public suites:

```bash
.venv/bin/ornith-mlx-eval list-suites
.venv/bin/ornith-mlx-eval validate-suite suites/smoke.json
```

Run the no-download mock runtime and inspect artifacts:

```bash
.venv/bin/ornith-mlx-eval run --runtime mock --suite smoke --output-root benchmark_results
.venv/bin/ornith-mlx-eval report benchmark_results/<run_id>
.venv/bin/ornith-mlx-eval compare benchmark_results/<run_a> benchmark_results/<run_b>
```

Successful runs create one isolated run directory with:

- `manifest.json`
- `results.jsonl`
- `summary.json`
- `report.md`

`report` reads only persisted files and rewrites only `<run_dir>/report.md`.
`compare` reads only persisted run directories and writes either the explicit
`--output` path or the documented default compare Markdown path.

## Real MLX Smoke

Real model smoke is opt-in because it can download several GiB of model
weights and use substantial unified memory.

4bit smoke:

```bash
ORNITH_MLX_ALLOW_MODEL_DOWNLOAD=1 \
.venv/bin/ornith-mlx-eval smoke \
  --model mlx-community/Ornith-1.0-9B-4bit \
  --allow-download \
  --max-tokens 32
```

The pinned 4bit revision is:

```text
1e980b9742a9e554a4d57e90b4c597811fb2fc4e
```

6bit promotion requires a fresh 4bit smoke artifact and explicit opt-in:

```bash
ORNITH_MLX_ALLOW_MODEL_DOWNLOAD=1 \
.venv/bin/ornith-mlx-eval smoke \
  --model mlx-community/Ornith-1.0-9B-6bit \
  --allow-download \
  --promotion-source benchmark_results/<4bit_run>/manifest.json
```

The pinned 6bit revision is:

```text
a2800933352a607ffbb1f814295fc3ff8e10ad69
```

8bit and 35B variants are intentionally rejected for normal local work on the
target 16 GB machine.

## Result Safety

- Case-level failures are recorded in artifacts and reports; completed valid
  runs still exit `0`.
- Systemic failures, invalid suites, unsafe output paths, schema failures,
  missing persisted files, and comparison mismatches exit nonzero.
- `--limit` marks a run smoke-only and disables benchmark-quality claims.
- Hidden expected-answer metadata is used only internally for grading and is
  not serialized as expected-answer fields in public artifacts.

## Local-Only Files

These files are planning/session artifacts and must stay unpublished:

- `plan.md`
- `status.md`
- `whatisdone.md`
- `whatisleft.md`
- `koder/`

Generated outputs and caches are also excluded from git:

- `benchmark_results/`
- `.factory/`
- `.venv/`
- `.pytest_cache/`
- `*.egg-info/`
- Hugging Face model caches and snapshots

The suite fixture in this repository is authored for this project. Do not copy
unlicensed upstream code, prompts, suite JSON, reports, answer keys, or
generated artifacts into this repository.
