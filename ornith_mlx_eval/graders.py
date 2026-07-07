"""Deterministic grading interfaces and implementations.

Owns:
  * grade() dispatch to per-type graders
  * exact_match, contains, numeric graders
  * parse-status guard: unclosed_think / ambiguous_think / empty_final fail
  * truncation guard: is_truncated responses fail by default
  * contradiction detection in final text
  * wrong-context and candidate-instruction resistance

Graders are deterministic and must avoid substring false positives.
Hidden expected-answer metadata must never be exposed in grader evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ornith_mlx_eval.parsing import ParseResult, CodeExtractionResult, extract_fenced_code
from ornith_mlx_eval.sandbox import run_code as run_in_sandbox


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    """Result of grading a parsed response against an expected answer.

    Attributes:
        passed: *True* if the candidate answer met the grading criteria.
        score: Numeric score, typically 0.0 or 1.0.
        reason: Human-readable explanation of the grading decision.
        evidence: Candidate text used for grading (the final text or
            extracted code).
        grader_type: The grader type that was used.
    """

    passed: bool = False
    score: float = 0.0
    reason: str = ""
    evidence: str = ""
    grader_type: str = ""


# ---------------------------------------------------------------------------
# Supported grader types
# ---------------------------------------------------------------------------

SUPPORTED_GRADERS: frozenset[str] = frozenset({
    "exact_match",
    "contains",
    "numeric",
    "json_match",
    "code",
})

# Negation patterns used to detect contradictory context
_NEGATION_WORDS: tuple[str, ...] = (
    "not", "never", "no", "neither", "nor",
)
_CONTRADICTION_WORDS: tuple[str, ...] = (
    "but actually", "however", "instead", "on the other hand",
    "alternatively", "rather", "though",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade(
    parse_result: ParseResult,
    expected: Any,
    grader_type: Optional[str],
    options: Optional[dict[str, Any]] = None,
) -> GradeResult:
    """Grade a parsed response against an expected answer.

    Pre-guards (applied before any grader logic):
    1. Parse failures (unclosed_think, ambiguous_think) fail.
    2. Reasoning-only responses (empty_final) fail.
    3. Truncated responses fail by default.

    Args:
        parse_result: The parsed response from :func:`parsing.parse_response`.
        expected: The expected answer (string for text graders, number for
            numeric, dict/list for JSON).
        grader_type: One of "exact_match", "contains", "numeric",
            "json_match", or "code".
        options: Grader-specific options dict (e.g. ignore_case, tolerance).

    Returns:
        A :class:`GradeResult` with pass/fail, score, reason, and evidence.
    """
    options = options or {}

    # -- Parse-status guard ---------------------------------------------------
    if parse_result.parse_status == "unclosed_think":
        return GradeResult(
            passed=False,
            score=0.0,
            reason="Parse failure: unclosed think block",
            evidence=parse_result.final_text or "",
            grader_type=grader_type or "unknown",
        )
    if parse_result.parse_status == "ambiguous_think":
        return GradeResult(
            passed=False,
            score=0.0,
            reason="Parse failure: ambiguous or malformed think tags",
            evidence=parse_result.final_text or "",
            grader_type=grader_type or "unknown",
        )
    if parse_result.parse_status == "empty_final":
        return GradeResult(
            passed=False,
            score=0.0,
            reason="Parse failure: reasoning-only response with empty final answer",
            evidence="",
            grader_type=grader_type or "unknown",
        )

    # -- Truncation guard -----------------------------------------------------
    if parse_result.is_truncated:
        return GradeResult(
            passed=False,
            score=0.0,
            reason="Truncated response: incomplete final answer",
            evidence=parse_result.final_text,
            grader_type=grader_type or "unknown",
        )

    final_text = parse_result.final_text

    # -- Dispatch to grader type ---------------------------------------------
    if grader_type == "exact_match":
        return grade_exact_match(final_text, expected, options)
    elif grader_type == "contains":
        return grade_contains(final_text, expected, options)
    elif grader_type == "numeric":
        return grade_numeric(final_text, expected, options)
    elif grader_type == "json_match":
        return grade_json_match(final_text, expected, options)
    elif grader_type == "code":
        return grade_code(final_text, expected, options)
    else:
        return GradeResult(
            passed=False,
            score=0.0,
            reason=f'Unsupported grader type: "{grader_type}"',
            evidence=final_text,
            grader_type=grader_type or "unknown",
        )


# ---------------------------------------------------------------------------
# Individual graders
# ---------------------------------------------------------------------------


def grade_exact_match(
    final_text: str,
    expected: Any,
    options: dict[str, Any],
) -> GradeResult:
    """Strict exact match grader.

    The candidate final text must match the expected string exactly
    (after optional case normalisation and whitespace stripping).

    Empty final text always fails (fail-closed); a model that produces
    no output should not be given credit, even if the expected answer
    happens to be empty.
    """
    if not final_text:
        return _fail("exact_match", "Empty final text", final_text)

    if not isinstance(expected, str):
        return _fail("exact_match",
                     f"Expected answer must be a string, got {type(expected).__name__}",
                     final_text)

    ignore_case = bool(options.get("ignore_case", False))
    strip_ws = bool(options.get("strip_whitespace", False))

    candidate = final_text
    answer = expected

    if strip_ws:
        candidate = candidate.strip()
        answer = answer.strip()

    if ignore_case:
        candidate = candidate.lower()
        answer = answer.lower()

    if candidate == answer:
        return GradeResult(
            passed=True,
            score=1.0,
            reason="Exact match",
            evidence=final_text,
            grader_type="exact_match",
        )

    return _fail("exact_match",
                 f'Expected "{expected}" but got "{final_text}"',
                 final_text)


def grade_contains(
    final_text: str,
    expected: Any,
    options: dict[str, Any],
) -> GradeResult:
    """Contains grader with contradiction detection.

    The expected answer must appear in the final text as a standalone token
    or phrase, not inside a negated or contradictory context.
    """
    if not final_text:
        return _fail("contains", "Empty final text", final_text)

    if not isinstance(expected, str):
        return _fail("contains",
                     f"Expected answer must be a string, got {type(expected).__name__}",
                     final_text)

    ignore_case = bool(options.get("ignore_case", False))
    text = final_text.lower() if ignore_case else final_text
    answer = expected.lower() if ignore_case else expected

    # Check if answer appears as a word/token in the text
    pattern = re.compile(r'(?<!\w)' + re.escape(answer) + r'(?!\w)')
    if not pattern.search(text):
        return _fail("contains",
                     f'Answer "{expected}" not found in final text',
                     final_text)

    # -- Contradiction / negation detection -----------------------------------
    lower_text = text.lower()
    answer_lower = answer.lower()

    # Check for negation immediately before the answer (e.g., "not Paris")
    neg_words = '|'.join(re.escape(w) for w in _NEGATION_WORDS)
    neg_pattern = re.compile(
        r'(?:\b(?:' + neg_words + r')\b\s+' + re.escape(answer_lower) + r')'
    )
    if neg_pattern.search(lower_text):
        return _fail("contains",
                     f'Answer "{expected}" found in negated context',
                     final_text)

    # Check for contradictory alternatives (e.g., "Paris or London")
    alt_pattern = re.compile(
        re.escape(answer_lower) + r'\s+(?:or|but|however|alternatively)\s+',
    )
    if alt_pattern.search(lower_text):
        return _fail("contains",
                     f'Answer "{expected}" found in ambiguous/contradictory context',
                     final_text)

    alt_pattern2 = re.compile(
        r'\b(?:or|but)\s+' + re.escape(answer_lower)
    )
    if alt_pattern2.search(lower_text):
        return _fail("contains",
                     f'Answer "{expected}" found in alternative/contradictory context',
                     final_text)

    return GradeResult(
        passed=True,
        score=1.0,
        reason="Answer found in final text",
        evidence=final_text,
        grader_type="contains",
    )


def grade_numeric(
    final_text: str,
    expected: Any,
    options: dict[str, Any],
) -> GradeResult:
    """Numeric tolerance grader.

    Extracts the first number from the final text and compares it against
    the expected value within an optional tolerance.
    """
    if not final_text:
        return _fail("numeric", "Empty final text", final_text)

    # Parse expected as float
    try:
        expected_val = float(expected)
    except (ValueError, TypeError):
        return _fail("numeric",
                     f'Expected answer must be numeric, got "{expected}"',
                     final_text)

    # Extract first number from final text
    candidate_val = _extract_first_number(final_text)
    if candidate_val is None:
        return _fail("numeric",
                     f'No numeric value found in "{final_text}"',
                     final_text)

    tolerance_raw = options.get("tolerance", 0.0)
    if isinstance(tolerance_raw, bool):
        return _fail("numeric",
                     f"Invalid tolerance: boolean values are not allowed",
                     final_text)
    try:
        tolerance = float(tolerance_raw)
    except (ValueError, TypeError):
        return _fail("numeric",
                     f"Invalid tolerance value: {tolerance_raw!r}",
                     final_text)

    if abs(candidate_val - expected_val) <= tolerance:
        return GradeResult(
            passed=True,
            score=1.0,
            reason=f"Numeric match: {candidate_val} within {tolerance} of {expected_val}",
            evidence=final_text,
            grader_type="numeric",
        )

    return _fail("numeric",
                 f"Value {candidate_val} not within {tolerance} of {expected_val}",
                 final_text)


def grade_json_match(
    final_text: str,
    expected: Any,
    options: dict[str, Any],
) -> GradeResult:
    """JSON match grader.

    Parses the final text as JSON and compares structurally against
    the expected value.
    """
    if not final_text:
        return _fail("json_match", "Empty final text", final_text)

    try:
        candidate = json.loads(final_text)
    except json.JSONDecodeError as exc:
        return _fail("json_match",
                     f"Failed to parse final text as JSON: {exc}",
                     final_text)

    if candidate == expected:
        return GradeResult(
            passed=True,
            score=1.0,
            reason="JSON structural match",
            evidence=final_text,
            grader_type="json_match",
        )

    return _fail("json_match",
                 f"JSON mismatch: expected {expected}, got {candidate}",
                 final_text)


def grade_code(
    final_text: str,
    expected: Any,
    options: dict[str, Any],
) -> GradeResult:
    """Code grading via fenced Python code extraction and sandbox execution.

    Extracts a single fenced Python code block from the final text and
    executes it inside the bounded subprocess sandbox.

    Options:
        *test_input* (str): Optional text piped to the candidate process
            stdin.
        *expected_output* (str): Optional expected stdout for comparison.
        *timeout* (float): Per-run timeout in seconds (default 10.0).
        *max_output_bytes* (int): Max captured stdout bytes.

    If neither *test_input* nor *expected_output* is provided, the code
    passes when it exits with code 0.
    """
    if not final_text:
        return _fail("code", "Empty final text", final_text)

    extraction = extract_fenced_code(final_text)

    if extraction.status != "success":
        # Map extraction failure statuses to grader reasons
        reason_map: dict[str, str] = {
            "no_fence": "No fenced Python code block found",
            "multiple_fences": "Multiple conflicting Python code blocks found",
            "unterminated_fence": "Unterminated code fence found",
            "unsupported_language": "Only Python code blocks are supported",
            "no_language": "Code fence missing language specifier; use ```python",
            "empty_code": "Extracted code block is empty",
        }
        return _fail("code",
                     reason_map.get(extraction.status, f"Code extraction failed: {extraction.status}"),
                     final_text)

    # -- Sandbox execution ----------------------------------------------------
    code_text = extraction.code
    test_input = options.get("test_input", None)
    expected_output = options.get("expected_output", None)
    sandbox_timeout = float(options.get("timeout", 10.0))
    max_output = int(options.get("max_output_bytes", 1024 * 1024))

    sandbox_result = run_in_sandbox(
        code=code_text,
        test_input=test_input,
        expected_output=expected_output,
        timeout=sandbox_timeout,
        max_output_bytes=max_output,
    )

    if sandbox_result.passed:
        return GradeResult(
            passed=True,
            score=1.0,
            reason=f"Code executed successfully in sandbox: {sandbox_result.reason}",
            evidence=code_text,
            grader_type="code",
        )
    else:
        return _fail(
            "code",
            f"Sandbox execution failed: {sandbox_result.reason}",
            code_text,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(grader_type: str, reason: str, evidence: str) -> GradeResult:
    """Shorthand for a failed grade result."""
    return GradeResult(
        passed=False,
        score=0.0,
        reason=reason,
        evidence=evidence,
        grader_type=grader_type,
    )


def _extract_first_number(text: str) -> Optional[float]:
    """Extract the first numeric value (integer or float) from *text*.

    Supports optional leading minus sign, decimal points, and scientific
    notation.  Returns *None* if no number is found.
    """
    match = re.search(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None
