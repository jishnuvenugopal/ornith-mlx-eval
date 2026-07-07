"""Tests for deterministic graders: strict text, numeric, and parse-status
integration.  Covers VAL-EVAL-013, VAL-EVAL-014, VAL-EVAL-021.

Coverage:
  VAL-EVAL-013 - Final-text rule ignores reasoning-only answers
  VAL-EVAL-014 - Contradictory final answers fail strict grading
  VAL-EVAL-021 - Truncated responses cannot pass by accident
  Integration with VAL-EVAL-035 (reasoning-only -> empty_final -> fail)
"""

from __future__ import annotations

import pytest

from ornith_mlx_eval.parsing import ParseResult, parse_response
from ornith_mlx_eval.graders import (
    GradeResult,
    grade,
    grade_exact_match,
    grade_contains,
    grade_numeric,
)

# Tag helpers for test construction
_O = "<think>"
_C = "</think>"


# ======================================================================
# VAL-EVAL-013 - Final-text rule ignores reasoning-only answers
# ======================================================================

class TestFinalTextOnlyGrading:
    """Default grading scores only final text, not reasoning."""

    def test_correct_in_reasoning_wrong_in_final_fails(self):
        """Correct answer only in reasoning, wrong final -> FAIL."""
        raw = _O + "The answer is Paris" + _C + "The answer is London."
        pr = parse_response(raw)
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False
        assert result.score == 0.0

    def test_correct_in_final_wrong_in_reasoning_passes(self):
        """Correct final text passes even if reasoning is wrong."""
        raw = _O + "The answer is London" + _C + "Paris"
        pr = parse_response(raw)
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is True
        assert result.score == 1.0

    def test_correct_in_both_passes(self):
        """Correct in both reasoning and final -> PASS."""
        raw = _O + "The answer is Paris" + _C + "Paris"
        pr = parse_response(raw)
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is True

    def test_reasoning_only_response_fails(self):
        """Parse status empty_final forces a fail."""
        raw = _O + "Paris" + _C
        pr = parse_response(raw)
        assert pr.parse_status == "empty_final"
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False
        assert result.score == 0.0


# ======================================================================
# VAL-EVAL-014 - Contradictory final answers fail strict grading
# ======================================================================

class TestContradictoryAnswers:
    """Final text containing both expected and contradictory answers fails."""

    def test_both_paris_and_london_in_final_fails_exact(self):
        """Final text: 'Paris or London' -> fails exact_match."""
        pr = parse_response("The answer is Paris or maybe London.")
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False

    def test_multiple_candidates_in_final(self):
        """Final text lists multiple options -> fails strict grading."""
        pr = parse_response("Possible answers: Paris, London, Berlin")
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False

    def test_negated_correct_answer_fails(self):
        """'Not Paris' should not pass as 'Paris'."""
        pr = parse_response("The answer is not Paris.")
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False

    def test_contradiction_detected_in_contains_grader(self):
        """Contains grader with contradictory answers in final text."""
        pr = parse_response("It could be Paris but actually it's London.")
        result = grade(pr, "Paris", "contains")
        # "Paris" appears in the text but in contradictory context
        # Strict contains should flag this
        assert result.passed is False or "contradict" in result.reason.lower()


# ======================================================================
# VAL-EVAL-021 - Truncated responses cannot pass by accident
# ======================================================================

class TestTruncatedResponseGrading:
    """Truncated responses fail if they lack a complete valid answer."""

    def test_truncated_response_fails(self):
        """Truncated final text without complete answer -> FAIL."""
        pr = parse_response("The capital of France is")
        assert pr.is_truncated is True
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False

    def test_truncated_with_close_but_incomplete(self):
        """Correct truncated answer still fails because incomplete."""
        raw = _O + "thinking" + _C + "The capital of Fr"
        pr = parse_response(raw)
        assert pr.is_truncated is True
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False

    def test_truncated_parse_failure_prevents_pass(self):
        """Unclosed think is a parse failure -> cannot pass."""
        raw = _O + "reasoning about Paris"
        pr = parse_response(raw)
        assert pr.parse_status == "unclosed_think"
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False
        assert result.score == 0.0

    def test_truncated_empty_after_think(self):
        """Empty final after think -> cannot pass."""
        raw = _O + "Paris is the answer" + _C
        pr = parse_response(raw)
        assert pr.parse_status == "empty_final"
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False


# ======================================================================
# Exact match grader
# ======================================================================

class TestExactMatchGrader:
    """Strict exact_match grader."""

    def test_exact_match_passes(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is True
        assert result.score == 1.0

    def test_exact_match_case_sensitive_by_default(self):
        pr = parse_response("paris")
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False

    def test_exact_match_case_insensitive_option(self):
        pr = parse_response("paris")
        result = grade(pr, "Paris", "exact_match", {"ignore_case": True})
        assert result.passed is True

    def test_exact_match_whitespace_diff(self):
        pr = parse_response("  Paris  ")
        # ParseResponse normalizes whitespace, so final_text is "Paris"
        assert pr.final_text == "Paris"
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is True

    def test_exact_match_wrong_answer(self):
        pr = parse_response("London")
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False
        assert result.score == 0.0

    def test_exact_match_substring_no_pass(self):
        """'Paris, France' should not match exact 'Paris'."""
        pr = parse_response("Paris, France")
        result = grade(pr, "Paris", "exact_match")
        assert result.passed is False

    def test_exact_match_strip_whitespace_option(self):
        pr = parse_response("  Paris  ")
        result = grade(pr, "Paris", "exact_match", {"strip_whitespace": True})
        assert result.passed is True

    def test_empty_final_fails_exact(self):
        pr = parse_response("")
        result = grade(pr, "", "exact_match")
        assert result.passed is False


# ======================================================================
# Contains grader
# ======================================================================

class TestContainsGrader:
    """Contains grader with context awareness."""

    def test_contains_finds_substring(self):
        pr = parse_response("The capital of France is Paris.")
        result = grade(pr, "Paris", "contains")
        assert result.passed is True

    def test_contains_case_insensitive_option(self):
        pr = parse_response("the capital is paris.")
        result = grade(pr, "Paris", "contains", {"ignore_case": True})
        assert result.passed is True

    def test_contains_missing_fails(self):
        pr = parse_response("The capital is London.")
        result = grade(pr, "Paris", "contains")
        assert result.passed is False

    def test_contains_with_negation_fails(self):
        """'not Paris' contains 'Paris' but is negated -> fail."""
        pr = parse_response("The answer is not Paris.")
        result = grade(pr, "Paris", "contains")
        # Should detect negation context
        assert result.passed is False

    def test_contains_multiple_possible_answers(self):
        """'Paris or London' is contradictory context."""
        pr = parse_response("It is either Paris or London.")
        result = grade(pr, "Paris", "contains")
        # Ambiguous/contradictory context
        assert result.passed is False

    def test_contains_empty_final_fails(self):
        pr = parse_response("")
        result = grade(pr, "Paris", "contains")
        assert result.passed is False


# ======================================================================
# Numeric grader
# ======================================================================

class TestNumericGrader:
    """Numeric tolerance grader."""

    def test_numeric_exact(self):
        pr = parse_response("42")
        result = grade(pr, "42", "numeric")
        assert result.passed is True

    def test_numeric_within_tolerance(self):
        pr = parse_response("42.1")
        result = grade(pr, "42", "numeric", {"tolerance": 0.2})
        assert result.passed is True

    def test_numeric_outside_tolerance(self):
        pr = parse_response("43")
        result = grade(pr, "42", "numeric", {"tolerance": 0.5})
        assert result.passed is False

    def test_numeric_non_numeric_final(self):
        """Non-numeric final text fails numeric grader."""
        pr = parse_response("about forty two")
        result = grade(pr, "42", "numeric")
        assert result.passed is False

    def test_numeric_extracts_number_from_text(self):
        """Extracts the first number from text."""
        pr = parse_response("The answer is 42.")
        result = grade(pr, "42", "numeric")
        assert result.passed is True

    def test_numeric_zero_tolerance(self):
        pr = parse_response("42.0001")
        result = grade(pr, "42", "numeric", {"tolerance": 0})
        assert result.passed is False

    def test_numeric_negative_numbers(self):
        pr = parse_response("-5")
        result = grade(pr, "-5", "numeric")
        assert result.passed is True

    def test_numeric_with_unit(self):
        """Numeric value with unit should still be extractable."""
        pr = parse_response("42 km")
        result = grade(pr, "42", "numeric")
        assert result.passed is True


# ======================================================================
# GradeResult data model
# ======================================================================

class TestGradeResult:
    """GradeResult dataclass properties."""

    def test_passed_is_boolean(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", "exact_match")
        assert isinstance(result.passed, bool)

    def test_score_in_range(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", "exact_match")
        assert 0.0 <= result.score <= 1.0

    def test_reason_is_nonempty(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", "exact_match")
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_evidence_includes_final_text(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", "exact_match")
        assert "Paris" in result.evidence

    def test_parse_failure_grade_result(self):
        raw = _O + "unclosed"
        pr = parse_response(raw)
        result = grade(pr, "anything", "exact_match")
        assert result.passed is False
        assert result.score == 0.0

    def test_grade_result_repr(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", "exact_match")
        rep = repr(result)
        assert "GradeResult" in rep
        assert "True" in rep


# ======================================================================
# Unsupported grader type
# ======================================================================

class TestUnsupportedGrader:
    """Unknown grader types fail closed."""

    def test_unknown_grader_fails(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", "unknown_grader")
        assert result.passed is False
        assert "unsupported" in result.reason.lower()

    def test_none_grader_type_fails(self):
        pr = parse_response("Paris")
        result = grade(pr, "Paris", None)
        assert result.passed is False
