"""Tests for response parsing, think-block extraction, final-text
normalization, truncation detection, and ambiguous-tag handling.

Coverage:
  VAL-EVAL-010 - Closed think blocks parse into reasoning and final text
  VAL-EVAL-011 - Missing think block uses whole response as final text
  VAL-EVAL-012 - Unclosed think blocks fail parsing
  VAL-EVAL-021 - Truncated responses cannot pass by accident
  VAL-EVAL-034 - Ambiguous think tags fail closed
  VAL-EVAL-035 - Reasoning-only response has empty final answer
"""

from __future__ import annotations

import pytest

from ornith_mlx_eval.parsing import (
    ParseResult,
    parse_response,
)

# Tag constants for test construction
_OPEN = "<think>"
_CLOSE = "</think>"


# ======================================================================
# VAL-EVAL-010 - Closed think blocks parse into reasoning and final text
# ======================================================================

class TestClosedThinkBlocks:
    """Closed think blocks split reasoning and final text correctly."""

    def test_basic_closed_think_block(self):
        """OPEN...CLOSE final text -> reasoning extracted, final text after close."""
        raw = _OPEN + "reasoning here" + _CLOSE + "final text"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text == "reasoning here"
        assert result.final_text == "final text"

    def test_closed_think_multiline(self):
        """Multiline reasoning block with final text."""
        raw = _OPEN + "line one\nline two" + _CLOSE + "final answer is 42"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text == "line one\nline two"
        assert result.final_text == "final answer is 42"

    def test_closed_think_with_trailing_spaces(self):
        """Whitespace after closing tag is trimmed from final text."""
        raw = _OPEN + "thought" + _CLOSE + "   answer  \n"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text == "thought"
        assert result.final_text == "answer"

    def test_closed_think_only_whitespace_between_tags(self):
        """Empty reasoning block still splits correctly."""
        raw = _OPEN + "  " + _CLOSE + "the answer"
        result = parse_response(raw)
        assert result.parse_status == "success"
        # Reasoning text is whitespace-only between tags, normalized to empty
        assert result.reasoning_text is not None
        assert result.final_text == "the answer"

    def test_closed_think_reasoning_with_newlines(self):
        """Reasoning text preserves internal newlines."""
        raw = _OPEN + "\nFirst thought\nSecond thought\n" + _CLOSE + "Final."
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text == "\nFirst thought\nSecond thought\n"
        assert result.final_text == "Final."


# ======================================================================
# VAL-EVAL-011 - Missing think block uses whole response as final text
# ======================================================================

class TestNoThinkTags:
    """Responses without think tags use whole normalized response."""

    def test_plain_text_no_tags(self):
        """Plain text becomes final text with no reasoning."""
        raw = "The capital of France is Paris."
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text is None
        assert result.final_text == "The capital of France is Paris."

    def test_plain_text_with_whitespace(self):
        """Leading/trailing whitespace is normalized."""
        raw = "  \n  Hello world  \n  "
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text is None
        assert result.final_text == "Hello world"

    def test_newlines_only_normalized(self):
        """Whitespace-only content becomes empty final text."""
        raw = "  \n \t  \n  "
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text is None
        assert result.final_text == ""

    def test_text_with_angle_brackets_not_tags(self):
        """Angle brackets that aren't think tags are kept in final text."""
        raw = "Use <div> for HTML, and 5 < 10 is true."
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text is None
        assert "div" in result.final_text
        assert "5 < 10" in result.final_text


# ======================================================================
# VAL-EVAL-012 - Unclosed think blocks fail parsing
# ======================================================================

class TestUnclosedThinkBlocks:
    """Opening think tag without matching close fails parsing."""

    def test_unclosed_think_at_end(self):
        """OPEN... with no close tag at end -> parse failure."""
        raw = _OPEN + "some reasoning but never closed"
        result = parse_response(raw)
        assert result.parse_status == "unclosed_think"
        # Final text should be empty or None on failure
        assert not result.final_text or result.final_text == ""

    def test_unclosed_think_mid_text(self):
        """OPEN opening but no close in the entire response."""
        raw = _OPEN + "reasoning about the question"
        result = parse_response(raw)
        assert result.parse_status == "unclosed_think"

    def test_unclosed_think_no_text_after_open(self):
        """Bare OPEN with nothing after."""
        raw = _OPEN
        result = parse_response(raw)
        assert result.parse_status == "unclosed_think"

    def test_unclosed_think_multiple_lines(self):
        """Multiline reasoning without close tag."""
        raw = _OPEN + "line1\nline2\nline3"
        result = parse_response(raw)
        assert result.parse_status == "unclosed_think"


# ======================================================================
# VAL-EVAL-034 - Ambiguous think tags fail closed
# ======================================================================

class TestAmbiguousThinkTags:
    """Malformed or ambiguous reasoning tags fail closed."""

    # -- Text before think block ------------------------------------------

    def test_text_before_think_block(self):
        """Text appearing before OPEN is ambiguous -> fail closed."""
        raw = "Here is my answer: " + _OPEN + "reasoning" + _CLOSE + "final"
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_whitespace_before_think_is_fine(self):
        """Leading whitespace before OPEN is allowed."""
        raw = "  \n " + _OPEN + "reasoning" + _CLOSE + "final"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text == "reasoning"
        assert result.final_text == "final"

    # -- Multiple think blocks --------------------------------------------

    def test_multiple_think_blocks(self):
        """Two separate OPEN...CLOSE blocks -> ambiguous."""
        raw = _OPEN + "first" + _CLOSE + "text" + _OPEN + "second" + _CLOSE + "end"
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_consecutive_think_tags(self):
        """OPEN directly followed by another OPEN -> ambiguous."""
        raw = _OPEN + _OPEN + "double" + _CLOSE + "final"
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    # -- Nested think tags ------------------------------------------------

    def test_nested_think_tags(self):
        """Nested OPEN tags -> ambiguous."""
        raw = _OPEN + "outer " + _OPEN + "inner" + _CLOSE + " more" + _CLOSE + "final"
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_deeply_nested_think_tags(self):
        """Deeply nested tags -> ambiguous."""
        raw = _OPEN + "a " + _OPEN + "b " + _OPEN + "c" + _CLOSE + _CLOSE + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    # -- Late think tags after final text ---------------------------------

    def test_late_think_tag_after_final_text(self):
        """A OPEN block appearing after text already been output."""
        raw = "final answer here " + _OPEN + "late reasoning" + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_text_between_blocks_is_ambiguous(self):
        """OPEN a CLOSE text OPEN b CLOSE -> ambiguous (multiple blocks)."""
        raw = _OPEN + "a" + _CLOSE + "text" + _OPEN + "b" + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    # -- Stray closing tags -----------------------------------------------

    def test_stray_closing_tag_no_open(self):
        """CLOSE without preceding OPEN -> ambiguous."""
        raw = "final answer" + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_opening_after_closing_without_new_open(self):
        """OPEN x CLOSE extra CLOSE -> ambiguous (stray close after already closed)."""
        raw = _OPEN + "x" + _CLOSE + "extra" + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    # -- Case and formatting variants -------------------------------------

    def test_uppercase_think_tag_not_recognized(self):
        """Uppercase THINK tag is not recognized; treated as plain text."""
        raw = "<THINK>some reasoning</THINK> answer"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text is None
        # The whole thing is treated as plain text since uppercase isn't a valid tag
        assert "THINK" in result.final_text

    def test_mixed_case_think_tags(self):
        """Mixed case tags are not recognized as think tags."""
        raw = "<Think>thought</Think> answer"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text is None

    def test_self_closing_think_tag(self):
        """<think/> is not a valid think tag."""
        raw = "<think/> reasoning here final"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text is None

    def test_think_with_attributes(self):
        """<think attr='val'> is not recognized as a valid think tag."""
        # Neither the open nor close think tags appear as exact substrings here
        raw = "<think attr='val'> reasoning... and the answer is 42"
        result = parse_response(raw)
        # Treated as plain text - no valid think tags found
        assert result.parse_status == "success"
        assert result.reasoning_text is None

    def test_only_opening_tag_no_close_no_text(self):
        """Just OPEN -> unclosed."""
        raw = _OPEN
        result = parse_response(raw)
        assert result.parse_status == "unclosed_think"

    def test_only_closing_tag(self):
        """Just CLOSE -> ambiguous (stray close)."""
        raw = _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_close_before_open(self):
        """CLOSE before OPEN -> ambiguous."""
        raw = _CLOSE + "stray close" + _OPEN + "and open"
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_unclosed_nested(self):
        """OPEN outer OPEN inner CLOSE -> ambiguous (nested tags detected first)."""
        raw = _OPEN + "outer " + _OPEN + "inner" + _CLOSE
        result = parse_response(raw)
        # Nested tags are detected before unclosed is determined,
        # so this is ambiguous_think rather than unclosed_think.
        assert result.parse_status == "ambiguous_think"

    def test_only_newlines_and_think(self):
        """OPEN x CLOSE with no final text after CLOSE -> empty_final."""
        raw = "\n" + _OPEN + "x" + _CLOSE
        result = parse_response(raw)
        # After CLOSE, there is no non-whitespace content -> empty_final
        assert result.parse_status == "empty_final"
        assert result.reasoning_text == "x"


# ======================================================================
# VAL-EVAL-035 - Reasoning-only response has empty final answer
# ======================================================================

class TestReasoningOnly:
    """OPEN reasoning CLOSE with no final text records empty final answer."""

    def test_reasoning_only_no_final_text(self):
        """Closed think with nothing after -> empty final text."""
        raw = _OPEN + "I think the answer is 42" + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "empty_final"
        assert result.reasoning_text == "I think the answer is 42"
        assert result.final_text == ""

    def test_reasoning_only_whitespace_after(self):
        """Only whitespace after closing tag -> empty final text."""
        raw = _OPEN + "thought" + _CLOSE + "   \n\t  "
        result = parse_response(raw)
        assert result.parse_status == "empty_final"
        assert result.reasoning_text == "thought"
        assert result.final_text == ""

    def test_reasoning_only_empty_think(self):
        """Empty think block with no final text."""
        raw = _OPEN + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "empty_final"
        assert result.final_text == ""

    def test_reasoning_only_multiline(self):
        """Multiline reasoning with no final answer."""
        raw = _OPEN + "\nStep 1: analyze\nStep 2: conclude\n" + _CLOSE + "   "
        result = parse_response(raw)
        assert result.parse_status == "empty_final"
        assert "Step 1" in (result.reasoning_text or "")
        assert result.final_text == ""


# ======================================================================
# VAL-EVAL-021 - Truncated responses cannot pass by accident
# ======================================================================

class TestTruncationDetection:
    """Truncation is detected and flagged."""

    def test_truncated_mid_sentence(self):
        """Response ending mid-sentence without punctuation."""
        raw = "The capital of France is"
        result = parse_response(raw)
        assert result.is_truncated is True

    def test_truncated_mid_word(self):
        """Response ending mid-word."""
        raw = "The capital of Fr"
        result = parse_response(raw)
        assert result.is_truncated is True

    def test_truncated_in_think_block(self):
        """Unclosed think is truncation-like."""
        raw = _OPEN + "thinking about the"
        result = parse_response(raw)
        # Unclosed think is already a parse failure
        assert result.parse_status == "unclosed_think"

    def test_complete_response_not_truncated(self):
        """Properly terminated response is not truncated."""
        raw = "The capital of France is Paris."
        result = parse_response(raw)
        assert result.is_truncated is False

    def test_truncated_with_closing_think(self):
        """OPEN x CLOSE unfinished -> final text is truncated."""
        raw = _OPEN + "reasoning" + _CLOSE + "The answer is"
        result = parse_response(raw)
        assert result.is_truncated is True

    def test_question_ending_not_truncated(self):
        """A response ending with a question mark is complete."""
        raw = "What is the capital of France?"
        result = parse_response(raw)
        assert result.is_truncated is False

    def test_exclamation_ending_not_truncated(self):
        """A response ending with ! is complete."""
        raw = "Paris!"
        result = parse_response(raw)
        assert result.is_truncated is False

    def test_code_block_ending_not_truncated(self):
        """A response ending with backticks is complete."""
        raw = "Here is the code:\n```python\nprint(42)\n```"
        result = parse_response(raw)
        assert result.is_truncated is False


# ======================================================================
# Edge cases and normalization
# ======================================================================

class TestEdgeCases:
    """Additional edge cases for parsing robustness."""

    def test_empty_response(self):
        """Empty response produces empty final text."""
        result = parse_response("")
        assert result.final_text == ""
        assert result.reasoning_text is None
        assert result.parse_status == "success"

    def test_whitespace_only_response(self):
        """Whitespace-only produces empty final text."""
        result = parse_response("   \n\t  ")
        assert result.final_text == ""
        assert result.reasoning_text is None
        assert result.parse_status == "success"

    def test_only_think_tags_no_content(self):
        """OPEN CLOSE with nothing between or after."""
        result = parse_response(_OPEN + _CLOSE)
        assert result.parse_status == "empty_final"
        assert result.final_text == ""

    def test_think_with_special_characters(self):
        """Special characters in reasoning and final text are preserved."""
        raw = _OPEN + '<>&"\'\\' + _CLOSE + "Answer: <>&\"'\\"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.final_text.startswith("Answer:")

    def test_raw_response_preserved(self):
        """Raw response field contains the original text."""
        raw = _OPEN + "hi" + _CLOSE + "there"
        result = parse_response(raw)
        assert result.raw_response == raw

    def test_very_long_reasoning(self):
        """Long reasoning blocks are handled correctly."""
        long_reasoning = "x" * 10000
        raw = _OPEN + long_reasoning + _CLOSE + "final"
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert result.reasoning_text == long_reasoning
        assert result.final_text == "final"

    def test_think_tag_midline(self):
        """OPEN not at start of line is ambiguous."""
        raw = "Let me think: " + _OPEN + "reasoning" + _CLOSE + "answer"
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_close_tag_with_following_think(self):
        """OPEN first CLOSE middle OPEN second CLOSE -> ambiguous due to multiple blocks."""
        raw = _OPEN + "first" + _CLOSE + "middle" + _OPEN + "second" + _CLOSE
        result = parse_response(raw)
        assert result.parse_status == "ambiguous_think"

    def test_rationale_preserved_with_final_text(self):
        """Full response: reasoning + final text both extracted."""
        raw = (
            _OPEN
            + "First, I need to understand what the question asks.\n"
            + "The capital of France is a well-known fact.\n"
            + _CLOSE
            + "The capital of France is Paris."
        )
        result = parse_response(raw)
        assert result.parse_status == "success"
        assert "First, I need to understand" in (result.reasoning_text or "")
        assert result.final_text == "The capital of France is Paris."


class TestParseResultRepr:
    """ParseResult dataclass provides readable representation."""

    def test_repr_includes_status(self):
        result = parse_response(_OPEN + "x" + _CLOSE + "y")
        rep = repr(result)
        assert "success" in rep
        assert "ParseResult" in rep

    def test_failure_repr_includes_status(self):
        result = parse_response(_OPEN + "x")
        rep = repr(result)
        assert "unclosed_think" in rep
