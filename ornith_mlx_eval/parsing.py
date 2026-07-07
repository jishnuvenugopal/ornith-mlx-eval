"""Response parsing, think-block extraction, final-text normalization,
truncation detection, and parse-status classification.

Owns:
  * <think> ... </think> extraction
  * unclosed / ambiguous reasoning-block detection
  * final-text normalization
  * truncation-status detection
  * parse-result data model

Default grading uses parsed final text.  Answers that appear only in
reasoning fail unless a specific grader explicitly allows reasoning-aware
scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"

# Characters that indicate a complete sentence/response ending.
_SENTENCE_ENDERS: set[str] = {".", "!", "?", ")", "]", "}", '"', "'", "`", "\u201d", "\u2019"}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ParseResult:
    """Result of parsing a raw model response.

    Attributes:
        raw_response: The original unmodified response text.
        reasoning_text: Extracted reasoning from <think>...</think> blocks, or *None*
            if no think block was present.
        final_text: Normalised final text used for grading.  May be empty
            on parse failures or reasoning-only responses.
        parse_status: One of ``"success"``, ``"unclosed_think"``,
            ``"ambiguous_think"``, or ``"empty_final"``.
        is_truncated: *True* if the response appears to be cut off
            (ends mid-sentence, missing closing tag, etc.).
    """

    raw_response: str = ""
    reasoning_text: Optional[str] = None
    final_text: str = ""
    parse_status: str = "success"
    is_truncated: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_response(raw: str) -> ParseResult:
    """Parse a raw model response into reasoning and final text.

    Parse rules (ordered by priority):

    1. **No think tags** (VAL-EVAL-011):
       The entire normalised response is final text.  Reasoning is *None*.

    2. **Text before opening tag** (VAL-EVAL-034):
       If any non-whitespace text appears before the first <think>, the
       parse is *ambiguous_think*.

    3. **Unclosed think** (VAL-EVAL-012):
       An opening <think> without a matching </think> -> *unclosed_think*.

    4. **Multiple / nested / late think blocks** (VAL-EVAL-034):
       More than one <think>...</think> block, nested tags, stray </think> without
       preceding <think>, or <think> appearing after content outside
       the block -> *ambiguous_think*.

    5. **Reasoning-only** (VAL-EVAL-035):
       A closed <think>...</think> with no non-whitespace final text
       after the closing tag -> *empty_final*.

    6. **Closed think block** (VAL-EVAL-010):
       <think>...</think> followed by non-whitespace final text -> *success*
       with extracted reasoning and normalised final text.

    Args:
        raw: The raw model response string.

    Returns:
        A :class:`ParseResult` with the parsed and classified response.
    """
    result = ParseResult(raw_response=raw)

    # Quick check: if no open tag at all, it's a simple no-think response.
    first_open = raw.find(_OPEN_TAG)
    if first_open == -1:
        # But if a stray close tag is present without any open, that's ambiguous.
        if _CLOSE_TAG in raw:
            result.parse_status = "ambiguous_think"
            result.reasoning_text = None
            result.final_text = ""
            result.is_truncated = True
            return result
        _finalize_no_think(raw, result)
        return result

    # --- Text before the first open tag? ------------------------------------
    prefix = raw[:first_open]
    if prefix.strip():
        # Non-whitespace text before the think block -> ambiguous
        result.parse_status = "ambiguous_think"
        result.reasoning_text = None
        result.final_text = ""
        result.is_truncated = True
        return result

    # --- Count open/close tag positions -------------------------------------
    open_positions = _find_all_tags(raw, _OPEN_TAG)
    close_positions = _find_all_tags(raw, _CLOSE_TAG)

    # --- If no close tag -> unclosed -----------------------------------------
    if not close_positions:
        result.parse_status = "unclosed_think"
        # Extract whatever is after the opening tag as reasoning
        after_open = raw[open_positions[0] + len(_OPEN_TAG):]
        result.reasoning_text = after_open.strip() if after_open.strip() else after_open
        result.final_text = ""
        result.is_truncated = True
        return result

    # --- Stray close before first open? -------------------------------------
    if close_positions and close_positions[0] < open_positions[0]:
        result.parse_status = "ambiguous_think"
        result.reasoning_text = None
        result.final_text = ""
        result.is_truncated = True
        return result

    first_close = close_positions[0]

    # --- Nested open tags? (another open before first close) -----------------
    if len(open_positions) > 1 and open_positions[1] < first_close:
        result.parse_status = "ambiguous_think"
        result.reasoning_text = None
        result.final_text = ""
        result.is_truncated = True
        return result

    # --- Multiple open tags? -------------------------------------------------
    if len(open_positions) > 1:
        result.parse_status = "ambiguous_think"
        result.reasoning_text = None
        result.final_text = ""
        result.is_truncated = True
        return result

    # --- Multiple close tags? ------------------------------------------------
    if len(close_positions) > 1:
        result.parse_status = "ambiguous_think"
        result.reasoning_text = None
        result.final_text = ""
        result.is_truncated = True
        return result

    # --- Check for text after close that looks like another think block ------
    after_close = raw[first_close + len(_CLOSE_TAG):]
    if _find_all_tags(after_close, _OPEN_TAG):
        # Late think tag after final text -> ambiguous
        result.parse_status = "ambiguous_think"
        result.reasoning_text = None
        result.final_text = ""
        result.is_truncated = True
        return result

    # --- Single valid think block --------------------------------------------
    reasoning_content = raw[open_positions[0] + len(_OPEN_TAG):first_close]
    final_content = after_close.strip()

    result.reasoning_text = reasoning_content if reasoning_content else ""

    if not final_content:
        # Reasoning-only response (VAL-EVAL-035)
        result.parse_status = "empty_final"
        result.final_text = ""
        result.is_truncated = True
        return result

    result.parse_status = "success"
    result.final_text = final_content
    result.is_truncated = _detect_truncation(final_content)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _finalize_no_think(raw: str, result: ParseResult) -> None:
    """Handle responses with no think tags."""
    normalized = raw.strip()
    result.reasoning_text = None
    result.final_text = normalized
    result.parse_status = "success"
    result.is_truncated = _detect_truncation(normalized) if normalized else False


def _find_all_tags(text: str, tag: str) -> list[int]:
    """Return all start positions of *tag* in *text* (non-overlapping)."""
    positions: list[int] = []
    pos = 0
    tag_len = len(tag)
    while True:
        idx = text.find(tag, pos)
        if idx == -1:
            break
        positions.append(idx)
        pos = idx + tag_len
    return positions


def _detect_truncation(text: str) -> bool:
    """Heuristic: response appears truncated if it ends mid-sentence.

    A response is considered truncated if:
    - It is non-empty AND
    - The last non-whitespace character is NOT a sentence-ending character
      (period, exclamation, question mark, closing bracket/quote/backtick) AND
    - The text contains multiple words (space-separated) — single-word
      answers like "Paris" or short numeric answers like "42 km" are
      not considered truncated.
    """
    if not text:
        return False
    stripped = text.rstrip()
    if not stripped:
        return False
    last_char = stripped[-1]
    if last_char in _SENTENCE_ENDERS:
        return False
    # Single-word or very short answers are not truncated
    # (e.g. "Paris", "42", "42 km", "-5")
    word_count = len(stripped.split())
    if word_count <= 2:
        return False

    # Check if the last word looks like an incomplete word fragment
    # (e.g. "Fr" from "France", "The capital of Fr")
    last_word = stripped.split()[-1].rstrip(".,;:!?()[]{}'\"")
    if _looks_incomplete(last_word):
        return True

    # Check if the last word is a known truncation-indicating word
    # (auxiliary verbs, prepositions, articles, conjunctions that
    #  typically require a complement and are unlikely to end a complete
    #  thought).  This avoids flagging complete statements like
    #  "The capital is Paris" while still catching "The capital of France is".
    if last_word.lower() in _TRUNCATION_TAIL_WORDS:
        return True

    # If the text has many words but no sentence ender AND the last word
    # is not a truncation indicator, assume it's a valid terse answer.
    # Only flag truly long texts (>8 words) without sentence enders
    # as potentially truncated.
    if word_count > 8:
        return True

    return False


def _looks_incomplete(word: str) -> bool:
    """Heuristic: does *word* look like an incomplete word fragment?

    Returns True if the word is short (<=3 chars), contains no vowels,
    or ends with a consonant fragment that is unlikely to be a complete
    English word.
    """
    if not word:
        return False
    w = word.lower()
    # Very short fragments that aren't common short words
    if len(w) <= 2 and w not in _COMMON_SHORT_WORDS:
        return True
    # Words without any vowels are likely fragments (e.g. "Fr", "cnt")
    if not any(ch in "aeiouy" for ch in w):
        return True
    return False


# Common short English words that shouldn't be flagged as incomplete
_COMMON_SHORT_WORDS: frozenset[str] = frozenset({
    "a", "i", "is", "it", "am", "be", "do", "go", "he", "hi", "if",
    "in", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us",
    "we", "as", "at", "by", "an", "oh", "ok", "ah", "ha",
})


# ---------------------------------------------------------------------------
# Fenced-code extraction
# ---------------------------------------------------------------------------


# Words that, when appearing at the end of text, suggest truncation
_TRUNCATION_TAIL_WORDS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an",
    "of", "in", "to", "for", "on", "at", "by", "with", "from",
    "and", "or", "but", "if", "as", "so", "nor", "yet",
    "that", "this", "these", "those", "which", "what",
    "has", "have", "had", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "not", "also", "then", "than", "about", "into", "over",
    "such", "its", "it", "there",
})

# Supported code fence languages
_SUPPORTED_CODE_LANGUAGES: frozenset[str] = frozenset({"python", "py"})


@dataclass
class CodeExtractionResult:
    """Result of extracting a fenced code block from a response.

    Attributes:
        code: The extracted code text (empty on failure).
        language: The declared fence language (empty on failure).
        status: One of ``"success"``, ``"no_fence"``,
            ``"multiple_fences"``, ``"unterminated_fence"``,
            ``"unsupported_language"``, ``"no_language"``, or
            ``"empty_code"``.
        fence_start_line: 0-based line index where the opening fence
            appears, or -1 on failure.
    """

    code: str = ""
    language: str = ""
    status: str = "no_fence"
    fence_start_line: int = -1


def extract_fenced_code(text: str) -> CodeExtractionResult:
    """Extract a single fenced Python code block from *text*.

    Rules (VAL-EVAL-015, VAL-EVAL-016):

    * Only ``python`` or ``py`` language identifiers are supported.
    * Exactly one code block must be present; multiple blocks fail.
    * The fence must be properly closed (`` ``` ``).
    * Empty or whitespace-only blocks fail.
    * A missing language identifier fails as ``no_language``.

    Args:
        text: The response text (typically the final text from parsing).

    Returns:
        A :class:`CodeExtractionResult` describing the extraction outcome.
    """
    # Find all fenced code blocks: ```<lang>\n<code>\n```
    # Pattern: ^```(?:python|py)?\s*$(.*?)^```\s*$  (multiline)
    fence_pattern = re.compile(
        r'^```([^\n]*)$(.*?)^```\s*$',
        re.MULTILINE | re.DOTALL,
    )

    matches = list(fence_pattern.finditer(text))
    if not matches:
        return CodeExtractionResult(status="no_fence")

    # Extract all python/py blocks
    python_blocks: list[tuple[str, str, int]] = []  # (lang, code, start_line)

    for m in matches:
        lang_str = m.group(1).strip().lower()
        code = m.group(2)

        # Compute 0-based line index of the opening fence
        start_line = text[:m.start()].count('\n')

        if lang_str in _SUPPORTED_CODE_LANGUAGES:
            python_blocks.append((lang_str, code, start_line))

    if not python_blocks:
        # Check if there were any fenced blocks at all
        # If there were non-python fences, mark as unsupported_language
        for m in matches:
            lang_str = m.group(1).strip().lower()
            if lang_str:
                return CodeExtractionResult(status="unsupported_language")
        # Fences without language
        return CodeExtractionResult(status="no_language")

    if len(python_blocks) > 1:
        return CodeExtractionResult(status="multiple_fences")

    lang, code, start_line = python_blocks[0]

    # Check for empty code
    stripped_code = code.strip()
    if not stripped_code:
        return CodeExtractionResult(status="empty_code")

    return CodeExtractionResult(
        code=stripped_code,
        language=lang,
        status="success",
        fence_start_line=start_line,
    )
