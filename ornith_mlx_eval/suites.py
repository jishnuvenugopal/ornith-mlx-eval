"""Suite loading, validation, hashing, and leakage audit.

Owns suite discovery from the suites/ directory, JSON schema validation,
canonical suite hashing, prompt-template and per-case prompt hashing,
leakage-audit for hidden expected answers, case-ID uniqueness rules,
unknown-field rejection (except `_ext`), and run-time revalidation hooks.

All suites must be authored from scratch. No upstream content may be copied.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Known / supported values
# ---------------------------------------------------------------------------

SUPPORTED_GRADERS: frozenset[str] = frozenset({
    "exact_match", "contains", "numeric", "json_match", "code",
})

# Fields known at each level.  Unknown fields outside `_ext` fail validation.
KNOWN_TOP_LEVEL: frozenset[str] = frozenset({
    "suite_id", "suite_version", "description", "tags", "cases", "_ext",
})
KNOWN_CASE_FIELDS: frozenset[str] = frozenset({
    "case_id", "prompt", "expected_answer", "grader", "_ext",
})
KNOWN_PROMPT_FIELDS: frozenset[str] = frozenset({
    "system", "user_template", "_ext",
})
KNOWN_EXPECTED_ANSWER_FIELDS: frozenset[str] = frozenset({
    "answer_type", "hidden_answer", "_ext",
})
KNOWN_GRADER_FIELDS: frozenset[str] = frozenset({
    "type", "options", "_ext",
})

# Semantic fields that contribute to the suite hash (everything except _ext).
SEMANTIC_TOP_LEVEL: frozenset[str] = KNOWN_TOP_LEVEL - {"_ext"}
SEMANTIC_CASE_FIELDS: frozenset[str] = KNOWN_CASE_FIELDS - {"_ext"}
SEMANTIC_PROMPT_FIELDS: frozenset[str] = KNOWN_PROMPT_FIELDS - {"_ext"}
SEMANTIC_EXPECTED_ANSWER_FIELDS: frozenset[str] = KNOWN_EXPECTED_ANSWER_FIELDS - {"_ext"}
SEMANTIC_GRADER_FIELDS: frozenset[str] = KNOWN_GRADER_FIELDS - {"_ext"}

# Per-grader-type option validation schema.
#   allowed  — option keys that this grader type accepts
#   required — option keys that must be present (code grader needs
#              test_input + expected_output; numeric tolerance is optional)
#   types    — expected Python type(s) for each option value
GRADER_OPTIONS_SCHEMA: dict[str, dict[str, Any]] = {
    "exact_match": {
        "allowed": frozenset({"ignore_case", "strip_whitespace"}),
        "required": frozenset(),
        "types": {"ignore_case": bool, "strip_whitespace": bool},
    },
    "contains": {
        "allowed": frozenset({"ignore_case", "strip_whitespace"}),
        "required": frozenset(),
        "types": {"ignore_case": bool, "strip_whitespace": bool},
    },
    "numeric": {
        "allowed": frozenset({"tolerance"}),
        "required": frozenset(),  # tolerance is optional (defaults to 0)
        "types": {"tolerance": (int, float)},
    },
    "json_match": {
        "allowed": frozenset(),
        "required": frozenset(),
        "types": {},
    },
    "code": {
        "allowed": frozenset({"test_input", "expected_output"}),
        "required": frozenset({"test_input", "expected_output"}),
        "types": {},
    },
}

# ---------------------------------------------------------------------------
# Public API – discovery
# ---------------------------------------------------------------------------

DEFAULT_SUITES_DIR = Path(__file__).resolve().parent.parent / "suites"


def discover_suites(suites_dir: Path | None = None) -> list[Path]:
    """Return all discoverable suite JSON files in sorted order."""
    if suites_dir is None:
        suites_dir = DEFAULT_SUITES_DIR
    if not suites_dir.is_dir():
        return []
    paths = sorted(
        p for p in suites_dir.iterdir()
        if p.is_file() and p.suffix == ".json"
    )
    return paths


# ---------------------------------------------------------------------------
# Public API – loading
# ---------------------------------------------------------------------------

class SuiteValidationError(ValueError):
    """Raised when a suite fails validation.

    Attributes:
        message: Human-readable error description.
        field_path: Dot-notation path to the offending field (if applicable).
        suite_path: The path to the suite file that failed (if known).
    """

    def __init__(self, message: str, *, field_path: str = "", suite_path: str = "") -> None:
        super().__init__(message)
        self.field_path = field_path
        self.suite_path = suite_path


def load_suite(path: Path) -> dict[str, Any]:
    """Load and parse a suite JSON file.  Raises SuiteValidationError on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SuiteValidationError(f"Suite file not found: {path}", suite_path=str(path))
    except OSError as exc:
        raise SuiteValidationError(f"Cannot read suite file: {exc}", suite_path=str(path))

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SuiteValidationError(
            f"Invalid JSON in suite file: {exc}", suite_path=str(path)
        )
    if not isinstance(data, dict):
        raise SuiteValidationError(
            "Suite root must be a JSON object", suite_path=str(path)
        )
    return data


# ---------------------------------------------------------------------------
# Public API – validation
# ---------------------------------------------------------------------------

def validate_suite(
    suite: dict[str, Any],
    *,
    suite_path: str = "",
) -> list[str]:
    """Validate *suite* against the harness schema.

    Returns a (possibly empty) list of error messages.  An empty list means the
    suite is valid.

    Checks performed:
      * required top-level / case / prompt / expected-answer / grader fields
      * field types and value constraints
      * case-ID uniqueness (after normalisation)
      * known grader types
      * unknown-field rejection (except ``_ext`` sub-objects)
      * leakage of any hidden expected answer into visible fields
    """
    errors: list[str] = []

    # -- Top-level -----------------------------------------------------------
    if not isinstance(suite, dict):
        return ["suite must be a JSON object"]

    for field in ("suite_id", "suite_version", "cases"):
        if field not in suite:
            errors.append(f"missing required top-level field: '{field}'")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"].strip():
        errors.append("top-level 'suite_id' must be a non-empty string")
    if not isinstance(suite.get("suite_version"), str) or not suite["suite_version"].strip():
        errors.append("top-level 'suite_version' must be a non-empty string")
    if not isinstance(suite.get("cases"), list):
        errors.append("top-level 'cases' must be a non-empty array")
    elif len(suite["cases"]) == 0:
        errors.append("top-level 'cases' must have at least one case")

    _check_unknown_fields(suite, KNOWN_TOP_LEVEL, "top-level", errors)

    if errors:
        _add_path_context(errors, suite_path)
        return errors

    # -- Cases ---------------------------------------------------------------
    cases: list[dict[str, Any]] = suite["cases"]
    seen_ids: dict[str, str] = {}  # normalised -> original

    for i, case in enumerate(cases):
        prefix = f"case[{i}] "
        if not isinstance(case, dict):
            errors.append(f"{prefix}must be a JSON object")
            continue

        # required fields
        for field in ("case_id", "prompt", "expected_answer", "grader"):
            if field not in case:
                errors.append(f"{prefix}missing required field: '{field}'")

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{prefix}'case_id' must be a non-empty string")
        else:
            norm = case_id.strip().lower()
            if norm in seen_ids:
                errors.append(
                    f"{prefix}duplicate case_id '{case_id}' "
                    f"(collides with '{seen_ids[norm]}' after normalisation)"
                )
            else:
                seen_ids[norm] = case_id

        _check_unknown_fields(case, KNOWN_CASE_FIELDS, prefix, errors)

        # prompt
        prompt = case.get("prompt")
        if isinstance(prompt, dict):
            if "user_template" not in prompt:
                errors.append(f"{prefix}'prompt' missing required field: 'user_template'")
            elif not isinstance(prompt.get("user_template"), str) or not prompt["user_template"].strip():
                errors.append(f"{prefix}'prompt.user_template' must be a non-empty string")
            _check_unknown_fields(prompt, KNOWN_PROMPT_FIELDS, f"{prefix}prompt.", errors)

        # expected_answer
        ea = case.get("expected_answer")
        if isinstance(ea, dict):
            for field in ("hidden_answer", "answer_type"):
                if field not in ea:
                    errors.append(f"{prefix}'expected_answer' missing required field: '{field}'")
            if not isinstance(ea.get("hidden_answer"), str) or not ea["hidden_answer"].strip():
                errors.append(f"{prefix}'expected_answer.hidden_answer' must be a non-empty string")
            if not isinstance(ea.get("answer_type"), str) or not ea["answer_type"].strip():
                errors.append(f"{prefix}'expected_answer.answer_type' must be a non-empty string")
            _check_unknown_fields(ea, KNOWN_EXPECTED_ANSWER_FIELDS, f"{prefix}expected_answer.", errors)

        # grader
        grader = case.get("grader")
        if isinstance(grader, dict):
            if "type" not in grader:
                errors.append(f"{prefix}'grader' missing required field: 'type'")
            else:
                gtype = grader["type"]
                if not isinstance(gtype, str) or not gtype.strip():
                    errors.append(f"{prefix}'grader.type' must be a non-empty string")
                elif gtype not in SUPPORTED_GRADERS:
                    errors.append(
                        f"{prefix}unsupported grader type '{gtype}'; "
                        f"supported: {', '.join(sorted(SUPPORTED_GRADERS))}"
                    )
            _check_unknown_fields(grader, KNOWN_GRADER_FIELDS, f"{prefix}grader.", errors)
            _validate_grader_options(grader, prefix, errors)

    if errors:
        _add_path_context(errors, suite_path)
        return errors

    # -- Leakage audit -------------------------------------------------------
    leakage_errors = _audit_leakage(suite)
    if leakage_errors:
        errors.extend(leakage_errors)

    _add_path_context(errors, suite_path)
    return errors


def _check_unknown_fields(
    obj: dict[str, Any],
    known: frozenset[str],
    prefix: str,
    errors: list[str],
) -> None:
    """Reject unknown fields that are not under the _ext namespace."""
    p = prefix.rstrip()  # normalise so we always control the trailing space
    for key in obj:
        if key.startswith("_"):
            if key == "_ext":
                if isinstance(obj[key], dict):
                    continue
                errors.append(f"{p} field '_ext' must be an object")
                continue
            # Any _-prefixed key other than _ext is unknown
            errors.append(f"{p} unknown field: '{key}'")
            continue
        if key not in known:
            errors.append(f"{p} unknown field: '{key}'")


def _validate_grader_options(
    grader: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> None:
    """Validate grader options against the per-type option schema.

    Checks:
      * Required options are present
      * Unknown options are rejected
      * Option value types are compatible
    """
    gtype = grader.get("type", "")
    if gtype not in SUPPORTED_GRADERS:
        return  # unknown grader type already reported by caller

    schema = GRADER_OPTIONS_SCHEMA.get(gtype)
    if schema is None:
        return

    p = prefix.rstrip()
    opt_prefix = f"{p} grader.options"

    options = grader.get("options")
    if not isinstance(options, dict):
        # Missing or non-dict options – check required
        if schema["required"]:
            for key in sorted(schema["required"]):
                errors.append(
                    f"{opt_prefix} missing required option '{key}' "
                    f"for grader type '{gtype}'"
                )
        return

    # Check required options
    for key in sorted(schema["required"]):
        if key not in options:
            errors.append(
                f"{opt_prefix} missing required option '{key}' "
                f"for grader type '{gtype}'"
            )

    # Check for unknown options
    for key in options:
        if key.startswith("_"):
            if key == "_ext":
                if isinstance(options[key], dict):
                    continue
                errors.append(f"{opt_prefix}._ext must be an object")
                continue
            errors.append(
                f"{opt_prefix} unknown option '{key}' "
                f"for grader type '{gtype}'"
            )
            continue
        if key not in schema["allowed"]:
            errors.append(
                f"{opt_prefix} unknown option '{key}' "
                f"for grader type '{gtype}'"
            )

    # Check option value types
    for key, expected_type in schema["types"].items():
        if key not in options:
            continue
        value = options[key]
        if value is None:
            errors.append(
                f"{opt_prefix}.{key} must not be null "
                f"for grader type '{gtype}'"
            )
            continue
        # bool is a subclass of int in Python, so isinstance(True, (int,float))
        # passes even though we want to reject booleans for numeric tolerance.
        if isinstance(value, bool) and (int, float) == expected_type:
            errors.append(
                f"{opt_prefix}.{key} must be int or float, "
                f"got bool for grader type '{gtype}'"
            )
            continue
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_names = " or ".join(
                    t.__name__ for t in expected_type
                )
            else:
                type_names = expected_type.__name__
            errors.append(
                f"{opt_prefix}.{key} must be {type_names}, "
                f"got {type(value).__name__} "
                f"for grader type '{gtype}'"
            )


def _add_path_context(errors: list[str], suite_path: str) -> None:
    """Optionally prepend suite path to first error."""
    pass  # path is already embedded in the error message prefix


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Normalise text for leakage comparison.

    Strips whitespace, lowercases, strips markdown formatting noise
    (bold, italic, code, etc.).
    """
    t = text.strip().lower()
    # Remove markdown bold/italic/heading markers
    t = re.sub(r'\*{1,3}|_{1,3}|#{1,6}', '', t)
    # Remove markdown code backticks
    t = re.sub(r'`{1,3}', '', t)
    t = t.strip()
    return t


def _normalise_json_escapes(text: str) -> str:
    """Decode JSON escape sequences so that hidden answers encoded as
    \\uXXXX or \\\" variants still trigger leakage detection (VAL-EVAL-033).

    Handles the common escapes that could mask an answer:
      * Unicode escapes:  \\u0050\\u0061...  ->  "Pa..."
      * Quote escapes:    \\\"Paris\\\"        ->  "\"Paris\""
      * Backslash:        \\\\                 ->  "\\"
      * Control chars:    \\n \\t \\r
    """
    def _replace_unicode(m: re.Match[str]) -> str:
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)

    result = re.sub(r'\\u([0-9a-fA-F]{4})', _replace_unicode, text)
    result = result.replace('\\"', '"')
    result = result.replace('\\\\', '\\')
    result = result.replace('\\n', '\n')
    result = result.replace('\\t', '\t')
    result = result.replace('\\r', '\r')
    return result


def _answer_appears_in(text: str, answer: str) -> bool:
    """Check whether *answer* appears in *text* after normalisation.

    Uses word-boundary-aware matching to avoid substring false positives.
    Additionally checks JSON-unescaped variants (VAL-EVAL-033) so that
    \\uXXXX-encoded answers are still detected.
    """
    answer_norm = _normalise_text(answer)
    if not answer_norm:
        return False

    pattern = re.compile(r'\b' + re.escape(answer_norm) + r'\b')

    def _matches(t: str) -> bool:
        if pattern.search(t):
            return True
        return answer_norm in t

    # 1. Check normalised raw text
    text_norm = _normalise_text(text)
    if _matches(text_norm):
        return True

    # 2. Check JSON-unescaped version (handles \\uXXXX, \\\", etc.)
    text_unescaped = _normalise_json_escapes(text)
    if text_unescaped != text:
        text_unescaped_norm = _normalise_text(text_unescaped)
        if _matches(text_unescaped_norm):
            return True

    return False


def _gather_visible_texts(suite: dict[str, Any]) -> list[tuple[str, str]]:
    """Collect all visible (non-hidden) text fields from the suite.

    Returns list of (field_path, text_value).
    """
    pairs: list[tuple[str, str]] = []

    if isinstance(suite.get("suite_id"), str):
        pairs.append(("suite_id", suite["suite_id"]))
    if isinstance(suite.get("description"), str):
        pairs.append(("description", suite["description"]))
    if isinstance(suite.get("tags"), list):
        for i, tag in enumerate(suite["tags"]):
            if isinstance(tag, str):
                pairs.append((f"tags[{i}]", tag))

    for i, case in enumerate(suite.get("cases", [])):
        if not isinstance(case, dict):
            continue
        case_prefix = f"case[{i}]"

        if isinstance(case.get("case_id"), str):
            pairs.append((f"{case_prefix}.case_id", case["case_id"]))

        prompt = case.get("prompt")
        if isinstance(prompt, dict):
            if isinstance(prompt.get("system"), str):
                pairs.append((f"{case_prefix}.prompt.system", prompt["system"]))
            if isinstance(prompt.get("user_template"), str):
                pairs.append((f"{case_prefix}.prompt.user_template", prompt["user_template"]))

        grader = case.get("grader")
        if isinstance(grader, dict):
            gtype = grader.get("type")
            if isinstance(gtype, str):
                pairs.append((f"{case_prefix}.grader.type", gtype))
            opts = grader.get("options")
            if isinstance(opts, dict):
                for ok, ov in opts.items():
                    if isinstance(ov, str):
                        pairs.append((f"{case_prefix}.grader.options.{ok}", ov))
                    elif isinstance(ov, (int, float, bool)):
                        pairs.append((f"{case_prefix}.grader.options.{ok}", str(ov)))

    return pairs


def _audit_leakage(suite: dict[str, Any]) -> list[str]:
    """Check that hidden answers do not appear in visible fields.

    Returns a list of error messages, one per leakage instance.
    """
    errors: list[str] = []

    # Gather all hidden answers
    hidden_answers: list[tuple[str, str]] = []  # (case_id, answer)
    for i, case in enumerate(suite.get("cases", [])):
        if not isinstance(case, dict):
            continue
        ea = case.get("expected_answer")
        if isinstance(ea, dict) and isinstance(ea.get("hidden_answer"), str):
            case_id = case.get("case_id", f"case[{i}]")
            hidden_answers.append((str(case_id), ea["hidden_answer"]))

    if not hidden_answers:
        return errors

    visible_texts = _gather_visible_texts(suite)

    for case_id, answer in hidden_answers:
        if not answer.strip():
            continue
        for field_path, text in visible_texts:
            if _answer_appears_in(text, answer):
                errors.append(
                    f"leakage: hidden answer for case '{case_id}' found in "
                    f"visible field '{field_path}'"
                )
                # Only report first leak per answer to avoid spam
                break

    return errors


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> bytes:
    """Serialize *obj* to canonical JSON bytes (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _strip_ext(obj: dict[str, Any], semantic_keys: frozenset[str]) -> dict[str, Any]:
    """Return a copy of *obj* containing only *semantic_keys*."""
    return {k: v for k, v in obj.items() if k in semantic_keys}


def _canonical_case(case: dict[str, Any]) -> dict[str, Any]:
    """Build a canonical representation of a single case for suite hashing.

    Prompt assembly fields (system, user_template) are deliberately excluded
    from the suite hash (VAL-EVAL-007).  The prompt-template hash captures
    those independently so that prompt changes do not invalidate suite identity.
    """
    canon: dict[str, Any] = {}
    if "case_id" in case:
        canon["case_id"] = case["case_id"]
    ea = case.get("expected_answer", {})
    if isinstance(ea, dict):
        canon["expected_answer"] = _strip_ext(ea, SEMANTIC_EXPECTED_ANSWER_FIELDS)
    grader = case.get("grader", {})
    if isinstance(grader, dict):
        canon["grader"] = _strip_ext(grader, SEMANTIC_GRADER_FIELDS)
    return canon


def compute_suite_hash(suite: dict[str, Any]) -> str:
    """Compute a canonical SHA-256 hash of the suite's semantic content.

    The hash is stable across JSON formatting differences (whitespace,
    indentation, key ordering) but changes when any semantic field changes
    (cases, prompts, expected answers, grader config, tags, version, etc.).
    """
    canon: dict[str, Any] = {}
    canon["suite_id"] = suite.get("suite_id", "")
    canon["suite_version"] = suite.get("suite_version", "")
    if "description" in suite:
        canon["description"] = suite["description"]
    if "tags" in suite and isinstance(suite["tags"], list):
        canon["tags"] = sorted(suite["tags"])

    canon["cases"] = []
    for case in suite.get("cases", []):
        if isinstance(case, dict):
            canon["cases"].append(_canonical_case(case))

    return hashlib.sha256(_canonical_json(canon)).hexdigest()


def compute_prompt_template_hash(suite: dict[str, Any]) -> str:
    """Compute a hash of the prompt-template assembly rules.

    This captures the structure of how prompts are assembled (system prompt,
    user template) across all cases.  Changing a system prompt or user_template
    changes this hash.
    """
    templates: list[dict[str, str]] = []
    for case in suite.get("cases", []):
        if not isinstance(case, dict):
            continue
        prompt = case.get("prompt", {})
        if isinstance(prompt, dict):
            tpl: dict[str, str] = {}
            if isinstance(prompt.get("system"), str):
                tpl["system"] = prompt["system"]
            if isinstance(prompt.get("user_template"), str):
                tpl["user_template"] = prompt["user_template"]
            templates.append(tpl)
    return hashlib.sha256(_canonical_json(templates)).hexdigest()


def render_prompt(case: dict[str, Any]) -> str:
    """Render the prompt text for a single case.

    Does NOT include hidden expected-answer metadata.
    """
    prompt = case.get("prompt", {})
    parts: list[str] = []
    if isinstance(prompt.get("system"), str) and prompt["system"].strip():
        parts.append(prompt["system"].strip())
    if isinstance(prompt.get("user_template"), str) and prompt["user_template"].strip():
        parts.append(prompt["user_template"].strip())
    return "\n\n".join(parts)


def compute_case_prompt_hash(case: dict[str, Any]) -> str:
    """Compute a hash of the exact rendered prompt text for a single case."""
    rendered = render_prompt(case)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Suite listing (used by CLI list-suites)
# ---------------------------------------------------------------------------

def list_suites_info(suites_dir: Path | None = None) -> list[dict[str, Any]]:
    """Discover suites and return summary info for each.

    Each dict contains: path, suite_id, suite_version, description, case_count,
    valid (bool), errors (list[str]).
    """
    results: list[dict[str, Any]] = []
    for path in discover_suites(suites_dir):
        info: dict[str, Any] = {"path": str(path)}
        try:
            suite = load_suite(path)
            errors = validate_suite(suite, suite_path=str(path))
            info["suite_id"] = suite.get("suite_id", "unknown")
            info["suite_version"] = suite.get("suite_version", "0.0.0")
            info["description"] = suite.get("description", "")
            if isinstance(suite.get("cases"), list):
                info["case_count"] = len(suite["cases"])
            else:
                info["case_count"] = 0
            if errors:
                info["valid"] = False
                info["errors"] = errors
            else:
                info["valid"] = True
                info["suite_hash"] = compute_suite_hash(suite)
                info["prompt_template_hash"] = compute_prompt_template_hash(suite)
        except SuiteValidationError as exc:
            info["valid"] = False
            info["errors"] = [str(exc)]
            info["suite_id"] = "unknown"
            info["suite_version"] = "0.0.0"
            info["case_count"] = 0
        results.append(info)
    return results
