#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure-python grader for the held-out MiniMind evaluation benchmark.

No model / torch / numpy dependencies: this file must stay importable anywhere
(CI, reward functions, offline analysis) with only the Python stdlib.

Public API
----------
normalize(text)            -> cleaned completion text
extract(task, completion)  -> (canonical_answer_string | None, reason)
check(task, completion)    -> (bool, reason)
grade(task, completion)    -> dict-friendly result

Design notes (see reports/eval_harness.md for the full rationale)
----------------------------------------------------------------
* Completions are raw decoded model text: they may contain special tokens such
  as ``<|im_end|>`` / ``<|im_start|>`` / ``<|endoftext|>`` and thinking blocks.
  ``normalize`` removes both before matching.
* Unicode NFKC folds full-width digits/punctuation (７ -> 7, ＋ -> +, ？ -> ?)
  to their ASCII forms; Chinese numerals are accepted for ``exact_int`` only.
* Matching is deliberately strict about *what* is extracted but lenient about
  *how* the answer is wrapped (whitespace, trailing punctuation, a short
  "答案是 ..." prefix). Every task in heldout_tasks.jsonl has exactly one
  programmatically verifiable answer; there is no LLM judge and no fuzzy
  semantic matching.
* ``choice`` follows the benchmark spec: the first occurrence of any allowed
  choice string wins. Compliant answers ("只回答 A 或 C") are unambiguous;
  verbose non-compliant answers may be penalised, which is intended.
* ``exact_int`` extracts the *last* integer in the completion, so
  "3+5=8", "答案是 8", "8。" all grade as 8.
* ``exact_json`` extracts the first balanced ``{...}`` object, parses it with
  json (falling back to ast.literal_eval), and compares only the required keys
  named by ``task["json_keys"]`` (numeric strings are coerced).
* ``regex`` uses ``re.fullmatch`` on the normalized text, i.e. fixed-format
  tasks require the whole answer to match the pattern.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import unicodedata

__version__ = "1.0.0"
__all__ = [
    "normalize",
    "extract",
    "check",
    "grade",
    "validate_tasks",
    "TASK_FAMILIES",
    "CHECKERS",
]

TASK_FAMILIES = ("arithmetic", "format", "logic", "multistep")
CHECKERS = ("exact_int", "exact_json", "choice", "regex")

# Special tokens the MiniMind tokenizer can leak into decoded completions.
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|object_ref_start|object_ref_end|"
    r"box_start|box_end|quad_start|quad_end|vision_start|vision_end|"
    r"image_pad|video_pad)\|>"
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

# Trailing punctuation that models commonly append after a short answer.
_TRAILING_PUNCT = " \t\r\n.。,，;；:：!！?？、·…\"'`”’）)】]】》"

# "答案是 7" / "结果：7" / "= 7" style wrappers stripped before int parsing.
_WRAPPER_PREFIXES = (
    "最终答案", "最终结果", "答案是", "答案", "结果", "所以", "因此",
    "答", "result", "answer",
)
_WRAPPER_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in _WRAPPER_PREFIXES) + r")\s*[:：=是为]?\s*",
    re.IGNORECASE,
)

# Integer token inside free text. Allows "- 9" / "-9" (minus may be separated).
_INT_RE = re.compile(r"[-\u2212]?\s*\d+")

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_NEG = ("负", "負")


def normalize(text) -> str:
    """Normalize a raw model completion into a comparable string.

    Steps: drop special tokens, drop thinking content, NFKC, strip whitespace
    and trailing punctuation. Safe on ``None`` / non-str input.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SPECIAL_TOKEN_RE.sub("", text)
    # Keep only the final answer: text after the last closed </think>, and if a
    # <think> is still open, everything after it is discarded.
    if _THINK_CLOSE_RE.search(text):
        text = _THINK_CLOSE_RE.split(text)[-1]
    else:
        text = _THINK_OPEN_RE.split(text)[0]
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2212", "-")  # minus sign -> hyphen-minus
    text = text.strip()
    # Remove trailing punctuation repeatedly ("8.。" -> "8").
    while text and text[-1] in _TRAILING_PUNCT:
        text = text[:-1].rstrip()
    return text


def _strip_wrappers(text: str) -> str:
    """Strip leading '答案是' / '结果：' / '=' style wrappers (repeatedly)."""
    prev = None
    while prev != text:
        prev = text
        text = _WRAPPER_RE.sub("", text).strip()
        text = text.lstrip("=:：").strip()
    return text


def _parse_int_token(token: str):
    """Parse '- 9' / '-9' / ' 7' style token into an int."""
    token = token.replace(" ", "").replace("\u2212", "-").strip()
    try:
        return int(token)
    except (TypeError, ValueError):
        return None


def parse_chinese_int(text: str):
    """Parse a simple Chinese numeral (0..99, optional 负) or return None.

    Supported: 零 一 二 两 三 四 五 六 七 八 九 十, 十五, 二十, 二十三, 负七.
    Not a general Chinese-number parser; sufficient for the benchmark answers.
    """
    if not text:
        return None
    text = text.strip()
    neg = False
    if text and text[0] in _CN_NEG:
        neg, text = True, text[1:]
    if not text:
        return None
    if "十" in text:
        left, _, right = text.partition("十")
        if left == "":
            tens = 1
        elif len(left) == 1 and left in _CN_DIGITS:
            tens = _CN_DIGITS[left]
        else:
            return None
        if right == "":
            ones = 0
        elif len(right) == 1 and right in _CN_DIGITS:
            ones = _CN_DIGITS[right]
        else:
            return None
        value = tens * 10 + ones
    elif len(text) == 1 and text in _CN_DIGITS:
        value = _CN_DIGITS[text]
    elif len(text) == 2 and all(ch in _CN_DIGITS for ch in text):
        value = _CN_DIGITS[text[0]] * 10 + _CN_DIGITS[text[1]]
    else:
        return None
    return -value if neg else value


def _extract_int(task, completion):
    text = normalize(completion)
    if not text:
        return None, "empty_completion"
    candidates = [text, _strip_wrappers(text)]
    for cand in candidates:
        if not cand:
            continue
        compact = cand.replace(" ", "")
        m = re.fullmatch(r"[-\u2212]?\d+", compact)
        if m:
            value = _parse_int_token(m.group(0))
            if value is not None:
                return str(value), "integer:%s" % value
        value = parse_chinese_int(compact)
        if value is not None:
            return str(value), "chinese_numeral:%s" % value
    matches = _INT_RE.findall(text)
    if matches:
        value = _parse_int_token(matches[-1])
        if value is not None:
            return str(value), "last_integer:%s" % value
    return None, "no_integer_in_completion"


def _balanced_json_slice(text: str):
    """Return the first balanced {...} substring (string-aware), else None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_json_object(text: str):
    obj = None
    errors = []
    try:
        obj = json.loads(text)
    except Exception as exc:  # noqa: BLE001 - fall through to literal_eval
        errors.append("json:%s" % exc)
    if not isinstance(obj, dict):
        try:
            obj = ast.literal_eval(text)
        except Exception as exc:  # noqa: BLE001
            errors.append("literal_eval:%s" % exc)
    return (obj if isinstance(obj, dict) else None), errors


def _extract_json(task, completion):
    text = normalize(completion)
    if not text:
        return None, "empty_completion"
    blob = _balanced_json_slice(text)
    if blob is None:
        return None, "no_json_object"
    obj, errors = _parse_json_object(blob)
    if obj is None:
        return None, "json_parse_failed:" + "|".join(errors[:2])
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return canonical, "json_object:%s" % canonical


def _match_choice(task, completion):
    text = normalize(completion)
    if not text:
        return None, "empty_completion"
    choices = task.get("choices") or []
    if not choices:
        return None, "task_has_no_choices"
    lowered = text.casefold()
    best = None  # (index, order, choice)
    for order, choice in enumerate(choices):
        needle = str(choice).casefold()
        idx = lowered.find(needle)
        if idx < 0:
            continue
        cand = (idx, order, str(choice))
        if best is None or cand < best:
            best = cand
    if best is None:
        return None, "none_of_choices_present:%s" % (choices,)
    return best[2], "choice:%s" % best[2]


def _extract_regex(task, completion):
    text = normalize(completion)
    pattern = task.get("regex")
    if not pattern:
        return None, "task_has_no_regex"
    if not text:
        return None, "empty_completion"
    try:
        m = re.fullmatch(pattern, text)
    except re.error as exc:
        return None, "bad_regex:%s" % exc
    if not m:
        return None, "regex_fullmatch_failed"
    return text, "regex_match"


_EXTRACTORS = {
    "exact_int": _extract_int,
    "exact_json": _extract_json,
    "choice": _match_choice,
    "regex": _extract_regex,
}


def extract(task, completion):
    """Extract the canonical answer string from a completion.

    Returns ``(canonical_or_None, reason)``. Used by ``check`` and by
    run_eval's majority-vote mode.
    """
    checker = task.get("checker")
    fn = _EXTRACTORS.get(checker)
    if fn is None:
        return None, "unknown_checker:%r" % (checker,)
    try:
        return fn(task, completion)
    except Exception as exc:  # noqa: BLE001 - never crash an eval run
        return None, "checker_error:%s:%s" % (type(exc).__name__, exc)


def _coerce_equal(expected, actual):
    """Loose-but-safe equality for JSON values (numeric strings allowed)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        if isinstance(expected, bool) and isinstance(actual, bool):
            return expected == actual
        if isinstance(actual, str):
            low = actual.strip().casefold()
            if low in ("true", "yes", "是"):
                return expected is True
            if low in ("false", "no", "否"):
                return expected is False
        return False
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            return float(expected) == float(actual)
        if isinstance(actual, str):
            try:
                return float(expected) == float(actual.strip())
            except ValueError:
                return False
        return False
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()
    return expected == actual


def _check_json(task, completion):
    canonical, reason = _extract_json(task, completion)
    if canonical is None:
        return False, reason
    try:
        obj = json.loads(canonical)
    except Exception as exc:  # noqa: BLE001
        return False, "json_reparse_failed:%s" % exc
    try:
        expected = json.loads(task["answer"])
    except Exception as exc:  # noqa: BLE001
        return False, "bad_task_answer_json:%s" % exc
    keys = task.get("json_keys") or list(expected.keys())
    missing = [k for k in keys if k not in obj]
    if missing:
        return False, "missing_keys:%s" % (missing,)
    mismatched = []
    for key in keys:
        if not _coerce_equal(expected.get(key), obj.get(key)):
            mismatched.append("%s:%r!=%r" % (key, obj.get(key), expected.get(key)))
    if mismatched:
        return False, "wrong_values:" + ";".join(mismatched)
    return True, "json_ok:%s" % canonical


def check(task, completion):
    """Grade one completion. Returns ``(bool, reason)``.

    Never raises for bad completions; a malformed *task* still raises
    ValueError so benchmark corruption is loud.
    """
    checker = task.get("checker")
    if checker not in CHECKERS:
        raise ValueError("unknown checker %r for task %r" % (checker, task.get("id")))
    if checker == "exact_int":
        value, reason = _extract_int(task, completion)
        if value is None:
            return False, reason
        expected = str(int(task["answer"]))
        if value == expected:
            return True, "ok:%s" % value
        return False, "mismatch:got=%s expected=%s" % (value, expected)
    if checker == "exact_json":
        return _check_json(task, completion)
    if checker == "choice":
        value, reason = _match_choice(task, completion)
        if value is None:
            return False, reason
        expected = str(task["answer"]).casefold()
        if value.casefold() == expected:
            return True, "ok:%s" % value
        return False, "mismatch:got=%s expected=%s" % (value, task["answer"])
    if checker == "regex":
        value, reason = _extract_regex(task, completion)
        if value is None:
            return False, reason
        return True, "ok:%s" % value
    raise ValueError("unreachable checker %r" % (checker,))


def grade(task, completion):
    """Dict-friendly wrapper around ``check`` for report rows."""
    correct, reason = check(task, completion)
    return {
        "id": task.get("id"),
        "family": task.get("family"),
        "checker": task.get("checker"),
        "completion": completion,
        "correct": bool(correct),
        "reason": reason,
    }


# --------------------------------------------------------------------------
# Task-file validation
# --------------------------------------------------------------------------
def validate_tasks(tasks, expected_counts=None):
    """Validate the held-out task list. Returns list of error strings."""
    expected_counts = expected_counts or {"arithmetic": 8, "format": 6, "logic": 6, "multistep": 4}
    errors = []
    if not isinstance(tasks, list):
        return ["tasks is not a list"]
    if len(tasks) != sum(expected_counts.values()):
        errors.append("expected %d tasks, found %d" % (sum(expected_counts.values()), len(tasks)))
    seen_ids = set()
    counts = {k: 0 for k in expected_counts}
    for i, task in enumerate(tasks):
        where = "task[%d]" % i
        for field in ("id", "family", "prompt", "answer", "checker"):
            if field not in task:
                errors.append("%s missing field %r" % (where, field))
        tid = task.get("id")
        if tid in seen_ids:
            errors.append("%s duplicate id %r" % (where, tid))
        seen_ids.add(tid)
        fam = task.get("family")
        if fam not in expected_counts:
            errors.append("%s bad family %r" % (where, fam))
        else:
            counts[fam] += 1
        if task.get("held_out") is not True:
            errors.append("%s held_out is not true" % where)
        checker = task.get("checker")
        if checker not in CHECKERS:
            errors.append("%s bad checker %r" % (where, checker))
            continue
        if not isinstance(task.get("answer"), str):
            errors.append("%s answer is not a string" % where)
        if checker == "exact_int":
            try:
                int(task["answer"])
            except Exception:  # noqa: BLE001
                errors.append("%s exact_int answer not int: %r" % (where, task.get("answer")))
        elif checker == "exact_json":
            try:
                obj = json.loads(task["answer"])
            except Exception as exc:  # noqa: BLE001
                errors.append("%s exact_json answer invalid: %s" % (where, exc))
                obj = None
            if obj is not None:
                keys = task.get("json_keys")
                if not keys:
                    errors.append("%s exact_json missing json_keys" % where)
                else:
                    for k in keys:
                        if k not in obj:
                            errors.append("%s json_keys %r not in answer" % (where, k))
        elif checker == "choice":
            choices = task.get("choices") or []
            if len(choices) < 2:
                errors.append("%s choice needs >=2 choices" % where)
            if task.get("answer") not in choices:
                errors.append("%s choice answer %r not in choices" % (where, task.get("answer")))
        elif checker == "regex":
            pattern = task.get("regex")
            if not pattern:
                errors.append("%s regex missing pattern" % where)
            else:
                try:
                    if not re.fullmatch(pattern, task["answer"]):
                        errors.append("%s answer %r does not fullmatch regex %r" % (where, task["answer"], pattern))
                except re.error as exc:
                    errors.append("%s bad regex: %s" % (where, exc))
    for fam, expected in expected_counts.items():
        if counts.get(fam) != expected:
            errors.append("family %r has %d tasks, expected %d" % (fam, counts.get(fam), expected))
    return errors


# --------------------------------------------------------------------------
# Self-test: every case below is hand-written and MUST pass.
# --------------------------------------------------------------------------
def _self_test():
    import os

    failures = []
    checks = 0

    def expect(cond, label, detail=""):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append("%s %s" % (label, detail))

    def task(tid, checker, answer, **params):
        rec = {"id": tid, "family": "arithmetic", "prompt": "x", "answer": answer,
               "checker": checker, "held_out": True}
        rec.update(params)
        return rec

    t_int = task("t_int", "exact_int", "8")
    t_neg = task("t_neg", "exact_int", "-9")
    t_json = task("t_json", "exact_json", '{"sum": 9}', json_keys=["sum"])
    t_yn = task("t_yn", "choice", "NO", choices=["YES", "NO"])
    t_abc = task("t_abc", "choice", "B", choices=["A", "B", "C"])
    t_cn = task("t_cn", "choice", "奇数", choices=["奇数", "偶数"])
    t_re = task("t_re", "regex", "2+6=8", regex=r"2\s*\+\s*6\s*=\s*8")

    # --- normalize -------------------------------------------------------
    expect(normalize("  8  ") == "8", "normalize:strip")
    expect(normalize("8<|im_end|>") == "8", "normalize:special_token")
    expect(normalize("8。") == "8", "normalize:trailing_cn_period")
    expect(normalize("８") == "8", "normalize:fullwidth_digit")
    expect(normalize("<think>\n3+5\n</think>\n8") == "8", "normalize:think_block")
    expect(normalize("8<|im_end|>。\n") == "8", "normalize:mixed_tail")
    expect(normalize(None) == "", "normalize:none")
    expect(normalize("答案是 7。") == "答案是 7", "normalize:cn_period_after_space")

    # --- exact_int: passing ---------------------------------------------
    for completion, label in [
        ("8", "plain"),
        ("  8  ", "whitespace"),
        ("8.", "trailing_period"),
        ("8。", "trailing_cn_period"),
        ("8<|im_end|>", "im_end"),
        ("答案是 8", "answer_prefix"),
        ("答案是 8。", "answer_prefix_period"),
        ("3+5=8", "leaked_expression_last_int"),
        ("８", "fullwidth_digit"),
        ("八", "chinese_numeral"),
        ("答案是八", "chinese_with_prefix"),
        ("8<|endoftext|>", "eot"),
    ]:
        ok, reason = check(t_int, completion)
        expect(ok, "exact_int pass[%s]" % label, "completion=%r reason=%s" % (completion, reason))
    for completion, label in [
        ("-9", "plain"),
        ("- 9", "minus_space"),
        ("负九", "chinese_negative"),
        ("答案是 -9", "prefix"),
        ("0-9=-9", "expression"),
        ("−9", "unicode_minus"),
    ]:
        ok, reason = check(t_neg, completion)
        expect(ok, "exact_int neg pass[%s]" % label, "completion=%r reason=%s" % (completion, reason))

    # --- exact_int: failing ---------------------------------------------
    for completion, label in [
        ("7", "wrong_value"),
        ("", "empty"),
        ("答案是七", "chinese_wrong"),
        ("不知道", "refusal"),
        ("9", "off_by_one"),
    ]:
        ok, reason = check(t_int, completion)
        expect(not ok, "exact_int fail[%s]" % label, "completion=%r reason=%s" % (completion, reason))
    ok, _ = check(t_neg, "9")
    expect(not ok, "exact_int fail[sign_only]")

    # --- exact_json ------------------------------------------------------
    for completion, label in [
        ('{"sum": 9}', "plain"),
        ('  {"sum": 9}  ', "whitespace"),
        ('{"sum": 9}<|im_end|>', "im_end"),
        ('```json\n{"sum": 9}\n```', "code_fence"),
        ('{"sum": "9"}', "numeric_string_coerced"),
        ('{"sum": 9, "extra": 1}', "extra_keys_ignored"),
        ('答案是 {"sum": 9}', "prefix"),
        ('{"sum": 9}。', "trailing_period"),
    ]:
        ok, reason = check(t_json, completion)
        expect(ok, "exact_json pass[%s]" % label, "completion=%r reason=%s" % (completion, reason))
    for completion, label in [
        ('{"sum": 8}', "wrong_value"),
        ('{"total": 9}', "missing_key"),
        ("sum=9", "not_json"),
        ("", "empty"),
        ('{"sum": 9', "truncated"),
    ]:
        ok, reason = check(t_json, completion)
        expect(not ok, "exact_json fail[%s]" % label, "completion=%r reason=%s" % (completion, reason))

    # --- choice ----------------------------------------------------------
    for completion, label in [
        ("NO", "plain"),
        ("no", "lowercase"),
        ("答案是 NO", "prefix"),
        ("NO<|im_end|>", "im_end"),
        ("NO。", "trailing_period"),
    ]:
        ok, reason = check(t_yn, completion)
        expect(ok, "choice yn pass[%s]" % label, "completion=%r reason=%s" % (completion, reason))
    ok, _ = check(t_yn, "YES")
    expect(not ok, "choice yn fail[wrong]")
    ok, _ = check(t_yn, "maybe")
    expect(not ok, "choice yn fail[none_present]")
    for completion, label in [
        ("B", "plain"),
        ("答案是 B。", "prefix_period"),
        ("B<|im_end|>", "im_end"),
        ("b", "lowercase"),
    ]:
        ok, reason = check(t_abc, completion)
        expect(ok, "choice abc pass[%s]" % label, "completion=%r reason=%s" % (completion, reason))
    ok, _ = check(t_abc, "C")
    expect(not ok, "choice abc fail[wrong]")
    for completion, label in [
        ("奇数", "plain"),
        ("答案是奇数。", "prefix"),
        ("奇数<|im_end|>", "im_end"),
    ]:
        ok, reason = check(t_cn, completion)
        expect(ok, "choice cn pass[%s]" % label, "completion=%r reason=%s" % (completion, reason))
    ok, _ = check(t_cn, "偶数")
    expect(not ok, "choice cn fail[wrong]")

    # --- regex -----------------------------------------------------------
    for completion, label in [
        ("2+6=8", "plain"),
        ("2 + 6 = 8", "spaces"),
        ("2+6=8。", "trailing_period"),
        ("2+6=8<|im_end|>", "im_end"),
    ]:
        ok, reason = check(t_re, completion)
        expect(ok, "regex pass[%s]" % label, "completion=%r reason=%s" % (completion, reason))
    for completion, label in [
        ("2+6=9", "wrong"),
        ("答案是 2+6=8", "extra_text_breaks_fullmatch"),
        ("", "empty"),
    ]:
        ok, reason = check(t_re, completion)
        expect(not ok, "regex fail[%s]" % label, "completion=%r reason=%s" % (completion, reason))

    # --- grade() is dict-friendly ---------------------------------------
    row = grade(t_int, "8")
    expect(row.get("correct") is True and row.get("id") == "t_int", "grade:dict_shape")

    # --- validate the real held-out file when present --------------------
    here = os.path.dirname(os.path.abspath(__file__))
    tasks_path = os.path.join(here, "heldout_tasks.jsonl")
    if os.path.exists(tasks_path):
        with open(tasks_path, "r", encoding="utf-8") as fh:
            tasks = [json.loads(line) for line in fh if line.strip()]
        errors = validate_tasks(tasks)
        checks += 1
        if errors:
            failures.extend("heldout_tasks.jsonl: " + e for e in errors)
        else:
            expect(True, "heldout_tasks.jsonl valid (%d tasks)" % len(tasks))
        # every task's canonical answer must pass its own checker
        for t in tasks:
            if t.get("checker") == "exact_json":
                probe = t["answer"]
            else:
                probe = t["answer"]
            ok, reason = check(t, probe)
            checks += 1
            if not ok:
                failures.append("task %r answer %r fails its own checker: %s" % (t.get("id"), t["answer"], reason))
    else:
        print("[self-test] heldout_tasks.jsonl not found next to grader.py; skipping file validation")

    if failures:
        print("[self-test] FAILED %d/%d checks" % (len(failures), checks))
        for failure in failures:
            print("  - " + failure)
        return 1
    print("[self-test] OK: %d checks passed (grader v%s)" % (checks, __version__))
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
