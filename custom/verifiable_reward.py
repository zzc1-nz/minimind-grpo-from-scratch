"""Programmatically-verifiable reward for MiniMind GRPO. No LLM judge, no learned reward model.

Every task carries an explicit `checker` + params, and the checker is a pure function
completion -> bool. This is the same contract used by the held-out benchmark, so the
training reward and the evaluation metric agree.

Checkers
  exact_int   : last integer appearing in the completion (optional leading '-') == answer
  exact_json  : first {...} block parses as JSON and its required keys match answer
  choice      : first standalone occurrence of one of `choices` (e.g. A/B/C)
  label       : completion, after stripping wrappers, equals one of `labels` (YES/NO, 奇数/偶数)
"""
import json
import re

_WRAPPERS = ['<|im_end|>', '<|endoftext|>', '<|im_start|>', '</think>', '<think>']
_FULLWIDTH = str.maketrans('０１２３４５６７８９．，：；（）', '0123456789.,:;()')

# Unicode minus signs the model can emit instead of ASCII '-'. Left un-normalised they
# make "-9" parse as "9" and silently fail a correct answer.
_MINUS = str.maketrans('\u2212\u2013\u2014\uff0d', '----')

# Chinese numerals. A 64M model trained on Chinese text does emit these, and the
# held-out grader accepts them, so the training reward must agree or GRPO is punished
# for a correct answer.
_CN_DIGITS = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_CN_UNITS = {'十': 10, '百': 100, '千': 1000}


def _cn_to_int(s: str):
    """Parse a simple Chinese numeral (0..9999, optional leading 负). Returns int|None."""
    if not s:
        return None
    neg = False
    if s.startswith('负'):
        neg = True
        s = s[1:]
    if not s:
        return None
    if all(ch in _CN_DIGITS for ch in s):
        # plain digit-by-digit form, e.g. 一二三
        val = 0
        for ch in s:
            val = val * 10 + _CN_DIGITS[ch]
        return -val if neg else val
    # unit form, e.g. 十 / 十八 / 二十 / 一百零五
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if number == 0:
                number = 1  # 十 alone means ten
            section += number * unit
            number = 0
        else:
            return None
    total = section + number
    return -total if neg else total


def normalize(text: str) -> str:
    """Strip chat wrappers / whitespace / full-width punctuation noise."""
    if text is None:
        return ''
    s = text
    for w in _WRAPPERS:
        s = s.replace(w, ' ')
    s = s.translate(_FULLWIDTH)
    s = s.translate(_MINUS)
    s = s.replace('\u3000', ' ')
    return s.strip()


def _clean(s: str) -> str:
    s = normalize(s)
    s = s.strip(' \t\r\n.。,，!！?？:：;；')
    return s


def _find_last_int(s: str):
    # accept forms like "8", "-9", "答案是 7", "= 12", and "- 9" (space after the sign,
    # which the model produces often; without allowing it "-9" parses as "9").
    nums = re.findall(r'-?\s?\d+', s)
    if nums:
        return int(nums[-1].replace(' ', '').replace('\u3000', ''))
    # fall back to a Chinese numeral if the completion contains no ASCII digits
    m = re.search(r'负?[零〇一二两三四五六七八九十百千]+', s)
    if m:
        v = _cn_to_int(m.group(0))
        if v is not None:
            return v
    return None


def _find_first_json(s: str):
    m = re.search(r'\{.*?\}', s, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # The model often emits a Python-dict style literal with single quotes; the grader
    # accepts it, so accept it here too.
    try:
        import ast
        v = ast.literal_eval(raw)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def check(task: dict, completion: str):
    """Return (correct: bool, reason: str)."""
    checker = task.get('checker')
    ans = task.get('answer')
    text = _clean(completion)

    if checker == 'exact_int':
        got = _find_last_int(text)
        try:
            want = int(str(ans).strip())
        except Exception:
            return False, 'bad_task_answer'
        if got is None:
            return False, 'no_integer_found'
        return got == want, f'got={got} want={want}'

    if checker == 'exact_json':
        obj = _find_first_json(text)
        if obj is None:
            return False, 'no_json_found'
        try:
            want = json.loads(ans) if isinstance(ans, str) else ans
        except Exception:
            return False, 'bad_task_answer'
        if not isinstance(obj, dict):
            return False, 'json_not_object'
        for k, v in want.items():
            if k not in obj:
                return False, f'missing_key={k}'
            if str(obj[k]).strip() != str(v).strip():
                return False, f'key={k} got={obj[k]} want={v}'
        return True, 'json_ok'

    if checker == 'choice':
        # Must match eval/grader.py semantics exactly:
        #   * the FIRST occurrence of any allowed choice wins (not the first choice in
        #     list order -- "NO, actually YES" must resolve to NO, as the grader does);
        #   * ASCII comparisons are case-insensitive ("b", "yes" are valid answers).
        # Getting either wrong means GRPO is rewarded for output the evaluator rejects,
        # or punished for output it accepts.
        choices = task.get('choices') or list('ABC')
        low = text.casefold()
        best = None  # (position, choice)
        for c in choices:
            cs = str(c)
            pat = (re.escape(cs.casefold()) if cs.isascii()
                   else re.escape(cs))
            m = re.search(r'(?<![A-Za-z0-9])' + pat + r'(?![A-Za-z0-9])',
                          low if cs.isascii() else text)
            if m:
                pos = m.start()
                if best is None or pos < best[0]:
                    best = (pos, cs)
        if best is None:
            return False, 'no_choice_found'
        return best[1] == str(ans).strip(), f'got={best[1]} want={ans}'

    if checker == 'label':
        # Same first-occurrence rule as `choice`; case-insensitive for ASCII labels.
        labels = task.get('labels') or [str(ans)]
        low = text.casefold()
        best = None
        for lb in labels:
            lbs = str(lb)
            pat = (re.escape(lbs.casefold()) if lbs.isascii() else re.escape(lbs))
            m = re.search(pat, low if lbs.isascii() else text)
            if m:
                if best is None or m.start() < best[0]:
                    best = (m.start(), lbs)
        if best is None:
            return False, 'no_label_found'
        return best[1] == str(ans).strip(), f'got={best[1]} want={ans}'

    if checker == 'regex':
        # Fixed-format answers ("a+b=c"). The held-out grader uses this checker, so the
        # training reward must support it too -- otherwise GRPO cannot be trained on
        # fixed-format tasks at all, and any such task is scored inconsistently between
        # training and evaluation.
        pat = task.get('regex')
        if not pat:
            return False, 'bad_task_regex'
        try:
            # fullmatch on the cleaned text, exactly as grader.py does
            return bool(re.fullmatch(pat, text)), f'regex_fullmatch={pat!r}'
        except re.error as exc:
            return False, f'bad_task_regex:{exc}'

    return False, f'unknown_checker={checker}'


def reward(task: dict, completion: str, mode: str = 'binary'):
    """Scalar reward. mode=binary -> 1/0 exactly as specified; mode=shaped -> +0.1 for
    a *format-valid but wrong* answer (still fully deterministic, no model judge)."""
    ok, reason = check(task, completion)
    if mode == 'binary':
        return 1.0 if ok else 0.0, ok, reason
    if ok:
        return 1.0, ok, reason
    # shaped: did the model at least emit something the checker could parse?
    text = _clean(completion)
    checker = task.get('checker')
    parsed = False
    if checker == 'exact_int':
        parsed = _find_last_int(text) is not None
    elif checker == 'exact_json':
        parsed = _find_first_json(text) is not None
    elif checker == 'choice':
        choices = task.get('choices') or list('ABC')
        parsed = any(re.search(r'(?<![A-Za-z])' + re.escape(c) + r'(?![A-Za-z])', text) for c in choices)
    elif checker == 'label':
        labels = task.get('labels') or [str(task.get('answer'))]
        parsed = any(lb in text for lb in labels)
    return (0.1 if parsed else 0.0), ok, reason


if __name__ == '__main__':
    cases = [
        ({'checker': 'exact_int', 'answer': '8'}, '8', True),
        ({'checker': 'exact_int', 'answer': '8'}, '答案是 8。<|im_end|>', True),
        ({'checker': 'exact_int', 'answer': '8'}, '7', False),
        ({'checker': 'exact_int', 'answer': '-5'}, '-5', True),
        ({'checker': 'exact_int', 'answer': '-5'}, '5', False),
        ({'checker': 'exact_json', 'answer': '{"sum": 8}'}, '{"sum": 8}', True),
        ({'checker': 'exact_json', 'answer': '{"sum": 8}'}, '{"sum": 9}', False),
        ({'checker': 'exact_json', 'answer': '{"sum": 8}'}, '结果如下 {"sum":8} 完毕', True),
        ({'checker': 'choice', 'answer': 'B', 'choices': ['A', 'B', 'C']}, 'B', True),
        ({'checker': 'choice', 'answer': 'B', 'choices': ['A', 'B', 'C']}, '答案是 B。', True),
        ({'checker': 'choice', 'answer': 'B', 'choices': ['A', 'B', 'C']}, 'A', False),
        ({'checker': 'label', 'answer': 'YES', 'labels': ['YES', 'NO']}, 'YES', True),
        ({'checker': 'label', 'answer': 'YES', 'labels': ['YES', 'NO']}, 'NO', False),
        ({'checker': 'label', 'answer': '奇数', 'labels': ['奇数', '偶数']}, '奇数', True),
    ]
    bad = 0
    for task, comp, want in cases:
        ok, reason = check(task, comp)
        flag = 'ok ' if ok == want else 'FAIL'
        if ok != want:
            bad += 1
        print(f'{flag} want={want} got={ok} | {comp!r} -> {reason}')
    print('SELF-TEST', 'PASS' if bad == 0 else f'{bad} FAILURES')
