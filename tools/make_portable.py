"""Make the published scripts portable: replace hardcoded `D:\\llm-rl-lab` paths.

The originals were written for one machine. For a GitHub repo they must derive the
project root from the script's own location, otherwise nobody can reproduce anything.

Strategy: for every `X = r'D:\\llm-rl-lab\\...'` (or `X = 'D:/llm-rl-lab/...'`) assignment,
rewrite it as an expression relative to a computed PROJECT_ROOT. A `PROJECT_ROOT` block is
inserted once per file, right after the imports, if not already present.

Usage: python make_portable.py --repo D:/llm-rl-lab/github-repo
"""
import argparse
import os
import re
import sys

ROOT_BLOCK = '''# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
'''

# matches: NAME = r'D:\llm-rl-lab\sub\path'   or   NAME = 'D:/llm-rl-lab/sub'
ASSIGN_RE = re.compile(
    r"""^(?P<indent>\s*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<q>r?['"])"""
    r"""D:[\\/]llm-rl-lab(?P<tail>[^'"]*)(?P=q)\s*$""",
    re.MULTILINE)


def convert(text, depth):
    """depth = how many directories up from the file to the project root."""
    up = '' if depth == 0 else '..' + '/..' * (depth - 1)
    changed = 0

    def repl(m):
        nonlocal changed
        tail = m.group('tail').replace('\\', '/').strip('/')
        indent, name = m.group('indent'), m.group('name')
        if not tail:
            expr = f"PROJECT_ROOT"
        else:
            expr = f"os.path.join(PROJECT_ROOT, {', '.join(repr(p) for p in tail.split('/'))})"
        changed += 1
        return f'{indent}{name} = {expr}'

    out = ASSIGN_RE.sub(repl, text)
    if changed and 'PROJECT_ROOT = ' not in out:
        # Insert the root block after the LAST module-level import (indent 0 only).
        # Scanning the first 80 lines blindly put it inside a function body and broke
        # indentation (IndentationError at the next indented statement).
        lines = out.split('\n')
        last_import = None
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            if ln.startswith(('import ', 'from ')):
                last_import = i
            elif ln[0] not in (' ', '\t', ')', ']', '}') and not ln.startswith(('#', '"', "'")):
                # first real statement after the import block -> stop looking
                if last_import is not None and i > last_import:
                    break
        if last_import is None:
            last_import = 0
        lines.insert(last_import + 1, '')
        lines.insert(last_import + 2, ROOT_BLOCK.rstrip())
        out = '\n'.join(lines)
        # ensure os is imported
        if not re.search(r'^\s*import os\b', out, re.MULTILINE):
            if re.search(r'^\s*import sys\b', out, re.MULTILINE):
                out = re.sub(r'^(\s*)import sys\b', r'\1import os\n\1import sys', out,
                             count=1, flags=re.MULTILINE)
            else:
                out = 'import os\n' + out
    return out, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(PROJECT_ROOT, 'github-repo'))
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)

    files = []
    for dirpath, _dirnames, filenames in os.walk(repo):
        for fn in filenames:
            if fn.endswith('.py'):
                files.append(os.path.join(dirpath, fn))

    total = 0
    for f in sorted(files):
        rel = os.path.relpath(f, repo)
        depth = rel.count(os.sep)          # e.g. custom/x.py -> 1
        with open(f, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        new, n = convert(text, depth)
        if n:
            with open(f, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(new)
            total += n
            print(f'  {rel:<50} {n} replacement(s)')
    print(f'\n{total} hardcoded path(s) rewritten across {len(files)} files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
