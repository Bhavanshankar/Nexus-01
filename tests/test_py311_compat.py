"""Python 3.11 compatibility guard (the project supports 3.11+; development uses 3.13).

PEP 701 (3.12) allows backslashes and same-quote nesting inside f-string replacement fields;
3.11 rejects both. This scans project sources with the tokenizer and fails on either pattern.
"""

import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SKIP = {"nexusvenv", ".venv", "venv", "__pycache__", "vector_store", ".git", "data", "logs", "knowledge_base"}
SOURCES = [p for p in ROOT.glob("*.py")] + [
    p for d in ROOT.iterdir() if d.is_dir() and d.name not in _SKIP for p in d.rglob("*.py") if "__pycache__" not in p.parts
]


def _violations(path: Path) -> list[str]:
    problems = []
    tokens = list(tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline))
    stack: list[str] = []  # quote style of each open f-string
    depth: list[int] = []  # brace depth inside the current f-string
    for tok in tokens:
        if tok.type == tokenize.FSTRING_START:
            quote = tok.string.lstrip("rRfFbBuU")
            if stack and depth[-1] > 0 and quote[0] == stack[-1][0]:
                problems.append(f"{path.name}:{tok.start[0]} nested f-string reuses quote {quote!r}")
            stack.append(quote)
            depth.append(0)
        elif tok.type == tokenize.FSTRING_END:
            stack.pop()
            depth.pop()
        elif stack:
            if tok.type == tokenize.OP and tok.string == "{":
                depth[-1] += 1
            elif tok.type == tokenize.OP and tok.string == "}":
                depth[-1] -= 1
            elif depth[-1] > 0 and tok.type == tokenize.STRING:
                if "\\" in tok.string:
                    problems.append(f"{path.name}:{tok.start[0]} backslash inside f-string expression")
                if tok.string.lstrip("rRbBuU")[0] == stack[-1][0]:
                    problems.append(f"{path.name}:{tok.start[0]} string reuses the enclosing f-string quote")
    return problems


def test_guard_detects_pep701_patterns(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        'ok = f"{\', \'.join([])}"\n'
        'bad1 = f"{"a"}"\n'
        "import re\nbad2 = f\"{re.sub(r'\\s', ' ', 'a b')}\"\n"
        "bad3 = f\"{f'{1}' + f'{'x'}'}\"\n",
        encoding="utf-8",
    )
    found = _violations(sample)
    assert not any(":1 " in p for p in found)
    assert any(":2 " in p for p in found) and any(":4 " in p for p in found) and any(":5 " in p for p in found)


def test_sources_avoid_pep701_only_fstring_syntax():
    problems = [p for path in SOURCES for p in _violations(path)]
    assert not problems, problems
