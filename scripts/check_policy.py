#!/usr/bin/env python3
"""Repository policy gate (pre-commit and CI).

1. Single-head generator (docs/adr/0002-single-head-generator.md): code must emit a single output
   tensor without spatial mask heads, spatial=True LPIPS maps, or external pattern banks.
2. Vocabulary (CONTEXT.md): the project's language is protective pattern, unauthorized style
   mimicry and transformation filter. Third-party names and quoted titles may stay on a line
   that carries the marker `policy: allow-term`.

Usage: check_policy.py [FILE ...]   (default: every git-tracked text file)
"""

import re
import subprocess
import sys
from pathlib import Path

CODE_RULES = [
    (re.compile(r"\b\w*(spatial_mask|region_mask|texture_mask|edge_mask|mask_head|routing_head)\w*", re.I),
     "spatial mask or routing head"),
    (re.compile(r"spatial\s*=\s*True"), "spatial LPIPS map"),
    (re.compile(r"\b\w*(pattern|perturbation|delta)_bank\w*", re.I), "stored pattern bank"),
]  # fmt: skip
TERM_RULES = [
    (re.compile(r"adversar", re.I), "say 'protective pattern' / 'protective signal'"),
    (re.compile(r"\battack", re.I), "say 'unauthorized style mimicry' or name the mimicry recipe"),
    (re.compile(r"purif", re.I), "say 'transformation filter'"),
    (re.compile(r"perturb", re.I), "say 'protective pattern'"),
    (re.compile(r"적대적|섭동|공격|정화"), "보호 패턴 / 무단 화풍 도용 / 전처리 변환 필터"),
]
ALLOW = "policy: allow-term"
SKIP = {"LICENSE", "NOTICE", "uv.lock", "CONTEXT.md", "scripts/check_policy.py", "tests/test_policy.py"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".txt", ".json", ".cfg", ".ini", ""}


def violations(path: str, text: str) -> list[str]:
    out = []
    is_code = path.endswith(".py")
    for n, line in enumerate(text.splitlines(), 1):
        if is_code:
            out += [f"{path}:{n}: {why}: {m.group(0)}" for rx, why in CODE_RULES if (m := rx.search(line))]
        if ALLOW not in line:
            out += [f"{path}:{n}: wording ({why}): {m.group(0)}" for rx, why in TERM_RULES if (m := rx.search(line))]
    return out


def tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()


def main(argv: list[str]) -> int:
    files = [f for f in (argv or tracked()) if f not in SKIP and Path(f).suffix in TEXT_SUFFIXES]
    found = []
    for f in files:
        try:
            found += violations(f, Path(f).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
    print("\n".join(found))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
