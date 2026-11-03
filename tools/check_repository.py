"""Cheap repository contracts: Python syntax, semantic blocks and local Markdown links."""

import ast
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
# Document anchors are explicitly registered; they need not have Python BEGIN/END markers.
DOCUMENT_ANCHORS = {"ARTIFACTS:LAYOUT"}


def check(root=ROOT):
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode().split("\0")
    anchors, references, errors = Counter(), [], []
    for relative in sorted(set(paths) - {""}):
        path = root / relative
        if not path.is_file():
            continue
        if relative.startswith(("src/", "tools/")) and path.suffix == ".py":
            content = path.read_text(encoding="utf-8-sig")
            try:
                ast.parse(content, filename=relative)
            except SyntaxError as exc:
                errors.append(str(exc))
            stack = []
            for number, line in enumerate(content.splitlines(), 1):
                m = re.match(r"(\s*)# \[SEMANTIC-(BEGIN|END)\] (\S+)", line)
                if m:
                    indent, kind, anchor = m.groups()
                    if kind == "BEGIN":
                        anchors[anchor] += 1
                        stack.append((anchor, indent))
                    elif not stack or stack.pop() != (anchor, indent):
                        errors.append(f"{relative}:{number}: unmatched END/indent for {anchor}")
                if "# @see:" in line:
                    references.extend((relative, a) for a in re.findall(r"[A-Z][A-Z_]*:[A-Z][A-Z_]*", line))
            if stack:
                errors.append(f"{relative}: unclosed blocks {stack}")
        if path.suffix == ".md":
            # Ignore fenced examples; check local file targets, not heading spelling or external websites.
            content = re.sub(r"(?ms)^```.*?^```[^\n]*", "", path.read_text(encoding="utf-8-sig"))
            for target in re.findall(r"\]\(([^)]+)\)", content):
                target = target.strip("<>").split("#", 1)[0]
                if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue
                if not (path.parent / unquote(target)).exists():
                    errors.append(f"{relative}: broken local link {target}")
    errors += [f"duplicate semantic anchor: {a}" for a, count in anchors.items() if count > 1]
    errors += [f"{p}: unknown @see {a}" for p, a in references if a not in anchors and a not in DOCUMENT_ANCHORS]
    return errors, len(anchors)


if __name__ == "__main__":
    errors, count = check()
    print("\n".join(errors) if errors else f"Repository contracts OK ({count} semantic anchors)")
    raise SystemExit(bool(errors))
