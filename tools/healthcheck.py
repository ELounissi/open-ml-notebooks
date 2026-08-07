"""Check every notebook in the book against the rules in STYLE.md.

Run it from the repository root:

    python tools/healthcheck.py

It reports and never repairs. A defect it cannot describe precisely is a defect
it should not be silently fixing. Exit code is 1 when anything is wrong, so CI
can gate on it.

What it checks, and why each one has bitten this repo at least once:

    executed          a notebook committed mid-run ships with empty outputs, and
                      GitHub then renders a page with no charts on it
    no errors         a cell that raised is a cell whose conclusions are missing
    has figures       a chapter with no figure is a chapter nobody will read
    author block      every notebook carries the attribution
    prose rules       the word and construction lists from STYLE.md
    dashes            an em dash is the loudest signature a machine leaves on
                      prose; 892 had accumulated before anything counted them,
                      and a colon, a comma or a full stop says the same thing
    README present    the project page is what a search engine lands on
    image links       a README pointing at a figure that does not exist

Author: Elyes Lounissi
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# From STYLE.md. Single words that are almost always padding.
BANNED_WORDS = [
    "delve", "showcase", "testament", "tapestry", "meticulous", "intricate",
    "garner", "bolster", "boasts", "vibrant", "enduring", "interplay",
    "valuable insights", "pivotal", "underscore",
]

# The rhetorical constructions, matched as patterns so that an ordinary
# "not just the training portion" is left alone while "not just X, it is Y" is not.
BANNED_PATTERNS = [
    (r"not just\b[^.;]{0,60}?,?\s*(?:but|it'?s|it is)\b", "not just X, but Y"),
    (r"\bit'?s not\b[^.;]{0,40},\s*it'?s\b", "it's not X, it's Y"),
    (r"\bserves as\b", "copula avoidance: serves as"),
    (r"\bstands as\b", "copula avoidance: stands as"),
]

# From STYLE.md. Prose gets a colon, a comma, a semicolon or a full stop; a
# range gets the word "to".
#
# Code cells are checked too, and that is deliberate. A dash inside a chart title
# is drawn into the figure, and a dash inside a print string is drawn into the
# output a reader scrolls past. Both are prose as far as the reader is concerned,
# even though they live in code. Fixing one means re-running the notebook, which
# is the cost of having said it in a string.
BANNED_CHARACTERS = [
    ("—", "em dash"),
    ("–", "en dash"),
]

SKIP = ("_scratch", ".ipynb_checkpoints")


def markdown_of(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def check_punctuation(text: str, where: str, problems: list[str]) -> None:
    marks = [f"{name} x{text.count(char)}"
             for char, name in BANNED_CHARACTERS if char in text]
    if marks:
        problems.append(f"{where}: banned punctuation -> {', '.join(marks)}")


def check_prose(text: str, where: str, problems: list[str]) -> None:
    lowered = text.lower()
    hits = [word for word in BANNED_WORDS if word in lowered]
    if hits:
        problems.append(f"{where}: banned word(s) -> {', '.join(hits)}")
    for pattern, name in BANNED_PATTERNS:
        if re.search(pattern, lowered):
            problems.append(f"{where}: banned construction -> {name}")
    check_punctuation(text, where, problems)


def main() -> int:
    problems: list[str] = []
    notebooks = figures = readmes = 0
    seen: set[pathlib.Path] = set()

    for path in sorted(ROOT.rglob("notebook.ipynb")):
        if any(part in str(path) for part in SKIP):
            continue

        notebooks += 1
        where = path.parent.relative_to(ROOT).as_posix()
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]

        errors = sum(1 for c in notebook["cells"] for o in c.get("outputs", [])
                     if o.get("output_type") == "error")
        unexecuted = sum(1 for c in code if c.get("execution_count") is None)
        inline = sum(1 for c in notebook["cells"] for o in c.get("outputs", [])
                     if "image/png" in o.get("data", {}))

        if errors:
            problems.append(f"{where}: {errors} cell error(s)")
        if unexecuted:
            problems.append(f"{where}: {unexecuted} unexecuted code cell(s)")
        if inline == 0:
            problems.append(f"{where}: no inline figures")

        prose = markdown_of(notebook)
        if "Elyes Lounissi" not in prose:
            problems.append(f"{where}: missing author attribution")
        if "linkedin.com/in/elyes-lounissi" not in prose:
            problems.append(f"{where}: missing LinkedIn link")
        check_prose(prose, where, problems)

        # A dash in a chart title is drawn into the figure and a dash in a print
        # string is drawn into the output, so both reach the reader even though
        # they live in code. Punctuation only: the banned-word list would fire on
        # variable names and on the very lists this file defines.
        source = "\n".join("".join(c["source"]) for c in code)
        printed = "\n".join(
            "".join(o.get("text", "")) for c in code for o in c.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        check_punctuation(source, f"{where} (code)", problems)
        check_punctuation(printed, f"{where} (output)", problems)

        readme = path.parent / "README.md"
        if not readme.exists():
            problems.append(f"{where}: no README.md")
        else:
            readmes += 1
            seen.add(readme)
            text = readme.read_text(encoding="utf-8")
            if "Elyes Lounissi" not in text:
                problems.append(f"{where}/README.md: missing author attribution")
            for ref in re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", text):
                if not (path.parent / ref).exists():
                    problems.append(f"{where}/README.md: broken image link -> {ref}")
            check_prose(text, f"{where}/README.md", problems)

        folder = path.parent / "figures"
        if folder.is_dir():
            figures += len(list(folder.glob("*.png")))

    # The chapter pages, the curriculum and the style guide are prose too, and
    # they are where the em dashes were thickest.
    pages = 0
    for page in sorted(ROOT.rglob("*.md")):
        if page in seen or any(part in str(page) for part in SKIP):
            continue
        pages += 1
        where = page.relative_to(ROOT).as_posix()
        text = page.read_text(encoding="utf-8")
        # STYLE.md is where the banned words are written down, so only its
        # punctuation can be checked against itself.
        if where == "STYLE.md":
            check_punctuation(text, where, problems)
        else:
            check_prose(text, where, problems)

    print(f"notebooks : {notebooks}")
    print(f"READMEs   : {readmes}")
    print(f"other .md : {pages}")
    print(f"figures   : {figures}")
    print(f"problems  : {len(problems)}")
    for problem in problems:
        print("  -", problem)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
