"""Dataset access, with the provenance rule built in.

Every file this module writes lands under `data/raw/<dataset>/` next to a
`SOURCE.txt` recording where it came from, when, and under what licence. That
file is what makes the repo auditable months later, when the URL has moved and
nobody remembers which version was downloaded.

Nothing here is clever. Downloads are cached by path, so re-running a notebook
does not re-hit the source, and a notebook that is offline still executes if the
data is already on disk.

Author: Elyes Lounissi
"""

from __future__ import annotations

import datetime as _dt
import urllib.request
from pathlib import Path

# A plain urllib request is refused by a few dataset hosts. This is the only
# reason a user agent is set.
_AGENT = "open-ml-notebooks/1.0 (+https://github.com/ELounissi/open-ml-notebooks)"


def repo_root() -> Path:
    """Walk up from this file until the directory holding `toolkit/` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "toolkit").is_dir():
            return parent
    raise RuntimeError("repo root not found above toolkit/data.py")


def raw_dir(dataset: str) -> Path:
    path = repo_root() / "data" / "raw" / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(url: str, dataset: str, filename: str, *, licence: str,
          source_page: str = "", force: bool = False) -> Path:
    """Download `url` into `data/raw/<dataset>/<filename>` once, and record it.

    Returns the local path. If the file is already there, the network is not
    touched, so an offline re-run of the notebook still works.

    `licence` is required rather than optional. A dataset whose licence nobody
    wrote down is a dataset that cannot be redistributed with confidence.
    """
    target = raw_dir(dataset) / filename
    if target.exists() and not force:
        return target

    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)

    _record(dataset, filename, url, licence, source_page, len(payload))
    return target


def _record(dataset: str, filename: str, url: str, licence: str,
            source_page: str, n_bytes: int) -> None:
    """Append one provenance line to the dataset's SOURCE.txt."""
    stamp = _dt.date.today().isoformat()
    lines = [
        f"file        : {filename}",
        f"retrieved   : {stamp}",
        f"size        : {n_bytes / 1e6:.2f} MB",
        f"url         : {url}",
        f"licence     : {licence}",
    ]
    if source_page:
        lines.append(f"source page : {source_page}")
    lines.append("")

    note = raw_dir(dataset) / "SOURCE.txt"
    with note.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def size_on_disk(dataset: str) -> float:
    """Megabytes currently committed for a dataset. Used to police the 20 MB rule."""
    total = sum(p.stat().st_size for p in raw_dir(dataset).rglob("*") if p.is_file())
    return total / 1e6
