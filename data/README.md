# Data

Every dataset used anywhere in this repository is listed here with its source and
its licence. If a dataset is not in this table, no notebook is allowed to use it.

## The rules

**Open licences only.** Each entry names the licence. Where a licence forbids
redistribution, the file is not committed and a script under [`fetch/`](fetch/)
rebuilds it.

**Twenty megabytes.** A dataset under 20 MB is committed to `raw/` so the repo
clones and runs with no network. Anything larger gets a fetch script instead. The
point is that a reader can open a notebook and see real output without a download
step, which is also why notebooks ship executed.

**Provenance is written by the code, not by me.** `toolkit.data.fetch` appends a
`SOURCE.txt` next to every file it downloads, recording the URL, the retrieval
date, the byte count, and the licence. Scripts under `fetch/` write the same
record. A dataset whose origin nobody wrote down cannot be defended later.

**Raw stays raw.** Nothing in `raw/` is edited by hand. Cleaning happens in the
notebook, in view, so a reader can disagree with a decision and change it.

**Nothing is chosen for convenience.** Titanic, Iris, and plain MNIST are not the
subject of any project here. A famous dataset is fair game when the angle on it
is not the one everybody uses; a neglected or recent dataset is better.

## What is here

| Dataset | Used by | Source | Licence | In repo | Size |
|---|---|---|---|---|---|
| USGS catalogued explosions vs earthquakes, contiguous US, 2015–2025 | [01-01](../01-foundations/01-01-three-splits-three-truths/) | [USGS FDSN event service](https://earthquake.usgs.gov/fdsnws/event/1/) | Public domain (U.S. Geological Survey) | yes, `raw/usgs-blasts/` | ~5 MB |

The table grows as projects ship. Datasets named in [`ROADMAP.md`](../ROADMAP.md)
but not listed above have been chosen and not yet pulled.

## Rebuilding anything

```bash
python data/fetch/usgs_blasts_vs_quakes.py
```

Each script is idempotent, writes only under `data/raw/<dataset>/`, and states at
the top what it builds and why the construction is the way it is.

## Citing the sources

**USGS explosions and earthquakes.** U.S. Geological Survey, Earthquake Hazards
Program, Advanced National Seismic System (ANSS) Comprehensive Catalog. Retrieved
through the FDSN event web service. Works of the U.S. Geological Survey are in
the public domain; attribution is courtesy, not a requirement.
