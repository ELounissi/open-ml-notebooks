"""Build the blast-versus-earthquake catalogue used by project 01-01.

Source
    USGS Earthquake Hazards Program, FDSN event web service.
    https://earthquake.usgs.gov/fdsnws/event/1/
    Works of the U.S. Geological Survey are in the public domain.

What this builds
    Every explosion the USGS catalogued in the contiguous United States between
    2015 and 2025 at magnitude 1.0 and above, matched one-for-one against
    naturally occurring earthquakes drawn from the same months.

Why matched within months rather than sampled globally
    Explosions are a working-hours activity and their yearly count drifts with
    construction demand, while earthquake counts are dominated by a handful of
    aftershock sequences. Sampling earthquakes globally would let a model score
    well by learning "2019 was a quiet quarry year", which is a fact about the
    sampling and not about the events. Drawing an equal number of earthquakes
    from each month removes that shortcut: within any month the two classes are
    exactly balanced, so calendar position carries no information about the label.

Runtime is a few minutes, almost all of it waiting on the service. The result is
written once to data/raw/usgs-blasts/events.csv and the notebook reads that.

Usage
    python data/fetch/usgs_blasts_vs_quakes.py

Author: Elyes Lounissi
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolkit import data as tkdata  # noqa: E402

SERVICE = "https://earthquake.usgs.gov/fdsnws/event/1/"
AGENT = {"User-Agent": "open-ml-notebooks/1.0 (+github.com/ELounissi/open-ml-notebooks)"}

# Contiguous United States. Alaska is excluded because its seismicity is an
# order of magnitude denser than anywhere else and would dominate the sample.
REGION = dict(minlatitude=24, maxlatitude=50, minlongitude=-125, maxlongitude=-66)

START, END = "2015-01-01", "2025-01-01"
MIN_MAG, MAX_MAG = 1.0, 4.2          # 4.1 is the largest catalogued blast
BLAST_TYPES = ["quarry blast", "mining explosion"]
SERVICE_LIMIT = 20_000               # hard cap the FDSN endpoint enforces

KEEP = [
    "time", "latitude", "longitude", "depth", "mag", "magType",
    "nst", "gap", "dmin", "rms", "horizontalError", "depthError",
    "magError", "magNst", "net", "id", "place", "type", "status",
    "locationSource", "magSource",
]


def _get(path: str, **params) -> bytes:
    url = SERVICE + path + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers=AGENT)
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def count(**params) -> int:
    """Ask how many events a query would return, without running it."""
    try:
        return json.loads(_get("count", format="geojson", **params))["count"]
    except urllib.error.HTTPError as err:            # over the limit, body still has the count
        return json.loads(err.read())["count"]


def query(**params) -> pd.DataFrame:
    raw = _get("query", format="csv", orderby="time-asc", **params)
    return pd.read_csv(io.StringIO(raw.decode("utf-8")))


def fetch_window(start: str, end: str, **params) -> pd.DataFrame:
    """Download a time window, halving it whenever the service cap is hit.

    Large aftershock sequences make event density wildly uneven in time, so a
    fixed chunk size either wastes requests or blows the cap. Splitting on
    demand handles both.
    """
    n = count(starttime=start, endtime=end, **params)
    if n == 0:
        return pd.DataFrame(columns=KEEP)
    if n <= SERVICE_LIMIT:
        return query(starttime=start, endtime=end, **params)

    midpoint = (pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2).floor("D")
    mid = midpoint.strftime("%Y-%m-%d")
    print(f"    {start}..{end} holds {n:,} events, splitting at {mid}")
    left = fetch_window(start, mid, **params)
    right = fetch_window(mid, end, **params)
    return pd.concat([left, right], ignore_index=True)


def main() -> None:
    rng = np.random.default_rng(20260806)

    print("Downloading explosions ...")
    blasts = []
    for event_type in BLAST_TYPES:
        frame = fetch_window(
            START, END, eventtype=event_type,
            minmagnitude=MIN_MAG, maxmagnitude=MAX_MAG, **REGION,
        )
        print(f"  {event_type:18s} {len(frame):6,}")
        blasts.append(frame)
    blast = pd.concat(blasts, ignore_index=True)
    blast["label"] = 1

    print("\nDownloading earthquakes month by month ...")
    months = pd.date_range(START, END, freq="MS", tz="UTC")
    blast_time = pd.to_datetime(blast["time"], format="ISO8601", utc=True)

    picked = []
    for start, end in zip(months[:-1], months[1:]):
        wanted = int(((blast_time >= start) & (blast_time < end)).sum())
        if wanted == 0:
            continue
        pool = fetch_window(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), eventtype="earthquake",
            minmagnitude=MIN_MAG, maxmagnitude=MAX_MAG, **REGION,
        )
        take = min(wanted, len(pool))
        picked.append(pool.iloc[rng.choice(len(pool), size=take, replace=False)])
        print(f"  {start:%Y-%m}  quarry {wanted:4,}  available {len(pool):6,}  kept {take:4,}")

    quake = pd.concat(picked, ignore_index=True)
    quake["label"] = 0

    events = pd.concat([blast, quake], ignore_index=True)[KEEP + ["label"]]
    events = events.sort_values("time").reset_index(drop=True)

    out = tkdata.raw_dir("usgs-blasts") / "events.csv"
    events.to_csv(out, index=False)

    (tkdata.raw_dir("usgs-blasts") / "SOURCE.txt").write_text(
        "dataset    : USGS catalogued explosions vs earthquakes, contiguous US, 2015-2025\n"
        f"built      : {pd.Timestamp.today():%Y-%m-%d} by data/fetch/usgs_blasts_vs_quakes.py\n"
        "service    : https://earthquake.usgs.gov/fdsnws/event/1/\n"
        "licence    : public domain (work of the U.S. Geological Survey)\n"
        "citation   : U.S. Geological Survey, Earthquake Hazards Program, "
        "Advanced National Seismic System (ANSS) Comprehensive Catalog.\n"
        f"rows       : {len(events):,}  ({int(events.label.sum()):,} explosions, "
        f"{int((1 - events.label).sum()):,} earthquakes)\n"
        f"size       : {out.stat().st_size / 1e6:.2f} MB\n"
        "sampling   : all explosions kept; earthquakes drawn at random from the same\n"
        "             calendar month, equal count, seed 20260806\n",
        encoding="utf-8",
    )

    print(f"\nWrote {out}  ({len(events):,} rows, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
