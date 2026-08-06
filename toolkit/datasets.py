"""The house datasets.

Every notebook in this book draws from the same handful of datasets. That is the
whole point: when you see a support vector machine score 0.93 in one chapter and
a random forest score 0.94 in another, the two numbers are comparable, because
the question was identical.

Each loader returns a plain `(X, y)` pair of pandas objects and caches its file
under `data/raw/`, so a notebook runs offline once the data is on disk.

    from toolkit import datasets
    X, y = datasets.dry_bean()

Made by Elyes Lounissi
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from . import data as _data

__all__ = ["california_housing", "breast_cancer", "dry_bean", "bike_sharing",
           "fashion_mnist", "HOUSE_DATASETS"]

HOUSE_DATASETS = {
    "california-housing": "Median house value across California districts. Regression, 20,640 rows.",
    "breast-cancer": "Malignant or benign from cell nucleus measurements. Binary, 569 rows.",
    "dry-bean": "Seven varieties of dry bean from grain images. Multiclass, 13,611 rows.",
    "bike-sharing": "Hourly bicycle hire counts in Washington DC. Regression and time series, 17,379 rows.",
    "fashion-mnist": "Ten clothing categories as 28x28 greyscale images. Image classification, 70,000 rows.",
}


def _cached(name: str, filename: str, build) -> pd.DataFrame:
    """Return a cached CSV, building it once if it is not there yet."""
    target = _data.raw_dir(name) / filename
    if not target.exists():
        frame = build()
        frame.to_csv(target, index=False)
    return pd.read_csv(target)


def california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """Median house value per district, from the 1990 US census.

    Regression. The target is in hundreds of thousands of dollars and is capped
    at 5.0, which matters: roughly a thousand districts sit exactly on that cap,
    and any model will look bad there. The notebooks point this out where it
    affects a result.

    Source: scikit-learn, derived from StatLib. Public domain.
    """
    def build():
        from sklearn.datasets import fetch_california_housing
        return fetch_california_housing(as_frame=True).frame

    frame = _cached("california-housing", "california_housing.csv", build)
    return frame.drop(columns="MedHouseVal"), frame["MedHouseVal"]


def breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    """Malignant versus benign, from measurements of cell nuclei.

    Binary classification, 569 rows and 30 numeric features. Small enough that
    every method in this book trains on it in under a second, which is why it
    appears whenever a chapter needs a quick demonstration rather than a
    challenge. Target is 1 for benign, 0 for malignant.

    Source: UCI, Wisconsin Diagnostic Breast Cancer. CC BY 4.0.
    """
    def build():
        from sklearn.datasets import load_breast_cancer
        bundle = load_breast_cancer(as_frame=True)
        return bundle.frame

    frame = _cached("breast-cancer", "breast_cancer.csv", build)
    return frame.drop(columns="target"), frame["target"]


def dry_bean() -> tuple[pd.DataFrame, pd.Series]:
    """Seven varieties of dry bean, measured from photographs of the grains.

    Multiclass classification, 13,611 rows and 16 shape features. Published in
    2020 and still uncommon in tutorials, which is why it is the workhorse of
    this book: results on it are not something you have already read elsewhere.

    The classes are unbalanced, from 522 Bombay to 3,546 Dermason, so it is also
    a natural place to talk about what accuracy hides.

    Source: UCI Machine Learning Repository, id 602. CC BY 4.0.
    """
    def build():
        payload = _data.fetch(
            "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
            "dry-bean", "dry_bean_dataset.zip",
            licence="CC BY 4.0",
            source_page="https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
        )
        with zipfile.ZipFile(payload) as archive:
            inner = next(n for n in archive.namelist() if n.endswith(".xlsx"))
            with archive.open(inner) as handle:
                return pd.read_excel(io.BytesIO(handle.read()))

    frame = _cached("dry-bean", "dry_bean.csv", build)
    return frame.drop(columns="Class"), frame["Class"]


def bike_sharing() -> tuple[pd.DataFrame, pd.Series]:
    """Hourly bicycle hire counts in Washington DC, 2011 and 2012.

    Regression, 17,379 rows. It carries a calendar, weather, and a strong daily
    and weekly cycle, so the same file serves the tabular regression chapters and
    the sequence-model chapters later on.

    Note that `casual` and `registered` sum exactly to the target `cnt`. Both are
    dropped here, because leaving them in turns the problem into addition.

    Source: UCI Machine Learning Repository, id 275. CC BY 4.0.
    """
    def build():
        payload = _data.fetch(
            "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
            "bike-sharing", "bike_sharing_dataset.zip",
            licence="CC BY 4.0",
            source_page="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        )
        with zipfile.ZipFile(payload) as archive:
            with archive.open("hour.csv") as handle:
                return pd.read_csv(handle)

    frame = _cached("bike-sharing", "bike_sharing_hourly.csv", build)
    features = frame.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
    return features, frame["cnt"]


def fashion_mnist(train: bool = True):
    """Ten clothing categories as 28x28 greyscale images.

    Image classification, 60,000 training and 10,000 test rows. A drop-in
    replacement for MNIST that is several times harder, so a convolutional
    network does not immediately hit 99% and stop being interesting.

    Too large to commit, so this downloads through torchvision on first use into
    `data/raw/fashion-mnist/`. Returns a torchvision dataset, not a DataFrame.

    Source: Zalando Research. MIT licence.
    """
    from torchvision import datasets as tvd
    from torchvision import transforms

    return tvd.FashionMNIST(
        root=str(_data.raw_dir("fashion-mnist")), train=train, download=True,
        transform=transforms.ToTensor(),
    )


def summary() -> pd.DataFrame:
    """One row per house dataset. Handy as the first cell of a new notebook."""
    return pd.DataFrame(
        [{"dataset": name, "description": text} for name, text in HOUSE_DATASETS.items()]
    )
