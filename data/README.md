# Data

Every notebook in this book draws from the same five datasets. That is the point:
when a support vector machine scores 0.93 in one chapter and a random forest
scores 0.94 in another, the numbers are comparable, because the question was
identical.

## The house datasets

| Dataset | Task | Rows × features | Licence | Committed |
|---|---|---|---|---|
| [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | regression | 20,640 × 8 | Public domain (StatLib / US Census) | yes, 1.9 MB |
| [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | binary classification | 569 × 30 | CC BY 4.0 | yes, 0.1 MB |
| [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | 7-class classification | 13,611 × 16 | CC BY 4.0 | yes, 3.8 MB |
| [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | regression, time series | 17,379 × 16 | CC BY 4.0 | yes, 1.2 MB |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | image classification | 70,000 × 28 × 28 | MIT | no, downloads on first use |

## Citing them

Three of the five are CC BY 4.0, which is not a licence to use the data quietly.
It requires credit to the people who collected it, so here it is. If you reuse a
dataset because you found it through this book, cite the dataset rather than the
book.

**UCI Dry Bean.** Koklu, M. and Ozkan, I. A. (2020). Multiclass classification of
dry beans using computer vision and machine learning techniques. *Computers and
Electronics in Agriculture*, 174, 105507. Dataset: UCI Machine Learning
Repository, <https://doi.org/10.24432/C50S4B>. CC BY 4.0.

**UCI Bike Sharing.** Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine
Learning Repository, <https://doi.org/10.24432/C5W894>. CC BY 4.0. The underlying
counts come from the Capital Bikeshare system in Washington, D.C., 2011 and 2012.

**Breast Cancer Wisconsin (Diagnostic).** Wolberg, W., Mangasarian, O., Street,
N., and Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI
Machine Learning Repository, <https://doi.org/10.24432/C5DW2B>. CC BY 4.0.

**California Housing.** Pace, R. K. and Barry, R. (1997). Sparse spatial
autoregressions. *Statistics and Probability Letters*, 33(3), 291 to 297. Derived
from the 1990 US Census and distributed with scikit-learn.

**Fashion-MNIST.** Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-MNIST: a
novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.
MIT licence.

None of these datasets were collected by me, and none of the chapters claim
otherwise. What is mine is the code, the experiments and the writing.

## Loading them

One function each, from anywhere in the repo:

```python
from toolkit import datasets

X, y = datasets.california_housing()   # regression
X, y = datasets.breast_cancer()        # binary
X, y = datasets.dry_bean()             # 7 classes
X, y = datasets.bike_sharing()         # regression and time series
train = datasets.fashion_mnist()       # torchvision dataset
```

Each loader caches under `data/raw/<name>/` on first call, so every later run is
offline and instant.

## Why these five

**California Housing** ships inside scikit-learn, so chapter one runs with no
download at all. Its target is capped at $500,000, which about 5% of districts hit
exactly, a real flaw, kept because noticing it is part of the lesson.

**Breast Cancer Wisconsin** is 569 rows. Every method in this book trains on it in
under a second, so it is used wherever a chapter needs a quick demonstration
rather than a challenge.

**UCI Dry Bean** is the workhorse. Published in 2020 and still uncommon in
tutorials, so results on it are not something you have already read somewhere
else. Its seven classes are unbalanced (522 Bombay against 3,546 Dermason),
which makes it a natural place to discuss what accuracy hides.

**UCI Bike Sharing** carries a calendar, weather, and a strong daily and weekly
cycle. One file therefore serves both the tabular regression chapters and the
sequence-model chapters later on. Note that `casual` and `registered` sum exactly
to the target; the loader drops both, because leaving them in turns the problem
into addition.

**Fashion-MNIST** has MNIST's shape and several times its difficulty, so a
convolutional network does not immediately reach 99% and stop being interesting.

## Rules

**Open licences only**, recorded above. **Provenance is written by the code**, not
by hand: `toolkit.data.fetch` refuses to download without a stated licence and
appends a `SOURCE.txt` next to every file it writes, with the URL, retrieval date,
and byte count. **Anything above 20 MB is not committed** and downloads on first
use instead, which is why Fashion-MNIST is the one exception in the table.

`data/raw/` is never edited by hand. Cleaning happens inside the notebook, in
view, so a reader can disagree with a decision and change it.

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
