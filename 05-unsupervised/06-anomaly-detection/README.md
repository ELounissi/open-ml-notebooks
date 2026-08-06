# Anomaly detection

### Four ways to say "this one is not like the others"

**[Open the notebook](notebook.ipynb)** · Part 5, Unsupervised learning ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | How Isolation Forest, One-Class SVM, Local Outlier Factor and a Gaussian-mixture density each define unusual, how to judge a detector with no labels and how to judge one honestly when you have them, and what `contamination` really does |
| **You should already know** | [Gaussian mixtures](../05-gaussian-mixture-models/), [density-based clustering](../04-dbscan-and-hdbscan/), [support vector machines](../../03-classification/05-support-vector-machines/), [random forests](../../04-ensembles/02-random-forest/) |
| **Datasets** | UCI Dry Bean, Wisconsin Breast Cancer, plus blobs with uniform noise scattered over them |
| **Runtime** | Two to three minutes on a laptop CPU |

---

## The unanimous flags were the empty ones

40 BARBUNYA beans hidden among 3,546 DERMASON, a base rate of **1.1154%**. Each of
the four detectors flags its top 40 rows. Then I count the overlap.

| | Rows | Of which really anomalies |
|---|---|---|
| Flagged by all four | 3 | **0** |
| Flagged by exactly one | 56 | **18** |

Every row the four detectors agreed on was an ordinary bean. Eighteen real anomalies
were seen by one detector and missed by the other three.

The obvious next move makes it worse. Rank-averaging the four scores — the cheapest
ensemble there is, and one that needs no labels to build — reaches **0.087** average
precision, against **0.949** for the best single detector and 0.077 for the worst.
The consensus landed **0.010 above the worst member and 0.862 below the best**.

The mechanism is worth being precise about, because this is the opposite of the
standard advice. Unanimous agreement selects rows that are extreme on every simple
criterion at once: cut off quickly by random splits, outside the enclosing surface,
sparse relative to their neighbours, unlikely under the fitted density. A row clearing
all four bars is usually a genuine bean at the far edge of the normal distribution. A
real anomaly has no obligation to look wrong in four ways — it only has to be wrong in
one, and then only the detector whose definition happens to match will see it.
Intersecting throws that row out; averaging buries it under the three that never
noticed. Consensus is not a safety property, and an ensemble is not automatically
safer than its best member.

## What each one is asking

![Isolation paths](figures/fig-01-isolation-paths.png)

Isolation Forest asks how few random cuts separate a point from the rest. On a cloud
of 120 points plus one stray, averaged over 200 random trees, the stray falls out
after **2.60** cuts and a central point after **11.57** — **4.5×** as many. No
distance metric, no scaling, no assumption about shape, which is why it is the
cheapest thing to try first.

The other three ask different questions. One-Class SVM: does the point fall outside a
boundary wrapped around the training data. Local Outlier Factor: is its neighbourhood
thinner than its neighbours'. Gaussian mixture: how unlikely is it under a fitted
density. Four different questions, which is why the four answers diverge.

## Drawn on 2-D data

![Decision surfaces](figures/fig-02-decision-surfaces.png)

30 uniform points hidden among 370 blob points, each detector allowed 32 flags.

| Detector | Noise caught | Blob points wrongly flagged |
|---|---|---|
| Isolation Forest | **22/30** | 10 |
| One-Class SVM | 21/30 | 12 |
| Local Outlier Factor | 19/30 | 13 |
| Gaussian mixture | **22/30** | 10 |

Within three points of each other, and the boundaries look nothing alike. Isolation
Forest draws a boxy region because it can only cut along the axes. The SVM draws a
smooth contour wrapping both blobs and the gap between them. LOF hugs each blob
separately, tight around the narrow one and loose around the wide one — that is what
*local* buys. The mixture draws two clean ellipses, which is exactly right here only
because I generated two ellipses. Some noise landed inside a blob: no detector flags
those and none should, because nothing about them is observably unusual. The ceiling
on this problem is set by the data.

## Judging a detector with no labels

Three things you can measure without labels, and none of them is correctness.

**Stability.** Top-40 overlap when the seed or the neighbourhood changes. Isolation
Forest across seeds: **98% / 95% / 98%**. LOF against its own k=20 answer: 72% at
k=10, 70% at k=40, **12% at k=80**. Isolation Forest is the random one and LOF has no
randomness at all, yet LOF's answer moves far more, because $k$ is a decision and
every value of it asks a different question.

**Resolution.** Score 2,000 uniform points drawn from the same box the data lives in.
**1.0%** of real rows clear the 99th percentile of real scores, against **88.6%** of
the uniform junk. Equal shares would mean the score carries no information.

**A describable difference.** The flagged rows against the rest, in standard
deviations: ConvexArea **+5.32**, Area +5.26, Perimeter +5.10, EquivDiameter +4.53,
MinorAxisLength +4.33. That is the honest deliverable of a label-free run — not
"these are wrong" but "these are much bigger, and here is by how much".

## Scored with labels

![Average precision](figures/fig-03-average-precision.png)

| Detector | Dry bean (base rate 0.011) | Breast cancer (base rate 0.048) |
|---|---|---|
| Isolation Forest | **0.949** (85× base) | **0.617** (13× base) |
| One-Class SVM | 0.537 (48× base) | 0.356 (7× base) |
| Gaussian mixture | 0.128 (11× base) | 0.225 (5× base) |
| Local Outlier Factor | 0.077 (7× base) | 0.476 (10× base) |

Every detector clears the random floor by a wide margin, so all four found something.
Isolation Forest won both. Below first place the order inverts: LOF goes from last on
the beans (0.077) to second on the cells (0.476), and the mixture goes the other way.

The structure of the anomaly decides most of that. The odd beans are one variety, so
they sit in their own tight clump of forty — and a clump is not sparse relative to
itself, which is precisely what a local method cannot see. The malignant cells spread
along several correlated measurements instead of clumping, so the local method has
room to work. You usually cannot see that structure before picking the detector, which
is why I distrust any post that names a best anomaly detector.

Report average precision with the base rate beside it. Accuracy is meaningless at 1.1%
positives, and ROC AUC flatters because the enormous normal class dominates the
false-positive rate.

## Contamination is not a parameter

![Disagreement and contamination](figures/fig-04-disagreement-and-contamination.png)

Sweeping `contamination` on Isolation Forest, and comparing every score against what
scikit-learn produces when it refits.

| contamination | Rows flagged | Precision | Recall | Largest score change vs the default fit |
|---|---|---|---|---|
| 0.01 | 36 | 0.917 | 0.825 | **0.00e+00** |
| 0.05 | 180 | 0.217 | 0.975 | **0.00e+00** |
| 0.2 | 717 | 0.056 | 1.000 | **0.00e+00** |

Not one score moved by a single bit. `contamination` only draws a line through a
ranking that was already fixed. It is your prior about how many anomalies exist,
entered as a number and handed back as a result. Set it to the true rate and you look
accurate, and the only way you knew the true rate is that you had labels, in which
case you had better options.

The One-Class SVM is the exception, because `nu` enters the optimisation itself. At
`nu=0.01` it keeps 101 support vectors and scores **0.186**; at `nu=0.20`, 607 support
vectors and **0.568**. Spearman between the two rankings is **0.6573** — two fits, two
rankings, a threefold difference in average precision. For the SVM the dial is real
and needs tuning; for the other three it is cosmetic, which is easy to miss because
scikit-learn spells both as one small float in the constructor.

Leave `contamination="auto"`, work with the raw `score_samples` output, and pick the
cutoff from outside the model: how many alerts a person can read in a day, or what a
false alarm costs.

## Cheat sheet

| | |
|---|---|
| **Try first** | Isolation Forest. No scaling, near-linear in rows, few knobs, and best on both labelled problems here |
| **Use LOF when** | Density varies and "unusual" means unusual for its own neighbourhood. Set `novelty=True` to score rows it was not fitted on |
| **Use One-Class SVM when** | Under a few thousand rows. It is quadratic, so subsample to fit and score everything afterwards |
| **Use a mixture when** | You can defend the shape assumption. Regularise the covariances (`reg_covar`) or it goes singular |
| **Scaling** | Everything except Isolation Forest needs it. Fit the scaler on the normal data only |
| **Metric** | Average precision with the base rate beside it |
| **Main dials** | `n_neighbors` for LOF (12% overlap between k=20 and k=80 here), `nu` for the SVM, `n_components` for the mixture |
| **Combining detectors** | Not a free win. Rank-averaging the four scored 0.087 against 0.949 for the best member, and the rows all four agreed on held none of the real anomalies |
| **Where to look first** | The rows a single detector flagged, and which detector flagged them. A lone flag is a lead |
| **No labels** | Check stability across seeds and neighbourhood sizes, check the score separates real rows from uniform junk, and describe the flagged rows in feature terms |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 5](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#AnomalyDetection` `#OutlierDetection` `#IsolationForest`
`#OneClassSVM` `#LocalOutlierFactor` `#UnsupervisedLearning` `#Python`
`#ScikitLearn` `#MLTutorial`
