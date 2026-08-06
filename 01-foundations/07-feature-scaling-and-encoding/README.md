# Feature scaling and encoding

### Getting the columns into a shape the model can actually read

**[Open the notebook](notebook.ipynb)** · Part 1, Foundations ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Which models need scaling and which cannot tell the difference, what the four scalers do to a column with a long tail, why ordinal encoding invents an order that was never in the data, how to put a cyclical column on a circle, and how target encoding leaks the answer if you compute it in the wrong place |
| **You should already know** | [Cross-validation](../04-cross-validation/), [k-nearest neighbours](../../03-classification/02-k-nearest-neighbours/), [linear regression](../../02-regression/01-linear-regression/) |
| **Datasets** | UCI Dry Bean, UCI Bike Sharing |
| **Runtime** | Under two minutes on a laptop CPU |

---

## Two columns out of sixteen own the distance

Four thousand random pairs of Dry Bean rows, squared per-column gap, share of the
total squared Euclidean distance:

| Column | Share, raw table |
|---|---|
| ConvexArea | 50.7632% |
| Area | 49.2335% |
| Perimeter | 0.0026% |
| The other thirteen together | **0.0007%** |

`Area` and `ConvexArea` are near-duplicates, so a nearest-neighbour search on the
raw table is a search on bean size alone — the widest column's standard deviation
is **49,968,404x** the narrowest's. After `StandardScaler` the shares run from
**6.4743%** down to **5.8836%**, against an equal split of 6.2500%.

## Which models care, measured

![Who cares](figures/fig-01-who-cares.png)

Same seven-class bean problem, 3,000 stratified rows, cross-validated accuracy:

| Model | Raw | Scaled | Change |
|---|---|---|---|
| k-NN (k=15) | 0.6327 | 0.9123 | **+0.2797** |
| SVM (rbf) | 0.6390 | 0.9197 | **+0.2807** |
| Logistic regression | 0.8517 | 0.9173 | +0.0657 |
| Random forest | 0.9147 | 0.9147 | **+0.0000** |
| Gradient boosting | 0.9160 | 0.9160 | **+0.0000** |

The tree ensembles print a change of **exactly zero**, not a small one: a split
asks whether a column is above a threshold, and a monotone transform moves the
threshold by exactly as much as it moves the data. Logistic regression fails
differently. On raw columns it used **1000 iterations, which is the cap**, against
**73** scaled, with a largest coefficient of 1.594e-01 against 3.113e+00. It ran
out of budget looking.

## The four scalers barely differ from each other

![Four scalers](figures/fig-02-four-scalers.png)

Median-to-largest-bean distance in interquartile ranges, and k-NN accuracy:

| | (max - median) / IQR | k-NN accuracy |
|---|---|---|
| Raw `Area`, no scaling | 8.397 | 0.6327 |
| StandardScaler | 8.397 | 0.9123 |
| MinMaxScaler | 8.397 | 0.9090 |
| RobustScaler | 8.397 | **0.9157** |
| QuantileTransformer | **3.822** | 0.9073 |

The ratio is identical for the raw column and all three affine scalers, because
an affine transform cannot move a point relative to the rest of its column:
**none of them does anything about an outlier**. `QuantileTransformer` keeps the
ranks and discards the values, which is why its number moves. Spread across the
four accuracies: **0.0083**. Gap from not scaling: **0.2747**, thirty-three times
as large.

## The ordering you did not mean to invent

![Invented order](figures/fig-03-invented-order.png)

Bike Sharing hands you categories already converted to integers, with no warning
that they are labels. Mean hourly hires by season code: spring **111.1**, summer
**208.3**, fall **236.0**, winter **198.9** — real, and not monotone. Given the
raw code a model has one coefficient, so it fits a straight ramp through four
points that are not on a line. There are exactly 24 ways to number four seasons,
none more correct than another, so I fitted all of them.

| Encoding | R squared |
|---|---|
| season, ordinal, worst of 24 numberings | -0.0003 |
| season, ordinal, best of 24 (average 0.0215) | 0.0555 |
| season, one-hot | **0.0653** |
| month, calendar order | **0.0142** |
| month, best of 30 random numberings | **0.0298** |
| month, one-hot (12 columns) | 0.0739 |

One-hot beats the *best* ordinal numbering of `season` by +0.0098 and is invariant
to a naming choice that carries no information. The month rows are sharper:
**calendar order lost to random relabellings**, scoring less than half the best
arbitrary one. The yearly pattern is a hump, a straight line through a hump is
nearly flat, and any numbering that splits busy months from quiet ones wins.

### Hour of day is a circle

![Hours on a circle](figures/fig-04-hours-on-a-circle.png)

As a plain number, 23:00 and midnight are the furthest apart of any pair in the
column: distance **23.000**, against **12.000** from midnight to noon. On a circle
they become **0.261** and **2.000**. R squared on hourly hires goes **0.1551** for
the raw hour number, **0.3363** for sine and cosine, **0.4111** and **0.4446** as
the second and third harmonics come in, and **0.5003** for 24 one-hot columns.
One-hot wins because Washington's demand has two daily peaks and one sine wave
has one hump; the harmonics close most of that gap with a quarter of the columns.

## Target encoding and its leak

![Target leak](figures/fig-05-target-leak.png)

Day of week crossed with hour gives 168 weekly slots, 72 training rows each.
Weather alone scores CV 0.2894 and held-out 0.3094; one-hot over the 168 levels
0.7802 and 0.7859; target encoding computed leakily 0.7802 and 0.7779; out of
fold 0.7736 and 0.7780. All three look fine, which is what makes the mistake
durable. So I add a column of pure noise and vary how many rows share a value:

| Rows per level | Levels | Leaky CV | Leaky test | Out-of-fold CV | Corr, leaky | Corr, safe |
|---|---|---|---|---|---|---|
| 2 | 6,082 | **0.5395** | **-0.0415** | 0.2893 | 0.6575 | -0.0029 |
| 8 | 1,520 | 0.3509 | 0.2327 | 0.2893 | 0.3544 | 0.0023 |
| 200 | 60 | 0.2920 | 0.3090 | 0.2894 | 0.0734 | 0.0090 |

A column containing no information reports a cross-validated **0.5395** against a
baseline of 0.2894, and the test set pays back **-0.0415**, worse than the model
that never saw it. The out-of-fold column never leaves the baseline, 0.2891 to
0.2894 across the whole sweep, because a row's encoding never saw its own target.

## Cheat sheet

| | |
|---|---|
| **Scale for** | k-NN, SVM, PCA, k-means, neural nets, any penalised linear model — anything using a distance or a gradient. Not trees: their change was 0.0000, not small |
| **Which one** | `StandardScaler`, or `RobustScaler` on a long tail. The four differed by 0.0083 and beat no scaling by 0.2747, and no affine scaler removes an outlier |
| **Always** | Fit the scaler inside a `Pipeline` so it never sees the validation fold |
| **Categories** | One-hot by default. Ordinal only when the order is real *and* the spacing means something |
| **Cyclical** | sin and cos of the angle, plus harmonics if the shape has more than one peak |
| **High cardinality** | Target encoding computed out of fold. `TargetEncoder` in a `Pipeline` does it for you |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 1](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#FeatureEngineering` `#DataPreprocessing` `#ScikitLearn`
`#OneHotEncoding` `#TargetEncoding` `#DataLeakage` `#Python` `#MLTutorial`
