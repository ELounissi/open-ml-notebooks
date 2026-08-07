# Imbalanced classes

### Resampling, class weights, and the threshold almost nobody moves

**[Open the notebook](notebook.ipynb)** · Part 3, Classification ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Why accuracy is unusable on a rare class, how to write undersampling, oversampling and SMOTE yourself in NumPy, what `class_weight="balanced"` actually does, why moving the decision threshold does the same job for a tenth of the work, and how much score you invent by resampling before the split |
| **You should already know** | [Logistic regression](../01-logistic-regression/), [decision trees](../06-decision-trees/) |
| **Datasets** | UCI Dry Bean and UCI Breast Cancer, both cut down to a rare positive class |
| **Runtime** | Two to three minutes on a laptop CPU |

---

## The result that made me write this

Same model, same folds, 3,606 beans with a **1.66%** positive rate. The only thing
that changes is how the decision is made.

| Recipe | F1 | Precision | Recall | Average precision |
|---|---|---|---|---|
| Undersample, cut at 0.5 | 0.205 | 0.117 | 0.873 | 0.496 |
| Oversample, cut at 0.5 | 0.227 | 0.130 | 0.873 | 0.624 |
| SMOTE, cut at 0.5 | 0.231 | 0.134 | 0.850 | 0.627 |
| `class_weight="balanced"`, cut at 0.5 | 0.226 | 0.130 | 0.880 | 0.625 |
| Plain model, cut at the base rate | 0.225 | 0.130 | 0.850 | **0.748** |
| Plain model, tuned cut | **0.723** | 0.917 | 0.610 | **0.748** |

Every resampler lands within 0.026 F1 of the plain model cut at its base rate,
which takes one line of arithmetic and no refitting. And a cut tuned on inner folds
reaches **0.723**, three times the best resampler, on a model nobody rebalanced at
all.

Average precision, which reads the whole ranking instead of one cut, is 0.748 for
the plain model and never beaten. Resampling did not improve the ranking. It moved
the cut, and it damaged the ranking on the way: every fix gave up between 0.12 and
0.25 average precision.

## Accuracy is a measurement of the common class

![Accuracy is useless](figures/fig-01-accuracy-is-useless.png)

| | Accuracy | Balanced accuracy | Recall | Precision | F1 | Average precision |
|---|---|---|---|---|---|---|
| Predict every row negative | 0.983 | 0.500 | 0.000 | 0.000 | 0.000 | 0.017 |
| Logistic regression at 0.5 | 0.993 | 0.783 | 0.567 | 0.971 | 0.716 | 0.735 |

A model that has never once said "positive" is **0.0092** behind a real one on
accuracy. Every other column separates them properly. Average precision has the
base rate as its floor, not one half, which is why the do-nothing model scores
0.017 rather than looking respectable. I avoid ROC-AUC here: with one positive per
60 rows the false positive rate has an enormous denominator, so a model can drown
you in false alarms and still post 0.963.

## Imbalance is not the problem: imbalance plus overlap is

Same rarity, same model, same 60 positives, different rare variety. SIRA overlaps
DERMASON heavily: average precision 0.735, recall at 0.5 of 0.567. BOMBAY beans are
several times larger: **1.000** and **1.000**.

The BOMBAY version is solved before anything is done about the imbalance. If your
rare class separates cleanly you can stop reading. If it overlaps badly, no
resampler invents a separation the features do not contain.

## The three families, and what they cost

![Three families](figures/fig-02-three-families.png)

| Method | Rows after | Positive rate | Distinct positive rows |
|---|---|---|---|
| As is | 3,606 | 1.7% | 60 |
| Undersample | 120 | 50.0% | 60 |
| Oversample | 7,092 | 50.0% | **60** |
| SMOTE | 7,092 | 50.0% | 3,546 |

Oversampling reports 3,546 positive rows and 60 distinct ones. SMOTE's rows are all
distinct, but every one is a convex combination of the same 60.

Across 25 folds, F1 at a fixed 0.5 cut spans **0.5076** between best and worst
method. Average precision spans **0.2518**, half as much. Most of what resampling
appears to buy is the cut moving rather than the model improving.

**But none of them are free, which is not what I expected.** Undersampling should
be the only method able to damage the ranking, since it is the only one that
deletes real rows. It is the worst, and it is not alone:

| Method | Average precision | Given up against plain |
|---|---|---|
| Plain | **0.748** | none, this is the reference |
| Undersample | 0.496 | -0.252 (34%) |
| Oversample | 0.624 | -0.124 |
| SMOTE | 0.627 | -0.121 |
| Class weights | 0.625 | -0.123 |

Duplicating rows, interpolating between them and reweighting the loss all pull the
boundary toward the rare class, and pulling the boundary reorders the scores near
it. Not deleting data is not the same as doing no harm.

## Why rebalancing is a threshold in disguise

![Threshold moving](figures/fig-03-threshold-moving.png)

For a calibrated model, rebalancing the training set to fifty-fifty and cutting at
0.5 is the same decision as cutting the original model at the base rate $\pi$:

$$\frac{p'(x)}{1-p'(x)} = r\cdot\frac{p(x)}{1-p(x)}, \qquad
  \frac{1}{1+r} = \frac{n_{\text{pos}}}{n_{\text{pos}}+n_{\text{neg}}} = \pi$$

The measured version confirms it: best resampler (SMOTE) **0.2313** F1, plain model
at the base rate **0.2252**. The prediction was right.

**What the prediction never promised is that either would be any good.** The
tuned cut averages **0.3381** across folds against a base rate of **0.0166**, a
factor of twenty apart, and the F1 that follows is **0.723** against **0.225**, a
factor of three. The base rate is not a cheap approximation to the tuned cut. It
is a different and much worse cut, because rebalancing to fifty-fifty is an
arbitrary target rather than an optimum, and on a 1.66% base rate it overshoots
far past where F1 peaks.

The left panel corrects one more thing. The out-of-fold F1 curve peaks at a cut of
**0.3468**, right beside the 0.5 line on that log axis, scoring **0.7600** against
**0.7158** at 0.5. The famous default is slightly wrong here, not badly wrong. The
base rate, at 0.0166, scores **0.2227**.

So the argument for the threshold is not that a formula beats resampling. It is
that a threshold is a **dial** rather than a formula. "Catch ninety percent of the
fraud" is a request you satisfy by reading a number off the recall curve. Neither
the base rate nor any resampling ratio takes that as input.

One caveat: the tuned cut wobbles fold to fold (sd 0.0630, range 0.2616 to
0.5569), because it is chosen from a handful of positives. Tune it, but do not
read the third decimal as meaningful.

### Second dataset, same shape of answer

Breast Cancer thinned to 375 rows, 18 positive (**4.80%**):

| Recipe | F1 | Precision | Recall | Average precision |
|---|---|---|---|---|
| Plain at 0.5 | 0.928 | 1.000 | 0.883 | 0.993 |
| Undersample | 0.726 | 0.610 | 0.980 | 0.900 |
| Oversample | 0.937 | 0.935 | 0.955 | 0.997 |
| SMOTE | 0.932 | 0.930 | 0.950 | 0.997 |
| Class weights | **0.945** | 0.963 | 0.940 | 0.994 |
| Plain at base rate | 0.823 | 0.720 | 0.995 | 0.993 |

Easier separation, higher base rate, and the methods still finish close together
while the ranking metric barely notices which one ran.

## Resampling before the split invents score

![Leakage](figures/fig-04-leakage.png)

| Model | Resampler | Before the split | Inside the fold | Inflation |
|---|---|---|---|---|
| Logistic regression | Undersample | 0.882 | 0.202 | +0.680 |
| Logistic regression | Oversample | **0.910** | **0.229** | **+0.681** |
| Logistic regression | SMOTE | 0.908 | 0.237 | +0.672 |
| Random forest | Undersample | 0.838 | 0.251 | +0.586 |
| Random forest | Oversample | 1.000 | 0.786 | +0.213 |
| Random forest | SMOTE | 0.991 | 0.640 | +0.351 |

Oversample first and **100.0%** of the positive test rows are exact copies of a
training row. Undersample first and the test fold arrives at a **50.00%** positive
rate instead of the real 1.66%, so precision is measured in a world that does not
exist.

**Read the inflation column, because it runs backwards from the usual story.** I
expected the forest to inflate most: a tree can put a duplicated row in its own
leaf, a linear model cannot. Every logistic inflation beats every forest one.
Mean inflation is **+0.678** for logistic regression against **+0.383** for the
forest, and the single largest is oversampling with logistic regression at
**+0.681**.

The answer is in the honest column, not the leaked one. The forest was **already
almost perfect on the training-fold task**, scoring 0.786 honestly with
oversampling against logistic regression's 0.229. A leaked duplicate cannot add
much to a model with that little headroom, so the forest climbs 0.786 to 1.000.
The linear model starts at 0.229 and the leak carries it to 0.910.

Inflation measures the room between honest performance and the ceiling, not
flexibility. **You cannot predict what a leak is worth from the model class.** Run
both orders and subtract: one extra cross-validation, and the only reliable
answer.

Class weights and thresholds cannot make this mistake, because they never create a
row.

## Cheat sheet

| | |
|---|---|
| **Use it when** | The class you care about is rare and overlaps the common one. If it separates cleanly, none of this is needed |
| **Do this first** | Stop reading accuracy. Use average precision, or precision and recall at a stated cut |
| **Do this second** | Move the threshold. Start at the base rate, tune on inner folds if you have positives to spare |
| **Scaling** | Needed for the model and for SMOTE, which picks neighbours by distance |
| **Main dials** | The decision threshold, then `class_weight="balanced"`. Resampling ratios last |
| **Watch out** | Resample inside the fold, never before the split. Leaked oversampling inflated F1 by +0.681 here |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 3](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#ImbalancedData` `#SMOTE` `#ClassWeights` `#Threshold`
`#PrecisionRecall` `#DataLeakage` `#Python` `#ScikitLearn` `#MLTutorial`
