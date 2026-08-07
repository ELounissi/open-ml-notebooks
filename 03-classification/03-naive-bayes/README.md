# Naive Bayes

### An assumption everyone knows is false, and a model that works anyway

**[Open the notebook](notebook.ipynb)** · Part 3, Classification ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | How Bayes' theorem becomes a classifier, what "naive" refers to, why the assumption being wrong often does not matter, and where it very much does |
| **You should already know** | [Logistic regression](../01-logistic-regression/) |
| **Datasets** | UCI Dry Bean, Breast Cancer Wisconsin |
| **Runtime** | Under a minute on a laptop CPU |

---

## The idea

Every other classifier here learns a boundary. Naive Bayes models **what each class
looks like**, then asks which class most likely produced the thing in front of it.

$$P(c \mid x) \propto P(c) \prod_{i} P(x_i \mid c)$$

That product is the **naive** part: it assumes features are independent given the
class. On Dry Bean, `Area` and `Perimeter` correlate at **0.9997**. The assumption
is not approximately true; it is wrong.

![The assumption](figures/fig-01-the-assumption.png)

![Independence](figures/fig-02-independence.png)

Mean absolute correlation between feature pairs is **0.498**, with **21 of 256
pairs above 0.9**.

## A thirteen-point detour worth taking

Our from-scratch version and scikit-learn's disagreed badly on identical data:

| | Dry Bean accuracy |
|---|---|
| Our implementation | **0.8954** |
| `GaussianNB()`, default, unscaled | 0.7629 |
| `GaussianNB(var_smoothing=1e-12)` | 0.8807 |
| `GaussianNB()`, default, **scaled** | 0.8951 |

Not a bug in either. Both add a small constant to every variance to avoid dividing
by zero: ours a flat `1e-9`, scikit-learn's **`1e-9` times the largest variance
across all features**.

On Dry Bean the largest variance is 8.87e+08 (`ConvexArea`), so the smoothing is
0.886. The smallest feature variance is 3.55e-07. **The smoothing is 2,496,841
times that feature's entire variance**, flattening it into uselessness.

**The textbook says Naive Bayes needs no feature scaling. The textbook is right
about the maths and wrong about the library.** The maths is scale-invariant; the
implementation is not, because a single global smoothing constant on features
spanning fifteen orders of magnitude cannot be small for all of them. Scale before
`GaussianNB`.

This is worth knowing because it shows up elsewhere in the book.
[03-04](../04-discriminant-analysis/) scores Gaussian naive Bayes at 0.7641 on the
same folds of the same data, because its leaderboard runs every estimator
unscaled. That is the same trap, not a disagreement.

The habit worth taking is broader than this one default. When a from-scratch
version and a library version of the same equations disagree by thirteen points,
the gap is a default you did not know about, and chasing it teaches more than
agreement would have.

## Where it lands

| Model | Dry Bean | Breast Cancer |
|---|---|---|
| Gaussian Naive Bayes (scaled) | 0.8972 | 0.9279 |
| Logistic regression | 0.9234 | 0.9807 |
| Gradient boosting | 0.9271 | 0.9385 |

![Comparison](figures/fig-03-comparison.png)

The bars carry fold standard deviations, and they decide which of those columns
can be read. Dry Bean has 13,611 rows, so the 0.03 that Naive Bayes trails by is
several times the fold spread and is a real gap. Logistic regression against
gradient boosting on the same column is 0.0037 apart, which is not, so do not read
that pair as an ordering. Breast Cancer has 569 rows split five ways, and only
logistic regression's lead there is wide enough to survive its own error bar.

The finding is the Dry Bean column, because that is where the independence
assumption is violated hardest. Naive Bayes trails, and it trails by far less than
a model built on a false premise has any right to.

**Classification only needs the right *ranking*, not the right probabilities.**
Correlated features make it count the same evidence several times, which pushes it
further toward the class that evidence already supported. The argmax is invariant
to that inflation. The probability is not.

That gives a rule with a condition rather than a slogan. Naive Bayes is fine
wherever the decision is the output and the probability is discarded: routing,
filtering, a first-pass triage. It stops being fine the moment something
downstream multiplies its probability by a cost, thresholds on it, or shows it to
a person.

## The probabilities do not survive

![Overconfidence](figures/fig-04-overconfidence.png)

**Nine predictions in ten claim above 99% confidence, and 85% claim above 99.9%**,
on a model that gets 89.5% right. A prediction stamped 99.9% should be wrong about
one time in a thousand. This one is wrong about one time in ten.

It has counted `Area`, `Perimeter`, `ConvexArea` and `EquivDiameter` as four
independent pieces of evidence when they are nearly one. The certainty compounds
four times over.

**Trust its labels. Do not trust its probabilities.**

## Cheat sheet

| | |
|---|---|
| **Use it when** | You want a baseline in one line; the data is enormous or streaming; text with bag-of-words (the classic case) |
| **Avoid it when** | You need probabilities you can act on |
| **Variants** | `GaussianNB` continuous, `MultinomialNB` counts, `BernoulliNB` binary |
| **Scaling needed** | Not by the maths, **but yes for `sklearn.GaussianNB`**. Unscaled it cost 13 points here |
| **Cost** | One pass over the data. Nothing iterates |
| **Watch out** | Wildly overconfident when features correlate |
| **If you need probabilities** | [Logistic regression](../01-logistic-regression/), or run this through [calibration](../09-probability-calibration/) |
| **Next** | [Discriminant analysis](../04-discriminant-analysis/), which keeps the generative idea and drops the independence assumption |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 3](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#NaiveBayes` `#BayesTheorem` `#Classification` `#Python`
`#ScikitLearn` `#DataScience` `#MLTutorial` `#LearnMachineLearning`
