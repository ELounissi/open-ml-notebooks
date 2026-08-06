# Probability calibration

### Getting a predicted 0.8 to mean eight times out of ten

**[Open the notebook](notebook.ipynb)** · Part 3, Classification ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Why ranking cases correctly and reporting honest probabilities are two separate skills, how to read a reliability diagram, how to write expected calibration error yourself in a few lines of NumPy, the shape of miscalibration each model family produces, and when Platt scaling beats isotonic regression |
| **You should already know** | [Logistic regression](../01-logistic-regression/), [naive Bayes](../03-naive-bayes/), [support vector machines](../05-support-vector-machines/), [classification metrics](../../01-foundations/05-classification-metrics/), [cross-validation](../../01-foundations/04-cross-validation/) |
| **Datasets** | UCI Breast Cancer Wisconsin, UCI Dry Bean |
| **Runtime** | Under two minutes on a laptop CPU |

---

The measurement I did not expect: fitting a calibrator on the model's own training
rows left a random forest **worse than never calibrating at all** — ECE **0.0651**
against **0.0171** uncalibrated — and knocked its AUC from 0.9820 to 0.9411. Platt
scaling then nearly tripled the error of a logistic regression that had arrived
calibrated already. Calibration is a separate axis from accuracy, and it is easy to
move the wrong way.

## Ranking is not probability

![Ranking is not probability](figures/fig-01-ranking-is-not-probability.png)

I built the two extremes by hand first. A model that orders every case correctly and
lies about every number: **AUC 1.0000, ECE 0.6109**, labelling everything positive at
the 0.5 threshold so accuracy lands on the base rate of 0.3000. Then a model that
ranks at chance and tells the truth on average: **AUC 0.5000, ECE 0.0000**. No single
number catches both failures.

On real data I took a logistic regression on breast cancer and divided its logits by
a temperature $T$, which cannot move the order.

| Variant | AUC | ECE | Brier |
|---|---|---|---|
| As fitted | 0.995283 | 0.0197 | 0.0195 |
| Over-confident, $T = 0.3$ | 0.995283 | 0.0184 | 0.0186 |
| Under-confident, $T = 3.0$ | 0.995283 | **0.1063** | 0.0381 |

AUC spread across the three: **0.0000000000**. ECE spread: **0.0878**. Sharpening
this model slightly *improved* its ECE — the fitted version was already a little
under-confident — so only the flattening did damage.

## Reliability diagrams and expected calibration error

![Reliability and bins](figures/fig-02-reliability-and-bins.png)

ECE is a weighted average of the per-bin gaps between predicted probability and
observed frequency, $\sum_b (n_b/n)\,|\bar{y}_b - \bar{p}_b|$, and nothing else is
going on. I wrote it twice, as a loop and vectorised, and checked both against
`toolkit.evaluate` — all three returned **0.0184214056**.

The bin populations are what most reliability plots hide:

| Mean predicted | Observed | Gap | Cases |
|---|---|---|---|
| 0.150 | 0.000 | −0.150 | 22 |
| 0.664 | 0.818 | +0.154 | 11 |
| 0.985 | 0.990 | +0.005 | 296 |

**79.3% of the data sits in the two end bins.** A bin holding 11 cases and a bin
holding 296 look identical on the diagram and should not. ECE is also not a proper
scoring rule — predicting the base rate for everything scores near zero, as above —
so quote Brier or log loss beside it.

## Each model family fails in its own shape

![Shapes of miscalibration](figures/fig-03-shapes-of-miscalibration.png)

Four models on breast cancer, all scored out of fold.

| Model | AUC | Accuracy | Brier | ECE |
|---|---|---|---|---|
| Logistic regression | 0.9953 | 0.9789 | 0.0195 | 0.0197 |
| Random forest | 0.9907 | 0.9578 | 0.0301 | 0.0371 |
| Naive Bayes | 0.9877 | 0.9385 | 0.0568 | 0.0603 |
| SVM margin, min-max squashed | 0.9953 | 0.9754 | 0.0654 | **0.1927** |

**The forest is under-confident.** Its score is the share of trees voting positive,
and near a boundary a few always dissent, so 0 and 1 are unreachable: **38.8%** of
its predictions sit strictly inside (0.02, 0.98), against 22.1% for logistic
regression.

**The SVM margin is not a probability.** `decision_function` returned a signed
distance in [−2.765, 2.592]. Min-max squashing and a plain sigmoid both hold the AUC
at exactly 0.9953 and both stay badly calibrated, at ECE 0.1927 and 0.1753. Neither
looked at a label.

**Naive Bayes is over-confident by the widest margin here.** It multiplies
correlated features as if they were independent, so the same evidence gets counted
repeatedly: **95.6% of its predictions came out above 0.99 confidence** at a mean
confidence of **0.9919**, against an overall accuracy of **0.9385**.

## Platt scaling against isotonic regression

![Before and after](figures/fig-04-before-and-after.png)

| Model | ECE none | ECE sigmoid | ECE isotonic | AUC, none → isotonic |
|---|---|---|---|---|
| Logistic regression | **0.0197** | 0.0571 | 0.0119 | 0.9953 → 0.9938 |
| Random forest | 0.0371 | 0.0358 | 0.0191 | 0.9907 → 0.9886 |
| Naive Bayes | 0.0603 | 0.0418 | 0.0332 | 0.9877 → 0.9854 |
| SVM (rbf) | 0.1927 | 0.0433 | **0.0159** | 0.9953 → 0.9931 |

The SVM is the clean win: ECE from 0.1927 to 0.0159, a **12× reduction**, for 0.0022
of AUC. That is what calibration is for. Two results argue against reaching for it
reflexively. Sigmoid on logistic regression made ECE **2.9× worse** — log loss is a
proper scoring rule, so that model was already fine and the wrapper solved a problem
that did not exist. And sigmoid on naive Bayes cost **0.0423 of AUC**; a calibrator
is monotone and cannot reorder anything, so that came from the refitting inside
`CalibratedClassifierCV`. Measure before you wrap.

### Where isotonic overfits

![Platt vs isotonic](figures/fig-05-platt-vs-isotonic.png)

Breast cancer is too small to sweep, so I switched to dry bean, kept the two
varieties that get confused most, and split 1730 train / 2597 calibration pool /
1855 test — uncalibrated Brier 0.0775, ECE 0.0736. Then I held the base model fixed
and grew only the calibration set.

| Calibration rows | Platt Brier | Isotonic Brier | Winner |
|---|---|---|---|
| 25 | **0.0678** | 0.0720 | Platt |
| 50 | **0.0613** | 0.0666 | Platt |
| 100 | **0.0602** | 0.0637 | Platt |
| 200 | **0.0583** | 0.0613 | Platt |
| 400 | **0.0582** | 0.0592 | Platt |
| 800 | 0.0580 | **0.0579** | isotonic |
| 1600 | 0.0576 | **0.0571** | isotonic |
| 2500 | 0.0577 | **0.0571** | isotonic |

**The crossover fell at about 800 calibration rows.** Read the margins as well as
the winner column: at 25 rows Platt wins by 0.0042, at 2500 isotonic wins by 0.0006.
Choosing Platt when isotonic was right costs seven times less than the reverse.
Sigmoid is the safer default and the crossover is a long way up.

## Fit the calibrator on data the model has not seen

This forest scored **1.0000 on its own training rows** and 0.9261 on test, at mean
confidence 0.9618 on those training rows. A calibrator fitted there learns that 0.95
means near-certain, because on those rows it was.

| Calibration data | ECE | Brier | AUC |
|---|---|---|---|
| No calibration | 0.0171 | 0.0515 | 0.9820 |
| Sigmoid, on the training rows | 0.0566 | 0.0614 | 0.9820 |
| Sigmoid, on held-out rows | 0.0344 | 0.0528 | 0.9820 |
| Isotonic, on the training rows | **0.0651** | 0.0671 | **0.9411** |
| Isotonic, on held-out rows | **0.0144** | 0.0523 | 0.9815 |
| Isotonic, cross-fitted on everything | 0.0152 | **0.0497** | **0.9836** |

Both methods came out worse than doing nothing when fitted on the training rows,
isotonic hardest at **3.8× the uncalibrated ECE** plus a 0.0409 AUC loss. Sigmoid on
held-out rows, at 0.0344, still failed to beat leaving this model alone.
Cross-fitting is the version to reach for: best Brier and best AUC of the six, and
no data set aside permanently.

The danger scales with overfitting. Repeated with naive Bayes, which fits its
training rows about as well as anything else (0.9035 train against 0.9121 test), the
two rows land at ECE 0.0176 and 0.0187. Barely a difference. Rather than reason
about that per model, always use held-out data.

## Cheat sheet

| | |
|---|---|
| **Discrimination** | Does the model order the cases. AUC. Invariant to any increasing transform of the score |
| **Calibration** | Does the number mean what it says. ECE, read against the diagonal of a reliability diagram |
| **Report together** | ECE plus Brier or log loss. ECE alone rewards a model that predicts the base rate and nothing else |
| **Random forest** | Under-confident. Votes average away from 0 and 1 — 38.8% of scores inside (0.02, 0.98) here |
| **Naive Bayes** | Over-confident. 95.6% of predictions above 0.99 confidence at 0.9385 accuracy |
| **SVM** | `decision_function` is a distance. Squashing it by hand fixes nothing — ECE stayed at 0.1927 |
| **Logistic regression** | Usually arrives calibrated. Wrapping it made ECE 2.9× worse here |
| **Platt vs isotonic** | Sigmoid below roughly 800 calibration rows, isotonic past it — and by a small margin even then |
| **Where to fit it** | Never the training rows. Held-out split, or `CalibratedClassifierCV(estimator=..., cv=5)` to cross-fit |
| **Prefit models** | `FrozenEstimator(model)` in scikit-learn 1.8. The old `cv="prefit"` has been removed |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 3](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#Calibration` `#ProbabilityCalibration` `#ReliabilityDiagram`
`#PlattScaling` `#IsotonicRegression` `#Python` `#ScikitLearn` `#DataScience`
`#MLTutorial`
