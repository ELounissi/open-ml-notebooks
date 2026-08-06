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
against **0.0171** uncalibrated — and knocked its AUC from 0.9820 down to 0.9411.
Two more from the same notebook. Platt scaling nearly tripled the error of a
logistic regression that arrived calibrated already, 0.0197 to 0.0571. And three
models whose AUC agreed to ten decimal places differed by **0.0878** in ECE.

Calibration is a separate axis from accuracy, and it is easy to move the wrong way.

## Ranking is not probability

![Ranking is not probability](figures/fig-01-ranking-is-not-probability.png)

I built the two extreme cases by hand first. A model that orders every case
correctly and lies about every number: **AUC 1.0000, ECE 0.6109**, and at the
default 0.5 threshold it labels everything positive, so accuracy lands on the base
rate of 0.3000. Then a model that ranks at chance and tells the truth on average:
**AUC 0.5000, ECE 0.0000**. Neither is useful and no single number catches both
failures.

On real data I took one logistic regression on breast cancer and divided its logits
by a temperature $T$, which is an increasing function of the score, so the order
cannot move.

| Variant | AUC | ECE | Brier |
|---|---|---|---|
| As fitted | 0.995283 | 0.0197 | 0.0195 |
| Over-confident, $T = 0.3$ | 0.995283 | 0.0184 | 0.0186 |
| Under-confident, $T = 3.0$ | 0.995283 | **0.1063** | 0.0381 |

AUC spread across the three: **0.0000000000**. ECE spread: **0.0878**. Worth
noticing that sharpening this model slightly *improved* its ECE — the fitted
version was a little under-confident to begin with. Only the flattening did real
damage.

## Reliability diagrams and expected calibration error

![Reliability and bins](figures/fig-02-reliability-and-bins.png)

ECE is a weighted average of per-bin gaps between predicted probability and
observed frequency:

$$\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{n} \left| \; \bar{y}_b - \bar{p}_b \; \right|$$

Nothing else is going on. I wrote it twice, as a loop and vectorised, and checked
both against `toolkit.evaluate` — all three returned **0.0184214056**.

The bin populations are the part most reliability plots hide. Here is the top and
the tail of the table:

| Mean predicted | Observed | Gap | Cases |
|---|---|---|---|
| 0.014 | 0.000 | −0.014 | 155 |
| 0.150 | 0.000 | −0.150 | 22 |
| 0.664 | 0.818 | +0.154 | 11 |
| 0.985 | 0.990 | +0.005 | 296 |

**79.3% of the data sits in the two end bins.** A bin holding 11 cases and a bin
holding 296 look the same on the diagram and should not. ECE is also not a proper
scoring rule — predicting the base rate for everything scores near zero, as the
section above showed — so quote Brier or log loss next to it.

## Each model family fails in its own shape

![Shapes of miscalibration](figures/fig-03-shapes-of-miscalibration.png)

Four models on breast cancer, all scored out of fold.

| Model | AUC | Accuracy | Brier | ECE |
|---|---|---|---|---|
| Logistic regression | 0.9953 | 0.9789 | 0.0195 | 0.0197 |
| Random forest | 0.9907 | 0.9578 | 0.0301 | 0.0371 |
| Naive Bayes | 0.9877 | 0.9385 | 0.0568 | 0.0603 |
| SVM margin, min-max squashed | 0.9953 | 0.9754 | 0.0654 | **0.1927** |

**The forest is under-confident.** A forest score is the share of trees voting
positive, and near a boundary a few trees always dissent, so 0 and 1 are
unreachable. Measured: **38.8%** of its predictions sit strictly inside
(0.02, 0.98), against 22.1% for logistic regression.

**The SVM margin is not a probability.** `decision_function` returned a signed
distance in the range [−2.765, 2.592]. Min-max squashing and a plain sigmoid both
preserve the AUC exactly (0.9953) and both stay badly calibrated — ECE 0.1927 and
0.1753. Neither looked at a label.

**Naive Bayes is over-confident, by the widest margin here.** It multiplies
correlated features as if they were independent, so the same evidence gets counted
several times. **95.6% of its predictions came out above 0.99 confidence** and its
mean confidence was **0.9919**, while its overall accuracy was **0.9385**.

## Platt scaling against isotonic regression

![Before and after](figures/fig-04-before-and-after.png)

| Model | Calibration | AUC | Brier | ECE |
|---|---|---|---|---|
| Logistic regression | none | 0.9953 | 0.0195 | **0.0197** |
| Logistic regression | sigmoid | 0.9957 | 0.0269 | 0.0571 |
| Logistic regression | isotonic | 0.9938 | 0.0178 | 0.0119 |
| Random forest | none | 0.9907 | 0.0301 | 0.0371 |
| Random forest | sigmoid | 0.9913 | 0.0281 | 0.0358 |
| Random forest | isotonic | 0.9886 | 0.0300 | 0.0191 |
| Naive Bayes | none | 0.9877 | 0.0568 | 0.0603 |
| Naive Bayes | sigmoid | **0.9454** | 0.0551 | 0.0418 |
| Naive Bayes | isotonic | 0.9854 | 0.0444 | 0.0332 |
| SVM (rbf) | none | 0.9953 | 0.0654 | 0.1927 |
| SVM (rbf) | sigmoid | 0.9945 | 0.0240 | 0.0433 |
| SVM (rbf) | isotonic | 0.9931 | 0.0206 | **0.0159** |

The SVM is the clean win: ECE from 0.1927 to 0.0159, a **12× reduction**, with AUC
moving 0.0022. That is what calibration is for.

Two rows argue against reaching for it reflexively. Sigmoid on logistic regression
made ECE **2.9× worse**, because log loss is a proper scoring rule and the model
was already fine — the wrapper spent effort on a problem that did not exist. And
sigmoid on naive Bayes cost **0.0423 of AUC**, from 0.9877 to 0.9454; a calibrator
is monotone and cannot reorder, so that loss came from the refitting inside
`CalibratedClassifierCV`. Measure before you wrap.

### Where isotonic overfits

![Platt vs isotonic](figures/fig-05-platt-vs-isotonic.png)

Breast cancer is too small to sweep, so I switched to dry bean, kept the two
varieties that get confused most, and split 1730 train / 2597 calibration pool /
1855 test. Uncalibrated test Brier 0.0775, ECE 0.0736. Then I held the base model
fixed and grew only the calibration set.

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

**The crossover fell at about 800 calibration rows.** Read the margins, not just
the winner column: at 25 rows Platt wins by 0.0042, at 2500 isotonic wins by
0.0006. The penalty for choosing Platt when isotonic was right is seven times
smaller than the penalty for the reverse. Sigmoid is the safer default and the
crossover is a long way up.

## Fit the calibrator on data the model has not seen

The forest here scored **1.0000 on its own training rows** and 0.9261 on test, with
mean confidence 0.9618 on those training rows. A calibrator fitted there learns
that 0.95 means near-certain, because on those rows it was.

| Calibration data | ECE | Brier | AUC |
|---|---|---|---|
| No calibration | 0.0171 | 0.0515 | 0.9820 |
| Sigmoid, fitted on the training rows | 0.0566 | 0.0614 | 0.9820 |
| Sigmoid, fitted on held-out rows | 0.0344 | 0.0528 | 0.9820 |
| Isotonic, fitted on the training rows | **0.0651** | 0.0671 | **0.9411** |
| Isotonic, fitted on held-out rows | **0.0144** | 0.0523 | 0.9815 |
| Isotonic, cross-fitted on everything | 0.0152 | **0.0497** | **0.9836** |

Both methods came out worse than doing nothing when fitted on the training rows.
Isotonic did it hardest, at **3.8× the uncalibrated ECE** plus a 0.0409 AUC loss.
Note also that sigmoid on held-out rows, 0.0344, still failed to beat leaving this
model alone. Cross-fitting is the version to reach for: best Brier and best AUC of
the six, no data set aside permanently.

The danger scales with overfitting. Repeating the experiment with naive Bayes,
which fits its training rows about as well as anything else — 0.9035 train against
0.9121 test — the two rows land at ECE 0.0176 and 0.0187. Barely a difference.
Rather than reason about that per model, always use held-out data.

## Cheat sheet

| | |
|---|---|
| **Discrimination** | Does the model order the cases. AUC. Invariant to any increasing transform of the score |
| **Calibration** | Does the number mean what it says. ECE, read against the diagonal of a reliability diagram |
| **Report together** | ECE plus Brier or log loss. ECE alone rewards a model that predicts the base rate and nothing else |
| **Random forest** | Under-confident. Votes average away from 0 and 1 — 38.8% of scores inside (0.02, 0.98) here |
| **Naive Bayes** | Over-confident. 95.6% of predictions above 0.99 confidence at 0.9385 accuracy |
| **SVM** | `decision_function` is a distance, not a probability. Squashing it by hand fixes nothing |
| **Logistic regression** | Usually arrives calibrated. Wrapping it made ECE 2.9× worse here |
| **Platt (`method="sigmoid"`)** | Two parameters. Safer below roughly 800 calibration rows, exact when the distortion is a smooth S |
| **Isotonic** | Any monotone shape. Won past 800 rows, and by a small margin even then |
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
