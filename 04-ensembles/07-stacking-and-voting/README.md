# Stacking and voting

### Six models combined, and an honest accounting of what the combination bought

**[Open the notebook](notebook.ipynb)** · Part 4, Ensembles ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Hard against soft voting, why an ensemble only pays when its members fail on different rows, how a meta-model learns to weight base models, why its training features must be out-of-fold, and what all of it costs |
| **You should already know** | [Bagging](../01-bagging/), [cross-validation](../../01-foundations/04-cross-validation/), [logistic regression](../../03-classification/01-logistic-regression/) |
| **Datasets** | UCI Dry Bean (6,300 train / 2,700 test, 16 features, 7 classes), California Housing |
| **Runtime** | Two to three minutes on a laptop CPU |

---

## Start here: the ensemble barely paid, and one version lost

Stacking six models did beat the best single model, by **+0.0022** accuracy, for
**14.7× the training time**, a gain of **0.4 standard errors**. It fixed **60 rows**
the best member got wrong and **broke 54** it had right. Soft voting did not beat the
best member at all.

| | Accuracy | Time | Against the best member |
|---|---|---|---|
| Best single member (gradient boosting) | 0.9185 | 1.5 s | |
| Soft vote | 0.9181 | 3.7 s | **-0.0004** at 2.5× time |
| Stacking | **0.9207** | 21.5 s | **+0.0022** at 14.7× time |

One standard error on a test accuracy at n = 2700 is **0.0053**, so the stacking gain
is smaller than the noise. Stacking is not useless, but on a well-tuned gradient
booster with five look-alike friends this is how the numbers come out, and nobody
prints the "broke 54" line on a leaderboard.

## The six members, and the vote that hard voting won

| Member | Test accuracy | Fit time | Mean confidence in its own winner |
|---|---|---|---|
| Gradient boosting | **0.9185** | 1.47 s | 0.970 |
| Logistic regression | 0.9170 | 0.12 s | 0.906 |
| k-NN (15) | 0.9148 | 0.01 s | 0.912 |
| Random forest | 0.9100 | 1.91 s | 0.910 |
| Decision tree | 0.8844 | 0.16 s | **1.000** |
| Naive Bayes | 0.7493 | 0.01 s | 0.902 |

The textbook rule is that soft voting beats hard voting, because averaging
probabilities keeps the confidence that counting labels throws away.

**Hard voting won.** It matched the best member exactly at **0.9185 (+0.0000)**;
soft voting came in at **0.9181 (-0.0004)**. Neither beat the best single member.
The two rules disagreed on only **58 of 2,700 rows**, with the hard vote tied on
**53**, so everything here is decided by a handful of beans.

The mechanism does work where it is supposed to. Test row 69: true class SIRA, hard
voting said HOROZ 4-2 and was wrong, soft voting said SIRA and was right. Three of
those four HOROZ votes were cast at **0.508**, **0.520** and **0.607**, barely
leaning, against SIRA votes of **0.667** and **0.998**.

It loses elsewhere for the reason in the confidence column. **Confidence is only
information if it is honest.** The decision tree reports **1.000** on every row it
names, because an unrestrained tree's leaves are pure by construction, while
scoring 0.8844. Naive Bayes averages **0.902** while scoring **0.7493**. Soft
voting hands both a megaphone sized to a confidence neither earned; hard voting
caps every member at one vote, which on these six is the safer cap.

So the rule is conditional: **soft voting beats hard voting when the members are
calibrated**, and two of these six are badly uncalibrated. See
[probability calibration](../../03-classification/09-probability-calibration/).
Calibrate first, then measure both rules. It costs one line. Stacking has a third
answer, since a meta-model can learn to discount a loud member.

## Diversity is necessary, and this split could not show it

![Error correlation](figures/fig-01-error-correlation.png)

Mean pairwise correlation between members' error indicators: **0.540**. Most alike were
logistic and k-NN at **0.740**, least alike gradient boosting and naive Bayes at
**0.287**.

The control was meant to confirm the rule. **It came out backwards.**

| Family mix | Mean error correlation | Vote | Best member | Gain |
|---|---|---|---|---|
| Five k-NN variants | **0.888** | 0.9144 | 0.9137 | **+0.0007** |
| Six different families | **0.540** | 0.9181 | 0.9185 | **-0.0004** |

The near-identical set, about as undiversified as five models can be, is the only
one of the two that gained. The trio scatter agrees: the rule predicts a negative
slope, less gain as members get more alike, and the fitted correlation is
**r = 0.36**, the wrong sign. The chart title says so honestly.

Before rewriting the theory, check the scale. The best trio gained **+0.0037** and
the worst lost **0.0159**, while one standard error at 2,700 test rows is
**0.0053**. Almost every number here is smaller than its own error bar, and the
twenty trios are not independent, being drawn from the same six models on the same
rows. An r of 0.36 across twenty overlapping points is not evidence either way.

The honest reading: this experiment **could not resolve** the effect it was built
to show. Low error correlation really is necessary, since averaging cannot cancel a
mistake every member makes, and that is arithmetic rather than a measurement. What
this split shows is that it is not sufficient. Check the correlation matrix first,
because a high one rules the ensemble out and a low one does not rule it in.

## The out-of-fold rule, and the leak done on purpose

Fit the base models on the training set and predict that same training set, and this is
what those columns claim about themselves:

| Member | In-sample | Out-of-fold | Gap |
|---|---|---|---|
| Logistic | 0.9251 | 0.9229 | +0.0022 |
| k-NN (15) | 0.9322 | 0.9216 | +0.0106 |
| **Decision tree** | **1.0000** | 0.8921 | **+0.1079** |
| Random forest | 1.0000 | 0.9200 | +0.0800 |
| Gradient boosting | 1.0000 | 0.9192 | +0.0808 |
| Naive Bayes | 0.7724 | 0.7694 | +0.0030 |

Three members look infallible on rows they were fitted on. Building the honest
out-of-fold columns took **16.9 s**. Trained on those the meta-model claimed 0.9310 and
got **0.9207**; trained on the in-sample columns it claimed **1.0000** and got
**0.9130**. The leak invents **+0.0793** of reported accuracy no held-out data will
confirm, and separately costs **-0.0078** of real accuracy. You do not merely misjudge
the model; you ship a worse one.

![Meta-model weights](figures/fig-02-meta-weights.png)

| Member | Honest mean weight | Leaky mean weight | Trusted most on |
|---|---|---|---|
| Logistic | **+1.75** | +0.81 | HOROZ (+2.42) |
| k-NN (15) | **+1.66** | +0.88 | HOROZ (+2.60) |
| **Decision tree** | **+0.36** | **+1.88** | BOMBAY (+1.08) |
| Random forest | +1.19 | +1.48 | DERMASON (+1.70) |
| **Gradient boosting** | **+0.44** | **+1.86** | BOMBAY (+1.11) |
| Naive Bayes | +0.58 | +0.52 | BOMBAY (+1.33) |

**Read the table, not the summary statistic.** The obvious way to describe a leak is
that trust piles onto one member, and the spread of trust refuses to say that:
**1.39 honest against 1.37 leaky**, essentially unchanged. On that diagnostic alone
you would conclude nothing happened.

What the leak does is **reorder** who gets believed. Honestly, the meta-model leans
on logistic (**+1.75**) and k-NN (**+1.66**), the two members that cannot memorise,
and trusts the decision tree (**+0.36**) and gradient boosting (**+0.44**) least.
Feed it in-sample columns and that inverts: the tree jumps to **+1.88** and boosting
to **+1.86**, both now above everything else, while logistic falls to **+0.81**. The
three members that scored a perfect 1.0000 on their own training rows are exactly
the ones the leak promotes. That inverted weighting is what gets shipped.

The lesson is about diagnostics as much as leakage: a statistic that averages over
the thing you are hunting will hide it.

Per-class weights are also what voting cannot express: a vote gives every member the
same say on every class forever, and can never subtract a member. `StackingClassifier`
reproduced my hand-built version to **0.0000** difference.

## The board, the same rule in regression, and the bill

![Members against ensembles](figures/fig-03-members-vs-ensembles.png)

On California Housing the pattern is starker. Boosting alone reached **R² 0.7898** and
stacking reached **0.7897**, a hair behind. The plain average, the regression version
of a vote, managed **0.7224**, worse than boosting alone, because it counted ridge
(0.5845) and k-NN (0.6296) equally. The meta-model's weights explain both: **ridge
+0.02, k-NN +0.05, boosting +0.93**. It learned to be gradient boosting.

![Cost and gain](figures/fig-04-cost-and-gain.png)

Stacking with $k$ folds and $M$ members costs $M(k+1)$ base fits plus the meta-model,
and every member has to be trained, versioned, loaded and served. Voting is nearly
free once the members exist and rarely hurts much, so it is a reasonable default,
but pick hard or soft by measuring both rather than by reputation. Stacking earns
its cost when the members are genuinely different, and here its 0.0022 win is 0.4
standard errors, which is a result to report as "no measurable difference".

## Cheat sheet

| | |
|---|---|
| **Hard voting** | Counts labels and ties on an even number of members (53 of 2,700 rows here), but caps every member at one vote, which won: 0.9185 against 0.9181 |
| **Soft voting** | Averages probabilities. Better in theory and only in practice once members are calibrated: an overconfident member dominates it. Calibrate first, then try both |
| **Diversity** | Check pairwise error correlation first. High rules the ensemble out; low does not rule it in. Five k-NN variants at 0.888 gained +0.0007 while six families at 0.540 gained -0.0004 |
| **Stacking** | A small meta-model, logistic or ridge, learns per-class weights and may weight a member negatively |
| **The rule** | Meta-model features must be out-of-fold. The leak invented +0.0793 and cost -0.0078 |
| **Cost** | $M(k+1)$ fits. Compare the gain to one standard error (0.0053 here) before believing it |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 4](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#Ensemble` `#Stacking` `#VotingClassifier` `#ModelBlending`
`#DataLeakage` `#Python` `#ScikitLearn` `#DataScience` `#MLTutorial`
