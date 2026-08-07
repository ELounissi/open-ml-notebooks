# Lasso regression

### The penalty that deletes features instead of shrinking them

**[Open the notebook](notebook.ipynb)** · Part 2, Regression ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Why swapping a square for an absolute value produces exact zeros, what that geometry looks like, how Lasso behaves on correlated features, and why `LassoCV` is **not** a feature selector |
| **You should already know** | [Ridge regression](../04-ridge-regression/) |
| **Datasets** | California Housing, with noise columns added to test feature selection |
| **Runtime** | About a minute on a laptop CPU |

---

## One character of difference

$$J_{\text{ridge}} = \|Xw - y\|^2 + \alpha \sum_i w_i^2 \qquad J_{\text{lasso}} = \|Xw - y\|^2 + \alpha \sum_i |w_i|$$

**Lasso sets weights to exactly zero. Ridge never does.**

The reason is the derivative. Near zero the square's gradient is $2w$, which
vanishes as $w$ does: the push toward zero weakens exactly when it is needed to
finish, so the weight glides in without arriving. The absolute value's gradient is
$\pm\alpha$, constant all the way down. It keeps pushing until the weight hits zero.

![Geometry](figures/fig-01-geometry.png)

Geometrically: Ridge's budget region is a **circle**, Lasso's a **diamond** with
corners on the axes. A curve touching a circle almost never touches at exactly the
top. A curve touching a diamond very often touches at a corner, and a corner is a
point where one coordinate is exactly zero.

![Paths](figures/fig-02-paths.png)

**Read the two x axes before the lines.** They are different, and that is half the
finding. Lasso's coefficients hit zero one by one:

| alpha | Features kept |
|---|---|
| 0.001 | all 8 |
| 0.05 | 4: MedInc, HouseAge, Latitude, Longitude |
| 0.3 | 1: MedInc |
| 1 | none |

The first version of this figure drew ridge on that same range and captioned it
"ridge's lines glide toward zero and none arrive". Over that range ridge does not
glide. It sits still: at alpha 1, where lasso has already emptied the model,
ridge's weight vector is at **||w|| 1.560** against 1.561 at the far left. Getting
a visible glide takes **alpha 100,000**, four decades further out, and even there
all 8 features survive with the smallest coefficient at **0.00385**, small and not
zero. So each panel now gets the range where its penalty actually acts.

The wider point: **for the same nominal alpha the two penalties are nowhere near
the same strength**, because one charges $|w|$ and the other $w^2$, and for the
small weights that decide selection $w^2$ is much the smaller number. "I used
alpha=0.1" means nothing until you say which penalty.

## The result that matters: LassoCV is not a feature selector

![Feature selection](figures/fig-03-feature-selection.png)

I added **30 pure noise columns** to California Housing and let `LassoCV` choose
alpha. It kept every real feature, and **17 of the 30 noise columns**. More than
half the garbage survived a method sold as automatic feature selection.

The reason is a mismatch of objectives nobody mentions. Cross-validation chose
alpha to **minimise prediction error**, landing on 0.0032, tiny. At that setting
the penalty barely bites, so useless columns get small non-zero weights instead of
exact zeros. Those small weights cost almost nothing in error, which is precisely
why cross-validation is indifferent to them.

| alpha | Real kept | Noise kept | CV RMSE |
|---|---|---|---|
| 0.0032 *(CV choice)* | 8/8 | **17/30** | 0.7267 |
| 0.01 | 7/8 | **1/30** | 0.7291 |
| 0.03 | 6/8 | 0/30 | 0.7497 |
| 0.1 | 3/8 | 0/30 | 0.8211 |

Raising alpha cleans the list out, and prediction error barely moves until real
features start going too. **Selecting features and predicting well are different
goals, and `LassoCV` optimises the second.** If you want a defensible shortlist,
choose alpha yourself: around 0.01 here buys a nearly clean list for 0.0024 RMSE.

## Where Lasso struggles

![Correlated features](figures/fig-04-correlated-features.png)

Given three near-identical copies of `MedInc`, Lasso keeps one **arbitrarily** and
the choice jumps from sample to sample. Twelve bootstrap refits:

| | Spread per coefficient | Zeros | Range | Sum of the three |
|---|---|---|---|---|
| Lasso | 0.360, 0.257, 0.191 | **17 of 36** | 0.000 to 0.791 | 0.756 ± 0.031 |
| Ridge | 0.113, 0.113, 0.128 | 0 of 36 | 0.051 to 0.574 | 0.849 ± 0.037 |

I had this panel captioned "ridge splits the weight evenly and stays put". Ridge
does neither. Its three lines wander between **0.051 and 0.574**, cross each other
repeatedly, and all three twins take a turn as the largest, exactly as lasso's do.
What ridge does is never zero anything and move about two to three times less.
Steadier is not still.

The genuinely stable quantity is the **total**. Both methods pin the sum of the
three twin coefficients to within a few per cent, and neither has any idea which
of the three columns earned it, because that question has no answer. Lasso
expresses that ignorance by picking one and calling the others zero, which reads
like a finding. Ridge expresses it by spreading the weight unevenly and zeroing
nothing, which reads like what it is.

Fine if you only want predictions. Misleading if you then say "the model selected
this feature, so it is the important one". For correlated groups **and** selection,
use [Elastic Net](../06-elastic-net/).

## Cheat sheet

| | |
|---|---|
| **Use it when** | Many features, most probably useless; you want a smaller model |
| **Prefer Ridge when** | Features are correlated and you want stable coefficients |
| **Scaling needed** | Yes, always. The penalty is unit-blind |
| **No closed form** | The absolute value is not differentiable at zero, so it is solved by coordinate descent, hence `max_iter` |
| **Watch out** | Among correlated features it picks one nearly at random |
| **Watch out more** | `LassoCV` tunes for prediction, not sparsity. It kept 17 of 30 noise columns here |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 2](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#Lasso` `#L1Regularization` `#FeatureSelection` `#Regression`
`#Python` `#ScikitLearn` `#DataScience` `#MLTutorial` `#SparseModels`
