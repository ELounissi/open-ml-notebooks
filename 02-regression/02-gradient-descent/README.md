# Gradient descent

### The learning rate has an exact ceiling, and full batch won on the clock

**[Open the notebook](notebook.ipynb)** · Part 2, Regression ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Why an iterative solver exists when an exact formula already does, how to derive and code the update rule, the exact stable ceiling on the learning rate, what batch size actually buys, and what feature scaling saves |
| **You should already know** | [Linear regression](../01-linear-regression/) and the normal equation, NumPy arrays, and what a derivative is |
| **Datasets** | California Housing (20,640 districts, 8 features), plus random matrices for the timings |
| **Runtime** | Under two minutes on a laptop CPU |

---

## Start here: the folklore about small batches did not survive the stopwatch

Everyone repeats that stochastic gradient descent is the fast one. Counted in seconds
rather than passes over the data, here it was the slow one *and* the inaccurate one.
Full batch finished 1,500 updates in **0.181 s** and landed **1.000e-16** from the best
achievable loss; stochastic descent spent **0.355 s** on 41,250 updates and stopped
**3.699e-02** away, fourteen orders of magnitude worse, for **96% more wall-clock
time**. That is not an argument against small batches, it is an argument about what
makes them fast: memory pressure and model shape, neither of which applies here.

## Why iterate when a formula exists

![Cost of solving exactly](figures/fig-01-cost-of-solving-exactly.png)

| Features | Form $X^\top X$ | Solve | One gradient step |
|---|---|---|---|
| 100 | 0.00162 s | 0.00940 s | 0.00028 s |
| 200 | 0.00561 s | 0.08703 s | 0.00048 s |
| 400 | 0.00602 s | 0.27528 s | 0.00067 s |
| 800 | 0.02150 s | 0.58247 s | 0.00147 s |
| 1600 | 0.07450 s | 0.73075 s | 0.00384 s |

Theory says every doubling of the feature count multiplies the solve by 8. Measured,
the four doublings gave **9.3×, 3.2×, 2.1×, 1.2×**, and the exponent fitted on the
largest three sizes came out at **0.70** against a theoretical 3. Forming $X^\top X$
fitted **1.81** (theory 2) and one gradient step **1.25** (theory 1).

Say the solve one plainly: at these sizes a multithreaded LAPACK solve is nowhere near
its asymptotic regime, so the exponent is meaningless and any time projection built on
it is nonsense. The notebook prints both projections to 100,000 features so you can
see how bad it gets: **0.00 hours** from the measured exponent against **50 hours**
from the theoretical one, off the same last data point. The memory projection is the one that
survives, because it is arithmetic on the shape rather than a fitted slope:
**$X^\top X$ alone would need 80 GB** at 100,000 features.

The two exact-solution columns also never cross. `solve` is above `form XtX` at every
width measured, 5.8× at 100 features, 45.7× at 400, 9.8× at 1600. Forming the Gram
matrix is $O(mn^2)$ with 5,000 rows against an $O(n^3)$ solve, so below about 5,000
features the forming should dominate and it never does. At this scale you are timing
constants and parallel efficiency, not exponents.

Cost is the reason people quote. Rank is the reason that stops you. Duplicate one
column and the matrix has 9 columns at **rank 8**; numpy refused with `Singular
matrix`. Gradient descent walked the same problem to **MSE 0.52432099**, matching
`lstsq` exactly, and put **+0.414810** on each duplicate, without being told anything
about the duplication.

## The update rule, checked against the closed form

$$\nabla J(w) = \frac{2}{m}X^\top(Xw - y), \qquad w \leftarrow w - \eta\,\nabla J(w)$$

On eight standardised features, run to a flat gradient in **447 steps**, gradient
descent reached **MSE 0.5243209862** against the closed form's **0.5243209862**. The
largest disagreement in the weights was **2.700e-09**, in the intercept **4.441e-16**,
in the predictions **2.524e-08**. Nothing about the destination is approximate, only
the route.

## The learning rate is a cliff, not a curve

![Three learning rates](figures/fig-02-three-learning-rates.png)

Error along an eigendirection of curvature $\lambda$ is multiplied by
$(1-\eta\lambda)$ every step, so the safe range is exactly $\eta < 2/\lambda_{\max}$.
Here that ceiling is **0.4934**. Multiples of it, 600 steps each:

| Multiple | Learning rate | Final loss | Outcome |
|---|---|---|---|
| 0.90 | 0.44402 | 5.2432e-01 | converging |
| **0.99** | 0.48842 | **5.2432e-01** | converging |
| **1.01** | 0.49829 | **3.3577e+07** | growing |
| 1.10 | 0.54269 | 1.6724e+92 | growing |
| 1.50 | 0.74003 | inf | overflowed at step 510 |

There is no gentle degradation between 0.99 and 1.01. You are not hunting a broad
optimum, you are finding the edge of a cliff and standing back from it. A loss of
`nan` in a training log is almost always this.

**Stop on the gradient, not on the loss.** Same problem, rate 0.01039, tolerance 1e-6:
the loss rule quit after **2,203 steps**, still **6.525e-02** from the exact answer.
The gradient rule ran **8,742 steps** and finished **2.316e-06** away. The loss rule
cannot tell "nowhere left to go" from "barely moving".

## Batch, stochastic, mini-batch

![Batch, stochastic and mini-batch](figures/fig-03-batch-stochastic-minibatch.png)

Row lengths are the wrinkle: median squared length **5.1**, max **14307.4**. Clipping
at ±5 touches **282 of 20,640 rows** and brings the max to **70.8**. Each method then
gets the largest step safe for its own gradient estimate, so this compares batch sizes
and not learning rates.

| Method | Updates | Seconds | Final gap | Uphill steps | Uphill above 1e-15 | Wobble |
|---|---|---|---|---|---|---|
| Batch, all 20,640 rows | 1,500 | **0.181** | **1.000e-16** | 21 / 299 | **0** | 0.808 |
| Mini-batch, 256 rows | 16,000 | 0.352 | 1.711e-05 | 85 / 319 | 85 | 0.281 |
| Stochastic, 1 row | 41,250 | 0.355 | 3.699e-02 | 140 / 274 | 140 | 0.293 |

Per pass over the data the stochastic curve is far ahead, and the right panel shows it:
that is why nobody trains a network full-batch. Per second the ordering reverses,
because one batch step is a single BLAS call across every core on a cached matrix while
41,250 single-row steps are 41,250 trips through the Python interpreter.

**Two of those columns lie, and the sixth one is there because of it.** Read the raw
uphill count and the wobble on their own and full batch looks like the twitchiest of
the three, 21 uphill moves and the largest wobble at 0.808. Batch descent under the
stable ceiling is monotone and cannot do that. The sixth column says what happened:
**none** of batch's 21 uphill moves happened while a gap above 1e-15 remained, and the
largest of them is 1.220e-16. Batch is the only run that reached the exact minimum, so
for its last 50 recorded points the plotted "distance from the best achievable loss" is
a subtraction of two doubles agreeing to every bit, and rounding goes up as often as
down. The wobble column, being the spread of the log of that quantity, then hands its
biggest number to the run that converged best.

The general form is worth keeping: **a convergence diagnostic measured against a known
optimum stops meaning anything once the difference reaches machine precision**, and it
gives no warning, it just starts printing numbers that read like instability.
Mini-batch's 85 and stochastic's 140 are real. And at a constant learning rate
stochastic descent never converges to a point; it settles into a cloud, and decaying
the rate shrinks it.

## What scaling saves

![Scaling changes the shape](figures/fig-04-scaling-changes-the-shape.png)

Two centred features, same model, same minimum loss of **0.65363196** both ways.
Original units: condition **44.5**, **439 steps**. Standardised: condition **1.3**,
**81 steps**. On all eight features with a shared 20,000-iteration budget it stops
being cosmetic:

| Version | Condition number | Steps | Gradient left | Loss gap |
|---|---|---|---|---|
| Original units | **42,145,624** | 20,000 (budget spent) | 4.52e-02 | 6.567e-01 |
| Standardised | 44 | **359** | 9.99e-09 | 1.443e-15 |

The normal equation does not care about units: the exact solution is the exact
solution. The moment you walk downhill, units decide whether the walk finishes at all.

## Cheat sheet

| | |
|---|---|
| **Use it when** | The exact solution is too expensive, does not exist, or was never available: every model after this chapter |
| **Update rule** | $w \leftarrow w - \eta\,\frac{2}{m}X^\top(Xw-y)$, at $O(mn)$ per step against $O(n^3)$ for one exact solve |
| **Safe learning rate** | Strictly below $2/\lambda_{\max}$. At 0.99× it converges, at 1.01× it grows to 3.4e7 in 600 steps |
| **Scaling** | Not optional. It sets the condition number, and the condition number sets the step count |
| **Batch size** | Small batches win per pass over the data. On one cached matrix product they lost on the clock |
| **Stopping rule** | Relative gradient length, plus an iteration cap, plus a finite-loss check |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 2](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#GradientDescent` `#Optimization` `#LearningRate`
`#LinearRegression` `#Python` `#NumPy` `#DataScience` `#MLTutorial` `#SGD`
