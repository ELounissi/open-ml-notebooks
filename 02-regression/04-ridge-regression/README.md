# Ridge regression

### Paying a penalty for large weights, and getting stability back

**[Open the notebook](notebook.ipynb)** · Part 2, Regression ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | What the L2 penalty does, why correlated features make ordinary regression unstable, how to read a regularisation path, and what ridge does **not** buy you |
| **You should already know** | [Linear regression](../01-linear-regression/), [overfitting](../../01-foundations/03-overfitting-and-underfitting/) |
| **Datasets** | California Housing, plus a deliberately collinear construction |
| **Runtime** | About a minute on a laptop CPU |

---

## The idea

Least squares minimises error and nothing else. Ridge adds a second term — the
size of the weights themselves:

$$J(w) = \underbrace{\|Xw - y\|^2}_{\text{fit the data}} + \underbrace{\alpha \|w\|^2}_{\text{stay small}}$$

It has an exact solution, with $\alpha$ added down the diagonal — the "ridge" the
name refers to:

$$w = (X^\top X + \alpha I)^{-1} X^\top y$$

That also fixes a numerical problem for free: with perfectly collinear features
$X^\top X$ has no inverse and least squares has no unique answer, but adding
$\alpha$ makes it invertible for **any** $\alpha > 0$.

## What I expected, and what happened

I added a column equal to `MedInc` plus a whisper of noise (correlation
**0.999986**) expecting the textbook pathology — one weight at +1000, its twin at
−997.

**It did not happen.** The two weights came out around −0.3 and +1.2. The textbook
example quietly assumes scarce data; with 20,640 rows and 9 columns there is
enough evidence to keep the solution bounded. **Collinearity is only catastrophic
when rows are few relative to columns.**

## What ridge actually bought

![Stability](figures/fig-01-stability.png)

| | Cross-validated RMSE |
|---|---|
| OLS, original 8 features | 0.7263 |
| OLS, with the near-copy | 0.7263 |
| Ridge, with the near-copy | 0.7263 |

**No accuracy at all** — identical to four decimal places. But refit on 30
bootstrap samples:

| | Spread of the `MedInc` weight |
|---|---|
| OLS | 2.682 |
| Ridge | **0.084** |

**Thirty-two times more stable.** The OLS coefficient is not a number you could
report to anyone, because a different sample of the same data gives a different
answer.

So the honest pitch is not "ridge predicts better". It is **"ridge gives you
coefficients that mean something"**. If you only need predictions and have plenty
of rows, plain least squares was fine here.

## The regularisation path

![Path](figures/fig-02-path.png)

Every weight shrinks toward zero and **none of them reaches it**. That is the L2
signature. [Lasso](../05-lasso-regression/) is the version that sets weights to
exactly zero.

![Choosing alpha](figures/fig-03-choosing-alpha.png)

Cross-validation picked **alpha ≈ 11.7**. Push it to 10⁵ and the model degrades to
RMSE 1.02 — so much penalty it is nearly a constant.

## The scaling argument, which surprised me

Sweeping alpha with and without scaling, on data where I deliberately re-expressed
income in dollars and population in millions:

| alpha | Unscaled | Scaled |
|---|---|---|
| 1 | 0.7263 | 0.7263 |
| 10,000 | **0.7682** | 0.8178 |
| 1,000,000 | **0.8084** | 1.1355 |

**The unscaled version scored better**, which is the opposite of the moral I
expected to write.

Follow the reason, because it is the actual argument for scaling. Inflating
`MedInc` by 10,000 forced its coefficient to shrink by the same factor, and a tiny
coefficient is nearly free under an $\alpha\|w\|^2$ penalty. So the unscaled model
kept the most predictive feature almost unregularised while the penalty flattened
everything else. It won **by accident**, because of a unit I chose arbitrarily.

That is the problem, not a defence. **Without scaling, your choice of units decides
which features the penalty protects.** Scaling does not promise a better score — it
promises the penalty responds to usefulness rather than to measurement units.

## Cheat sheet

| | |
|---|---|
| **Use it when** | Features are correlated; many features relative to rows; you want interpretable, stable coefficients |
| **Prefer Lasso when** | You want features actually removed |
| **Scaling needed** | Yes — so the penalty tracks usefulness, not units |
| **Main dial** | `alpha`. `RidgeCV` picks it efficiently |
| **Free bonus** | Always has a unique solution, even with perfect collinearity |
| **Watch out** | Choose `alpha` inside cross-validation, never by peeking at the test set |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 2](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#RidgeRegression` `#Regularization` `#L2` `#LinearRegression`
`#Python` `#ScikitLearn` `#DataScience` `#MLTutorial` `#Multicollinearity`
