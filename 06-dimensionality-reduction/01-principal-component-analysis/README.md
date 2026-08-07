# Principal Component Analysis

### Fewer columns, almost the same information

**[Open the notebook](notebook.ipynb)** · Part 6, Dimensionality reduction ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | What a principal component is, how to compute PCA from the covariance matrix by hand, how many components to keep, and whether reducing dimensions actually helps a downstream model |
| **You should already know** | Matrix multiplication. Eigenvectors help, but the notebook explains what is needed |
| **Datasets** | UCI Dry Bean (13,611 × 16), Breast Cancer Wisconsin (569 × 30) |
| **Runtime** | About a minute on a laptop CPU |

---

## The idea

Dry Bean has sixteen measurements per bean, but not sixteen independent facts.
`Area`, `Perimeter`, `ConvexArea` and `EquivDiameter` all essentially say *how big
is it*: several pairs correlate above **0.99**.

PCA finds new axes that are combinations of the old ones, ordered so the first
captures as much variation as possible. It is a rotation, chosen so variance
concentrates in the earliest axes.

Components are the **eigenvectors** of the covariance matrix; each eigenvalue is
the variance along it. The from-scratch version agrees with scikit-learn to
**2.2 × 10⁻¹⁶**.

## How much do you keep?

![Scree](figures/fig-01-scree.png)

| Variance explained | Components needed |
|---|---|
| 80% | 2 of 16 |
| 90% | 4 of 16 |
| 95% | **4 of 16** |
| 99% | 7 of 16 |

## What the components mean

![Projection](figures/fig-02-projection.png)

PCA is **unsupervised**: it never saw the varieties. It only looked for
directions of large variance, and the varieties came out largely separated along
them.

![Loadings](figures/fig-03-loadings.png)

Reading the weights names them. **PC1 loads heavily and in the same direction on
`Area`, `Perimeter`, `ConvexArea`, `EquivDiameter` and the axis lengths: it is
measuring how big the bean is**, which is why it carries so much variance: those
six columns were always saying the same thing. **PC2 separates elongation from
bulk: it is measuring shape.**

Naming components is not always possible, and forcing it is a good way to fool
yourself. When it works it is a real insight.

## Does it actually help? Not the way I expected

![Does it help](figures/fig-04-does-it-help.png)

| Dataset | All raw columns | Best PCA result |
|---|---|---|
| UCI Dry Bean | 0.9234 (16 cols) | 0.9239 at 10 components |
| Breast Cancer | 0.9807 (30 cols) | 0.9807 at **30** components |

**PCA never beat using every column.** Both curves climb toward the baseline and
flatten onto it; neither rises above it.

I expected Breast Cancer to show a clear gain: thirty correlated columns against
only 569 rows is exactly the crowded regime where dropping dimensions is supposed
to pay. It did not. The best score came from keeping all thirty components, which
is just a rotation and no reduction at all.

What PCA *did* buy is **compression at no cost in accuracy**: six components match
all sixteen exactly on Dry Bean, so ten columns can go for free. That is worth
having when a model is slow, when storage matters, or when you need a
two-dimensional picture of something with sixteen axes. It is not an accuracy
trick, and material presenting it as one is overselling it.

## Cheat sheet

| | |
|---|---|
| **Use it when** | Columns are correlated; you want a 2-D picture; you need to speed up a downstream model |
| **Avoid it when** | You need to explain individual features; the structure is non-linear (try [t-SNE](../03-t-sne/) or [UMAP](../04-umap/)); the informative direction has low variance |
| **Scaling needed** | Yes, whenever units differ. Otherwise the largest-unit column becomes PC1 |
| **Main dials** | `n_components`, an integer, or a variance fraction like `0.95` |
| **Watch out** | Fit PCA **inside** the pipeline. Fitting on all data before splitting leaks test information |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 6](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#PCA` `#DimensionalityReduction` `#UnsupervisedLearning`
`#Python` `#ScikitLearn` `#DataScience` `#MLTutorial` `#Eigenvectors`
