# Gaussian Mixture Models

### Clustering that admits it is uncertain

**[Open the notebook](notebook.ipynb)** · Part 5, Unsupervised learning ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | How a mixture model differs from k-means, what expectation-maximisation does, what the covariance types buy, and how BIC picks the number of components |
| **You should already know** | [k-Means](../01-k-means/) |
| **Datasets** | UCI Dry Bean, plus the stretched blobs that broke k-means |
| **Runtime** | About a minute on a laptop CPU |

---

## The idea

[k-means](../01-k-means/) has two limits: every cluster is a sphere, and every
point belongs to exactly one. A Gaussian mixture relaxes both.

$$p(x) = \sum_{i=1}^{k} \pi_i \, \mathcal{N}(x \mid \mu_i, \Sigma_i)$$

Three things per component: how common it is, where it sits, and **what shape it
has**. That covariance matrix is the whole upgrade — a cluster can be a stretched,
tilted ellipse instead of a ball.

**k-means is the special case** where every covariance is the identity and every
assignment is hard.

**Expectation-maximisation** breaks the circularity: given components, compute the
probability each point came from each (E-step); given those probabilities,
recompute each component as a weighted average (M-step). Same shape as Lloyd's
algorithm, with soft assignment.

## The shapes k-means could not see

![Shapes](figures/fig-01-shapes.png)

On stretched blobs where k-means slices across the grain:

| Method | Adjusted Rand |
|---|---|
| k-means | 0.589 |
| Gaussian mixture, full covariance | **0.796** |

The third panel is what k-means genuinely cannot do. **17.3% of points are
assigned with under 80% confidence** — the model telling you where it does not
know. k-means assigns every point with equal, unearned certainty.

## The covariance types

![Covariance types](figures/fig-02-covariance-types.png)

**spherical** is k-means with soft assignment. **diag** allows axis-aligned
ellipses. **tied** shares one shape across clusters. **full** lets every cluster
tilt freely, and costs the most parameters. Step down from `full` when you have few
points per cluster — that is how a mixture model overfits.

## Choosing k, and where BIC misleads

![Choosing k](figures/fig-03-choosing-k.png)

$$\text{BIC} = -2\log L + p \log n$$

Unlike inertia, BIC can have a genuine minimum, because every extra component costs
$p \log n$.

**But BIC did not pick 7.** It preferred **twelve** components with full
covariance, and that is worth being straight about because it is the usual outcome.

BIC answers "how many Gaussians describe this density best", which is a different
question from "how many varieties of bean are there". Seven real varieties, each
slightly non-Gaussian, are described better by twelve Gaussians than by seven — so
BIC buys extra components to patch the shape mismatch.

Use it to narrow the range, then look at the clusters. At the true k of 7:

| Method | Adjusted Rand |
|---|---|
| Gaussian mixture (full) | **0.682** |
| k-means | 0.669 |

## Cheat sheet

| | |
|---|---|
| **Use it when** | Clusters are elliptical or differently shaped; you want probabilities not hard labels |
| **Avoid it when** | Clusters are non-convex — crescents and rings need [DBSCAN](../04-dbscan-and-hdbscan/); very few points per cluster |
| **Scaling needed** | Yes, as for k-means |
| **Main dials** | `n_components`, `covariance_type`, `n_init` (EM finds local optima) |
| **Choosing k** | BIC narrows the range. It does not answer your question for you |
| **Bonus** | `score_samples` gives a density, so it doubles as an [anomaly detector](../06-anomaly-detection/) |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 5](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#GaussianMixture` `#GMM` `#Clustering` `#UnsupervisedLearning`
`#ExpectationMaximization` `#Python` `#ScikitLearn` `#MLTutorial` `#BIC`
