# UMAP

### Faster than t-SNE, and it keeps more of the shape — but check that yourself

**[Open the notebook](notebook.ipynb)** · Part 6, Dimensionality reduction ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | What UMAP optimises and how that differs from t-SNE, what `n_neighbors` and `min_dist` change, how to measure the "preserves global structure" claim instead of repeating it, and why having a `transform` changes where you are allowed to use it |
| **You should already know** | [t-SNE](../03-t-sne/), [PCA](../01-principal-component-analysis/) |
| **Datasets** | UCI Dry Bean, UCI Breast Cancer |
| **Runtime** | Two to four minutes on a laptop CPU |

---

## Read this first: what actually ran

`umap-learn` is not part of scikit-learn and it was **not installed** on the machine
that executed this notebook. The import guard printed:

```
umap-learn available : False
neighbour method used: Isomap
running the scikit-learn fallback - pip install umap-learn for the real thing
```

So every number below comes from **`Isomap`**, with `SpectralEmbedding` alongside it
in one figure. Both are neighbour-graph methods that ship with scikit-learn, both
make the same argument about local versus global structure, and Isomap has a
`transform` the way UMAP does. What is missing is the `min_dist` dial, which has no
scikit-learn equivalent, and the stochastic layout — Isomap is deterministic.

Install `umap-learn` and re-run the notebook and every cell produces the UMAP
version instead. I would rather tell you which method produced 0.9499 than let you
assume.

## The measurement that matters

1,000 dry beans, 16 features, every pair of points scored two ways.

| Method | Global (distance rank correlation) | Local (trustworthiness, k=15) |
|---|---|---|
| PCA | **0.9524** | 0.9434 |
| t-SNE | 0.8516 | **0.9875** |
| Isomap | 0.9499 | 0.9445 |

This is the whole local-versus-global trade in six numbers. t-SNE wins the local
column by 0.043 and loses the global column by 0.098. The neighbour-graph method
sits within **0.0025** of PCA on global structure while giving up almost nothing
locally.

Two caveats worth more than the table. This is one dataset at one setting —
`n_neighbors=15` against `perplexity=30` is one point on two different curves. And
"more global structure than t-SNE" is a comparison to a method that discards nearly
all of it. Clearing that bar does not make the map a distance-preserving
projection. If you need distances, use PCA.

## Three maps of the same beans

![Three maps](figures/fig-01-three-maps.png)

3,000 beans sampled from 13,611, 16 features, timed on the same CPU:

| Method | Seconds |
|---|---|
| PCA | 0.0 |
| t-SNE | **19.4** |
| Isomap | 3.2 |

Six times faster than t-SNE at 3,000 rows. The gap you see quoted is larger because
most of the reported speed-up only appears at tens of thousands of rows, where
t-SNE's all-pairs normalisation starts to hurt. Compare your own timings before
repeating anyone's.

## The dials

![The dials](figures/fig-02-dials.png)

`n_neighbors` is the dial that matters. It sets how many neighbours define "local"
when the graph is built, and it is the closest thing UMAP has to t-SNE's
`perplexity`. Small values (2 to 10) splinter the layout into islands; large values
(50 to 200) force a broader arrangement and smooth fine structure away.

`min_dist` does something narrower than people assume. It never touches the graph —
it only sets a floor on how close two points may sit in the output. Changing it
changes how a plot looks without changing what the algorithm concluded.

Because `umap-learn` was absent, this figure shows one dial across three settings
(5, 15, 50) for two methods, Isomap on the top row and `SpectralEmbedding` on the
bottom, rather than a `n_neighbors` × `min_dist` grid. Widening the neighbourhood
merges groups that a narrow one kept apart, for both methods. A reader shown one
panel would count a different number of clusters depending on which panel you
picked.

**Try three settings of `n_neighbors` before you believe any of them.**

## Where the long distances go

![Global structure](figures/fig-03-global-structure.png)

Each panel plots distance in the original 16 features against distance in the map.
A tight diagonal cloud means the long distances survived; a shapeless one means they
did not. PCA 0.952, t-SNE 0.852, Isomap 0.950.

Spearman rather than Pearson, because no method here promises to preserve distances
on a linear scale — only their ordering.

## It has a transform, so it can go in a pipeline

![Transform](figures/fig-04-transform.png)

| Estimator | Has `.transform`? |
|---|---|
| PCA | True |
| t-SNE | **False** |
| SpectralEmbedding | **False** |
| Isomap | True |

This is the practical difference, and it is bigger than the speed. Fit on 398
training rows of Breast Cancer, `transform` the 171 held-out rows, then ask each
held-out point what class its nearest training neighbour in the map belongs to:
**90.6%** agree. The test rows never influenced the map.

Being allowed in a pipeline is not the same as earning a place in it:

| Pipeline | Accuracy | Time for 5 folds |
|---|---|---|
| All 30 features | **0.9789** | 0.1 s |
| PCA to 2D | 0.9578 | 0.1 s |
| Isomap to 2D | 0.9561 | 0.9 s |

Squeezing 30 features into 2 costs **0.0211** accuracy, and the non-linear reduction
finishes 0.0017 *behind* plain PCA while taking nine times as long. The reducer was
refit inside every fold — which is only possible because it has a `transform`.

## The same warnings as t-SNE, and one honest surprise

Three planted clusters with spreads and gaps I chose myself, so the right answer is
known.

| | Real | Isomap |
|---|---|---|
| Radius, cluster 0 | 0.56 | 0.62 |
| Radius, cluster 1 | 4.25 | 4.18 |
| Radius, cluster 2 | 0.59 | 0.41 |
| Gap ratio (0→2 over 0→1) | 4.99 | **5.02** |

Here the fallback changes the story, and I am not going to paper over it. Isomap
runs classical MDS on geodesic distances, so it is explicitly a global method, and
it reproduced the seven-fold difference in cluster spread and the five-fold gap
ratio almost exactly. UMAP would not: its per-point distance rescaling is precisely
what destroys cluster size, which is why the standard warning exists. Run this cell
with `umap-learn` installed and expect the radii to flatten out.

Seed stability read **1.000** by construction — Isomap is deterministic. UMAP is
not. Set `random_state` and say what you set it to.

What does not change either way: **empty space still means nothing**, because gaps
come from the repulsion term rather than from the data.

## Cheat sheet

| | |
|---|---|
| **Use it when** | You want to look at high-dimensional data and have more rows than t-SNE will sit still for |
| **Also use it when** | You need a non-linear reduction inside a model. It has a `transform`, so this is legal |
| **Do not use it when** | You need distances. PCA scored 0.9524 on global structure and takes 0.0 s |
| **Scaling** | Required. It is distance-based |
| **Main dials** | `n_neighbors` (local versus global — try 5 / 15 / 50), `min_dist` (packing only), `random_state` always |
| **Install** | `pip install umap-learn`. Not in scikit-learn. `Isomap` is the nearest thing that ships with it |
| **Watch out** | Cluster sizes and empty space are artefacts. Measure the global claim on your own data before quoting it |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 6](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#UMAP` `#Isomap` `#DimensionalityReduction` `#DataVisualization`
`#Manifold` `#UnsupervisedLearning` `#Python` `#ScikitLearn` `#MLTutorial`
