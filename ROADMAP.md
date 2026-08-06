# Roadmap

Forty-four projects across nine parts. Each one states a question that could come
back "no", answers it on an openly licensed dataset, and ends with a verdict.

Status is one of **shipped** (executed, figures read, README written),
**drafting** (notebook exists, verdict not settled), **planned** (question and
dataset chosen, nothing written).

Datasets listed here are the intended ones. When a dataset turns out not to
support the question, I swap it and note the swap in the project README rather
than bending the question to fit the data.

---

## Part 1 — Foundations

Four projects on the things that decide whether a number means anything, before
any model is chosen.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 01-01 | Does the choice of cross-validation split move the reported score more than the choice of model? | USGS Earthquake Catalog (public domain, live API) | shipped |
| 01-02 | Does the textbook bias-variance curve actually appear on real data, or only on simulations? | UCI Superconductivity, 21k rows (CC BY 4.0) | planned |
| 01-03 | Can you predict where a learning curve plateaus from its first ten percent? | Six OpenML classification sets of varying size | planned |
| 01-04 | How much does the ranking of models change when you change only the metric? | UCI SECOM semiconductor yield (CC BY 4.0) | planned |

## Part 2 — Supervised learning, the classical core

Eight projects. Each pairs a from-scratch implementation with the library version,
so the reader can see what the library is hiding.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 02-01 | When features are correlated, do ridge and lasso disagree about which one matters, and does that disagreement matter for prediction? | UCI Communities and Crime (CC BY 4.0) | planned |
| 02-02 | Is a well-ranked classifier also a well-calibrated one? | NASA Kepler Objects of Interest, confirmed vs false positive (public domain) | planned |
| 02-03 | Does k-NN degrade with the number of columns, or with something else? | Five real datasets spanning 8 to 561 features, with intrinsic dimension estimated | planned |
| 02-04 | What does a decision tree learn that a linear model structurally cannot, and what does it pay for that? | UCI Dry Bean, 13k seeds, 7 classes (CC BY 4.0) | planned |
| 02-05 | Where does the variance reduction in a random forest actually come from: the bagging, or the feature subsampling? | UCI Steel Plates Faults (CC BY 4.0) | planned |
| 02-06 | Which LightGBM hyperparameter is doing the work, and can you find it without a grid search? | Open Food Facts nutrition grade prediction (ODbL) | planned |
| 02-07 | When does the kernel trick pay for itself, and when is it a slower linear model? | UCI Rice and UCI Raisin, small and separable by design (CC BY 4.0) | planned |
| 02-08 | For imbalanced data, does resampling beat simply moving the decision threshold? | Credit card fraud, 0.17 percent positives (ODbL) | planned |

## Part 3 — Unsupervised learning

Six projects. The theme is that unsupervised results are easy to believe and hard
to check, so each one builds the check.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 03-01 | k-means finds k clusters whether or not there are k clusters. How do you tell the difference? | Global Power Plant Database (CC BY 4.0) | planned |
| 03-02 | Does density-based clustering recover structure that k-means provably cannot, on data that was not designed to make the point? | GBIF species occurrence records (CC0) | planned |
| 03-03 | Can a Gaussian mixture tell you the number of components, or does it only tell you the number you asked for? | OpenAQ hourly air quality (CC BY 4.0) | planned |
| 03-04 | Do principal components correspond to anything a domain expert would name? | UCI Human Activity Recognition, 561 sensor features (CC BY 4.0) | planned |
| 03-05 | Does the cluster you found in a UMAP plot survive a change of seed? | Fashion-MNIST (MIT) and a tabular set, embedded 50 times each | planned |
| 03-06 | Do anomaly detectors agree with each other, and does agreement mean they are right? | NASA Kepler light curve summaries (public domain) | planned |

## Part 4 — Deep learning, the core

Seven projects. Two are written without a framework so the mechanics are visible.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 04-01 | Backpropagation from scratch in NumPy: does a hand-derived gradient match autograd to machine precision, and where does it stop matching? | UCI Dry Bean (CC BY 4.0) | planned |
| 04-02 | Do adaptive optimizers find better minima, or just find them faster? | Fashion-MNIST (MIT) | planned |
| 04-03 | Dropout, weight decay, and augmentation all reduce overfitting. Do they reduce the same overfitting? | CIFAR-10 (MIT) | planned |
| 04-04 | What does a convolution buy over a fully connected layer with the same parameter count? | Speech Commands spectrograms (CC BY 4.0) | planned |
| 04-05 | How little labelled data do you need before transfer learning stops helping? | Plant seedlings or an equivalent CC BY image set | planned |
| 04-06 | Do LSTMs beat a well-tuned linear model on real sequences, or only on synthetic ones? | UCR time series archive, ten datasets | planned |
| 04-07 | A transformer from scratch: what does each attention head end up looking at? | arXiv abstract metadata (CC0) | planned |

## Part 5 — Generative models

Five projects. Small models, trained to completion on a laptop, with the failure
modes shown rather than cropped out.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 05-01 | Is an autoencoder bottleneck doing compression, or memorisation? | Fashion-MNIST (MIT) | planned |
| 05-02 | Does the VAE latent space actually become the prior it was trained toward? | Fashion-MNIST (MIT) | planned |
| 05-03 | Mode collapse: can you see it coming from the training curves alone? | A two-class subset chosen so collapse is visible | planned |
| 05-04 | A minimal diffusion model: how many denoising steps are actually needed? | Fashion-MNIST (MIT) | planned |
| 05-05 | Normalizing flows give exact likelihoods. Is an exact likelihood a useful one? | UCI Superconductivity (CC BY 4.0) | planned |

## Part 6 — Self-supervised learning

Four projects. The question running through all of them is what a model learns
when nobody tells it the answer.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 06-01 | Contrastive pretraining needs augmentations. Which augmentation is carrying the result? | Speech Commands (CC BY 4.0) | planned |
| 06-02 | Masked modelling on sequences: does the mask ratio change what gets learned, or only how fast? | UCR time series archive | planned |
| 06-03 | A pretext task can be solved without learning anything useful. How do you detect that? | Rotation and jigsaw pretext on a CC BY image set | planned |
| 06-04 | Does self-supervision help on tabular data, where there is no obvious augmentation? | Open Food Facts (ODbL) | planned |

## Part 7 — Graphs and structure

Three projects on data where the relationships are the signal.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 07-01 | Does a graph neural network beat a plain MLP on node features alone, and by how much per hop? | ogbn-arxiv citation graph (ODC-BY) | planned |
| 07-02 | Link prediction: how much of the score is the model, and how much is the degree distribution? | Wikipedia clickstream (CC BY-SA 3.0) | planned |
| 07-03 | For sequence labelling, does a CRF still earn its place next to a neural tagger? | An openly licensed annotated corpus | planned |

## Part 8 — Reinforcement learning

Five projects. The environments are built from real data rather than from the
standard toy suite, which makes the results harder to hand-wave.

| ID | Question | Dataset / environment | Status |
|---|---|---|---|
| 08-01 | On a bandit built from a real classification log, does Thompson sampling beat UCB, and does the answer depend on the reward scale? | Supervised-to-bandit conversion of a multiclass open dataset | planned |
| 08-02 | Value iteration on real terrain: does the optimal path change more with the reward shaping or with the discount factor? | SRTM / Copernicus elevation tiles (open) | planned |
| 08-03 | Q-learning and SARSA differ only in one term. On what kind of map does that term decide the route? | Gridworld generated from open street and elevation data | planned |
| 08-04 | DQN adds a replay buffer and a target network. Remove each one: which removal breaks it? | A compact custom environment, CPU-trainable | planned |
| 08-05 | Policy gradients have high variance. Which variance reduction actually reduces it? | Same environment as 08-04, for a controlled comparison | planned |

## Part 9 — Beyond the model

Two projects on the questions that get asked after a model works.

| ID | Question | Dataset | Status |
|---|---|---|---|
| 09-01 | Conformal prediction promises coverage without distributional assumptions. Does the promise hold when the data shifts? | USGS Earthquake Catalog, split across time (public domain) | planned |
| 09-02 | Do SHAP and permutation importance agree, and when they disagree, which one is misleading you? | ACS PUMS via folktables, income prediction (public domain) | planned |

---

## Order of work

Parts are not built in numerical order. The order below front-loads the projects
that set the standard for the rest, and groups projects that share a dataset so
the loading code gets written once.

1. Foundations, all four. They define the evaluation vocabulary every later project uses.
2. Supervised classical, 02-03 and 02-08 first, since both push back on common advice.
3. Unsupervised, 03-05 first, for the same reason.
4. Deep learning 04-01 and 04-02, which establish the training loop the later parts reuse.
5. Everything else, three to four projects per batch.

## Definition of done, per project

A project moves to **shipped** only when every box in the checklist at the end of
[`STYLE.md`](STYLE.md) is ticked.
