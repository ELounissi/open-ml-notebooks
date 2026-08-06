# Curriculum

Eighty-six notebooks, twelve parts, one method each. Read them in order or jump
straight to the one you need — every notebook stands alone.

Each notebook follows the same five-part shape, so once you have read one you
know where to look in all of them:

1. **The idea** — what the method does, in plain language, with one picture
2. **The maths** — only what you need, written out, no hand-waving
3. **From scratch** — a minimal implementation in NumPy, so nothing is magic
4. **In practice** — the scikit-learn or PyTorch version, annotated line by line
5. **When it wins, when it loses** — on the house datasets, with the reason

Status: **shipped** means executed with charts and a README. **planned** means the
method and dataset are chosen.

---

## Part 1 — Foundations

The vocabulary everything else uses. Start here if you are new.

| # | Notebook | Status |
|---|---|---|
| 01-01 | What machine learning actually does — the workflow end to end | planned |
| 01-02 | Train, validation, test: why you need all three | planned |
| 01-03 | Overfitting and underfitting, seen rather than described | planned |
| 01-04 | Cross-validation, and which flavour to use when | planned |
| 01-05 | Metrics for classification: accuracy, precision, recall, F1, ROC-AUC | planned |
| 01-06 | Metrics for regression: MAE, RMSE, R squared, MAPE | planned |
| 01-07 | Feature scaling and encoding categorical variables | planned |
| 01-08 | Missing data: what to do and what it costs | planned |
| 01-09 | Hyperparameter tuning: grid, random, and Bayesian search | planned |

## Part 2 — Regression

| # | Notebook | Status |
|---|---|---|
| 02-01 | Linear regression, from the normal equation to scikit-learn | shipped |
| 02-02 | Gradient descent, watched step by step | planned |
| 02-03 | Polynomial regression and the bias-variance tradeoff | planned |
| 02-04 | Ridge regression (L2) | planned |
| 02-05 | Lasso regression (L1) and automatic feature selection | planned |
| 02-06 | Elastic Net | planned |
| 02-07 | Outlier-resistant regression: Huber and RANSAC | planned |
| 02-08 | Quantile regression: predicting a range, not a number | planned |
| 02-09 | Poisson and other generalised linear models | planned |

## Part 3 — Classification

| # | Notebook | Status |
|---|---|---|
| 03-01 | Logistic regression | planned |
| 03-02 | k-Nearest Neighbours | planned |
| 03-03 | Naive Bayes | planned |
| 03-04 | Linear and Quadratic Discriminant Analysis | planned |
| 03-05 | Support Vector Machines and the kernel trick | planned |
| 03-06 | Decision trees, and how a split is chosen | planned |
| 03-07 | Imbalanced classes: resampling, weights, and thresholds | planned |
| 03-08 | Multiclass and multilabel strategies | planned |
| 03-09 | Probability calibration | planned |

## Part 4 — Ensembles

| # | Notebook | Status |
|---|---|---|
| 04-01 | Bagging, and why averaging helps | planned |
| 04-02 | Random Forests | planned |
| 04-03 | Extra Trees | planned |
| 04-04 | AdaBoost | planned |
| 04-05 | Gradient Boosting from first principles | planned |
| 04-06 | XGBoost, LightGBM, CatBoost compared | planned |
| 04-07 | Stacking and voting | planned |

## Part 5 — Unsupervised learning

| # | Notebook | Status |
|---|---|---|
| 05-01 | k-Means | planned |
| 05-02 | Choosing k: elbow, silhouette, gap statistic | planned |
| 05-03 | Hierarchical clustering and dendrograms | planned |
| 05-04 | DBSCAN and HDBSCAN | planned |
| 05-05 | Gaussian Mixture Models | planned |
| 05-06 | Anomaly detection: Isolation Forest, One-Class SVM, LOF | planned |
| 05-07 | Association rules with Apriori | planned |

## Part 6 — Dimensionality reduction

| # | Notebook | Status |
|---|---|---|
| 06-01 | Principal Component Analysis | planned |
| 06-02 | Kernel PCA, ICA, and NMF | planned |
| 06-03 | t-SNE | planned |
| 06-04 | UMAP | planned |
| 06-05 | Feature selection: filter, wrapper, embedded | planned |

## Part 7 — Neural networks

| # | Notebook | Status |
|---|---|---|
| 07-01 | The perceptron | planned |
| 07-02 | Multilayer perceptron and backpropagation, in NumPy | planned |
| 07-03 | The same network in PyTorch | planned |
| 07-04 | Activation functions and why they matter | planned |
| 07-05 | Optimisers: SGD, Momentum, RMSProp, Adam | planned |
| 07-06 | Regularisation: dropout, batch norm, weight decay, early stopping | planned |
| 07-07 | A training loop you can reuse | planned |

## Part 8 — Computer vision

| # | Notebook | Status |
|---|---|---|
| 08-01 | Convolution, pooling, and what a filter learns | planned |
| 08-02 | A CNN on Fashion-MNIST, layer by layer | planned |
| 08-03 | Classic architectures: LeNet, VGG, ResNet | planned |
| 08-04 | Data augmentation | planned |
| 08-05 | Transfer learning and fine-tuning | planned |
| 08-06 | Image segmentation, an introduction | planned |
| 08-07 | Object detection, an introduction | planned |

## Part 9 — Sequences and language

| # | Notebook | Status |
|---|---|---|
| 09-01 | Text preprocessing, bag of words, and TF-IDF | planned |
| 09-02 | Word embeddings: Word2Vec and GloVe | planned |
| 09-03 | Recurrent neural networks | planned |
| 09-04 | LSTM | planned |
| 09-05 | GRU, and how it compares to LSTM | planned |
| 09-06 | Sequence to sequence with attention | planned |
| 09-07 | The Transformer, built from scratch | planned |
| 09-08 | Fine-tuning a pretrained transformer | planned |
| 09-09 | Time series forecasting: ARIMA against ML against deep learning | planned |

## Part 10 — Generative models

| # | Notebook | Status |
|---|---|---|
| 10-01 | Autoencoders | planned |
| 10-02 | Variational autoencoders | planned |
| 10-03 | Generative adversarial networks | planned |
| 10-04 | Diffusion models, the smallest working example | planned |

## Part 11 — Reinforcement learning

| # | Notebook | Status |
|---|---|---|
| 11-01 | The reinforcement learning setup: agents, states, rewards | planned |
| 11-02 | Multi-armed bandits | planned |
| 11-03 | Markov decision processes, value and policy iteration | planned |
| 11-04 | Q-learning | planned |
| 11-05 | SARSA, and how it differs from Q-learning | planned |
| 11-06 | Deep Q-Networks | planned |
| 11-07 | Policy gradients and REINFORCE | planned |
| 11-08 | Actor-critic and PPO | planned |

## Part 12 — Putting it together

| # | Notebook | Status |
|---|---|---|
| 12-01 | **The scoreboard**: every method on every house dataset | planned |
| 12-02 | Interpreting models: permutation importance, SHAP, LIME | planned |
| 12-03 | Pipelines, and never leaking again | planned |
| 12-04 | Saving, loading, and serving a model | planned |
| 12-05 | The mistakes everybody makes | planned |

---

## The house datasets

Every notebook uses the same small set, so a comparison across chapters means
something. You can see immediately how a support vector machine and a random
forest differ, because they were asked the same question.

| Dataset | Task | Size | Why this one |
|---|---|---|---|
| California Housing | regression | 20,640 × 8 | Ships with scikit-learn, so chapter one runs with no download |
| Breast Cancer Wisconsin | binary classification | 569 × 30 | Small enough that every method trains instantly |
| **UCI Dry Bean** | 7-class classification | 13,611 × 16 | Published 2020 and barely used in tutorials, which keeps the book off the beaten path |
| UCI Bike Sharing | regression, time series | 17,379 × 16 | One dataset that works for both tabular regression and sequence models |
| Fashion-MNIST | image classification | 70,000 × 28 × 28 | Same shape as MNIST, several times harder, so CNN results are not all 99% |

Sources, licences, and retrieval dates are in [`data/README.md`](data/README.md).

## The scoreboard

Part 12 collects every method's result on every house dataset into one table, and
then explains the pattern. Some of it is predictable and some of it is not:
gradient boosting usually leads on tabular data, k-nearest neighbours falls apart
as columns multiply, a linear model beats a neural network when rows are few. The
point of the book is that you finish it knowing *why*, not just *which*.

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
