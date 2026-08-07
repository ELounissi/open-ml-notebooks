# Curriculum

Eighty-six notebooks, twelve parts, one method each. Read them in order or jump
straight to the one you need: every notebook stands alone.

Each notebook follows the same five-part shape, so once you have read one you
know where to look in all of them:

1. **The idea**: what the method does, in plain language, with one picture
2. **The maths**: only what you need, written out, no hand-waving
3. **From scratch**: a minimal implementation in NumPy, so nothing is magic
4. **In practice**: the scikit-learn or PyTorch version, annotated line by line
5. **When it wins, when it loses**: on the house datasets, with the reason

Status: **shipped** means executed with charts and a README. **planned** means the
method and dataset are chosen.

---

## Part 1: Foundations

The vocabulary everything else uses. Start here if you are new.

| # | Notebook | Status |
|---|---|---|
| 01-01 | What machine learning actually does: the workflow end to end | [shipped](01-foundations/01-what-machine-learning-does/) |
| 01-02 | Train, validation, test: why you need all three | [shipped](01-foundations/02-train-validation-test/) |
| 01-03 | Overfitting and underfitting, seen rather than described | [shipped](01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | Cross-validation, and which flavour to use when | [shipped](01-foundations/04-cross-validation/) |
| 01-05 | Metrics for classification: accuracy, precision, recall, F1, ROC-AUC | [shipped](01-foundations/05-classification-metrics/) |
| 01-06 | Metrics for regression: MAE, RMSE, R squared, MAPE | [shipped](01-foundations/06-regression-metrics/) |
| 01-07 | Feature scaling and encoding categorical variables | [shipped](01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | Missing data: what to do and what it costs | [shipped](01-foundations/08-missing-data/) |
| 01-09 | Hyperparameter tuning: grid, random, and Bayesian search | [shipped](01-foundations/09-hyperparameter-tuning/) |

## Part 2: Regression

| # | Notebook | Status |
|---|---|---|
| 02-01 | Linear regression, from the normal equation to scikit-learn | shipped |
| 02-02 | Gradient descent, watched step by step | [shipped](02-regression/02-gradient-descent/) |
| 02-03 | Polynomial regression and the bias-variance tradeoff | [shipped](02-regression/03-polynomial-regression/) |
| 02-04 | Ridge regression (L2) | [shipped](02-regression/04-ridge-regression/) |
| 02-05 | Lasso regression (L1) and automatic feature selection | [shipped](02-regression/05-lasso-regression/) |
| 02-06 | Elastic Net | [shipped](02-regression/06-elastic-net/) |
| 02-07 | Outlier-resistant regression: Huber and RANSAC | [shipped](02-regression/07-outlier-resistant-regression/) |
| 02-08 | Quantile regression: predicting a range, not a number | [shipped](02-regression/08-quantile-regression/) |
| 02-09 | Poisson and other generalised linear models | [shipped](02-regression/09-generalised-linear-models/) |

## Part 3: Classification

| # | Notebook | Status |
|---|---|---|
| 03-01 | Logistic regression | [shipped](03-classification/01-logistic-regression/) |
| 03-02 | k-Nearest Neighbours | [shipped](03-classification/02-k-nearest-neighbours/) |
| 03-03 | Naive Bayes | [shipped](03-classification/03-naive-bayes/) |
| 03-04 | Linear and Quadratic Discriminant Analysis | [shipped](03-classification/04-discriminant-analysis/) |
| 03-05 | Support Vector Machines and the kernel trick | [shipped](03-classification/05-support-vector-machines/) |
| 03-06 | Decision trees, and how a split is chosen | [shipped](03-classification/06-decision-trees/) |
| 03-07 | Imbalanced classes: resampling, weights, and thresholds | [shipped](03-classification/07-imbalanced-classes/) |
| 03-08 | Multiclass and multilabel strategies | [shipped](03-classification/08-multiclass-and-multilabel/) |
| 03-09 | Probability calibration | [shipped](03-classification/09-probability-calibration/) |

## Part 4: Ensembles

| # | Notebook | Status |
|---|---|---|
| 04-01 | Bagging, and why averaging helps | [shipped](04-ensembles/01-bagging/) |
| 04-02 | Random Forests | [shipped](04-ensembles/02-random-forest/) |
| 04-03 | Extra Trees | [shipped](04-ensembles/03-extra-trees/) |
| 04-04 | AdaBoost | [shipped](04-ensembles/04-adaboost/) |
| 04-05 | Gradient Boosting from first principles | [shipped](04-ensembles/05-gradient-boosting/) |
| 04-06 | XGBoost, LightGBM, CatBoost compared | [shipped](04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | Stacking and voting | [shipped](04-ensembles/07-stacking-and-voting/) |

## Part 5: Unsupervised learning

| # | Notebook | Status |
|---|---|---|
| 05-01 | k-Means | [shipped](05-unsupervised/01-k-means/) |
| 05-02 | Choosing k: elbow, silhouette, gap statistic | [shipped](05-unsupervised/02-choosing-k/) |
| 05-03 | Hierarchical clustering and dendrograms | [shipped](05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | DBSCAN and HDBSCAN | [shipped](05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | Gaussian Mixture Models | [shipped](05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | Anomaly detection: Isolation Forest, One-Class SVM, LOF | [shipped](05-unsupervised/06-anomaly-detection/) |
| 05-07 | Association rules with Apriori | [shipped](05-unsupervised/07-association-rules/) |

## Part 6: Dimensionality reduction

| # | Notebook | Status |
|---|---|---|
| 06-01 | Principal Component Analysis | [shipped](06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | Kernel PCA, ICA, and NMF | [shipped](06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | t-SNE | [shipped](06-dimensionality-reduction/03-t-sne/) |
| 06-04 | UMAP | [shipped](06-dimensionality-reduction/04-umap/) |
| 06-05 | Feature selection: filter, wrapper, embedded | [shipped](06-dimensionality-reduction/05-feature-selection/) |

## Part 7: Neural networks

| # | Notebook | Status |
|---|---|---|
| 07-01 | The perceptron | [shipped](07-neural-networks/01-the-perceptron/) |
| 07-02 | Multilayer perceptron and backpropagation, in NumPy | [shipped](07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | The same network in PyTorch | [shipped](07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | Activation functions and why they matter | [shipped](07-neural-networks/04-activation-functions/) |
| 07-05 | Optimisers: SGD, Momentum, RMSProp, Adam | [shipped](07-neural-networks/05-optimisers/) |
| 07-06 | Regularisation: dropout, batch norm, weight decay, early stopping | [shipped](07-neural-networks/06-regularisation/) |
| 07-07 | A training loop you can reuse | [shipped](07-neural-networks/07-a-training-loop/) |

## Part 8: Computer vision

| # | Notebook | Status |
|---|---|---|
| 08-01 | Convolution, pooling, and what a filter learns | [shipped](08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | A CNN on Fashion-MNIST, layer by layer | [shipped](08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | Classic architectures: LeNet, VGG, ResNet | [shipped](08-computer-vision/03-classic-architectures/) |
| 08-04 | Data augmentation | [shipped](08-computer-vision/04-data-augmentation/) |
| 08-05 | Transfer learning and fine-tuning | [shipped](08-computer-vision/05-transfer-learning/) |
| 08-06 | Image segmentation, an introduction | [shipped](08-computer-vision/06-image-segmentation/) |
| 08-07 | Object detection, an introduction | [shipped](08-computer-vision/07-object-detection/) |

## Part 9: Sequences and language

| # | Notebook | Status |
|---|---|---|
| 09-01 | Text preprocessing, bag of words, and TF-IDF | [shipped](09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | Word embeddings: Word2Vec and GloVe | [shipped](09-sequences-and-language/02-word-embeddings/) |
| 09-03 | Recurrent neural networks | [shipped](09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | LSTM | [shipped](09-sequences-and-language/04-lstm/) |
| 09-05 | GRU, and how it compares to LSTM | [shipped](09-sequences-and-language/05-gru/) |
| 09-06 | Sequence to sequence with attention | [shipped](09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | The Transformer, built from scratch | [shipped](09-sequences-and-language/07-the-transformer/) |
| 09-08 | Fine-tuning a pretrained transformer | [shipped](09-sequences-and-language/08-fine-tuning/) |
| 09-09 | Time series forecasting: ARIMA against ML against deep learning | [shipped](09-sequences-and-language/09-time-series-forecasting/) |

## Part 10: Generative models

| # | Notebook | Status |
|---|---|---|
| 10-01 | Autoencoders | [shipped](10-generative-models/01-autoencoders/) |
| 10-02 | Variational autoencoders | [shipped](10-generative-models/02-variational-autoencoders/) |
| 10-03 | Generative adversarial networks | [shipped](10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | Diffusion models, the smallest working example | [shipped](10-generative-models/04-diffusion-models/) |

## Part 11: Reinforcement learning

| # | Notebook | Status |
|---|---|---|
| 11-01 | The reinforcement learning setup: agents, states, rewards | [shipped](11-reinforcement-learning/01-the-setup/) |
| 11-02 | Multi-armed bandits | [shipped](11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | Markov decision processes, value and policy iteration | [shipped](11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | Q-learning | [shipped](11-reinforcement-learning/04-q-learning/) |
| 11-05 | SARSA, and how it differs from Q-learning | [shipped](11-reinforcement-learning/05-sarsa/) |
| 11-06 | Deep Q-Networks | [shipped](11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | Policy gradients and REINFORCE | [shipped](11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | Actor-critic and PPO | [shipped](11-reinforcement-learning/08-actor-critic-and-ppo/) |

## Part 12: Putting it together

| # | Notebook | Status |
|---|---|---|
| 12-01 | **The scoreboard**: every method on every house dataset | [shipped](12-putting-it-together/01-the-scoreboard/) |
| 12-02 | Interpreting models: permutation importance, SHAP, LIME | [shipped](12-putting-it-together/02-interpreting-models/) |
| 12-03 | Pipelines, and never leaking again | [shipped](12-putting-it-together/03-pipelines/) |
| 12-04 | Saving, loading, and serving a model | [shipped](12-putting-it-together/04-saving-and-serving/) |
| 12-05 | The mistakes everybody makes | [shipped](12-putting-it-together/05-common-mistakes/) |

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
