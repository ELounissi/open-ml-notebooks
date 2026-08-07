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

Every notebook below is finished: executed, with its charts and its own page.

---

## Part 1: Foundations

The vocabulary everything else uses. Start here if you are new.

| # | Notebook |
|---|---|
| 01-01 | [What machine learning actually does: the workflow end to end](01-foundations/01-what-machine-learning-does/) |
| 01-02 | [Train, validation, test: why you need all three](01-foundations/02-train-validation-test/) |
| 01-03 | [Overfitting and underfitting, seen rather than described](01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | [Cross-validation, and which flavour to use when](01-foundations/04-cross-validation/) |
| 01-05 | [Metrics for classification: accuracy, precision, recall, F1, ROC-AUC](01-foundations/05-classification-metrics/) |
| 01-06 | [Metrics for regression: MAE, RMSE, R squared, MAPE](01-foundations/06-regression-metrics/) |
| 01-07 | [Feature scaling and encoding categorical variables](01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | [Missing data: what to do and what it costs](01-foundations/08-missing-data/) |
| 01-09 | [Hyperparameter tuning: grid, random, and Bayesian search](01-foundations/09-hyperparameter-tuning/) |

## Part 2: Regression

| # | Notebook |
|---|---|
| 02-01 | Linear regression, from the normal equation to scikit-learn | shipped |
| 02-02 | [Gradient descent, watched step by step](02-regression/02-gradient-descent/) |
| 02-03 | [Polynomial regression and the bias-variance tradeoff](02-regression/03-polynomial-regression/) |
| 02-04 | [Ridge regression (L2)](02-regression/04-ridge-regression/) |
| 02-05 | [Lasso regression (L1) and automatic feature selection](02-regression/05-lasso-regression/) |
| 02-06 | [Elastic Net](02-regression/06-elastic-net/) |
| 02-07 | [Outlier-resistant regression: Huber and RANSAC](02-regression/07-outlier-resistant-regression/) |
| 02-08 | [Quantile regression: predicting a range, not a number](02-regression/08-quantile-regression/) |
| 02-09 | [Poisson and other generalised linear models](02-regression/09-generalised-linear-models/) |

## Part 3: Classification

| # | Notebook |
|---|---|
| 03-01 | [Logistic regression](03-classification/01-logistic-regression/) |
| 03-02 | [k-Nearest Neighbours](03-classification/02-k-nearest-neighbours/) |
| 03-03 | [Naive Bayes](03-classification/03-naive-bayes/) |
| 03-04 | [Linear and Quadratic Discriminant Analysis](03-classification/04-discriminant-analysis/) |
| 03-05 | [Support Vector Machines and the kernel trick](03-classification/05-support-vector-machines/) |
| 03-06 | [Decision trees, and how a split is chosen](03-classification/06-decision-trees/) |
| 03-07 | [Imbalanced classes: resampling, weights, and thresholds](03-classification/07-imbalanced-classes/) |
| 03-08 | [Multiclass and multilabel strategies](03-classification/08-multiclass-and-multilabel/) |
| 03-09 | [Probability calibration](03-classification/09-probability-calibration/) |

## Part 4: Ensembles

| # | Notebook |
|---|---|
| 04-01 | [Bagging, and why averaging helps](04-ensembles/01-bagging/) |
| 04-02 | [Random Forests](04-ensembles/02-random-forest/) |
| 04-03 | [Extra Trees](04-ensembles/03-extra-trees/) |
| 04-04 | [AdaBoost](04-ensembles/04-adaboost/) |
| 04-05 | [Gradient Boosting from first principles](04-ensembles/05-gradient-boosting/) |
| 04-06 | [XGBoost, LightGBM, CatBoost compared](04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | [Stacking and voting](04-ensembles/07-stacking-and-voting/) |

## Part 5: Unsupervised learning

| # | Notebook |
|---|---|
| 05-01 | [k-Means](05-unsupervised/01-k-means/) |
| 05-02 | [Choosing k: elbow, silhouette, gap statistic](05-unsupervised/02-choosing-k/) |
| 05-03 | [Hierarchical clustering and dendrograms](05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | [DBSCAN and HDBSCAN](05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | [Gaussian Mixture Models](05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | [Anomaly detection: Isolation Forest, One-Class SVM, LOF](05-unsupervised/06-anomaly-detection/) |
| 05-07 | [Association rules with Apriori](05-unsupervised/07-association-rules/) |

## Part 6: Dimensionality reduction

| # | Notebook |
|---|---|
| 06-01 | [Principal Component Analysis](06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | [Kernel PCA, ICA, and NMF](06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | [t-SNE](06-dimensionality-reduction/03-t-sne/) |
| 06-04 | [UMAP](06-dimensionality-reduction/04-umap/) |
| 06-05 | [Feature selection: filter, wrapper, embedded](06-dimensionality-reduction/05-feature-selection/) |

## Part 7: Neural networks

| # | Notebook |
|---|---|
| 07-01 | [The perceptron](07-neural-networks/01-the-perceptron/) |
| 07-02 | [Multilayer perceptron and backpropagation, in NumPy](07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | [The same network in PyTorch](07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | [Activation functions and why they matter](07-neural-networks/04-activation-functions/) |
| 07-05 | [Optimisers: SGD, Momentum, RMSProp, Adam](07-neural-networks/05-optimisers/) |
| 07-06 | [Regularisation: dropout, batch norm, weight decay, early stopping](07-neural-networks/06-regularisation/) |
| 07-07 | [A training loop you can reuse](07-neural-networks/07-a-training-loop/) |

## Part 8: Computer vision

| # | Notebook |
|---|---|
| 08-01 | [Convolution, pooling, and what a filter learns](08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | [A CNN on Fashion-MNIST, layer by layer](08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | [Classic architectures: LeNet, VGG, ResNet](08-computer-vision/03-classic-architectures/) |
| 08-04 | [Data augmentation](08-computer-vision/04-data-augmentation/) |
| 08-05 | [Transfer learning and fine-tuning](08-computer-vision/05-transfer-learning/) |
| 08-06 | [Image segmentation, an introduction](08-computer-vision/06-image-segmentation/) |
| 08-07 | [Object detection, an introduction](08-computer-vision/07-object-detection/) |

## Part 9: Sequences and language

| # | Notebook |
|---|---|
| 09-01 | [Text preprocessing, bag of words, and TF-IDF](09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | [Word embeddings: Word2Vec and GloVe](09-sequences-and-language/02-word-embeddings/) |
| 09-03 | [Recurrent neural networks](09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | [LSTM](09-sequences-and-language/04-lstm/) |
| 09-05 | [GRU, and how it compares to LSTM](09-sequences-and-language/05-gru/) |
| 09-06 | [Sequence to sequence with attention](09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | [The Transformer, built from scratch](09-sequences-and-language/07-the-transformer/) |
| 09-08 | [Fine-tuning a pretrained transformer](09-sequences-and-language/08-fine-tuning/) |
| 09-09 | [Time series forecasting: ARIMA against ML against deep learning](09-sequences-and-language/09-time-series-forecasting/) |

## Part 10: Generative models

| # | Notebook |
|---|---|
| 10-01 | [Autoencoders](10-generative-models/01-autoencoders/) |
| 10-02 | [Variational autoencoders](10-generative-models/02-variational-autoencoders/) |
| 10-03 | [Generative adversarial networks](10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | [Diffusion models, the smallest working example](10-generative-models/04-diffusion-models/) |

## Part 11: Reinforcement learning

| # | Notebook |
|---|---|
| 11-01 | [The reinforcement learning setup: agents, states, rewards](11-reinforcement-learning/01-the-setup/) |
| 11-02 | [Multi-armed bandits](11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | [Markov decision processes, value and policy iteration](11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | [Q-learning](11-reinforcement-learning/04-q-learning/) |
| 11-05 | [SARSA, and how it differs from Q-learning](11-reinforcement-learning/05-sarsa/) |
| 11-06 | [Deep Q-Networks](11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | [Policy gradients and REINFORCE](11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | [Actor-critic and PPO](11-reinforcement-learning/08-actor-critic-and-ppo/) |

## Part 12: Putting it together

| # | Notebook |
|---|---|
| 12-01 | [**The scoreboard**: every method on every house dataset](12-putting-it-together/01-the-scoreboard/) |
| 12-02 | [Interpreting models: permutation importance, SHAP, LIME](12-putting-it-together/02-interpreting-models/) |
| 12-03 | [Pipelines, and never leaking again](12-putting-it-together/03-pipelines/) |
| 12-04 | [Saving, loading, and serving a model](12-putting-it-together/04-saving-and-serving/) |
| 12-05 | [The mistakes everybody makes](12-putting-it-together/05-common-mistakes/) |

---

## The house datasets

Every notebook uses the same small set, so a comparison across chapters means
something. You can see immediately how a support vector machine and a random
forest differ, because they were asked the same question.

| Dataset | Task | Size | Why this one |
|---|---|---|
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

---

## Using this, and helping it reach people

The code is MIT licensed. Copy it, adapt it, put it in your own projects or
your teaching, no permission needed and no attribution required. If a chapter
saves you an afternoon, that is what it was for.

If you find it useful, **a star helps other people find it**, which is the only
way a book like this travels. Corrections and disagreements are welcome in the
issues, especially if you re-run something and get a different answer.

---
