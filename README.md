# open-ml-notebooks

### Learn machine learning by reading the code

A free, open book of **86 Jupyter notebooks** covering machine learning, deep
learning, and reinforcement learning — one method per notebook, explained in
plain language, implemented from scratch, then done properly with the library.

Every notebook ships **already executed**. Open one on GitHub and the charts,
numbers, and outputs are right there. Nothing to install to start reading.

**Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)**

![A straight line cannot see two commutes](02-regression/01-linear-regression/figures/fig-04-where-it-loses.png)

*From [Linear regression](02-regression/01-linear-regression/): why the simplest
model fails on cyclical data, and the one-line fix that takes it from 0.39 to 0.68.*

---

## Why this book

Most tutorials show you which function to call. Few show you what the method is
actually doing, and almost none tell you **when it is the wrong choice**.

Every notebook here answers four questions:

1. **What does this method actually do?** In words first, with a picture.
2. **What is the maths?** Written out, only what you need.
3. **Can I build it myself?** A minimal NumPy version, so nothing is magic.
4. **When does it win, and when does it lose?** Measured, not asserted.

That fourth question is the one that matters and the one usually missing.

## The idea that ties it together

Every notebook uses the **same five datasets**. That is deliberate. When a
support vector machine scores 0.93 in chapter three and a random forest scores
0.94 in chapter four, the numbers are directly comparable, because the question
was identical.

| Dataset | Task | Size | Why this one |
|---|---|---|---|
| California Housing | regression | 20,640 × 8 | Ships with scikit-learn — chapter one runs with no download |
| Breast Cancer Wisconsin | binary classification | 569 × 30 | Small enough that every method trains instantly |
| **UCI Dry Bean** | 7-class classification | 13,611 × 16 | Published 2020, barely used in tutorials, and pleasantly unbalanced |
| UCI Bike Sharing | regression + time series | 17,379 × 16 | One dataset that serves both tabular and sequence chapters |
| Fashion-MNIST | image classification | 70,000 images | MNIST's shape, several times harder, so CNNs stay interesting |

At the end, [**the scoreboard**](CURRICULUM.md) puts every method against every
dataset in one table and explains the pattern — why gradient boosting usually
takes tabular data, why k-nearest neighbours falls apart as columns multiply, why
a linear model beats a neural network when rows are few.

---

## Contents

Full table of contents with all 86 notebooks: **[CURRICULUM.md](CURRICULUM.md)**

| Part | Covers |
|---|---|
| **1 — [Foundations](01-foundations/)** | The workflow, train/validation/test, overfitting, cross-validation, metrics, scaling, encoding, missing data, tuning |
| **2 — [Regression](02-regression/)** | Linear, gradient descent, polynomial, Ridge, Lasso, Elastic Net, Huber, RANSAC, quantile, GLMs |
| **3 — [Classification](03-classification/)** | Logistic regression, k-NN, Naive Bayes, LDA/QDA, SVM and kernels, decision trees, imbalance, calibration |
| **4 — [Ensembles](04-ensembles/)** | Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, stacking |
| **5 — [Unsupervised](05-unsupervised/)** | k-Means, choosing k, hierarchical, DBSCAN, HDBSCAN, Gaussian mixtures, anomaly detection, Apriori |
| **6 — [Dimensionality reduction](06-dimensionality-reduction/)** | PCA, Kernel PCA, ICA, NMF, t-SNE, UMAP, feature selection |
| **7 — [Neural networks](07-neural-networks/)** | Perceptron, MLP and backprop in NumPy, PyTorch, activations, optimisers, regularisation |
| **8 — [Computer vision](08-computer-vision/)** | Convolution, CNNs, LeNet/VGG/ResNet, augmentation, transfer learning, segmentation, detection |
| **9 — [Sequences and language](09-sequences-and-language/)** | TF-IDF, embeddings, RNN, LSTM, GRU, seq2seq, attention, Transformers, time series |
| **10 — [Generative models](10-generative-models/)** | Autoencoders, VAEs, GANs, diffusion |
| **11 — [Reinforcement learning](11-reinforcement-learning/)** | Bandits, MDPs, Q-learning, SARSA, DQN, policy gradients, actor-critic, PPO |
| **12 — [Putting it together](12-putting-it-together/)** | The scoreboard, SHAP and LIME, pipelines, deployment, common mistakes |

### Available now

- **[01-03 — Overfitting and underfitting](01-foundations/03-overfitting-and-underfitting/)** —
  the single most useful diagnostic in machine learning, drawn rather than defined;
  a degree-20 polynomial with a *better* training error than degree 4 and a test
  error **three hundred times worse**
- **[02-01 — Linear regression](02-regression/01-linear-regression/)** — the normal
  equation derived, a 30-line NumPy version matched against scikit-learn to 12
  decimal places, residual diagnostics that expose a bias the summary statistic
  hid, and the method failing on cyclical data then being rescued by encoding
  rather than by a bigger model
- **[02-04 — Ridge regression](02-regression/04-ridge-regression/)** — the L2
  penalty, and an honest finding that it bought **no accuracy at all** on this data
  while making the coefficients **32× more stable**; plus a scaling experiment that
  came out backwards and explains itself
- **[02-05 — Lasso regression](02-regression/05-lasso-regression/)** — why one
  exponent turns shrinkage into selection, and the discovery that **`LassoCV` kept
  17 of 30 pure noise columns**, because cross-validation optimises prediction and
  not sparsity
- **[03-01 — Logistic regression](03-classification/01-logistic-regression/)** —
  the sigmoid, log loss, gradient descent written out, softmax on seven unbalanced
  bean varieties, and why the *rarest* class turned out to be the easiest one
- **[03-02 — k-Nearest Neighbours](03-classification/02-k-nearest-neighbours/)** —
  the model that does no training at all; scaling alone is worth **+0.205
  accuracy**, `k=1` scores a meaningless perfect 1.000 on training data, and the
  curse of dimensionality gets measured rather than asserted
- **[03-03 — Naive Bayes](03-classification/03-naive-bayes/)** — why an assumption
  everyone knows is false still works, and a thirteen-point detour into
  scikit-learn's `var_smoothing` default that quietly destroys small-scale features
- **[03-05 — Support Vector Machines](03-classification/05-support-vector-machines/)** —
  the margin, what `C` actually controls, and the kernel trick shown by lifting two
  rings into a third dimension before explaining why you never need to build it;
  plus the cost nobody mentions, where 27× the rows cost **139× the time**
- **[03-06 — Decision trees](03-classification/06-decision-trees/)** — the split
  search written from scratch, the single best question in the whole dataset found
  by arithmetic, a readable depth-3 tree, and an unrestrained one that hits perfect
  training accuracy while getting *worse* on held-out data
- **[04-02 — Random forests](04-ensembles/02-random-forest/)** — the variance
  equation that explains the whole design, out-of-bag scoring, two feature-importance
  measures that disagree, and a head-to-head against logistic regression that the
  forest **does not win**
- **[04-05 — Gradient boosting](04-ensembles/05-gradient-boosting/)** — why fitting
  the residual *is* gradient descent, the learning-rate trade, and the sharpest
  difference from a forest: adding trees eventually makes boosting **worse**
- **[05-01 — k-Means](05-unsupervised/01-k-means/)** — Lloyd's algorithm from
  scratch, the elbow and silhouette both pointing at the *wrong* number of clusters
  on data where the truth is known, and four failure modes including k-means
  confidently carving pure noise into tidy groups
- **[05-04 — DBSCAN and HDBSCAN](05-unsupervised/04-dbscan-and-hdbscan/)** — density
  clustering that finds crescents k-means cannot, choosing `eps` from a k-distance
  elbow instead of guessing, and the discovery that **no `eps` setting reports "this
  data has no structure"**
- **[06-01 — PCA](06-dimensionality-reduction/01-principal-component-analysis/)** —
  eigendecomposition by hand agreeing with scikit-learn to 2.2×10⁻¹⁶, components
  that turn out to mean "size" and "shape", and an honest finding that PCA **never
  beat** simply keeping every column
- **[07-01 — The perceptron](07-neural-networks/01-the-perceptron/)** — one neuron,
  the learning rule that needs no calculus, and the four points of XOR that stalled
  neural networks for a decade
- **[07-02 — MLP and backpropagation](07-neural-networks/02-mlp-and-backpropagation/)** —
  every gradient derived by hand and **checked numerically to 1.9×10⁻¹⁰**, then a
  working network in NumPy alone
- **[07-03 — The same net in PyTorch](07-neural-networks/03-the-same-net-in-pytorch/)** —
  autograd demonstrated on something checkable by hand, the five-line training loop
  you reuse forever, and the three mistakes everyone makes once
- **[08-02 — A CNN, layer by layer](08-computer-vision/02-a-cnn-layer-by-layer/)** —
  weight sharing explained, and a convolutional net that beats a dense one on
  accuracy, parameter count **and** speed; its first-layer filters turn into edge
  detectors nobody asked for
- **[11-04 — Q-learning](11-reinforcement-learning/04-q-learning/)** — Cliff Walking
  built from scratch with no `gym` dependency, the Bellman update in one line, and
  the classic result where Q-learning finds the better policy while SARSA collects
  more reward

Between them these cover supervised regression, supervised classification,
ensembles, unsupervised clustering, dimensionality reduction, and reinforcement
learning. New notebooks land in batches — watch or star the repo to get them.

### The scoreboard so far

Because every notebook uses the same datasets, the comparisons accumulate. On
**UCI Dry Bean**, 5-fold cross-validated accuracy:

| Method | Accuracy |
|---|---|
| SVM, RBF kernel | **0.9301** |
| Gradient boosting | 0.9271 |
| SVM, linear kernel | 0.9262 |
| Random forest | 0.9244 |
| Logistic regression | 0.9234 |
| k-Nearest Neighbours | 0.9231 |
| Neural network (NumPy, from scratch) | 0.9306 * |
| Naive Bayes | 0.8972 |
| Decision tree | 0.8945 |

\* single held-out split rather than 5-fold, so not directly comparable — see the
[notebook](07-neural-networks/02-mlp-and-backpropagation/) for why I do not claim
it as a win.

Nine methods inside four hundredths of each other, and a straight line beats most
of them. Bean measurements are smooth correlated geometry, so the flexible models
have little non-linear structure to exploit.

Change the dataset and the ordering changes. On **California Housing**, gradient
boosting cuts RMSE from linear regression's 0.7263 to **0.4668** — a 36%
improvement — because that problem *is* full of interactions.

**Which method wins is a property of your data, not a ranking of algorithms.**
That is the thread running through the whole book, and several notebooks here
report a result I expected to go the other way and say so plainly.

---

## Getting started

```bash
git clone https://github.com/ELounissi/open-ml-notebooks
cd open-ml-notebooks
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Everything runs on a **laptop CPU in minutes**. A GPU speeds up the deep learning
chapters and is never required. Small datasets are committed, so most notebooks
run offline immediately.

**Complete beginner?** Start at [Part 1, Foundations](01-foundations/).
**Know the basics?** Jump to whichever method you need — every notebook stands alone.
**Preparing for interviews?** The "when it wins, when it loses" section of each
notebook is written for exactly that conversation.

---

## Reusing this

Code is **MIT**. Text and figures are **CC BY 4.0** — take them into a course, a
talk, or a study group, with attribution. Datasets keep their own licences,
recorded per dataset in [`data/README.md`](data/README.md).

If this saved you time, **a star helps other people find it**. Found a mistake?
Open an issue — corrections are genuinely welcome, especially ones that show a
claim here is wrong.

---

### Topics

`machine-learning` `deep-learning` `reinforcement-learning` `machine-learning-tutorial`
`jupyter-notebook` `python` `scikit-learn` `pytorch` `data-science` `neural-network`
`cnn` `rnn` `lstm` `transformer` `linear-regression` `logistic-regression`
`random-forest` `xgboost` `clustering` `pca` `learn-machine-learning`
`machine-learning-algorithms` `ml-tutorial` `data-science-tutorial` `education`

`#MachineLearning` `#DeepLearning` `#ReinforcementLearning` `#DataScience`
`#Python` `#ScikitLearn` `#PyTorch` `#LearnMachineLearning` `#MLTutorial`
`#JupyterNotebook` `#OpenSource` `#AI`

---

**Made by Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
