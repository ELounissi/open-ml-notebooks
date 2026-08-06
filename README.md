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

- **[02-01 — Linear regression](02-regression/01-linear-regression/)** — the normal
  equation derived, a 30-line NumPy version matched against scikit-learn to 12
  decimal places, residual diagnostics that expose a bias the summary statistic
  hid, and the method failing on cyclical data then being rescued by encoding
  rather than by a bigger model
- **[03-01 — Logistic regression](03-classification/01-logistic-regression/)** —
  the sigmoid, log loss, gradient descent written out, softmax on seven unbalanced
  bean varieties, and why the *rarest* class turned out to be the easiest one
- **[04-02 — Random forests](04-ensembles/02-random-forest/)** — the variance
  equation that explains the whole design, out-of-bag scoring, two feature-importance
  measures that disagree, and a head-to-head against logistic regression that the
  forest **does not win**
- **[11-04 — Q-learning](11-reinforcement-learning/04-q-learning/)** — Cliff Walking
  built from scratch with no `gym` dependency, the Bellman update in one line, and
  the classic result where Q-learning finds the better policy while SARSA collects
  more reward

Between them these cover supervised regression, supervised classification,
ensembles, and reinforcement learning. New notebooks land in batches — watch or
star the repo to get them.

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
