# open-ml-notebooks

**Machine learning, deep learning, and reinforcement learning explained through
experiments that could have failed.**

Forty-four self-contained Jupyter notebooks. Each one asks a single question about
a method, answers it on an openly licensed dataset, and ends with a verdict —
including the parts where the evidence did not cooperate.

Every notebook ships executed. Open one on GitHub and the charts are already
there, along with the numbers that produced them.

By [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/).

![Explosions sit in knots. Earthquakes trace faults.](01-foundations/01-01-three-splits-three-truths/figures/fig-01-where.png)

---

## Why another machine learning repository

Most collections teach you to run a model. Almost none teach you to distrust the
number it prints. That is the gap this one aims at.

**The question comes before the dataset.** I do not start with Titanic and look
for something to predict. I start with something I genuinely do not know the
answer to, then find open data that can settle it. If the answer was never in
doubt, it is not an experiment and it is not in here.

**Results are allowed to disappoint.** In the first project I predicted that the
block-size sweep would flatten out. It did not, which means one of my claims is
weaker than I wanted, and the notebook says so in the section where it matters
rather than in a footnote.

**Every figure was opened and read.** A chart that does not clearly support or
clearly refute the question gets the experiment redesigned, not the caption
reworded. One chart was cut from project 01-01 for exactly this reason, and the
notebook explains why.

**Nothing is copied.** Prior work for each project is searched, read, and recorded
in [`ORIGINALITY.md`](ORIGINALITY.md), including the work that comes closest to
mine and the parts where I am reproducing rather than extending.

The rules I hold myself to are written down in [`STYLE.md`](STYLE.md), covering
prose, charts, code, and datasets.

---

## Start here

**[01-01 — Three splits, three different truths](01-foundations/01-01-three-splits-three-truths/)**

Can a model tell a quarry blast from an earthquake using nothing but the USGS
catalogue row? Yes. But the same model and the same data report a balanced
accuracy of **0.98 or 0.86** depending only on how the folds are arranged, and
the split most people reach for by reflex — a time-based one — protects against
almost nothing here.

![Location collapses. The clock does not.](01-foundations/01-01-three-splits-three-truths/figures/fig-06-reversal.png)

Under a random split, location looks like the informative feature. Under a
spatially blocked split, location is the weakest and time of day is the
strongest. Same data, same model, opposite conclusions.

---

## The map

Nine parts, forty-four projects. Full detail, including the question and dataset
for each, is in [`ROADMAP.md`](ROADMAP.md).

| Part | What it covers | Projects | Shipped |
|---|---|---|---|
| [1 — Foundations](01-foundations/) | Evaluation, leakage, bias and variance, learning curves, metrics | 4 | 1 |
| 2 — Supervised, classical | Regularization, calibration, k-NN, trees, forests, boosting, SVMs, imbalance | 8 | 0 |
| 3 — Unsupervised | k-means, density clustering, mixtures, PCA, embeddings, anomalies | 6 | 0 |
| 4 — Deep learning core | Backprop from scratch, optimizers, regularization, CNNs, transfer, sequences, transformers | 7 | 0 |
| 5 — Generative | Autoencoders, VAEs, GANs, diffusion, normalizing flows | 5 | 0 |
| 6 — Self-supervised | Contrastive learning, masked modelling, pretext tasks, tabular self-supervision | 4 | 0 |
| 7 — Graphs and structure | Graph neural networks, link prediction, sequence labelling | 3 | 0 |
| 8 — Reinforcement learning | Bandits, dynamic programming, Q-learning, DQN, policy gradients | 5 | 0 |
| 9 — Beyond the model | Conformal prediction, explainability, fairness | 2 | 0 |

Projects are built in the order given at the end of `ROADMAP.md`, not in numerical
order.

---

## Running it

Everything runs on a laptop CPU in minutes. A GPU makes the deep learning parts
faster and is never required.

```bash
git clone https://github.com/ELounissi/open-ml-notebooks
cd open-ml-notebooks
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Small datasets are committed, so notebooks run with no network. Larger ones have a
rebuild script under [`data/fetch/`](data/fetch/). Sources and licences for
everything are in [`data/README.md`](data/README.md).

A GitHub Action re-executes every notebook on a clean machine each month, so a
notebook that only worked because of leftover kernel state gets caught.

---

## Reusing this

Code is MIT. The prose and the figures are CC BY 4.0 — take them into a course or
a talk, with attribution. Datasets keep their own licences, recorded per dataset.
Details in [`LICENSE`](LICENSE).

If it saved you time, a star helps other people find it.

---

## Topics

`machine-learning` `deep-learning` `reinforcement-learning` `self-supervised-learning`
`jupyter-notebook` `data-science` `scikit-learn` `pytorch` `open-data`
`reproducible-research` `data-leakage` `cross-validation` `machine-learning-tutorial`
`ml-portfolio` `data-visualization`

---

**Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Corrections are welcome, especially ones that show a conclusion here is wrong.
Open an issue.
