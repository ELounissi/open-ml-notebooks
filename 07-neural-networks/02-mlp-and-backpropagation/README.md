# Multilayer perceptron and backpropagation

### A neural network in NumPy, with every gradient derived

**[Open the notebook](notebook.ipynb)** · Part 7, Neural networks ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | What backpropagation actually computes, how to derive every gradient by hand, how to check them numerically, and how to train a working network in NumPy alone |
| **You should already know** | [The perceptron](../01-the-perceptron/), the chain rule |
| **Dataset** | UCI Dry Bean (13,611 × 16, 7 classes) |
| **Runtime** | About two minutes on a laptop CPU |

---

## The problem it solves

The [perceptron](../01-the-perceptron/) ended on a wall: a hidden layer can rebuild
the input space so a line separates the classes, but the perceptron rule cannot
train one, because **nobody tells you what a hidden unit should have output**.

Backpropagation is not a learning algorithm. It is a way of computing derivatives.
Gradient descent still does the learning; backprop works out efficiently how much
each weight contributed to the error, by applying the chain rule **from the output
backwards** so every intermediate result gets reused.

## The derivation

Softmax paired with cross entropy gives a result so clean it looks like a mistake:

$$\frac{\partial L}{\partial z_2} = \hat{p} - y$$

Prediction minus truth, the same expression that appeared in
[linear regression](../../02-regression/01-linear-regression/) and
[logistic regression](../../03-classification/01-logistic-regression/). The messy
softmax derivative and the messy log derivative cancel exactly.

From there, with $\delta_2 = \hat{p} - y$:

$$\delta_1 = (\delta_2 W_2^\top) \odot \mathbb{1}[z_1 > 0] \qquad \frac{\partial L}{\partial W_1} = x^\top \delta_1$$

ReLU's derivative is 1 where the input was positive and 0 elsewhere, so that term
is just a mask.

## Check the gradients before trusting them

A backprop bug does not crash. It trains slightly worse, and you never find out.
The notebook compares every analytic gradient against a numerical estimate:

```
worst gap between analytic and numerical gradient: 1.947e-10  →  the derivation is correct
```

**Always do this once on a new implementation.** It is slow, impractical for
training, and decisive as a one-off test.

## Training

![Training](figures/fig-01-training.png)

![Width](figures/fig-02-width.png)

Wider helps until it stops mattering.

## The comparison, and why I do not trust it

![Comparison](figures/fig-03-comparison.png)

| Model | Accuracy on the same held-out quarter |
|---|---|
| Our NumPy network | **0.9306** |
| Gradient boosting | 0.9257 |
| Random forest | 0.9251 |
| Logistic regression | 0.9245 |

Two hundred lines of NumPy came out **ahead of scikit-learn's boosted trees**,
which was not what I expected to be writing.

Before celebrating, read the gap properly. The difference is about half a point,
on a **single** held-out quarter. Every other notebook in this book used five-fold
cross-validation, and the fold-to-fold spread on this dataset ran to roughly that
size on its own. The honest reading is **a tie measured badly**, not a win. A
single split is a sample of size one.

It also does not change the advice. Gradient boosting needed one line and no
tuning; the network needed a derivation, an initialisation scheme, a learning rate,
a width, and an epoch count. Neural networks earn their keep where the input has
structure a tree cannot exploit: the grid of an image, the order of a sequence.

## Cheat sheet

| | |
|---|---|
| **Backprop is** | A way to compute derivatives, not a learning algorithm |
| **Key simplification** | Softmax + cross entropy gives $\partial L/\partial z = \hat{p} - y$. Never derive them separately |
| **Numerical stability** | Subtract the row max before `exp`; add `1e-12` before `log` |
| **Initialisation** | He (variance $2/\text{fan-in}$) for ReLU. Zeros make every unit identical forever |
| **Always** | Gradient-check once. A wrong gradient trains quietly and badly |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 7](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#DeepLearning` `#Backpropagation` `#NeuralNetwork` `#NumPy` `#MachineLearning`
`#Python` `#MLTutorial` `#GradientDescent` `#LearnMachineLearning` `#AI`
