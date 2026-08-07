# Overfitting and underfitting

### Seen, rather than described

**[Open the notebook](notebook.ipynb)** · Part 1, Foundations ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | What overfitting looks like on a chart rather than in a definition, how to spot it from two numbers, what the bias-variance tradeoff means concretely, and how more data moves the line |
| **You should already know** | [Linear regression](../../02-regression/01-linear-regression/) |
| **Datasets** | UCI Bike Sharing, plus a small synthetic curve |
| **Runtime** | About a minute on a laptop CPU |

---

## The one sentence version

**Underfitting** is a model too simple to capture the pattern, wrong on training
data and wrong on new data, in the same way.

**Underfitting** is visible. **Overfitting** is not: the model is *right* on the
training data and wrong on new data, and the gap between those two is the only
warning you get.

![Seeing it](figures/fig-01-seeing-it.png)

| Degree | Train RMSE | Test RMSE | |
|---|---|---|---|
| 1 | 0.720 | 0.777 | underfitting |
| 4 | 0.250 | **0.409** | about right |
| 20 | 0.302 | **146.102** | overfitting |

Degree 20 has a *better training error than degree 4* and a test error three
hundred times worse.

## The two numbers that matter

![The curve](figures/fig-02-the-curve.png)

This is the most useful diagnostic in machine learning, and its shape is worth
memorising.

**Training error only ever falls** as flexibility rises, so a low training error
tells you nothing on its own. **Test error falls, bottoms out, then rises.** The
turning point is the model you want. **The gap between the lines is the
overfitting**, not the height of either one.

## Bias and variance, without the definitions

![Bias and variance](figures/fig-03-bias-variance.png)

Each faint line is the same model refitted on a different sample of 20 points.

**Degree 1: high bias, low variance.** The forty fits sit on top of each other,
stable, and all wrong in the same way. Consistently mistaken.

**Degree 15: low bias, high variance.** The fits scatter everywhere. Their
*average* tracks the truth closely, so the model is not systematically wrong, but
any single fit is unreliable.

High bias is being wrong the same way every time. High variance is being wrong a
different way every time.

## More data moves the line

![More data](figures/fig-04-more-data.png)

At 15 points, degree 18 is a disaster. At 400 points it is fine. There is now
enough evidence to pin down its coefficients. Degree 1 does not improve at all,
because no amount of evidence fixes a model that structurally cannot represent the
pattern.

**More data cures variance, not bias.**

## The same shape on real data

![Real data](figures/fig-05-real-data.png)

| max_depth | Train RMSE | Held-out RMSE |
|---|---|---|
| 12 | 34.1 | **57.8** |
| unrestrained | **0.3** | 61.2 |

The unrestrained tree is essentially perfect on data it has already seen, and
worse on data it has not.

## What to do about it

**If you are overfitting**, in order of what to try first: get more data; reduce
flexibility; add [regularisation](../../02-regression/04-ridge-regression/); stop
training earlier; average several models
([bagging](../../04-ensembles/01-bagging/) exists almost entirely for this).

**If you are underfitting**, the list inverts: more flexibility, better features,
less regularisation, longer training.

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 1](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#Overfitting` `#BiasVarianceTradeoff` `#Underfitting`
`#Python` `#ScikitLearn` `#DataScience` `#MLTutorial` `#LearnMachineLearning`
