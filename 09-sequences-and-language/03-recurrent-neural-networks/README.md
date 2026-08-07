# Recurrent neural networks

### The architecture for data where order carries the meaning

**[Open the notebook](notebook.ipynb)** · Part 9, Sequences and language ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | The recurrence and why sharing weights across time is the same trick as sharing them across space, a vanilla RNN cell written by hand and checked against `nn.RNN`, backpropagation through time, the vanishing gradient measured rather than asserted, gradient clipping, and an hourly forecast where the RNN has to earn its place |
| **You should already know** | [The same network in PyTorch](../../07-neural-networks/03-the-same-net-in-pytorch/). [Convolution and pooling](../../08-computer-vision/01-convolution-and-pooling/) helps but is not required |
| **Datasets** | UCI Bike Sharing, 17,379 hourly hire counts |
| **Runtime** | Two to three minutes on a laptop CPU |

---

## The gradient at step one is not small, it is zero

A fixed random RNN, a random input sequence, one number read off the final state,
and autograd asked how much the *first* input step still influences it. Entries of
$W_h$ are drawn with standard deviation $g/\sqrt{H}$, so the spectral radius sits
near the gain $g$:

| Recurrent gain | Gradient at T=2 | Gradient at T=120 | Measured factor per step |
|---|---|---|---|
| 0.6 | 6.408e-01 | **0.000e+00** | underflowed |
| 1.0 | 1.060e+00 | 5.243e-03 | 0.956 |
| 1.6 | 1.662e+00 | **3.806e+03** | 1.060 |

At gain 0.6 the number did not get small. It reached float zero, and the notebook
prints a `divide by zero encountered in log` warning when it tries to fit a slope
through it. At gain 1.6 the same product ran the other way, to 3.806e+03: one
scalar moved by 1.0 separates those outcomes.

## The recurrence

![Unrolled](figures/fig-01-unrolled.png)

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

One matrix for reading the input, one for carrying the state forward, and that is
the whole architecture. The same two run at every step: weight sharing across
time, the convolution argument moved from space to sequence, and a parameter
count that does not grow with input length. Writing the loop by hand and copying
the weights into `nn.RNN` gives agreement across all 4 x 12 x 16 states:

```
largest disagreement, all steps : 6.706e-08
largest disagreement, last state: 5.960e-08
```

Float32 noise. Nothing in the loop is cyclic: once it has run, autograd holds a
feedforward graph `n_steps` layers deep in which every layer uses one matrix.

## The gradient that does not arrive

![Gradient through time](figures/fig-02-gradient-through-time.png)

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \operatorname{diag}\big(\tanh'(a_t)\big)\, W_h$$

A product of $T-1$ matrices, mostly the same matrix, and products of matrices
behave like powers of a number. Both panels are on a log scale, so a straight line
is an exponential whose slope is the per-step factor above. Where that factor is
gentler than the gain (1.060 from a gain of 1.6), saturation is the reason: once
the state is large, $\tanh'$ is small and damps every term.

```
gradient norm before clipping : 14326.79
gradient norm after clipping  : 1.00
cosine between the two        : 1.000000
```

Clipping is a 14,000x rescale with the direction untouched to six decimal places,
which is what makes it safe on every step whether or not it bites. It does nothing
for the vanishing half: you cannot rescale a gradient back into existence. That
needs the [LSTM](../04-lstm/).

## Forecasting bicycle hire

![Training](figures/fig-03-training.png)

17,379 hourly counts from Washington DC become 17,355 windows of 24 hours. Train
on the first **13,884**, test on the last **3,471**, split by time and never at
random: neighbouring windows share 23 of their 24 values. Scaling statistics come
from the training hours only: mean 174.72, sd 166.92, on a series running 1 to
977 with a mean of 189.5.

| Model | Parameters | Test RMSE (hires) |
|---|---|---|
| **RNN, 48 hidden** | 2,497 | **51.91** |
| Linear on 24 lags | 25 | 81.77 |
| Last value | 0 | 129.72 |
| Same hour yesterday | 0 | 134.85 |

**The RNN won**, by 36.5% RMSE against the linear model and 60.0% against
persistence. I did not assume it would, and on a one-hour horizon there was a good
reason it might not: the next hour is close to a fixed weighted average of the
last few, which is precisely what 25 coefficients are. Here the extra capacity
paid for itself, at a training RMSE of **37.69** against a test RMSE of 51.91,
with the gradient hitting the clip on **99 of 1,962** steps.

The last line is the one to read twice. "Same hour yesterday" finished **behind
plain persistence** on a series with a strong daily cycle: twenty-four hours back
is further away than one hour back, and proximity beat seasonality.

## Where the errors live

![Error by hour](figures/fig-04-error-by-hour.png)

| Hour | Last value | Linear | RNN |
|---|---|---|---|
| 02 | **22.9** | 43.5 | 23.2 |
| 04 | **14.5** | 24.1 | 15.1 |
| 05 | 25.9 | **20.4** | 20.7 |
| 09 | 295.1 | 193.3 | **53.6** |
| 17 | 267.2 | 123.0 | **88.1** |
| 19 | 186.1 | **65.2** | 65.6 |

The RNN's whole margin comes from the commute. At 09:00 it scores 53.6 against
193.3 for the linear model and 295.1 for persistence, 3.6x better in the hardest
hour of the day. In the quiet hours it does not win at all: persistence is ahead
at 02:00 and 04:00, the linear model at 05:00 and 19:00. Even there it never drops
below second: a model that beats everything on average can still be beaten hour by
hour where little happens, and it is worth checking by how much.

![Forecast week](figures/fig-05-forecast-week.png)

Over one week both models follow the two commuter peaks and the overnight trough
without being told that days exist, and both shave the tops off the sharpest
spikes, because a squared error loss prefers being slightly low on a peak to being
badly wrong on either side. Their errors are correlated, the tell that they use
the same information in nearly the same way. Two honest notes: a one-hour horizon
on a periodic series is not the case a recurrent layer is built for and it won
anyway, a result rather than a rule; and it saw the counts alone, with no
temperature, humidity or holiday flags.

## Cheat sheet

| | |
|---|---|
| **The recurrence** | `h = tanh(x_t @ W_x.T + h @ W_h.T + b)` in a loop, one $W_x$ and one $W_h$ for every step. `nn.RNN(batch_first=True)` takes `(batch, time, features)` and returns `(output, h_n)` |
| **Vanishing** | A product of $T$ Jacobians. At gain 0.6 the first-step gradient hit float zero by T=120; at gain 1.6 it hit 3.806e+03 |
| **Clipping** | `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. Fixes exploding only, keeps the direction to 1.000000 cosine |
| **Time series splits** | Split by time, never at random. Overlapping windows make a random split close to training on the test set |
| **Always** | Run persistence and a lag-based linear model first. Here the RNN beat both, and running them is how you find out |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 9](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#DeepLearning` `#RNN` `#PyTorch` `#TimeSeries` `#Forecasting`
`#MachineLearning` `#Python` `#MLTutorial` `#LearnMachineLearning` `#AI`
