# Optimisers

### SGD, Momentum, RMSProp, Adam: what each one actually fixes

**[Open the notebook](notebook.ipynb)** · Part 7, Neural networks ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | The named failure each optimiser was invented to fix, how the four differ on a surface you can draw, and whether adaptive methods find better minima or only reach them sooner |
| **You should already know** | [The PyTorch training loop](../03-the-same-net-in-pytorch/) |
| **Datasets** | A 2-D quadratic ravine, then Fashion-MNIST |
| **Runtime** | About fifteen minutes end to end. The four 8-epoch runs take 526 s of that, on torch 2.11.0 with CUDA |

---

## The answer, before the argument

On a real network the adaptive optimisers were **faster, and they were not
better.** Adam crossed 85% test accuracy in **3 epochs**, momentum needed **5**,
plain SGD never reached it in 8, yet the highest final accuracy belonged to **SGD
with momentum, 0.8745**, ahead of Adam's 0.8672. Two different claims, and only
the speed one survived the measurement.

## What plain gradient descent gets wrong

Every optimiser after SGD fixes one of two named failures. **Ravines:** when the
surface is far steeper in one direction than another, the gradient points across
the valley rather than along it, and the textbook account has the step bouncing
between the walls. The normal case, produced by correlated features or features on
different scales. **One rate for every parameter:** a rate safe in the steepest
direction is far too small for the flattest. Momentum averages past gradients, so
reversing components cancel and consistent ones accumulate. RMSProp divides each
step by the square root of that parameter's own running squared gradient, a
per-parameter rate. Adam runs both, plus a bias correction for the cold start.

Section 3 of the notebook counts what each optimiser does on such a surface instead
of describing it, and the count contradicts the account above.

## Four rules on a 20:1 ravine

![Paths on the ravine](figures/fig-01-ravine.png)

The bowl is `0.05x² + y²`, 60 steps, same learning rate of 0.28 for all four.

| | Final loss | Distance | Along the flat axis | Along the steep axis | Crossings of the valley floor |
|---|---|---|---|---|---|
| SGD | 0.134093 | 1.6376 | 1.6376 | 0.0000 | **0** |
| Momentum | 0.011805 | 0.3918 | 0.3862 | 0.0659 | **15** |
| RMSProp | **0.009567** | **0.0978** | 0.0002 | 0.0978 | 14 |
| Adam | 0.009721 | 0.4345 | 0.4341 | 0.0173 | 5 |

**SGD did not zig-zag. It crossed the valley floor zero times.** The path that
swings high above the floor, dives below it and overshoots the minimum on the far
side is momentum, with 15 crossings. The classic ravine picture is real, and at
this learning rate it belongs to the wrong optimiser.

The arithmetic is two lines. The steep direction has curvature 2, so SGD's update
there is $y \leftarrow y - 0.28 \cdot 2y = 0.44y$: a positive multiplier under one,
a monotone contraction that **cannot change sign**. Oscillation would need
$\eta > 1/\text{curvature} = 0.5$, and 0.28 is not it. The flat direction has
curvature 0.1, so the same rate gives $x \leftarrow 0.972x$, and sixty of those get
$-9$ only as far as 1.6376. **All** of SGD's final distance is flat and none of it
is steep. Its loss is high because it never crossed the valley, not because it
spent its steps crossing it. It is slow, not bouncy.

Momentum is the unstable one. Its velocity is $v \leftarrow 0.9v + g$, so the
steady-state step is $\eta/(1-\beta) = 2.8$, and $2.8 \times 2 = 5.6$ is past the
heavy-ball stability limit of $2(1+\beta) = 3.8$. That it still finishes at **11×
lower loss than SGD** is the finding: an oscillating optimiser that covers ground
beats a stable one that barely moves.

RMSProp is SGD's mirror image. It ends 0.0002 out along the flat axis and 0.0978
along the steep one, all of its error in the direction where SGD had none. Adam
finishes **further from the minimum** than momentum (0.4345 against 0.3918) and yet
at a **lower loss**, because what is left of its error lies along the cheap flat
axis. Distance and loss are not the same ruler on a stretched surface.

**The honest version of the ravine lesson: the classic picture needs a learning
rate large enough for the steep direction to oscillate.** Below that threshold
plain gradient descent does not zig-zag, it crawls, and the failure you get is the
slow flat direction rather than the bouncing steep one. Same underlying problem,
one rate for two curvatures, and the same fix.

## The same four on Fashion-MNIST

![Training curves and test accuracy](figures/fig-02-real-network.png)

A 128-64-10 network, batch size 256, 8 epochs each, one seed. SGD and momentum at
`lr=0.05`, RMSProp and Adam at `lr=0.001`.

| | Final train loss | Best test acc | Final test acc | Epochs to 0.85 | Seconds |
|---|---|---|---|---|---|
| SGD | 0.4496 | 0.8317 | 0.8317 | never | 130.6 |
| Momentum | **0.2956** | **0.8745** | **0.8745** | 5 | 151.9 |
| RMSProp | 0.3065 | 0.8662 | 0.8662 | 5 | 133.4 |
| Adam | 0.3091 | 0.8692 | 0.8672 | **3** | **110.5** |

The dramatic ordering from the ravine mostly evaporates: momentum, RMSProp and Adam
finish within **0.0083** of each other. Plain SGD is the only clear loser, **0.0428
below** momentum, and the only one that never crossed the line.

## Faster, or better

**Faster: yes.** Adam reached 0.85 in 3 epochs against 5 for the other two, in the
shortest wall-clock time of the four, 110.5 s, 41.4 s less than momentum, despite
being the more expensive update rule per step.

**Better: no.** Momentum's final 0.8745 beats Adam's best-ever epoch of 0.8692 by
**0.0053**, and momentum also reached the lowest training loss, 0.2956. What the
adaptive methods bought here was time, not quality. One seed and 8 epochs, so I
would not stake much on a 0.007 gap, which is the point. The accuracy difference
is inside the noise; the speed difference is not.

## What actually separates them

![Learning rate sensitivity](figures/fig-03-sensitivity.png)

Each optimiser swept across its own sensible range, 3 epochs per rate.

| Position | SGD rate | Test acc | Adam rate | Test acc |
|---|---|---|---|---|
| 1 | 0.005 | **0.589** | 0.0001 | 0.791 |
| 2 | 0.02 | 0.724 | 0.0005 | 0.843 |
| 3 | 0.05 | 0.797 | 0.001 | 0.854 |
| 4 | 0.2 | 0.801 | 0.005 | **0.865** |
| 5 | 0.5 | **0.812** | 0.02 | 0.860 |
| **spread** | | **0.223** | | **0.073** |

The chart's x-axis carries the rates rather than words like "default" and "much too
large", because the run does not support those words. See the two wrinkles below.

**SGD's spread is 0.223 across its sweep. Adam's is 0.073, three times tighter.**
Adam's worst rate, off by 10× in the wrong direction, still scored 0.791, within
0.006 of SGD's carefully chosen default and ahead of every SGD rate below 0.05.
This is why Adam is the default in most codebases: not that it finds better minima,
which the numbers above say it did not, but that a badly chosen rate costs you 7
points instead of 22.

Two wrinkles in my sweep. SGD never actually blew up: accuracy rose monotonically
to 0.812 at `lr=0.5`, the largest rate I tried, so the failure here was rates too
small rather than too large, and a wider sweep would find the divergence point.
And Adam peaked at `lr=0.005` (0.865), not at the `1e-3` default (0.854). Well-tuned
SGD with momentum still wins many vision benchmarks, and it won here too.

## Cheat sheet

| | |
|---|---|
| **Start with** | Adam at `lr=1e-3`. It is the default because it is forgiving, not because it is best |
| **Switch to SGD + momentum when** | You are tuning carefully, training a vision model, or chasing the last fraction of accuracy. It took the top accuracy here |
| **Momentum fixes** | The slow flat direction, by accumulating a consistent gradient. It damps zig-zag only when the rate is high enough to cause one: at lr=0.28 on my ravine, momentum was the one zig-zagging, 15 crossings against SGD's 0 |
| **RMSProp fixes** | One learning rate for all parameters, by scaling each by its own gradient history |
| **Adam** | Both at once, plus bias correction for the cold start. Use AdamW instead whenever you want weight decay |
| **Watch out** | The learning rate matters more than the optimiser. The 0.223 spread from the rate dwarfs the 0.007 gap between optimisers. Tune it first |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 7](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#DeepLearning` `#Optimizers` `#Adam` `#SGD` `#Momentum` `#RMSProp`
`#PyTorch` `#LearningRate` `#MLTutorial` `#GradientDescent`
