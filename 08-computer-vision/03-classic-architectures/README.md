# Classic architectures: LeNet, VGG, ResNet

### The degradation ResNet was built for showed up on cue, and the gradient measurement that is supposed to explain it pointed the other way

**[Open the notebook](notebook.ipynb)** · Part 8, Computer vision ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | How to write LeNet, a VGG block stack and a residual network from scratch in PyTorch, where each one keeps its parameters, how far each one can see computed from the layer list, and what a skip connection does and does not do to the gradient at the first layer |
| **You should already know** | [A CNN layer by layer](../02-a-cnn-layer-by-layer/) and the [size arithmetic](../01-convolution-and-pooling/) |
| **Dataset** | Fashion-MNIST, a fixed 8,000-image subset, 758 to 856 images per class, scored on all 10,000 test images |
| **Runtime** | 1.3 minutes on a laptop GPU. Nothing here is tuned per architecture, which is the only way the comparison means anything |

---

## The result I would lead with

Two stacks, identical down to the parameter count and the seed, differing by one
addition sign inside the block. Both keep He initialisation and batch norm, so
the only variable left is the skip connection.

| Conv layers | Stack | Parameters | Final train loss | Train accuracy | Test accuracy | Mean first-layer gradient, epoch 1 |
|---|---|---|---|---|---|---|
| 9 | plain | 19,034 | 0.8216 | **0.7686** | 0.7578 | 2.398e-01 |
| 9 | residual | 19,034 | 0.6655 | 0.7542 | 0.7415 | 3.850e-01 |
| 17 | plain | 37,722 | 0.8324 | 0.7331 | 0.7066 | 6.908e-01 |
| 17 | residual | 37,722 | 0.5250 | **0.8327** | **0.8117** | 5.850e-01 |
| 25 | plain | 56,410 | 1.1936 | **0.6142** | 0.5993 | **2.361e+00** |
| 25 | residual | 56,410 | 0.4963 | 0.8107 | 0.7697 | 6.989e-01 |

The headline held. The plain stack lost **0.1544 training accuracy** going from 9
to 25 layers while the residual one gained 0.0565 over the same range, worth
**+0.1965 at 25 layers**. Training accuracy falling as depth rises is the
degradation the ResNet paper was written about, and it cannot be overfitting,
because it is the training set.

The mechanism did not hold. The standard story is that the plain stack fails
because gradient stops reaching its early layers. Read the last column. At 25
layers the plain stack, the one that collapsed to 0.6142, received a first-layer
gradient of **2.361e+00** against the residual stack's **6.989e-01**. The network
that failed got **3.4x more** gradient than the network that worked, and the sign
of the comparison flips between 9 layers and 17.

So the sweep gives a clean result and takes away the explanation. Whatever went
wrong in the plain stack at 25 layers, "the first layer stopped hearing anything"
is not it on this budget, and any chapter that prints both tables and only
narrates one of them is choosing which half to believe.

![Depth, plain against residual](figures/fig-05-depth-plain-vs-residual.png)

Two more things in that figure that the summary line hides. At 9 layers the
residual stack is **worse** than the plain one on both splits, 0.7542 against
0.7686 on training and 0.7415 against 0.7578 on test, so the skip connection is
not free at shallow depth. And the residual stack peaks at 17 layers and then
falls, 0.8327 to 0.8107 on training and 0.8117 to 0.7697 on test. It degrades
too. It just degrades later. The single printed figure of +0.0565 is a 9-to-25
difference across a curve that goes up and then comes back down.

## What the initialisation measurement actually shows

One batch of 128 images, one backward pass, no training, four families read in
the order that separates three different fixes usually credited to each other.
Gradient norm at the first convolution:

| Conv layers | Plain, default init | Plain, He init | Plain, He + batch norm | Residual, He + batch norm |
|---|---|---|---|---|
| 5 | 7.928e-04 | 1.064e-01 | 1.805e-01 | 4.711e-01 |
| 9 | 1.183e-05 | 1.810e-02 | 2.887e-01 | 4.428e-01 |
| 17 | 6.907e-09 | 8.048e-03 | 9.482e-01 | 6.726e-01 |
| 33 | 4.485e-15 | 2.728e-03 | 1.089e+01 | 1.469e+00 |
| 65 | **0.000e+00** | 4.705e-04 | **4.013e+02** | 3.778e+00 |

![Gradient at the first layer](figures/fig-04-gradient-at-the-first-layer.png)

Three separate stories in one table, and the column order is what separates them.

PyTorch's default convolution initialisation, which is not He, underflows to
exactly zero by 65 layers. He initialisation alone slows the decay to a 226x fall
over the same range and does not stop it, because it fixes the variance at layer
one and the product over layers still drifts.

Batch norm stops the vanishing outright and overshoots into the opposite failure.
At 65 layers plain with He and batch norm reads **4.013e+02**, two orders of
magnitude above everyone else. **The largest number in that column is not the best
setup in it**, and any summary that ranks these families by magnitude will name
an exploding network the winner. Both failure modes live at the extremes; the
answer is the family that stays in the middle.

That is the residual one, and it is the only family in a usable band the whole
way, from 4.711e-01 to 3.778e+00 over a 13x change in depth. The identity term in
the gradient product is why: it neither grows nor shrinks with depth, so it puts
a floor and a ceiling on the whole expression at once.

One diagnostic worth stealing. The notebook also prints the standard deviation of
the logits at initialisation, with ln(10) = 2.3026 as the reference for a network
that outputs the same thing for every image. Nothing collapses. The default-init
family sits at 1.426e-01 at 65 layers against 1.343e-01 at 5, essentially
unchanged, while its gradient has gone to zero. The forward pass was healthy the
whole time the backward pass was dead, so a sane-looking activation histogram is
not evidence that a deep network can train.

## Where the parameters live

![Where the parameters live](figures/fig-01-where-the-parameters-live.png)

| | Convolution | Dense | Batch norm | Total | Dense share |
|---|---|---|---|---|---|
| LeNet-5 | 2,572 | 59,134 | 0 | 61,706 | **95.8%** |
| small VGG | 16,368 | 202,122 | 0 | 218,490 | 92.5% |
| small ResNet | 42,128 | 330 | 480 | 42,938 | **0.8%** |

The interesting number is not the total, it is the share sitting in the dense
classifier after the flatten, because that share is exactly what global average
pooling deletes. small VGG carries 5.1x the parameters of small ResNet and 92.5%
of them are in a layer doing position-specific bookkeeping over a 7x7 map. Swap
the flatten for an average and the same architecture family gets deeper and
smaller at the same time.

The skip connection itself costs nothing. Plain and residual stacks at 4 blocks
both print 19,034 parameters. It is an addition, not a layer.

## How far each one can see

![Receptive field](figures/fig-02-receptive-field.png)

Computed off the layer list rather than estimated, so the table cannot drift out
of step with the code:

| | Final receptive field | Verdict |
|---|---|---|
| LeNet-5 | 16 pixels | sees a window, never the whole garment |
| small VGG | 16 pixels | sees a window, never the whole garment |
| small ResNet | **48 pixels** | covers the image |

One of the three can see a whole 28x28 image from a single output unit. The other
two are deciding "sneaker or boot" from a patch. Stride is what buys the reach
cheaply: every pool multiplies the jump, so a 3x3 convolution placed after two
pools advances the reach by four pixels instead of two for the same nine weights.

The arithmetic behind VGG's one design decision, at a real layer width of 32
channels in and 32 out:

| Target reach | One large kernel | Stacked 3x3 | Weight ratio | Nonlinearities |
|---|---|---|---|---|
| 5 | 25,600 | 18,432 | 1.39x | 2 against 1 |
| 7 | 50,176 | 27,648 | 1.81x | 3 against 1 |
| 9 | 82,944 | 36,864 | 2.25x | 4 against 1 |
| 11 | 123,904 | 46,080 | **2.69x** | 5 against 1 |

Same reach, fewer weights, and the gap widens with the target. The single large
kernel gets one nonlinearity no matter how wide it is.

## Parameters are not accuracy, and they are not cost either

![Parameters against accuracy](figures/fig-03-parameters-vs-accuracy.png)

All four trained on the same 8,000 images with the same optimiser, learning rate,
batch size and seed:

| Model | Parameters | Train | Test | Seconds per epoch | Train minus test |
|---|---|---|---|---|---|
| LeNet-5 | 61,706 | 0.8086 | 0.7918 | 1.0 | +0.0168 |
| LeNet, ReLU + max pool | 61,706 | 0.7795 | **0.7710** | 0.7 | +0.0085 |
| small VGG | 218,490 | 0.8399 | **0.8310** | **0.6** | +0.0089 |
| small ResNet | 42,938 | 0.8536 | 0.8231 | **1.3** | +0.0305 |

The standard error on a test accuracy of 0.8310 over 10,000 images is 0.0037, and
the top two models are 0.0079 apart, close enough that this test set does not
separate them. So the honest reading is that small VGG and small ResNet are the
same accuracy here, reached with **5.1x the parameters** in one case and not the
other. Per ten thousand parameters, above the weakest model, small ResNet returns
+0.01213 against small VGG's +0.00275. That ratio is the result, not the accuracy
column, and it is what global average pooling buys.

The seconds column disagrees with the parameter column outright. small VGG has
the most parameters and the **fastest** epoch at 0.6 s. small ResNet has the
fewest and the **slowest** at 1.3 s. A dense layer holds many weights and touches
each once per image; a convolution holds few and touches each at every spatial
position, so the arithmetic bill follows feature map size, not weight count.

The unflattering row is the second one. Swapping LeNet's `tanh` and average
pooling for ReLU and max pooling, the substitution with the best claim to having
unlocked everything that came after it, **lost 0.0208 test accuracy** here, 5.6
standard errors, which is too large to wave away.

The textbook result is not wrong, it is about a situation this network is not in.
What ReLU fixes is saturation: `tanh` flattens at both ends, its derivative goes
to zero there, and in a deep stack that shrinks the gradient multiplicatively.
That is a claim about depth. LeNet is four layers with a 16-pixel receptive field,
and four `tanh` layers have no vanishing gradient problem, so the fix is aimed at
a disease this patient does not have.

Two things then push the other way at this scale. ReLU zeroes about half its
activations, cheap across hundreds of channels and expensive when the first
convolution has six. And max pooling keeps one pixel in four where average
pooling keeps a summary of all four, which suits a smooth greyscale silhouette
whose signal is the shape of a region rather than the brightest edge in it.

So: ReLU and max pooling are what make depth possible, and their advantage grows
with depth, width and input size. At four layers on 28x28 greyscale they are not
free. Adopt a modernisation when your network has the problem it solves. The
25-layer stacks at the top of this page are where the same components earn their
keep.

## Cheat sheet

| | |
|---|---|
| **LeNet-5** | Conv, pool, conv, pool, dense. Still the skeleton, and 95.8% of its weights are in the dense head |
| **VGG** | Only 3x3 convolutions and 2x2 pools, depth as the single dial. Cheap per epoch here, expensive in parameters |
| **Why 3x3** | Two reach as far as one 5x5 at 0.72x the weights with an extra nonlinearity. Three match a 7x7 at 0.55x |
| **ResNet** | `y = relu(F(x) + x)`. The addition costs zero parameters, measured, and puts an identity term in the gradient product |
| **Global average pooling** | Replaces the flatten and deletes the largest parameter block in the other two architectures. 0.8% dense share against 92.5% |
| **Receptive field** | `r += d(k-1)*jump` then `jump *= stride`. Compute it. Two of these three networks never see a whole garment |
| **Degradation** | A claim about training error, not test error. Confirmed here: the plain stack lost 0.1544 training accuracy from 9 to 25 layers |
| **Do not** | Assume the skip connection worked by keeping gradient alive during training. At 25 layers the failing plain stack had 3.4x the gradient |
| **Watch out** | PyTorch's default convolution init is not He. It underflowed to exactly zero at 65 layers while the forward pass still looked fine |
| **Larger is not better** | Batch norm without skips gave the largest first-layer gradient at depth, 4.013e+02. That is the other failure, not the fix |
| **Modernising** | Adopt a component when your network has the problem it solves. ReLU and max pooling cost LeNet 0.0208 here because four layers do not saturate |
| **Where these backbones get used** | [Transfer learning](../05-transfer-learning/) reuses a trained one instead of starting over |
| **Next** | [08-04 Data augmentation](../04-data-augmentation/), which changes the data instead of the architecture, and lost at every size it tried |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 8](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#ResNet` `#VGG` `#LeNet` `#SkipConnections` `#ResidualNetworks` `#BatchNorm`
`#ReceptiveField` `#CNN` `#ComputerVision` `#PyTorch` `#FashionMNIST`
`#DeepLearning` `#MachineLearning` `#MLTutorial` `#LearnMachineLearning`
`#DataScience` `#AI`
