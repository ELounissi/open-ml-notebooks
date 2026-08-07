# Missing data

### Three mechanisms, deleted on purpose so the damage can be measured

**[Open the notebook](notebook.ipynb)** · Part 1, Foundations ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Why the missingness mechanism decides whether a fix is safe, what `dropna()` costs, how constant filling shrinks a column, what a missingness indicator buys, and why the imputer belongs inside the fold |
| **You should already know** | [Cross-validation](../04-cross-validation/), [feature scaling and encoding](../07-feature-scaling-and-encoding/) |
| **Datasets** | California Housing (20,640 rows), UCI Dry Bean |
| **Runtime** | Two to three minutes on a laptop CPU |

---

## The result I would lead with

The usual framing is that MCAR is harmless and MNAR is a catastrophe. I deleted
30% of `MedInc` three times, once under each mechanism, filled with the median,
and measured what each cost. The complete-data model scores R squared 0.6037.

| Mechanism | Fill error | Mean bias | R squared | R squared lost | MedInc coefficient |
|---|---|---|---|---|---|
| MCAR | 1.9256 | -0.1008 | 0.4976 | 0.1061 | 0.2987 |
| MAR | 2.3753 | -0.3313 | 0.4699 | 0.1338 | 0.2589 |
| MNAR | 2.5041 | -0.4473 | 0.4589 | 0.1448 | 0.2407 |

MNAR is worse, but only by **1.4x on R squared lost and 1.3x on fill error**.
The bigger number is the one nobody mentions: MCAR, the mechanism everyone calls
safe, still cost 0.1061 of R squared, about a sixth of the model, purely from
filling a third of one column with a constant. Getting the mechanism right does
not rescue you from the missing rate.

The coefficient column tells the sharper story. The true MedInc coefficient is
0.4367; under MNAR the model learns 0.2407. Squashing every high income to the
median flattens the relationship, and nothing in a score labelled "R squared"
tells you it happened.

## Which values disappeared

![Three mechanisms](figures/fig-01-three-mechanisms.png)

One deletion rule, three drivers: noise for MCAR, `AveRooms` for MAR, the column
itself for MNAR. All three delete about the same amount: 30.14%, 30.38% and
30.19% of 20,640 rows. The MedInc/AveRooms correlation is **+0.327**, and that is
what makes MAR partly recoverable later.

Against a true mean of 3.871, the survivors average 3.872 under MCAR, 3.637 under
MAR and 3.525 under MNAR. Only MCAR left the distribution's shape alone.

![Mechanism damage](figures/fig-02-mechanism-damage.png)

## Dropping rows

![Rows surviving](figures/fig-03-rows-surviving.png)

With $k$ columns each losing a fraction $p$, the complete fraction is $(1-p)^k$,
and the measured survivors track it: 65.81% for 4 columns at 10% (theory 65.61%),
42.89% for 8 at 10% (43.05%), 43.92% for 16 at 5% (44.01%), 2.72% for 16 at 20%
(2.81%).

Five per cent missing in each of sixteen columns sounds clean and leaves 43.92%
of the rows. That is the first cost, and at least you can count it. The second
cost is invisible, because dropping is only unbiased under MCAR:

Dropping kept 69.9%, 69.6% and 69.8% of rows under MCAR, MAR and MNAR, the same
number every time. Against a true mean of 3.8707 the survivors averaged 3.8719
(+0.0013), 3.6368 (**-0.2339**) and 3.5254 (**-0.3452**). Only one of the three
threw away a random selection.

## What constant filling does to a column

![Mean imputation](figures/fig-04-mean-imputation.png)

Replacing a fraction $p$ with a constant multiplies the column's variance by
$1-p$ and its correlations by $\sqrt{1-p}$. With $p = 0.3014$ the predicted factor
is **0.8358**. The standard deviation went 1.8998 to 1.5891, a ratio of 0.8365,
and the correlation with the target went 0.6881 to 0.5760, a ratio of 0.8371.

Those two land on the formula, and they are the two the derivation actually
promises. The mean ratio across all eight correlations is **0.9412**, well above
0.8358, and that average is the noisy number rather than the interesting one: a
correlation that started near zero, divided by another near zero, is noise with a
ratio attached. Apply $\sqrt{1-p}$ to a column you already know matters, not to a
whole correlation matrix.

The planning use is the point. Before filling a column you can already say how much
of its relationship with the target you are giving away, and that figure holds
whichever constant imputer you pick. It is also why "that feature turned out not to
matter" is a claim worth checking against the feature's missing rate.

## The missingness indicator

![Indicator gain](figures/fig-05-indicator-gain.png)

Adding `SimpleImputer(add_indicator=True)` moved R squared from 0.4976 to 0.4989
under MCAR (+0.0013), 0.4699 to 0.4778 under MAR (+0.0079), and 0.4589 to 0.4849
under MNAR (**+0.0260**).

Under MCAR the indicator is a column of noise and buys nothing, which is correct.
Under MNAR it buys the most, but be precise about how much: 0.0260 recovered out
of 0.1448 lost is **18% of the damage**, not most of it. It costs one boolean
column and no thought, so it is still worth adding by default. It is not a repair.

## Clever imputers, on a 3,000-row subsample

One column of this table is broken and I want to say so before quoting any of it.
The complete-data model on this subsample scores R squared **-4.3744**: a plain
linear regression, on 3,000 California rows across 3 folds, losing to a flat line.
A few districts have absurd `AveOccup` values, and on a subsample this size one of
them landing in a validation fold is enough to swamp that fold.

| Mechanism | Imputer | Fill error | Mean bias | R squared lost |
|---|---|---|---|---|
| MCAR | median | 1.8313 | -0.1028 | 1.6467 |
| MCAR | KNN (k=5) | 1.0938 | 0.0156 | 0.1134 |
| MAR | median | 2.3019 | -0.3102 | 1.5486 |
| MAR | KNN (k=5) | 1.1772 | -0.0205 | -0.0173 |
| MNAR | median + indicator | 2.5239 | -0.4534 | 1.4274 |
| MNAR | KNN (k=5) | 1.2723 | -0.1133 | -0.6963 |
| MNAR | iterative | **7.3979** | -0.1734 | 2.8929 |

Look at the last column. Two rows are negative, which says a model missing 30% of
its strongest feature beat the model that had all of it, by 0.70 of R squared in
one case. That cannot happen. It is the outlier landing in different folds under
different imputations, and it means **the R squared column here cannot rank
anything**, so I have not used it to. Ranking off a column like that is the
easiest paragraph in the chapter to write, which is why it is worth refusing.

The other two columns are direct measurements over every deleted entry, they do
not depend on a fold assignment, and they hold up. Two things in them contradict
the tidy version of this story.

**KNN under MNAR does not fail alongside the median.** Fill error 1.2723 against
2.5239, and the mean bias cut from -0.4534 to -0.1133. It never sees a deleted
income, but it can see that a district with those rooms, that occupancy and that
location ought to earn more than the surviving average, and that is enough to push
part of the tail back. The individual values stay wrong; the aggregate stops being
systematically wrong. Those are different achievements and only the second one
survives MNAR.

**`IterativeImputer` under MNAR is the worst method on the table**, at 7.3979
against the median's 2.5239. There is a mechanism. It regresses the missing column
on the others, feeds its own guesses back in and repeats. Under MNAR the observed
part of the column is the wrong part, so the regression is fitted to a truncated
relationship, and each cycle re-imputes from a column its own guesses have already
dragged downward. Nothing anchors the loop to the values that were deleted, so it
converges confidently on the wrong answer. Do not reach for the cleverest imputer
on a column you suspect is MNAR.

## Imputing inside the fold

Paired comparison, same subsample, same folds, same model. Only the placement of
the cleaning step changes. Fitting the imputer before the split inflated R
squared by **+0.1487 for a mean** (higher in 72% of 60 repeats) and **+1.2226 for
KNN** (higher in 70%).

A KNN imputer fitted before the split fills held-out rows from their own
neighbours, and a row partly reconstructed from its own fold is easier to
predict. +1.2226 in R squared is not a rounding error, and it went the other way
in 30% of repeats, so a single run will not reliably reveal it.

On the full 20,640 rows with a median, the same test gives honest 0.497618
against leaky 0.497600, a difference of **-0.000018**. The leak is invisible
when the data is large and the imputer is simple. It does not stay that way.

## Cheat sheet

| | |
|---|---|
| **First question** | Why is it missing? Ask the people who collected the data |
| **MCAR** | Dropping is unbiased. Filling still cost 0.1061 R squared here |
| **MAR** | Constants are biased. Imputers reading other columns recover it |
| **MNAR** | Nothing recovers the values. KNN still cut the aggregate bias 4x here |
| **Dropping rows** | $(1-p)^k$ survive. Count them before you commit |
| **Constant fill** | Variance times $1-p$, correlations times $\sqrt{1-p}$ |
| **Indicator** | One boolean column, recovered 18% of the MNAR loss. Free, not a fix |
| **IterativeImputer** | Came last under MNAR, worse than a median. Its loop refits on a column it has already pulled down |
| **KNNImputer** | Scale first. It leaked +1.2226 R squared when fitted outside the fold |
| **Always** | Imputer inside the `Pipeline`, refitted on every training fold |
| **Next** | [Hyperparameter tuning](../09-hyperparameter-tuning/), where the same inside-the-fold rule decides whether a search result is real |

---

If this chapter was useful, a star on the repository helps other people find it.
The code is yours to use, copy and adapt in your own work, no permission needed.

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 1](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#MissingData` `#Imputation` `#MCAR` `#MNAR`
`#DataCleaning` `#Python` `#ScikitLearn` `#DataScience` `#MLTutorial`
