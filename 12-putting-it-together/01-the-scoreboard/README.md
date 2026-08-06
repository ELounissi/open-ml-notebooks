# The scoreboard

### Every method in this book, on every dataset in this book, scored the same way

**[Open the notebook](notebook.ipynb)** · Part 12, Putting it together ·
Made by [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)

| | |
|---|---|
| **What you will learn** | Which method actually wins on which kind of data, why the answer changes, and how to pick a default when you cannot try everything |
| **You should already know** | Anything from Parts 2 to 7. This chapter assumes the methods and compares them |
| **Datasets** | All four house datasets: Dry Bean, Breast Cancer, California Housing, Bike Sharing |
| **Runtime** | About one minute. 5-fold cross-validation throughout, one seed, identical preprocessing for every method |

---

## The result I would lead with

Before running it I wrote down seven predictions. The notebook checks each one
against its own output and prints whether it held. **Two did not.**

| Prediction | Verdict |
|---|---|
| A tree ensemble wins on every dataset | **did not hold** |
| Breast Cancer has the smallest linear gap | held |
| Bike Sharing has the largest linear gap | held |
| Bike Sharing has the largest step advantage | **did not hold** |
| Breast Cancer has the fewest rows per column | held |
| The MLP gains nothing over logistic regression on Breast Cancer | held |
| The winner is never also the fastest method | held |

The first failure is the one worth the chapter. Tree ensembles are the default
advice for tabular data, and they won **two of four** here:

| Dataset | Winner | Score | Per fit |
|---|---|---|---|
| Dry Bean | **RBF SVM** | 0.9308 accuracy | 0.15 s |
| Breast Cancer | **logistic regression** | 0.9789 accuracy | 0.00 s |
| California Housing | gradient boosting | 0.7916 R² | 2.22 s |
| Bike Sharing | random forest | 0.9243 R² | 4.57 s |

On Breast Cancer, plain logistic regression beat every ensemble and every kernel
method in the book, and the MLP **lost to it by 0.0404**. With 569 rows and 30
columns there are about nineteen rows per column, and at that ratio the flexible
methods spend their capacity on noise.

![Classification scoreboard](figures/fig-01-classification-scoreboard.png)

![Regression scoreboard](figures/fig-02-regression-scoreboard.png)

## The winner is never the fastest

That prediction held on all four datasets, without exception:

| Dataset | Winner | Fastest |
|---|---|---|
| Dry Bean | RBF SVM | k-NN |
| Breast Cancer | logistic regression | k-NN |
| California Housing | gradient boosting | lasso |
| Bike Sharing | random forest | ridge |

So the useful question is not which method wins, it is what the win costs. On
Dry Bean, k-NN gives up **0.0092 accuracy** and runs **51x cheaper** than the
RBF SVM. If that trade is worth making, it is worth making deliberately.

![Quality against time](figures/fig-03-quality-against-time.png)

## What makes the datasets different

The notebook computes four properties of each dataset and lines them up against
which method won, so the ranking has something to be explained by:

| | Rows per column | Mean abs correlation | Columns for 95% variance | Step advantage | Linear gap |
|---|---|---|---|---|---|
| Dry Bean | 375.0 | 0.498 | 5 of 16 | -0.014 | 0.007 |
| Breast Cancer | 19.0 | 0.395 | 10 of 30 | -0.007 | 0.000 |
| California Housing | 750.0 | 0.152 | 6 of 8 | 0.074 | 0.148 |
| Bike Sharing | 500.0 | 0.097 | 10 of 12 | 0.031 | 0.542 |

**Linear gap** is how far the best linear model sits behind the winner, and it
is the column that explains the most. On Breast Cancer it is 0.000: nothing in
the book beats a line, so the problem is linearly separable and the extra
machinery has nothing to buy. On Bike Sharing it is **0.542**, the largest in
the table, because hourly demand depends on time of day in a way no straight
line can express.

**Step advantage** is where my second prediction failed. I expected Bike Sharing
to reward step functions most, since its structure is sharp. California Housing
rewarded them more (0.074 against 0.031). Two of the four are slightly negative,
meaning the axis-aligned splitters were marginally worse there.

![What makes them different](figures/fig-04-what-makes-them-different.png)

## The k-NN result I got wrong

The standard claim is that k-NN falls apart as columns are added. Measuring
k-NN's gap to the winner against the column count gives a correlation of
**-0.41**, which points the wrong way: the dataset with the most columns
(Breast Cancer, 30) has one of k-NN's smallest gaps, and the dataset where it
loses badly (Bike Sharing, 0.3906) has 12.

I am not going to claim the textbook is wrong from four points. **A correlation
over four datasets is not evidence of anything**, and the sign could flip with a
fifth. What the four points do show is that on this collection, column count is
not what decides whether k-NN works. What decides it is whether distance in the
feature space means anything, and on Bike Sharing an hour index and a
temperature do not combine into a meaningful distance.

## No free lunch, ranked

Every family's rank within each dataset, out of the six families:

| Family | Dry Bean | Breast Cancer | California | Bike Sharing | Best | Worst | Mean |
|---|---|---|---|---|---|---|---|
| gradient boosting | 3 | 3 | **1** | 3 | #1 | **#3** | **2.50** |
| neural net | **1** | 5 | 3 | 2 | #1 | #5 | 2.75 |
| random forest | 5 | 4 | 2 | **1** | #1 | #5 | 3.00 |
| linear model | 2 | **1** | 5 | 6 | #1 | #6 | 3.50 |
| k-NN | 4 | 2 | 4 | 5 | #2 | #5 | 3.75 |
| decision tree | 6 | 6 | 6 | 4 | #4 | #6 | 5.50 |

**No family is top-ranked on all four.** Four of the six take first place
somewhere, and four of the six also land fifth or sixth somewhere.

Gradient boosting has the best mean rank, and the reason is the "worst" column
rather than the "best" one: it is **the only family that never finishes below
third**. That is what makes it a good default. Not that it usually wins, because
here it won once, but that it never loses badly.

The linear model is the widest swing in the table, moving **5 places** between
first on Breast Cancer and sixth on Bike Sharing. The single decision tree is
the steadiest at 2 places, and steady is not worth much when the range is fourth
to sixth.

![No free lunch](figures/fig-05-no-free-lunch.png)

## Cheat sheet

| | |
|---|---|
| **Start with** | Gradient boosting on tabular data. It never finished below third here |
| **Always run** | A linear model. It won a dataset, and its gap to the winner tells you how much non-linearity the problem actually has |
| **Few rows per column** | Prefer the linear model. At 19 rows per column the MLP lost to logistic regression |
| **Sharp thresholds** | Tree ensembles. Bike Sharing's linear gap of 0.542 is what that looks like |
| **Before you accept a win** | Check the fold standard deviation. Some gaps here are smaller than the spread across folds |
| **Cost** | The winner was never the fastest, on any of the four. Decide what the win is worth |
| **Do not** | Read a four-dataset correlation as a law. This page contains one that points the wrong way |

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

Back to [Part 12](../) · [the curriculum](../../CURRICULUM.md) · [the book](../../README.md)

`#MachineLearning` `#ModelSelection` `#NoFreeLunch` `#ScikitLearn`
`#TabularData` `#GradientBoosting` `#Benchmark` `#MLTutorial`
`#LearnMachineLearning` `#DataScience` `#AI`
