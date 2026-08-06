# Part 1 — Foundations

Four projects about the things that decide whether a number means anything, before
any model gets chosen.

Most machine learning material starts with a model. This part starts with the
measurement, because a score computed the wrong way is worse than no score: it is
a wrong answer carrying a decimal point.

| Project | Question | Status |
|---|---|---|
| [01-01 Three splits, three different truths](01-01-three-splits-three-truths/) | Does the choice of cross-validation split move the reported score more than the choice of model? | shipped |
| 01-02 Where does the error live? | Does the textbook bias-variance curve actually appear on real data, or only on simulations? | planned |
| 01-03 How much data do you need? | Can you predict where a learning curve plateaus from its first ten percent? | planned |
| 01-04 The metric is the model | How much does the ranking of models change when you change only the metric? | planned |

## Why these four, in this order

**01-01** establishes that an evaluation protocol is a modelling decision. It is
first because every later project in the repository reports a cross-validated
score, and the reader should already be suspicious of what that means.

**01-02** takes the most-drawn curve in machine learning teaching and checks
whether it survives contact with a real dataset.

**01-03** is the question every practitioner is actually asked — "would more data
help?" — treated as something measurable rather than something to guess at.

**01-04** closes the part by showing that the metric is not a reporting choice
made after the fact. It selects the model.
