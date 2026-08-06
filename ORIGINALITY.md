# Originality

This file exists so that anyone — a reviewer, an interviewer, or me in a year —
can check where each project's idea came from.

## The procedure

Before I write a project I do four things and record the result here.

1. **Search for the pairing.** Not the method alone and not the dataset alone, but
   the two together. A method is never new; the question asked of a particular
   dataset can be.
2. **Read the closest prior work.** If a paper or post has already answered my
   question, I either find a different question or I state plainly that I am
   reproducing something and say what I add.
3. **Write down the near misses.** The entries below name the work that comes
   closest, including work that weakens my claim to novelty. Leaving those out
   would make this file worthless.
4. **Check the prose.** Every README and every markdown cell goes through the
   word and construction lists in [`STYLE.md`](STYLE.md). Nothing is
   paraphrased from a source; where an idea is borrowed, it is cited inline.

I do not claim any of these projects is a research contribution. The claim is
narrower and, I think, more useful: the question, the framing, the code, and the
prose are mine, and where they overlap with existing work I say so.

---

## 01-01 — Three splits, three different truths

**The pairing.** USGS catalogue metadata (no waveforms) for separating quarry
blasts from natural earthquakes, used as a vehicle for comparing random,
temporal, and spatially blocked cross-validation on the same model.

**Closest prior work, and how this differs.**

*Blast-versus-earthquake discrimination* is a long-standing seismology problem
with a substantial literature. The nearest open-access example I found is
[Enhancing the classification of seismic events with supervised machine learning
and feature importance](https://pmc.ncbi.nlm.nih.gov/articles/PMC11668824/)
(Scientific Reports, 2024), which classifies 837 events from northern and
central Egypt using waveform-derived features — corner frequency, seismic
moment, spectral ratio, event complexity — and reports 99.68 percent accuracy
under seven-fold cross-validation.

Two things separate that work from this project. It uses **waveform physics**,
which requires the seismograms; I use only the **catalogue row**, which exists
for every event the USGS publishes and costs nothing to obtain. And its
evaluation is a random k-fold, with no discussion of spatial autocorrelation. I
did not audit its splitting protocol beyond what the paper states, and I make no
claim that its result is wrong — the point of my project is that a random fold
cannot distinguish a model that learned event physics from one that learned a
list of quarry coordinates, and that distinction is worth measuring.

Other work in the same area — including
[deep learning and transfer learning applied to southern California and eastern
Kentucky](https://academic.oup.com/gji/article/236/2/979/7453668) — is likewise
waveform-based.

*Spatially blocked cross-validation* is not my idea. It is established practice
in ecology and soil science; the standard reference is Roberts et al.,
[Cross-validation strategies for data with temporal, spatial, hierarchical, or
phylogenetic structure](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881)
(Ecography, 2017), and the reported gap between random and spatial estimates in
that literature runs as high as tens of percentage points. What I have not found
is this machinery pointed at a seismic catalogue, or a demonstration that the
*ranking of features* — not just the score — reverses when the split changes.

**What is mine.** The question, the dataset construction (including the
within-month balancing that removes calendar shortcuts), the decision to drop
analyst-assigned depth as label-contaminated, the three-way split comparison,
every chart, and every word.

---

## Template for the remaining projects

Each project adds a section here before it ships, with the same four headings:
the pairing, closest prior work and how this differs, what is mine, and anything
I found that weakens the novelty claim.
