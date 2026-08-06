# Style

Rules I hold myself to across this repo. They exist because a collection like this
fails in two predictable ways: the prose turns into filler, and the charts turn into
decoration. Both are avoidable if the rules are written down.

---

## Voice

I write in first person. I say what I tried, what I expected, and what came back.
When a result surprised me, I say so instead of rewriting history to look like I
planned it.

I don't address the reader as "we" when I mean "I", and I don't address them as
"you" in a coaching tone. A notebook is a lab record that happens to be readable.

Where I'm unsure, I write that I'm unsure. A hedge that names its reason is useful.
A hedge that exists to avoid commitment is noise.

---

## Words and constructions I don't use

Most of this list comes from the Wikipedia page
[Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing).
I read it as a style guide rather than a detector, because the constructions it
names are genuinely weak writing regardless of who produced them.

**Vocabulary that gets cut on sight**

> delve, showcase, testament, tapestry, landscape (as an abstract noun), robust,
> crucial, pivotal, meticulous, intricate, garner, foster, underscore (as a verb),
> highlight (as a verb), bolster, boasts (meaning "has"), vibrant, enduring,
> align with, interplay, valuable insights, key (as a filler adjective)

**Sentence shapes that get rewritten**

- `not just X, but Y` and `it's not X, it's Y`. Say the thing you mean once.
- `X rather than Y` used for emphasis rather than for an actual contrast.
- Copula avoidance: `serves as`, `stands as`, `represents`, `functions as`,
  `marks` where `is` was the honest verb.
- Trailing participles that pretend to analyse: `..., highlighting the importance
  of ...`, `..., reflecting broader trends in ...`, `..., contributing to ...`.
  If the clause has content, promote it to its own sentence. If it doesn't, delete it.
- The rule of three used as padding. Three items because there are three, not
  because three sounds complete.
- Significance inflation: `a pivotal moment`, `underscores its importance`,
  `left an indelible mark`, `setting the stage for`.

**Structures that don't appear**

- A closing section shaped like *"Despite its limitations, X faces challenges, and
  future work may ..."*. Every project ends with a verdict on the stated question,
  and the limitations are named where they bite, not swept into a tidy final block.
- Bulleted lists of the form `- **Header**: sentence`. If items need headers they
  need subsections or a table.
- Bold used to mark "key takeaways". Bold marks a term being defined, and nothing else.
- Emoji as bullets or section dividers.
- Title Case In Headings. Sentence case only.
- Skipped heading levels, and horizontal rules immediately before a heading.

**Punctuation**

Straight quotes and apostrophes, never curly. Em dashes are allowed but rationed;
if a paragraph has two, one of them becomes a full stop.

---

## How a project is built

Every project answers one question that can come back "no". If the answer was never
in doubt, it isn't an experiment and it doesn't go in.

The order is fixed, and it's the order I actually worked in:

1. **The question.** One or two sentences. Stated so that a specific result would refute it.
2. **The data.** What it is, where it came from, what's wrong with it. Every dataset
   gets a row in [`data/README.md`](data/README.md) with source, licence, and retrieval date.
3. **The setup.** The model, the split, the metric, and why each was chosen over the
   obvious alternative.
4. **The run.** Code with outputs kept, so GitHub renders the figures without anyone
   installing anything.
5. **The verdict.** What the charts support, what they don't, and what I'd need to
   settle the parts they leave open.

Each project folder carries its own `README.md` with the figures embedded, so a
visitor arriving from a search engine gets the whole story without opening the notebook.

---

## Charts

A figure earns its place by changing what the reader believes. Decoration gets deleted.

- One shared theme, from `toolkit/style.py`, so the repo looks like one book.
- Axes are labelled with units. Titles state the finding, not the variable names:
  "Random splits overstate accuracy by 14 points", not "Accuracy by split type".
- Uncertainty is drawn. A bare point estimate is a claim without evidence, so
  error bars, bootstrap bands, or the raw fold-level points appear wherever a
  comparison is being made.
- Colour is never the only channel carrying meaning. The palette is checked
  against deuteranopia and works in greyscale.
- Every figure is rendered, opened, and read before it ships. If it doesn't clearly
  support or clearly refute the stated question, the experiment changes — different
  data, different framing, different design — until the picture is legible.
  Ambiguous figures are a signal that the experiment is wrong, not a formatting problem.

---

## Code

- Notebooks run top to bottom on a fresh kernel. No hidden state, no cells that
  only work in the order I happened to click them.
- One idea per cell. A cell that does three things gets split.
- Comments explain why, not what. `# scale before PCA, otherwise variance follows units`
  is worth writing. `# fit the model` is not.
- Everything runs on CPU in a few minutes. A GPU makes it faster and is never required.
- Seeds are set and stated. Where a result depends on the seed, that dependence is
  the finding and gets its own chart.
- Shared logic lives in `toolkit/`. Copy-pasted helper functions across notebooks
  get pulled up on the second occurrence.
- Dependencies are pinned in `requirements.txt`.

---

## Datasets

Open licence, no exceptions, and the licence is recorded.

Nothing is used because it is convenient. Titanic, Iris, and plain MNIST do not
appear as the subject of a project. A famous dataset can appear when the angle on
it is one that hasn't been worked to death; a new or neglected dataset is better still.

If a file is large or its licence forbids redistribution, `data/fetch/` gets a
script and `data/raw/` stays empty for that entry.

---

## Originality

Before writing a project I search for existing work on the same
question-and-dataset pairing, and I record what I found in
[`ORIGINALITY.md`](ORIGINALITY.md) — including the near misses, and what makes mine
different. A project that turns out to be a restatement of a well-known tutorial
gets a new angle or gets dropped.

Prose is mine. Where an idea comes from a paper or a post, it is cited inline with
a link, and the citation resolves to a real page that says what I claim it says.

---

## Checklist before a project ships

- [ ] The question could have come back "no"
- [ ] Every figure was opened and read, and supports or refutes the question without ambiguity
- [ ] Notebook re-runs clean on a fresh kernel
- [ ] Dataset row exists in `data/README.md` with a licence
- [ ] Originality note exists in `ORIGINALITY.md`
- [ ] Prose passes the word and construction lists above
- [ ] Project `README.md` stands alone with figures embedded
- [ ] Verdict states what the evidence does *not* cover
