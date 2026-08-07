# How this book is written

Rules I hold myself to, so that eighty-six notebooks read like one book instead of
eighty-six blog posts.

## Every notebook has the same five parts

1. **The idea**: what the method does, in plain language, with one picture. No
   equations yet.
2. **The maths**: written out, only what you need to understand the method. No
   step is skipped with "it can be shown that".
3. **From scratch**: a minimal NumPy implementation, checked against the library
   version so you can see that nothing is magic.
4. **In practice**: the scikit-learn or PyTorch version, annotated line by line,
   with the mistakes that are easy to make called out where they happen.
5. **When it wins, when it loses**: measured on the house datasets, with the
   reason explained. This section is the point of the book.

Every notebook opens with a table saying what you will learn, what you should
already know, which datasets it uses, and how long it runs. Every notebook closes
with a cheat sheet and a "what to remember" list.

## Code

- Runs top to bottom on a fresh kernel. No hidden state.
- One idea per cell. A cell doing three things gets split.
- Comments explain **why**, not what. `# scale before PCA, otherwise variance
  follows units` earns its place; `# fit the model` does not.
- Everything runs on a laptop CPU in minutes. A GPU only makes it faster.
- Seeds are set and stated.
- Shared code lives in `toolkit/`. A helper copied into a second notebook gets
  pulled up.
- Notebooks ship executed, so GitHub renders the charts without anyone installing
  anything.

## Charts

- One theme, from `toolkit/style.py`, so the whole book looks like one object.
- Titles state the finding, not the axis names: "Location outweighs income" beats
  "Coefficients by feature".
- Axes are labelled with units.
- Colour is never the only channel carrying meaning. The palette is Okabe-Ito and
  survives colour blindness and greyscale.
- Bars start at zero. If the interesting range does not include zero, it becomes a
  dot plot instead.
- Every figure is opened and read before it ships. A chart that does not clearly
  make its point gets redesigned, not recaptioned.

## Numbers

No number appears in prose unless the executed notebook printed it. When a result
contradicts what I expected, the text says so: the linear regression notebook
says plainly that I expected income to have the largest coefficient and it did
not. Being wrong in public and correcting it is more useful to a reader than a
tidy story.

## Prose

I write in first person, plainly, and I do not pad. Most of the list below comes
from [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing),
which I treat as a style guide rather than a detector, because the constructions
it names are weak writing regardless of who produced them.

**Words that get cut on sight**

> delve, showcase, testament, tapestry, landscape (as an abstract noun), robust,
> crucial, pivotal, meticulous, intricate, garner, foster, underscore, highlight
> (as a verb), bolster, boasts (meaning "has"), vibrant, enduring, align with,
> interplay, valuable insights, key (as a filler adjective)

**Constructions that get rewritten**

- `not just X, but Y` and `it's not X, it's Y`. Say the thing once.
- Copula avoidance: `serves as`, `stands as`, `represents`, `functions as` where
  `is` was the honest verb.
- Trailing participles that pretend to analyse: `..., highlighting the importance
  of ...`, `..., reflecting broader trends in ...`. Promote to a real sentence or delete.
- The rule of three used as padding.
- Significance inflation: `a pivotal moment`, `underscores its importance`.

**Formatting**

- Sentence case in headings, never Title Case.
- Bold marks a term being defined. It does not mark "key takeaways".
- No emoji as bullets or dividers.
- No bulleted lists shaped `- **Header**: sentence`. If items need headers they
  need a table or subsections.
- Straight quotes, never curly. No em dashes or en dashes anywhere in prose: a
  colon, a comma, a full stop or a bracket always says it more plainly, and the
  em dash is the most recognisable marker of machine-written text. Enforced by
  `tools/healthcheck.py`.

## Datasets

Open licences only, recorded in [`data/README.md`](data/README.md) with source and
retrieval date. The five house datasets are used everywhere so results stay
comparable across chapters. Nothing above 20 MB is committed.

---

Made by **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
